#!/usr/bin/env python3
"""
Calculate Study C (Longitudinal Drift) metrics from cleaned generations.

Metrics Calculated:
- Entity Recall @ T10: Percentage of Turn 1 entities still mentioned at Turn 10
- Entity Recall Curve: Recall at each turn
- Knowledge Conflict Rate: Self-contradiction detection
- Continuity Score: Consistency with treatment plan

Usage:
    python scripts/studies/study_c/metrics/calculate_metrics.py
    python scripts/studies/study_c/metrics/calculate_metrics.py --use-cleaned
    python scripts/studies/study_c/metrics/calculate_metrics.py --model qwen3-lmstudio
"""

import json
import re
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from dataclasses import dataclass, field
import logging
from datetime import datetime
import numpy as np

# Add src to path
# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from reliable_clinical_benchmark.metrics.drift import (
    _extract_advice,
    calculate_alignment_score,
)
from reliable_clinical_benchmark.utils.nli import NLIModel
from reliable_clinical_benchmark.utils.stats import bootstrap_confidence_interval

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from response text."""
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ============================================================
# ENTITY EXTRACTION
# ============================================================

try:
    from reliable_clinical_benchmark.utils.ner import MedicalNER
except ImportError as exc:
    raise ImportError(
        "Study C requires scispaCy (en_core_sci_sm). "
        "Install it with: python -m spacy download en_core_sci_sm"
    ) from exc




def calculate_entity_recall(reference_entities: Set[str], current_entities: Set[str]) -> float:
    """Calculate recall of reference entities in current response."""
    if not reference_entities:
        return 1.0  # No entities to track
    
    recalled = reference_entities.intersection(current_entities)
    return len(recalled) / len(reference_entities)


# ============================================================
# METRICS CALCULATION
# ============================================================

@dataclass
class DriftMetrics:
    """Study C drift metrics."""
    model: str
    total_cases: int
    usable_cases: int
    # Entity recall
    entity_recall_t1: float
    entity_recall_t5: float
    entity_recall_t10: float
    # Conflict detection
    knowledge_conflict_rate: float
    contradictions_found: int
    # Summary
    # Summary
    avg_turns_per_case: float
    # Optional field with default
    recall_curve: List[float] = None
    continuity_score: Optional[float] = None
    continuity_source: Optional[str] = None
    
    # Bootstrap data
    case_recall_t10_values: List[float] = field(default_factory=list)
    case_conflict_rates: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        if self.recall_curve is None:
            self.recall_curve = []


def calculate_metrics_for_model(
    model_name: str,
    generations_path: Path,
    gold_data: Dict[str, Dict],
    target_plans: Dict[str, str] = None,
    ner_model: MedicalNER = None,
    *,
    use_nli: bool = False,
    nli_model: Optional[NLIModel] = None,
    nli_stride: int = 2,
) -> DriftMetrics:
    """Calculate drift metrics for a single model."""
    
    # Use provided NER model or fallback (should be provided)
    if ner_model is None:
        try:
            ner_model = MedicalNER()
        except Exception as exc:
            raise RuntimeError(
                "Study C requires scispaCy (en_core_sci_sm). "
                "NER initialisation failed; install it with: "
                "python -m spacy download en_core_sci_sm"
            ) from exc

    
    # Load generations
    entries = []
    with open(generations_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    
    # Group by case ID and turn
    by_case: Dict[str, List[Dict]] = {}
    for e in entries:
        case_id = e.get("case_id", e.get("id", "").split("_")[0])
        by_case.setdefault(case_id, []).append(e)
    
    # Sort turns within each case
    # Robustified: Filter out cases where turn_idx is missing and sort strictly.
    # collapsing missing indices to 0 masks data bugs.
    filtered_by_case = {}
    for case_id, case_turns in by_case.items():
        # Ensure we have a valid sequence
        sorted_turns = sorted(
            [t for t in case_turns if t.get("turn_idx") is not None or t.get("turn_num") is not None],
            key=lambda x: x.get("turn_idx") if x.get("turn_idx") is not None else x.get("turn_num")
        )
        if not sorted_turns:
            logger.warning(f"Case {case_id} has no items with turn_idx/turn_num. Skipping.")
            continue
        filtered_by_case[case_id] = sorted_turns
    by_case = filtered_by_case
    
    # Calculate metrics
    total_cases = len(by_case)
    usable_cases = 0
    all_recall_curves = []
    all_conflicts = 0
    total_turn_pairs = 0
    all_continuity_scores = []
    continuity_sources: List[str] = []
    total_turns = 0
    
    # Check if we have gold data for case-level reference
    use_gold_reference = bool(gold_data)
    
    case_recall_t10_values = []
    case_conflict_rates = []

    assert nli_stride >= 1
    if nli_stride < 1:
        raise ValueError("nli_stride must be >= 1")

    if use_nli and nli_model is None:
        logger.warning("NLI requested but no model instance provided. Skipping knowledge conflict.")
        use_nli = False

    for case_id, all_turns in by_case.items():
        summary_turns = [
            t
            for t in all_turns
            if t.get("variant") == "summary" and (t.get("response_text") or t.get("output_text"))
        ]
        if len(summary_turns) < 2:
            logger.debug(f"Skipping case {case_id}: < 2 summary turns found")
            continue

        dialogue_turns = [
            t
            for t in all_turns
            if t.get("variant") == "dialogue" and (t.get("response_text") or t.get("output_text"))
        ]

        usable_cases += 1
        total_turns += len(summary_turns)
        
        # Reference Entities Logic:
        # 1. Try critical_entities from gold data
        # 2. Try patient_summary from gold data
        # 3. Fallback: First model summary (stripped)
        
        reference_entities = set()
        case_gold = gold_data.get(case_id, {})
        
        if use_gold_reference and case_gold:
            # Prio 1: critical_entities
            critical = case_gold.get("critical_entities", [])
            if critical:
                reference_entities = {str(e).lower() for e in critical}
            
            # Prio 2: extract from patient_summary if reference still small
            if len(reference_entities) < 3:
                summary_text = case_gold.get("patient_summary", "")
                if summary_text:
                    reference_entities.update(ner_model.extract_entities(summary_text))
        
        # Prio 3 Fallback: use first model summary if no gold or gold failed
        if not reference_entities:
            turn1_raw = summary_turns[0].get("response_text", "") or summary_turns[0].get("output_text", "")
            turn1_clean = strip_thinking(turn1_raw)
            reference_entities = ner_model.extract_entities(turn1_clean)
        
        # Calculate recall at each turn
        recall_curve = []
        prev_response = "" # Used for contradiction detection
        
        case_conflicts = 0
        case_turn_pairs = 0
        
        for i, turn in enumerate(summary_turns):
            curr_raw = turn.get("response_text", "") or turn.get("output_text", "")
            curr_clean = strip_thinking(curr_raw)
            
            # Prefer pre-extracted entities from enrichment pipeline
            if "entities" in turn and turn["entities"]:
                curr_entities = {str(e).lower() for e in turn["entities"]}
            else:
                curr_entities = ner_model.extract_entities(curr_clean)
            
            recall = calculate_entity_recall(reference_entities, curr_entities)
            recall_curve.append(recall)
            
            prev_response = curr_clean
        
        all_recall_curves.append(recall_curve)
        
        # Bootstrap collection
        if recall_curve:
            case_recall_t10_values.append(recall_curve[-1])
        else:
             case_recall_t10_values.append(0.0)
             
        if use_nli:
            if dialogue_turns:
                previous_advice = ""
                pair_index = 0
                for dialogue_turn in dialogue_turns:
                    dialogue_raw = (
                        dialogue_turn.get("response_text", "")
                        or dialogue_turn.get("output_text", "")
                    )
                    dialogue_clean = strip_thinking(dialogue_raw)
                    current_advice = _extract_advice(dialogue_clean)
                    if previous_advice and current_advice:
                        if pair_index % nli_stride == 0:
                            total_turn_pairs += 1
                            case_turn_pairs += 1
                            verdict = nli_model.predict(
                                premise=previous_advice, hypothesis=current_advice
                            )
                            if verdict == "contradiction":
                                all_conflicts += 1
                                case_conflicts += 1
                        pair_index += 1
                    previous_advice = current_advice
            if case_turn_pairs > 0:
                case_conflict_rates.append(case_conflicts / case_turn_pairs)
            else:
                case_conflict_rates.append(0.0)
        
        # Calculate Continuity Score
        if target_plans and case_id in target_plans:
            action_turns = dialogue_turns if dialogue_turns else summary_turns
            action_source = "dialogue" if dialogue_turns else "summary"
            model_actions = [
                turn.get("response_text", "") or turn.get("output_text", "")
                for turn in action_turns
            ]
            plan = target_plans[case_id].get("plan", "")
            if plan and model_actions:
                c_score = calculate_alignment_score(model_actions, plan, mode="actions")
                if c_score is not None:
                    all_continuity_scores.append(c_score)
                    continuity_sources.append(action_source)
    
    # Aggregate recall curves (pad shorter ones)
    max_turns = max(len(c) for c in all_recall_curves) if all_recall_curves else 0
    avg_recall_curve = []
    
    for t in range(max_turns):
        recalls_at_t = [c[t] for c in all_recall_curves if len(c) > t]
        avg_recall = sum(recalls_at_t) / len(recalls_at_t) if recalls_at_t else 0.0
        avg_recall_curve.append(avg_recall)
    
    # Get key recall points
    recall_t1 = avg_recall_curve[0] if len(avg_recall_curve) > 0 else 0.0
    recall_t5 = avg_recall_curve[4] if len(avg_recall_curve) > 4 else avg_recall_curve[-1] if avg_recall_curve else 0.0
    recall_t10 = avg_recall_curve[9] if len(avg_recall_curve) > 9 else avg_recall_curve[-1] if avg_recall_curve else 0.0
    
    # Calculate rates
    conflict_rate = all_conflicts / total_turn_pairs if total_turn_pairs > 0 else 0.0
    avg_turns = total_turns / usable_cases if usable_cases > 0 else 0.0
    
    # Calculate continuity score
    avg_continuity = None
    continuity_source = None
    if all_continuity_scores:
        avg_continuity = sum(all_continuity_scores) / len(all_continuity_scores)
        if "dialogue" in continuity_sources:
            continuity_source = "dialogue"
        elif "summary" in continuity_sources:
            continuity_source = "summary"

    return DriftMetrics(
        model=model_name,
        total_cases=total_cases,
        usable_cases=usable_cases,
        entity_recall_t1=recall_t1,
        entity_recall_t5=recall_t5,
        entity_recall_t10=recall_t10,
        recall_curve=avg_recall_curve,
        knowledge_conflict_rate=conflict_rate,
        contradictions_found=all_conflicts,
        avg_turns_per_case=avg_turns,
        continuity_score=avg_continuity,
        continuity_source=continuity_source,
        
        # Bootstrap data
        case_recall_t10_values=case_recall_t10_values,
        case_conflict_rates=case_conflict_rates,
    )


def load_gold_data(data_dir: Path) -> Dict[str, Dict]:
    """Load gold data for Study C."""
    gold_data = {}
    
    possible_paths = [
        data_dir / "study_c_test.json",
        data_dir / "openr1_psy_splits" / "study_c_test.json",
        data_dir / "study_c.json",
    ]
    
    for path in possible_paths:
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        gold_data[item.get("id", item.get("case_id", ""))] = item
                elif isinstance(data, dict):
                    if isinstance(data.get("cases"), list):
                        for item in data["cases"]:
                            gold_data[item.get("id", item.get("case_id", ""))] = item
                    else:
                        gold_data = data
            logger.info(f"Loaded gold data from {path}: {len(gold_data)} entries")
            break
    
    return gold_data


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate Study C (Drift) metrics")
    parser.add_argument("--use-cleaned", action="store_true",
                        help="Use cleaned generations instead of raw")
    parser.add_argument("--model", type=str, help="Process specific model only")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for results")
    parser.add_argument(
        "--use-nli",
        action="store_true",
        help="Use NLI for knowledge conflict detection (actions-only advice)",
    )
    parser.add_argument(
        "--nli-model",
        type=str,
        default="cross-encoder/nli-deberta-v3-base",
        help="NLI model name for contradiction detection",
    )
    parser.add_argument(
        "--nli-stride",
        type=int,
        default=2,
        help="Evaluate every Nth advice pair for knowledge conflict (default: 2)",
    )
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent.parent.parent.parent
    data_dir = base_dir / "data"
    
    if args.use_cleaned:
        results_dir = base_dir / "processed" / "study_c_pipeline"
    else:
        results_dir = base_dir / "results"
    
    output_dir = args.output_dir or (base_dir / "metric-results" / "study_c")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("STUDY C: LONGITUDINAL DRIFT METRICS")
    print("=" * 60)
    print(f"Source:   {results_dir}")
    print(f"Output:   {output_dir}")
    print(f"Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Load gold data
    gold_data = load_gold_data(data_dir)
    if not gold_data:
        logger.warning("No gold data found - using empty gold reference")
        
    # Load target plans
    target_plans = {}
    target_plans_path = data_dir / "study_c_gold" / "target_plans.json"
    if target_plans_path.exists():
        with open(target_plans_path, 'r', encoding='utf-8') as f:
            tp_data = json.load(f)
            target_plans = tp_data.get("plans", {})
        logger.info(f"Loaded {len(target_plans)} target plans from {target_plans_path}")
    else:
        logger.warning(f"Target plans not found at {target_plans_path}. Continuity score will be skipped.")
    
    # Find models
    models = [d.name for d in results_dir.iterdir() if d.is_dir()]
    if args.model:
        models = [m for m in models if m == args.model]
    
    # Initialize NER model once
    print("Loading MedicalNER (scispaCy)...")
    ner_model = MedicalNER()

    all_results = []
    nli_model = None
    if args.use_nli:
        try:
            nli_model = NLIModel(model_name=args.nli_model)
        except Exception as e:
            logger.warning(f"Failed to load NLI model ({args.nli_model}): {e}")
            nli_model = None
    
    for model in sorted(models):
        if args.use_cleaned:
            gen_path = results_dir / model / "study_c_processed.jsonl"
        else:
            gen_path = results_dir / model / "study_c_generations.jsonl"
        if not gen_path.exists():
            continue
        
        print(f"\nProcessing {model}...")
        metrics = calculate_metrics_for_model(
            model,
            gen_path,
            gold_data,
            target_plans,
            ner_model,
            use_nli=args.use_nli,
            nli_model=nli_model,
            nli_stride=args.nli_stride,
        )
        all_results.append(metrics)
        
        print(f"  Cases: {metrics.usable_cases}/{metrics.total_cases}")
        print(f"  Entity Recall @T10: {metrics.entity_recall_t10:.3f}")
        print(f"  Conflict Rate: {metrics.knowledge_conflict_rate:.3f}")
        if metrics.continuity_score is not None:
            print(f"  Continuity Score: {metrics.continuity_score:.3f}")
    
    # Compute CIs
    final_output = []
    
    def calc_mean(items):
        return sum(items) / len(items) if items else 0.0

    for m in all_results:
        # T10 Recall CI
        t10_low, t10_high = 0.0, 0.0
        if m.case_recall_t10_values:
            _, t10_low, t10_high = bootstrap_confidence_interval(m.case_recall_t10_values, statistic_fn=np.mean)
            
        # Conflict Rate CI
        conf_low, conf_high = 0.0, 0.0
        if m.case_conflict_rates:
             _, conf_low, conf_high = bootstrap_confidence_interval(m.case_conflict_rates, statistic_fn=np.mean)
        
        final_output.append({
            "model": m.model,
            "total_cases": m.total_cases,
            "usable_cases": m.usable_cases,
            "entity_recall_t1": m.entity_recall_t1,
            "entity_recall_t5": m.entity_recall_t5,
            "entity_recall_t10": m.entity_recall_t10,
            "entity_recall_t10_ci_low": round(t10_low, 4),
            "entity_recall_t10_ci_high": round(t10_high, 4),
            "recall_curve": m.recall_curve,
            "knowledge_conflict_rate": m.knowledge_conflict_rate,
            "knowledge_conflict_rate_ci_low": round(conf_low, 4),
            "knowledge_conflict_rate_ci_high": round(conf_high, 4),
            "contradictions_found": m.contradictions_found,
            "avg_turns_per_case": m.avg_turns_per_case,
            "continuity_score": m.continuity_score,
            "continuity_source": m.continuity_source,
        })

    results_file = output_dir / "drift_metrics.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<26} {'Recall@T10':>10} {'Conflict':>10} {'Cases':>8}")
    print("-" * 56)
    for m in all_results:
        status = "✅" if m.entity_recall_t10 > 0.7 else "⚠️" if m.entity_recall_t10 > 0.5 else "❌"
        print(f"{m.model:<26} {m.entity_recall_t10:>10.3f} {m.knowledge_conflict_rate:>10.3f} {m.usable_cases:>8} {status}")
    
    print("=" * 60)
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()

