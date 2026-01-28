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

# Add src to path
# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from reliable_clinical_benchmark.metrics.drift import calculate_continuity_score

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

from reliable_clinical_benchmark.utils.ner import MedicalNER




def calculate_entity_recall(reference_entities: Set[str], current_entities: Set[str]) -> float:
    """Calculate recall of reference entities in current response."""
    if not reference_entities:
        return 1.0  # No entities to track
    
    recalled = reference_entities.intersection(current_entities)
    return len(recalled) / len(reference_entities)


# ============================================================
# KNOWLEDGE CONFLICT DETECTION
# ============================================================

CONTRADICTION_PAIRS = [
    ("recommend", "not recommend"),
    ("should", "should not"),
    ("increase", "decrease"),
    ("start", "stop"),
    ("continue", "discontinue"),
    ("beneficial", "harmful"),
    ("safe", "unsafe"),
    ("effective", "ineffective"),
]


def detect_contradiction(prev_response: str, curr_response: str) -> bool:
    """Detect if current response contradicts previous response."""
    if not prev_response or not curr_response:
        return False
    
    prev_lower = prev_response.lower()
    curr_lower = curr_response.lower()
    
    for pos, neg in CONTRADICTION_PAIRS:
        # Check if previous says positive and current says negative (or vice versa)
        prev_has_pos = pos in prev_lower and neg not in prev_lower
        curr_has_neg = neg in curr_lower and pos not in curr_lower
        
        prev_has_neg = neg in prev_lower and pos not in prev_lower
        curr_has_pos = pos in curr_lower and neg not in curr_lower
        
        if (prev_has_pos and curr_has_neg) or (prev_has_neg and curr_has_pos):
            return True
    
    return False


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
    
    def __post_init__(self):
        if self.recall_curve is None:
            self.recall_curve = []


def calculate_metrics_for_model(
    model_name: str,
    generations_path: Path,
    gold_data: Dict[str, Dict],
    target_plans: Dict[str, str] = None,
    ner_model: MedicalNER = None,
) -> DriftMetrics:
    """Calculate drift metrics for a single model."""
    
    # Use provided NER model or fallback (should be provided)
    if ner_model is None:
        ner_model = MedicalNER()

    
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
    for case_id in by_case:
        by_case[case_id].sort(key=lambda x: x.get("turn_idx", x.get("turn", 0)))
    
    # Calculate metrics
    total_cases = len(by_case)
    usable_cases = 0
    all_recall_curves = []
    all_conflicts = 0
    total_turn_pairs = 0
    all_conflicts = 0
    total_turn_pairs = 0
    total_turns = 0
    all_continuity_scores = []
    
    # Check if we have gold data for case-level reference
    use_gold_reference = bool(gold_data)
    
    for case_id, all_turns in by_case.items():
        # FILTER: Only use 'summary' variant for consistent drift tracking
        # Also filter out turns without content
        turns = [
            t for t in all_turns 
            if t.get("variant") == "summary" and (t.get("response_text") or t.get("output_text"))
        ]
        
        if len(turns) < 2:
            logger.debug(f"Skipping case {case_id}: < 2 summary turns found")
            continue
        
        usable_cases += 1
        total_turns += len(turns)
        
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
            turn1_raw = turns[0].get("response_text", "") or turns[0].get("output_text", "")
            turn1_clean = strip_thinking(turn1_raw)
            reference_entities = ner_model.extract_entities(turn1_clean)
        
        # Calculate recall at each turn
        recall_curve = []
        prev_response = "" # Used for contradiction detection
        
        for i, turn in enumerate(turns):
            curr_raw = turn.get("response_text", "") or turn.get("output_text", "")
            curr_clean = strip_thinking(curr_raw)
            curr_entities = ner_model.extract_entities(curr_clean)
            
            recall = calculate_entity_recall(reference_entities, curr_entities)
            recall_curve.append(recall)
            
            # Check for contradictions with previous turn
            if i > 0 and prev_response:
                total_turn_pairs += 1
                if detect_contradiction(prev_response, curr_clean):
                    all_conflicts += 1
            
            prev_response = curr_clean
        
        all_recall_curves.append(recall_curve)
        
        # Calculate Continuity Score
        if target_plans and case_id in target_plans:
            # Extract clinical actions/advice from all turns
            # We use the raw text as the 'action' for simplicity, or could extract advice
            model_actions = [
                turn.get("response_text", "") or turn.get("output_text", "") 
                for turn in turns
            ]
            plan = target_plans[case_id].get("plan", "")
            if plan:
                c_score = calculate_continuity_score(model_actions, plan)
                if c_score is not None:
                    all_continuity_scores.append(c_score)
    
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
    if all_continuity_scores:
        avg_continuity = sum(all_continuity_scores) / len(all_continuity_scores)

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
    
    for model in sorted(models):
        if args.use_cleaned:
            gen_path = results_dir / model / "study_c_processed.jsonl"
        else:
            gen_path = results_dir / model / "study_c_generations.jsonl"
        if not gen_path.exists():
            continue
        
        print(f"\nProcessing {model}...")
        metrics = calculate_metrics_for_model(model, gen_path, gold_data, target_plans, ner_model)
        all_results.append(metrics)
        
        print(f"  Cases: {metrics.usable_cases}/{metrics.total_cases}")
        print(f"  Entity Recall @T10: {metrics.entity_recall_t10:.3f}")
        print(f"  Conflict Rate: {metrics.knowledge_conflict_rate:.3f}")
        if metrics.continuity_score is not None:
            print(f"  Continuity Score: {metrics.continuity_score:.3f}")
    
    # Save results
    results_file = output_dir / "drift_metrics.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump([{
            "model": m.model,
            "total_cases": m.total_cases,
            "usable_cases": m.usable_cases,
            "entity_recall_t1": m.entity_recall_t1,
            "entity_recall_t5": m.entity_recall_t5,
            "entity_recall_t10": m.entity_recall_t10,
            "recall_curve": m.recall_curve,
            "knowledge_conflict_rate": m.knowledge_conflict_rate,
            "contradictions_found": m.contradictions_found,
            "avg_turns_per_case": m.avg_turns_per_case,
            "continuity_score": m.continuity_score,
        } for m in all_results], f, indent=2)
    
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


