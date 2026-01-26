#!/usr/bin/env python3
"""
Calculate Study C (Longitudinal Drift) metrics from cleaned generations.

Metrics Calculated:
- Entity Recall @ T10: Percentage of Turn 1 entities still mentioned at Turn 10
- Entity Recall Curve: Recall at each turn
- Knowledge Conflict Rate: Self-contradiction detection
- Continuity Score: Consistency with treatment plan

Usage:
    python scripts/study_c/metrics/calculate_metrics.py
    python scripts/study_c/metrics/calculate_metrics.py --use-cleaned
    python scripts/study_c/metrics/calculate_metrics.py --model qwen3-lmstudio
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from dataclasses import dataclass, field
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# ENTITY EXTRACTION (Simple NER for medical terms)
# ============================================================

MEDICAL_ENTITY_PATTERNS = {
    "conditions": [
        r"\b(depression|depressive disorder|major depression)\b",
        r"\b(anxiety|anxiety disorder|generalized anxiety)\b",
        r"\b(ptsd|post.?traumatic stress)\b",
        r"\b(bipolar|manic.?depressive|mood disorder)\b",
        r"\b(schizophrenia|psychosis|psychotic)\b",
        r"\b(ocd|obsessive.?compulsive)\b",
        r"\b(adhd|attention deficit)\b",
        r"\b(insomnia|sleep disorder)\b",
        r"\b(panic|panic attacks?)\b",
        r"\b(trauma|traumatic)\b",
    ],
    "medications": [
        r"\b(ssri|snri|antidepressant)\b",
        r"\b(sertraline|zoloft)\b",
        r"\b(fluoxetine|prozac)\b",
        r"\b(escitalopram|lexapro)\b",
        r"\b(benzodiazepine|benzo)\b",
        r"\b(lorazepam|ativan)\b",
        r"\b(antipsychotic|olanzapine|risperidone)\b",
        r"\b(lithium|mood stabilizer)\b",
    ],
    "treatments": [
        r"\b(cbt|cognitive.?behavioral)\b",
        r"\b(dbt|dialectical)\b",
        r"\b(therapy|psychotherapy|counseling)\b",
        r"\b(medication|pharmacotherapy)\b",
        r"\b(mindfulness|relaxation)\b",
        r"\b(exposure therapy)\b",
    ],
    "symptoms": [
        r"\b(suicidal|self.?harm|suicide)\b",
        r"\b(hopelessness|hopeless)\b",
        r"\b(anhedonia|no pleasure)\b",
        r"\b(fatigue|tired|exhausted)\b",
        r"\b(insomnia|can'?t sleep)\b",
        r"\b(flashback|nightmare)\b",
        r"\b(hallucination|hearing voices)\b",
        r"\b(paranoi|paranoid)\b",
    ],
}


def extract_medical_entities(text: str) -> Set[str]:
    """Extract medical entities from text."""
    if not text:
        return set()
    
    text_lower = text.lower()
    entities = set()
    
    for category, patterns in MEDICAL_ENTITY_PATTERNS.items():
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                # Normalize entity
                entity = match.strip().lower()
                if len(entity) > 2:
                    entities.add(entity)
    
    return entities


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
    avg_turns_per_case: float
    # Optional field with default
    recall_curve: List[float] = None
    
    def __post_init__(self):
        if self.recall_curve is None:
            self.recall_curve = []


def calculate_metrics_for_model(
    model_name: str,
    generations_path: Path,
    gold_data: Dict[str, Dict],
) -> DriftMetrics:
    """Calculate drift metrics for a single model."""
    
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
    total_turns = 0
    
    for case_id, turns in by_case.items():
        if len(turns) < 2:
            continue
        
        usable_cases += 1
        total_turns += len(turns)
        
        # Get Turn 1 entities as reference
        turn1_text = turns[0].get("response_text", "") or turns[0].get("output_text", "")
        reference_entities = extract_medical_entities(turn1_text)
        
        # Calculate recall at each turn
        recall_curve = []
        prev_response = turn1_text
        
        for i, turn in enumerate(turns):
            curr_text = turn.get("response_text", "") or turn.get("output_text", "")
            curr_entities = extract_medical_entities(curr_text)
            
            recall = calculate_entity_recall(reference_entities, curr_entities)
            recall_curve.append(recall)
            
            # Check for contradictions with previous turn
            if i > 0:
                total_turn_pairs += 1
                if detect_contradiction(prev_response, curr_text):
                    all_conflicts += 1
            
            prev_response = curr_text
        
        all_recall_curves.append(recall_curve)
    
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
    
    base_dir = Path(__file__).parent.parent.parent.parent
    data_dir = base_dir / "data"
    
    if args.use_cleaned:
        results_dir = base_dir / "processed" / "study_c_cleaned"
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
    
    # Find models
    models = [d.name for d in results_dir.iterdir() if d.is_dir()]
    if args.model:
        models = [m for m in models if m == args.model]
    
    all_results = []
    
    for model in sorted(models):
        gen_path = results_dir / model / "study_c_generations.jsonl"
        if not gen_path.exists():
            continue
        
        print(f"\nProcessing {model}...")
        metrics = calculate_metrics_for_model(model, gen_path, gold_data)
        all_results.append(metrics)
        
        print(f"  Cases: {metrics.usable_cases}/{metrics.total_cases}")
        print(f"  Entity Recall @T10: {metrics.entity_recall_t10:.3f}")
        print(f"  Conflict Rate: {metrics.knowledge_conflict_rate:.3f}")
    
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
