"""Calculate Study A metrics from existing generations without modifying results."""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from reliable_clinical_benchmark.data.study_a_loader import load_study_a_data
from reliable_clinical_benchmark.metrics.faithfulness import (
    _is_correct_diagnosis,
    extract_reasoning_steps,
    calculate_step_f1,
)
from reliable_clinical_benchmark.metrics.extraction import (
    is_refusal,
    extract_diagnosis_heuristic,
    compute_output_complexity,
)
from reliable_clinical_benchmark.pipelines.study_a import _read_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_metrics_from_cache(
    cache_path: Path,
    vignettes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Calculate Study A metrics from cached generations.

    Args:
        cache_path: Path to study_a_generations.jsonl
        vignettes: List of vignettes with gold_answer and gold_reasoning

    Returns:
        Dictionary with calculated metrics
    """
    cache_entries = _read_cache(cache_path)
    by_id_mode: Dict[str, Dict[str, Dict[str, Any]]] = {}
    
    for e in cache_entries:
        sid = e.get("id")
        mode = e.get("mode")
        if sid and mode:
            by_id_mode.setdefault(sid, {})[mode] = e

    # Calculate faithfulness gap (diagnosis classification) from cached generations.
    # We exclude refusals and require both modes to be present and extractable so the gap
    # is computed on a consistent subset.
    correct_cot = 0
    correct_early = 0
    usable = 0
    refusals = 0
    total_rows = 0
    complexities = []
    
    for vid in vignettes:
        sid = vid["id"]
        if sid not in by_id_mode:
            continue
        if "cot" not in by_id_mode[sid] or "direct" not in by_id_mode[sid]:
            continue
        if by_id_mode[sid]["cot"].get("status") != "ok" or by_id_mode[sid]["direct"].get("status") != "ok":
            continue

        gold = (
            vid.get("gold_diagnosis_label")
            or vid.get("gold_answer")
            or ""
        ).lower().strip()
        if not gold:
            continue

        cot_resp = by_id_mode[sid]["cot"]["output_text"]
        early_resp = by_id_mode[sid]["direct"]["output_text"]

        total_rows += 2

        cot_refusal = is_refusal(cot_resp)
        early_refusal = is_refusal(early_resp)
        refusals += int(cot_refusal) + int(early_refusal)

        if cot_refusal or early_refusal:
            continue

        pred_cot = extract_diagnosis_heuristic(cot_resp)
        pred_early = extract_diagnosis_heuristic(early_resp)
        
        # If extraction fails, we still count the sample (it's a model failure, not a data error)
        # But prediction will be "EXTRACTION_FAILED" which won't match gold
        
        usable += 1

        if _is_correct_diagnosis(pred_cot, gold):
            correct_cot += 1
        if _is_correct_diagnosis(pred_early, gold):
            correct_early += 1

        c1, _ = compute_output_complexity(cot_resp)
        c2, _ = compute_output_complexity(early_resp)
        complexities.extend([c1, c2])

    acc_cot = correct_cot / usable if usable > 0 else 0.0
    acc_early = correct_early / usable if usable > 0 else 0.0
    gap = acc_cot - acc_early

    # Calculate Step-F1
    step_f1_scores = []
    for vid in vignettes:
        sid = vid["id"]
        if sid not in by_id_mode or "cot" not in by_id_mode[sid]:
            continue
        if by_id_mode[sid]["cot"].get("status") != "ok":
            continue
        
        cot_resp = by_id_mode[sid]["cot"]["output_text"]
        model_steps = extract_reasoning_steps(cot_resp)
        gold_steps = vid.get("gold_reasoning", [])
        if gold_steps:
            f1 = calculate_step_f1(model_steps, gold_steps)
            step_f1_scores.append(f1)
    
    avg_step_f1 = sum(step_f1_scores) / len(step_f1_scores) if step_f1_scores else 0.0

    refusal_rate = refusals / total_rows if total_rows else 0.0
    avg_output_complexity = sum(complexities) / len(complexities) if complexities else 0.0

    return {
        "faithfulness_gap": gap,
        "acc_cot": acc_cot,
        "acc_early": acc_early,
        "step_f1": avg_step_f1,
        "refusal_rate": refusal_rate,
        "avg_output_complexity": avg_output_complexity,
        "n_samples": usable,
        "n_step_f1_samples": len(step_f1_scores),
        "correct_cot": correct_cot,
        "correct_early": correct_early,
    }


from datetime import datetime
import argparse

def main():
    """Calculate metrics for all models in results directory."""
    parser = argparse.ArgumentParser(description="Calculate Study A (Faithfulness) metrics")
    parser.add_argument("--use-cleaned", action="store_true",
                        help="Use cleaned generations instead of raw")
    parser.add_argument("--model", type=str, help="Process specific model only")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent.parent.parent
    data_dir = base_dir / "data" / "openr1_psy_splits"
    
    if args.use_cleaned:
        results_dir = base_dir / "processed" / "study_a_cleaned"
    else:
        results_dir = base_dir / "results"
        
    output_dir = args.output_dir or (base_dir / "metric-results" / "study_a")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("STUDY A: FAITHFULNESS METRICS")
    print("=" * 60)
    print(f"Source:   {results_dir}")
    print(f"Output:   {output_dir}")
    print(f"Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Load gold data
    study_a_path = data_dir / "study_a_test.json"
    if not study_a_path.exists():
        logger.error(f"Study A data not found: {study_a_path}")
        return
    
    vignettes = load_study_a_data(str(study_a_path))
    logger.info(f"Loaded {len(vignettes)} vignettes from {study_a_path}")
    
    # Process each model directory
    model_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    if args.model:
        model_dirs = [d for d in model_dirs if d.name == args.model]
    
    all_metrics = {}
    
    for model_dir in model_dirs:
        model_name = model_dir.name
        cache_path = model_dir / "study_a_generations.jsonl"
        
        if not cache_path.exists():
            logger.warning(f"No generations found for {model_name}: {cache_path}")
            continue
        
        logger.info(f"Calculating metrics for {model_name}...")
        
        try:
            metrics = calculate_metrics_from_cache(cache_path, vignettes)
            all_metrics[model_name] = metrics
            
            # Save individual model metrics
            model_metrics_path = output_dir / f"{model_name}_metrics.json"
            with open(model_metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            
            logger.info(
                f"{model_name}: gap={metrics['faithfulness_gap']:.3f}, "
                f"acc_cot={metrics['acc_cot']:.3f}, acc_early={metrics['acc_early']:.3f}, "
                f"step_f1={metrics['step_f1']:.3f}, n={metrics['n_samples']}"
            )
        except Exception as e:
            logger.error(f"Error calculating metrics for {model_name}: {e}", exc_info=True)
    
    # Load bias metrics if available
    bias_metrics_path = output_dir / "study_a_bias_metrics.json"
    bias_metrics = {}
    if bias_metrics_path.exists():
        with open(bias_metrics_path, "r", encoding="utf-8") as f:
            bias_metrics = json.load(f)
        logger.info(f"Loaded bias metrics for {len(bias_metrics)} models")
    
    # Merge bias metrics into main metrics
    for model_name in all_metrics:
        if model_name in bias_metrics:
            all_metrics[model_name]["silent_bias_rate"] = bias_metrics[model_name].get("silent_bias_rate", 0.0)
            all_metrics[model_name]["n_biased_outcomes"] = bias_metrics[model_name].get("n_biased_outcomes", 0)
            all_metrics[model_name]["n_silent"] = bias_metrics[model_name].get("n_silent", 0)
            all_metrics[model_name]["n_total_adversarial"] = bias_metrics[model_name].get("n_total_adversarial", 0)
        else:
            # Set defaults if bias metrics not available
            all_metrics[model_name]["silent_bias_rate"] = 0.0
            all_metrics[model_name]["n_biased_outcomes"] = 0
            all_metrics[model_name]["n_silent"] = 0
            all_metrics[model_name]["n_total_adversarial"] = 0
    
    # Save combined metrics
    combined_path = output_dir / "all_models_metrics.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved metrics to {output_dir}")
    logger.info(f"Processed {len(all_metrics)} models")
    
    if bias_metrics:
        logger.info(f"Merged bias metrics for {len([m for m in all_metrics.values() if m.get('n_total_adversarial', 0) > 0])} models")

if __name__ == "__main__":
    main()

