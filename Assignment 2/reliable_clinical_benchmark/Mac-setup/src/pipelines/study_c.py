"""Study C: Longitudinal Drift Evaluation Pipeline."""

import json
from pathlib import Path
from typing import Optional
import logging

from models.base import ModelRunner
from metrics.drift import (
    compute_entity_recall_curve,
    calculate_knowledge_conflict_rate,
    calculate_continuity_score,
    compute_drift_slope,
    DriftResult,
)
from data.study_c_loader import load_study_c_data
from utils.ner import MedicalNER
from utils.nli import NLIModel
from utils.stats import bootstrap_confidence_interval

logger = logging.getLogger(__name__)


def run_study_c(
    model: ModelRunner,
    data_dir: str = "data/openr1_psy_splits",
    max_cases: Optional[int] = None,
    output_dir: str = "results",
    model_name: str = "unknown",
    use_nli: bool = True,
) -> DriftResult:
    """
    Run Study C longitudinal drift evaluation.

    Args:
        model: ModelRunner instance
        data_dir: Directory containing study_c_test.json
        max_cases: Maximum number of cases to evaluate (None = all)
        output_dir: Directory to save results
        model_name: Name of the model being evaluated
        use_nli: Whether to compute knowledge conflict (requires NLI model)

    Returns:
        DriftResult with all metrics
    """
    logger.info(f"Starting Study C evaluation for {model_name}")

    # Load data
    study_c_path = Path(data_dir) / "study_c_test.json"
    cases = load_study_c_data(str(study_c_path))

    if max_cases:
        cases = cases[:max_cases]
        logger.info(f"Limited to {max_cases} cases")

    if not cases:
        logger.error("No Study C data loaded. Check data paths.")
        return DriftResult(
            entity_recall_at_t10=0.0,
            knowledge_conflict_rate=0.0,
            continuity_score=0.0,
            n_cases=0,
        )

    # Initialise NER
    try:
        ner = MedicalNER()
    except Exception as e:
        logger.error(f"Failed to load NER model: {e}")
        return DriftResult(
            entity_recall_at_t10=0.0,
            knowledge_conflict_rate=0.0,
            continuity_score=0.0,
            n_cases=0,
        )

    # Compute entity recall curves
    all_recalls_at_t10 = []
    all_recall_curves = []

    for case in cases:
        try:
            recall_curve = compute_entity_recall_curve(model, case, ner)
            if recall_curve:
                all_recall_curves.append(recall_curve)
                # Get recall at turn 10 (or last turn if fewer than 10)
                recall_at_t10 = (
                    recall_curve[9] if len(recall_curve) > 9 else recall_curve[-1]
                )
                all_recalls_at_t10.append(recall_at_t10)
        except Exception as e:
            logger.warning(f"Entity recall calculation failed for case {case.id}: {e}")

    mean_recall_at_t10 = (
        sum(all_recalls_at_t10) / len(all_recalls_at_t10)
        if all_recalls_at_t10
        else 0.0
    )

    # Calculate knowledge conflict rate
    k_conflict = 0.0
    if use_nli:
        try:
            nli_model = NLIModel()
            k_conflict = calculate_knowledge_conflict_rate(model, cases, nli_model)
        except Exception as e:
            logger.warning(f"NLI model not available, skipping knowledge conflict: {e}")

    # Calculate continuity score (placeholder - requires target plans)
    continuity_score = 0.0
    # Note: This would require target_plan data in the case structure
    # For now, we skip it or use a placeholder

    result = DriftResult(
        entity_recall_at_t10=mean_recall_at_t10,
        knowledge_conflict_rate=k_conflict,
        continuity_score=continuity_score,
        n_cases=len(cases),
    )

    # Save results
    output_path = Path(output_dir) / model_name / "study_c_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_dict = {
        "model": model_name,
        "study": "C",
        "entity_recall_at_t10": mean_recall_at_t10,
        "knowledge_conflict_rate": k_conflict,
        "continuity_score": continuity_score,
        "n_cases": len(cases),
    }

    # Add bootstrap CIs if we have enough cases
    if len(all_recalls_at_t10) > 10:
        recall_ci = bootstrap_confidence_interval(all_recalls_at_t10)
        result_dict["entity_recall_ci"] = {
            "lower": recall_ci[1],
            "upper": recall_ci[2],
        }

    # Save average recall curve
    if all_recall_curves:
        # Average across all cases
        max_turns = max(len(curve) for curve in all_recall_curves)
        avg_curve = []
        for turn_idx in range(max_turns):
            turn_recalls = [
                curve[turn_idx] for curve in all_recall_curves if len(curve) > turn_idx
            ]
            if turn_recalls:
                avg_curve.append(sum(turn_recalls) / len(turn_recalls))
        result_dict["average_recall_curve"] = avg_curve

    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent=2)

    logger.info(f"Study C results saved to {output_path}")
    logger.info(
        f"Entity Recall (T=10): {mean_recall_at_t10:.3f}, "
        f"Knowledge Conflict: {k_conflict:.3f}"
    )

    return result

