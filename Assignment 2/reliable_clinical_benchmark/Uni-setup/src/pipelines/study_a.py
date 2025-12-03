"""Study A: Faithfulness Evaluation Pipeline."""

import json
from pathlib import Path
from typing import Optional
import logging

from ..models.base import ModelRunner
from ..metrics.faithfulness import (
    calculate_faithfulness_gap,
    calculate_step_f1,
    calculate_silent_bias_rate,
    extract_reasoning_steps,
    FaithfulnessResult,
)
from ..data.study_a_loader import load_study_a_data
from ..data.adversarial_loader import load_adversarial_bias_cases
from ..utils.stats import bootstrap_confidence_interval

logger = logging.getLogger(__name__)


def run_study_a(
    model: ModelRunner,
    data_dir: str = "data/openr1_psy_splits",
    adversarial_data_path: str = "data/adversarial_bias/biased_vignettes.json",
    max_samples: Optional[int] = None,
    output_dir: str = "results",
    model_name: str = "unknown",
) -> FaithfulnessResult:
    """
    Run Study A faithfulness evaluation.
    """
    logger.info(f"Starting Study A evaluation for {model_name}")

    study_a_path = Path(data_dir) / "study_a_test.json"
    vignettes = load_study_a_data(str(study_a_path))

    if max_samples:
        vignettes = vignettes[:max_samples]
        logger.info(f"Limited to {max_samples} samples")

    if not vignettes:
        logger.error("No Study A data loaded. Check data paths.")
        return FaithfulnessResult(
            faithfulness_gap=0.0,
            acc_cot=0.0,
            acc_early=0.0,
            step_f1=0.0,
            silent_bias_rate=0.0,
            n_samples=0,
        )

    gap, acc_cot, acc_early = calculate_faithfulness_gap(model, vignettes)

    step_f1_scores = []
    for vignette in vignettes:
        try:
            resp_cot = model.generate(vignette["prompt"], mode="cot")
            model_steps = extract_reasoning_steps(resp_cot)
            gold_steps = vignette.get("gold_reasoning", [])
            if gold_steps:
                f1 = calculate_step_f1(model_steps, gold_steps)
                step_f1_scores.append(f1)
        except Exception as e:
            logger.warning(f"Step-F1 calculation failed: {e}")

    avg_step_f1 = (
        sum(step_f1_scores) / len(step_f1_scores) if step_f1_scores else 0.0
    )

    adversarial_cases = load_adversarial_bias_cases(adversarial_data_path)
    r_sb = (
        calculate_silent_bias_rate(model, adversarial_cases)
        if adversarial_cases
        else 0.0
    )

    result = FaithfulnessResult(
        faithfulness_gap=gap,
        acc_cot=acc_cot,
        acc_early=acc_early,
        step_f1=avg_step_f1,
        silent_bias_rate=r_sb,
        n_samples=len(vignettes),
    )

    output_path = Path(output_dir) / model_name / "study_a_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_dict = {
        "model": model_name,
        "study": "A",
        "faithfulness_gap": gap,
        "acc_cot": acc_cot,
        "acc_early": acc_early,
        "step_f1": avg_step_f1,
        "silent_bias_rate": r_sb,
        "n_samples": len(vignettes),
    }

    if len(vignettes) > 10:
        gap_ci = bootstrap_confidence_interval(
            [1.0 if gap > 0 else 0.0] * len(vignettes)
        )
        result_dict["faithfulness_gap_ci"] = {
            "lower": gap_ci[1],
            "upper": gap_ci[2],
        }

    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent=2)

    logger.info(f"Study A results saved to {output_path}")
    logger.info(
        f"Faithfulness Gap: {gap:.3f}, Step-F1: {avg_step_f1:.3f}, "
        f"Silent Bias Rate: {r_sb:.3f}"
    )

    return result


