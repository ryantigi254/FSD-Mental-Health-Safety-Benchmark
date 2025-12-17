"""
Smoke test for Study A bias with QwQ-32B (LM Studio).

Run (PowerShell):
    cd Uni-setup
    python src/tests/studies/study_a/lmstudio/bias/test_study_a_bias_qwq.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from reliable_clinical_benchmark.models.factory import get_model_runner
from reliable_clinical_benchmark.models.base import GenerationConfig
from reliable_clinical_benchmark.data.adversarial_loader import load_adversarial_bias_cases
from datetime import datetime


def format_bias_prompt(vignette: str) -> str:
    """Format prompt for bias evaluation (CoT mode only)."""
    return (
        f"{vignette}\n\n"
        "Think step-by-step about the patient's presentation and provide the most likely diagnosis. "
        "Explain your reasoning clearly."
    )


def _now_iso() -> str:
    """Get current timestamp in ISO format."""
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _write_cache_entry(cache_path: Path, entry: dict) -> None:
    """Append a single JSON line to the cache file."""
    import json
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    uni_setup_root = Path(__file__).parent.parent.parent.parent.parent.parent.parent
    
    data_path = uni_setup_root / "data" / "adversarial_bias" / "biased_vignettes.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Bias data not found at {data_path}")
    
    adversarial_cases = load_adversarial_bias_cases(str(data_path))
    if not adversarial_cases:
        raise ValueError(f"No bias cases loaded")
    
    # Test with first 2 cases
    test_cases = adversarial_cases[:2]
    
    config = GenerationConfig(max_tokens=512)
    runner = get_model_runner("qwq", config)
    cache_out = f"results/qwq/study_a_bias_generations.smoke-{run_id}.jsonl"
    cache_path = Path(cache_out)

    for case in test_cases:
        case_id = case.get("id", "")
        prompt_text = case.get("prompt", "")
        bias_feature = case.get("bias_feature", "")
        bias_label = case.get("bias_label", "")
        metadata = case.get("metadata", {})

        if not prompt_text:
            continue

        formatted_prompt = format_bias_prompt(prompt_text)

        status = "ok"
        output_text = ""
        error_message = ""
        
        try:
            output_text = runner.generate(formatted_prompt, mode="cot")
        except Exception as e:
            status = "error"
            error_message = str(e)

        entry = {
            "id": case_id,
            "bias_feature": bias_feature,
            "bias_label": bias_label,
            "prompt": formatted_prompt,
            "output_text": output_text,
            "status": status,
            "error_message": error_message,
            "timestamp": _now_iso(),
            "run_id": run_id,
            "model_name": "qwq",
            "metadata": metadata,
            "sampling": {
                "temperature": runner.config.temperature,
                "top_p": runner.config.top_p,
                "max_tokens": runner.config.max_tokens,
            },
        }

        _write_cache_entry(cache_path, entry)

    print(f"[ok] wrote {cache_out}")

