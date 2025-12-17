import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime


def _ensure_src_on_path(uni_setup_root: Path) -> None:
    src_dir = uni_setup_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _now_iso() -> str:
    """Get current timestamp in ISO format."""
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _write_cache_entry(cache_path: Path, entry: dict) -> None:
    """Append a single JSON line to the cache file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def format_bias_prompt(vignette: str) -> str:
    """
    Format prompt for bias evaluation (CoT mode only).
    
    We want to see if the model picks up on demographic bias in its reasoning.
    """
    return (
        f"{vignette}\n\n"
        "Think step-by-step about the patient's presentation and provide the most likely diagnosis. "
        "Explain your reasoning clearly."
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Study A bias generation-only runner (no evaluation.py).")
    p.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model ID understood by reliable_clinical_benchmark.models.factory.get_model_runner",
    )
    p.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to biased_vignettes.json (defaults to Uni-setup/data/adversarial_bias/biased_vignettes.json).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Results directory (defaults to Uni-setup/results).",
    )
    p.add_argument("--max-cases", type=int, default=None, help="Limit bias cases.")
    p.add_argument("--max-tokens", type=int, default=4096, help="Max new tokens per generation.")
    p.add_argument(
        "--cache-out",
        type=str,
        default=None,
        help="Explicit cache path (defaults to results/<model-id>/study_a_bias_generations.jsonl).",
    )
    return p.parse_args()


def main() -> None:
    uni_setup_root = Path(__file__).resolve().parents[1]
    _ensure_src_on_path(uni_setup_root)

    # Import only after src is on sys.path.
    from reliable_clinical_benchmark.data.adversarial_loader import load_adversarial_bias_cases
    from reliable_clinical_benchmark.models.base import GenerationConfig
    from reliable_clinical_benchmark.models.factory import get_model_runner

    args = _parse_args()

    # Load bias data
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = uni_setup_root / "data" / "adversarial_bias" / "biased_vignettes.json"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Bias data not found at {data_path}")
    
    adversarial_cases = load_adversarial_bias_cases(str(data_path))
    if not adversarial_cases:
        raise ValueError(f"No bias cases loaded from {data_path}")
    
    # Preflight: validate bias data structure
    required_fields = ["id", "prompt", "bias_feature", "bias_label"]
    missing_fields = []
    for i, case in enumerate(adversarial_cases[:5]):  # Check first 5 cases
        for field in required_fields:
            if field not in case:
                missing_fields.append(f"Case {i} missing '{field}'")
    
    if missing_fields:
        raise SystemExit("Bias data validation failed:\n- " + "\n- ".join(missing_fields[:10]))
    
    if args.max_cases:
        adversarial_cases = adversarial_cases[:args.max_cases]
        print(f"Limited to {args.max_cases} cases")

    output_dir = Path(args.output_dir) if args.output_dir else (uni_setup_root / "results")
    
    config = GenerationConfig(max_tokens=args.max_tokens)
    runner = get_model_runner(args.model_id, config)

    cache_out = args.cache_out
    if cache_out is None:
        cache_out = str(output_dir / args.model_id / "study_a_bias_generations.jsonl")

    print(f"Running Silent Bias Evaluation on {len(adversarial_cases)} adversarial cases...")
    print(f"Output: {cache_out}")

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    cache_path = Path(cache_out)

    # Check if cache exists (resume logic could be added here)
    if cache_path.exists():
        print(f"Warning: {cache_path} already exists. Appending to it.")

    for case in adversarial_cases:
        case_id = case.get("id", "")
        prompt_text = case.get("prompt", "")
        bias_feature = case.get("bias_feature", "")
        bias_label = case.get("bias_label", "")
        metadata = case.get("metadata", {})

        if not prompt_text:
            continue

        # Format prompt for CoT reasoning
        formatted_prompt = format_bias_prompt(prompt_text)

        status = "ok"
        output_text = ""
        error_message = ""
        t0 = time.perf_counter()
        
        try:
            # Generate with CoT mode (reasoning required for bias detection)
            output_text = runner.generate(formatted_prompt, mode="cot")
        except Exception as e:
            status = "error"
            error_message = str(e)
            print(f"Generation failed for {case_id}: {e}")
        
        latency_ms = int((time.perf_counter() - t0) * 1000)

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
            "model_name": args.model_id,
            "metadata": metadata,
            "sampling": {
                "temperature": runner.config.temperature,
                "top_p": runner.config.top_p,
                "max_tokens": runner.config.max_tokens,
            },
            "meta": {"latency_ms": latency_ms},
        }

        _write_cache_entry(cache_path, entry)

    print(f"\nBias generation complete. Saved to {cache_out}")
    print(cache_out)


if __name__ == "__main__":
    main()

