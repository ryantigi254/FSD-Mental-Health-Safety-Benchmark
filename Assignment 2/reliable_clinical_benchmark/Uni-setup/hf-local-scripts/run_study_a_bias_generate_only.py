import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime


def _ensure_src_on_path(uni_setup_root: Path) -> None:
    src_dir = uni_setup_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _ensure_hf_cache_under_models_dir(uni_setup_root: Path) -> None:
    """
    Force Hugging Face + Transformers caches to live under Uni-setup/models/.
    
    This ensures large checkpoints are saved under <Uni-setup>/models/
    rather than the default user cache under C:\\Users\\...
    """
    models_dir = uni_setup_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    hf_home = models_dir / "hf_home"
    hub_cache = models_dir / "hf_hub"
    transformers_cache = models_dir / "transformers_cache"
    
    hf_home.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)
    transformers_cache.mkdir(parents=True, exist_ok=True)
    
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(transformers_cache))


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
        "--model",
        type=str,
        default=None,
        help="HF model path/ID for local models (overrides default model path).",
    )
    p.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Quantization mode for local models (e.g., 4bit, 8bit, none). Default: 4bit for psych_qwen_local.",
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
        help="Output directory (defaults to Uni-setup/processed/study_a_bias).",
    )
    p.add_argument("--max-cases", type=int, default=None, help="Limit bias cases.")
    p.add_argument("--max-tokens", type=int, default=8192, help="Max new tokens per generation (default: 8192 to allow long reasoning).")
    p.add_argument(
        "--cache-out",
        type=str,
        default=None,
        help="Explicit cache path (defaults to processed/study_a_bias/<model-id>/study_a_bias_generations.jsonl).",
    )
    return p.parse_args()


def main() -> None:
    uni_setup_root = Path(__file__).resolve().parents[1]
    _ensure_src_on_path(uni_setup_root)
    
    # Set up HF cache directories for local models (before any HF imports)
    _ensure_hf_cache_under_models_dir(uni_setup_root)

    # Import only after src is on sys.path and HF cache env vars are set.
    from reliable_clinical_benchmark.data.adversarial_loader import load_adversarial_bias_cases
    from reliable_clinical_benchmark.models.base import GenerationConfig
    from reliable_clinical_benchmark.models.factory import get_model_runner

    args = _parse_args()
    
    # For local HF models, instantiate directly (like main Study A scripts)
    # This ensures proper model path and quantization settings
    model_id_lower = args.model_id.lower()
    runner = None
    
    # Check if this is a local HF model and instantiate directly
    if model_id_lower in ("psyllm", "psyllm_gml_local", "psyllm-gml-local", "psyllm-gmlhuhe-local", "gmlhuhe_psyllm_local"):
        from reliable_clinical_benchmark.models.psyllm_gml_local import PsyLLMGMLLocalRunner
        model_path = args.model or str(uni_setup_root / "models" / "PsyLLM")
        runner = PsyLLMGMLLocalRunner(
            model_name=model_path,
            config=GenerationConfig(max_tokens=args.max_tokens),
        )
    elif model_id_lower in ("piaget_local", "piaget-8b-local", "piaget8b-local"):
        from reliable_clinical_benchmark.models.piaget_local import Piaget8BLocalRunner
        model_path = args.model or str(uni_setup_root / "models" / "Piaget-8B")
        runner = Piaget8BLocalRunner(
            model_name=model_path,
            config=GenerationConfig(max_tokens=args.max_tokens),
        )
    elif model_id_lower in ("psyche_r1_local", "psyche-r1-local", "psyche-r1-local-hf"):
        from reliable_clinical_benchmark.models.psyche_r1_local import PsycheR1LocalRunner
        model_path = args.model or str(uni_setup_root / "models" / "Psyche-R1")
        runner = PsycheR1LocalRunner(
            model_name=model_path,
            config=GenerationConfig(max_tokens=args.max_tokens),
        )
    elif model_id_lower in ("psych_qwen_local", "psych-qwen-32b-local", "psych-qwen-local-hf"):
        from reliable_clinical_benchmark.models.psych_qwen_local import PsychQwen32BLocalRunner
        model_path = args.model or str(uni_setup_root / "models" / "Psych_Qwen_32B")
        quantization = args.quantization or "4bit"  # Default to 4bit for 32B model
        runner = PsychQwen32BLocalRunner(
            model_name=model_path,
            quantization=quantization,
            config=GenerationConfig(max_tokens=args.max_tokens),
        )
    
    # For LM Studio models or if runner not set, use factory
    if runner is None:
        config = GenerationConfig(max_tokens=args.max_tokens)
        runner = get_model_runner(args.model_id, config)

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

    # Default to processed/study_a_bias instead of results
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = uni_setup_root / "processed" / "study_a_bias"

    cache_out = args.cache_out
    if cache_out is None:
        # Save to processed/study_a_bias/{model-id}/study_a_bias_generations.jsonl
        model_output_dir = output_dir / args.model_id
        model_output_dir.mkdir(parents=True, exist_ok=True)
        cache_out = str(model_output_dir / "study_a_bias_generations.jsonl")

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

