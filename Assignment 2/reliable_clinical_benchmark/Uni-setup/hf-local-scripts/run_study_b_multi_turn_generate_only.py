import argparse
import sys
from pathlib import Path


def _ensure_src_on_path(uni_setup_root: Path) -> None:
    src_dir = uni_setup_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Study B Multi-Turn Runner (Turn-of-Flip evaluation).")
    p.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model ID understood by reliable_clinical_benchmark.models.factory.get_model_runner",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing study_b_multi_turn.json (defaults to Uni-setup/data/openr1_psy_splits).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Results directory (defaults to Uni-setup/results).",
    )
    p.add_argument("--max-samples", type=int, default=None, help="Limit number of multi-turn cases.")
    p.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Max new tokens per generation turn (default: 16384 to avoid truncation).",
    )
    p.add_argument(
        "--cache-out",
        type=str,
        default=None,
        help="Explicit cache path (defaults to results/<model-id>/study_b_multi_turn_generations.jsonl).",
    )
    return p.parse_args()


def _normalize_model_id_for_path(model_id: str, output_dir: Path) -> str:
    """
    Normalize model_id to match existing folder names in results directory.
    """
    # Explicit aliases
    alias_map = {
        "gpt_oss": "gpt-oss-20b",
        "deepseek_r1_lmstudio": "deepseek-r1-lmstudio",
        "piaget_local": "piaget-8b-local",
        "psych_qwen_local": "psych-qwen-32b-local",
    }
    alias_target = alias_map.get(model_id)
    if alias_target:
        alias_path = output_dir / alias_target
        if alias_path.exists() and alias_path.is_dir():
            return alias_target

    hyphen_preferred = [
        model_id.replace("_", "-"),
        model_id.lower().replace("_", "-"),
    ]
    for variant in hyphen_preferred:
        variant_path = output_dir / variant
        if variant_path.exists() and variant_path.is_dir():
            return variant

    exact_path = output_dir / model_id
    if exact_path.exists() and exact_path.is_dir():
        return model_id
    
    variations = [
        model_id.replace("-", "_"),
        model_id.lower(),
        model_id.lower().replace("-", "_"),
    ]
    
    for variant in variations:
        variant_path = output_dir / variant
        if variant_path.exists() and variant_path.is_dir():
            return variant
    
    return model_id.replace("_", "-").lower()


def main() -> None:
    uni_setup_root = Path(__file__).resolve().parents[1]
    _ensure_src_on_path(uni_setup_root)

    # Import only after src is on sys.path.
    from reliable_clinical_benchmark.eval.runtime_checks import validate_study_b_schema
    from reliable_clinical_benchmark.models.base import GenerationConfig
    from reliable_clinical_benchmark.models.factory import get_model_runner
    from reliable_clinical_benchmark.pipelines.study_b import run_study_b

    args = _parse_args()

    base_data_dir = Path(args.data_dir) if args.data_dir else (uni_setup_root / "data" / "openr1_psy_splits")
    output_dir = Path(args.output_dir) if args.output_dir else (uni_setup_root / "results")

    # Simple schema check - currently validates single-turn mostly, but useful for basic readiness
    # ok, errors = validate_study_b_schema(str(base_data_dir.parent))
    # if not ok:
    #     raise SystemExit("Study B split validation failed:\n- " + "\n- ".join(errors[:30]))

    config = GenerationConfig(max_tokens=args.max_tokens)
    runner = get_model_runner(args.model_id, config)

    normalized_model_id = _normalize_model_id_for_path(args.model_id, output_dir)

    cache_out = args.cache_out
    if cache_out is None:
        cache_out = str(output_dir / normalized_model_id / "study_b_multi_turn_generations.jsonl")

    print(f"Running Study B Multi-Turn for {normalized_model_id}...")
    run_study_b(
        model=runner,
        data_dir=str(base_data_dir),
        max_samples=args.max_samples,
        output_dir=str(output_dir),
        model_name=normalized_model_id,
        use_nli=False,
        generate_only=True,
        cache_out=cache_out,
        do_single_turn=False,
        do_multi_turn=True,
    )

    print(f"Done. Cache at: {cache_out}")


if __name__ == "__main__":
    main()
