import argparse
import sys
from pathlib import Path


def _ensure_src_on_path(uni_setup_root: Path) -> None:
    src_dir = uni_setup_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study A generation-only runner (no evaluation.py).")
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model ID understood by reliable_clinical_benchmark.models.factory.get_model_runner",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing study_a_test.json (defaults to Uni-setup/data/openr1_psy_splits).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Results directory (defaults to Uni-setup/results).",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit Study A samples.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Max new tokens per generation (default: 16384 for long reasoning traces).",
    )
    parser.add_argument(
        "--cache-out",
        type=str,
        default=None,
        help="Explicit cache path (defaults to results/<model-id>/study_a_generations.jsonl).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of parallel generation workers. "
            "Default is auto: 4 for LM Studio runners, 1 for non-LM Studio runners."
        ),
    )
    parser.add_argument(
        "--progress-interval-seconds",
        type=int,
        default=10,
        help="Heartbeat interval for progress logging while waiting for workers.",
    )
    return parser.parse_args()


def _normalize_model_id_for_path(model_id: str, output_dir: Path) -> str:
    """
    Normalize model_id to match existing folder names in results directory.
    """
    alias_map = {
        "gpt_oss": "gpt-oss-20b",
        "deepseek_r1_lmstudio": "deepseek-r1-lmstudio",
        "piaget_local": "piaget-8b-local",
        "psych_qwen_local": "psych-qwen-32b-local",
        "psyllm_gml_local": "psyllm-gml-local",
        "psyche_r1_local": "psyche-r1-local",
        "qwen3_lmstudio": "qwen3-lmstudio",
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

    from reliable_clinical_benchmark.models.base import GenerationConfig
    from reliable_clinical_benchmark.models.factory import get_model_runner
    from reliable_clinical_benchmark.pipelines.study_a import run_study_a
    from reliable_clinical_benchmark.utils.worker_runtime import resolve_worker_count

    args = _parse_args()

    base_data_dir = Path(args.data_dir) if args.data_dir else (uni_setup_root / "data" / "openr1_psy_splits")
    output_dir = Path(args.output_dir) if args.output_dir else (uni_setup_root / "results")

    study_a_path = base_data_dir / "study_a_test.json"
    if not study_a_path.exists():
        raise SystemExit(f"Study A split not found: {study_a_path}")

    config = GenerationConfig(max_tokens=args.max_tokens)
    runner = get_model_runner(args.model_id, config)
    worker_count = resolve_worker_count(args.workers, runner, lmstudio_default=4, non_lm_default=1)

    normalized_model_id = _normalize_model_id_for_path(args.model_id, output_dir)

    cache_out = args.cache_out
    if cache_out is None:
        cache_out = str(output_dir / normalized_model_id / "study_a_generations.jsonl")

    run_study_a(
        model=runner,
        data_dir=str(base_data_dir),
        max_samples=args.max_samples,
        output_dir=str(output_dir),
        model_name=normalized_model_id,
        generate_only=True,
        cache_out=cache_out,
        workers=worker_count,
        progress_interval_seconds=args.progress_interval_seconds,
    )

    print(f"Workers: {worker_count}")
    print(cache_out)


if __name__ == "__main__":
    main()
