import argparse
import sys
from pathlib import Path


def _ensure_src_on_path(uni_setup_root: Path) -> None:
    src_dir = uni_setup_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Study B generation-only runner (no evaluation.py).")
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
        help="Directory containing study_b_test.json (defaults to Uni-setup/data/openr1_psy_splits).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Results directory (defaults to Uni-setup/results).",
    )
    p.add_argument("--max-samples", type=int, default=None, help="Limit Study B samples.")
    p.add_argument("--max-tokens", type=int, default=4096, help="Max new tokens per generation.")
    p.add_argument(
        "--cache-out",
        type=str,
        default=None,
        help="Explicit cache path (defaults to results/<model-id>/study_b_generations.jsonl).",
    )
    return p.parse_args()


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

    # Preflight: ensure persona IDs + IDs are valid before generating.
    ok, errors = validate_study_b_schema(str(base_data_dir.parent))
    if not ok:
        raise SystemExit("Study B split validation failed:\n- " + "\n- ".join(errors[:30]))

    config = GenerationConfig(max_tokens=args.max_tokens)
    runner = get_model_runner(args.model_id, config)

    cache_out = args.cache_out
    if cache_out is None:
        cache_out = str(output_dir / args.model_id / "study_b_generations.jsonl")

    run_study_b(
        model=runner,
        data_dir=str(base_data_dir),
        max_samples=args.max_samples,
        output_dir=str(output_dir),
        model_name=args.model_id,
        use_nli=False,
        generate_only=True,
        cache_out=cache_out,
    )

    print(cache_out)


if __name__ == "__main__":
    main()


