import argparse
import sys
from pathlib import Path



def _ensure_src_on_path(uni_setup_root: Path) -> None:
    src_dir = uni_setup_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def main() -> None:
    uni_setup_root = Path(__file__).resolve().parents[1]
    _ensure_src_on_path(uni_setup_root)

    # Import after sys.path is updated
    from reliable_clinical_benchmark.models.base import GenerationConfig
    from reliable_clinical_benchmark.models.psyche_r1_local import PsycheR1LocalRunner
    from reliable_clinical_benchmark.pipelines.study_a import run_study_a

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default=str(uni_setup_root / "models" / "Psyche-R1"),
        help="Local HF model directory.",
    )
    parser.add_argument(
        "--model-name",
        default="psyche-r1-local",
        help="Folder name under results/ (and model_name field in JSONL).",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=16384)

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prompt = sub.add_parser("prompt")
    p_prompt.add_argument("prompt")
    p_prompt.add_argument("--mode", choices=["cot", "direct"], default="cot")

    p_study_a = sub.add_parser("study-a")
    p_study_a.add_argument("--max-samples", type=int, default=None)
    p_study_a.add_argument(
        "--data-dir",
        default=str(uni_setup_root / "data" / "openr1_psy_splits"),
    )
    p_study_a.add_argument(
        "--output-dir",
        default=str(uni_setup_root / "results"),
    )
    p_study_a.add_argument(
        "--generate-only",
        action="store_true",
        help="Write generations JSONL only (no metrics).",
    )
    p_study_a.add_argument(
        "--from-cache",
        default=None,
        help="Path to cache JSONL (metrics-from-cache mode).",
    )
    p_study_a.add_argument(
        "--cache-out",
        default=None,
        help="Path to write cache JSONL (generation-only mode).",
    )

    args = parser.parse_args()

    runner = PsycheR1LocalRunner(
        model_name=args.model_dir,
        config=GenerationConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
        ),
    )

    if args.cmd == "prompt":
        print(runner.generate(args.prompt, mode=args.mode))
        return

    cache_out = args.cache_out
    if cache_out is None and args.generate_only:
        cache_out = str(Path(args.output_dir) / args.model_name / "study_a_generations.jsonl")

    run_study_a(
        model=runner,
        data_dir=args.data_dir,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        model_name=args.model_name,
        generate_only=bool(args.generate_only),
        from_cache=args.from_cache,
        cache_out=cache_out,
    )


if __name__ == "__main__":
    main()

