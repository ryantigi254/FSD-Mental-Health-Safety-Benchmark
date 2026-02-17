import argparse
import sys
import warnings
from pathlib import Path

# Suppress spacy FutureWarnings about regex set unions in newer Python versions
warnings.filterwarnings("ignore", category=FutureWarning, module="spacy")



def _ensure_src_on_path(uni_setup_root: Path) -> None:
    src_dir = uni_setup_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def main() -> None:
    uni_setup_root = Path(__file__).resolve().parents[1]
    _ensure_src_on_path(uni_setup_root)

    # Import after sys.path is updated
    from reliable_clinical_benchmark.models.base import GenerationConfig
    from reliable_clinical_benchmark.models.lmstudio_qwen3 import Qwen3LMStudioRunner
    from reliable_clinical_benchmark.pipelines.study_a import run_study_a
    from reliable_clinical_benchmark.pipelines.study_b import run_study_b
    from reliable_clinical_benchmark.pipelines.study_c import run_study_c

    # Define shared args in a parent parser so they can be placed
    # either before or after the subcommand (prompt/study-a).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--api-base",
        default="http://127.0.0.1:1234/v1",
        help="LM Studio API base URL.",
    )
    common.add_argument(
        "--api-identifier",
        default="deepseek-r1-distill-qwen-7b",  # Default for DeepSeek model
        help="LM Studio API Identifier for the loaded model.",
    )
    common.add_argument(
        "--model-name",
        default="deepseek-r1-distill-qwen-7b",  # Clean folder name
        help="Folder name under results/ (and model_name field in JSONL).",
    )
    common.add_argument("--temperature", type=float, default=0.7)
    common.add_argument("--top-p", type=float, default=0.9)
    common.add_argument("--max-new-tokens", type=int, default=4096)

    parser = argparse.ArgumentParser(parents=[common])

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prompt = sub.add_parser("prompt", parents=[common])
    p_prompt.add_argument("prompt")
    p_prompt.add_argument("--mode", choices=["cot", "direct"], default="cot")

    # Study A arguments
    p_study_a = sub.add_parser("study-a", parents=[common])
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

    # Study B arguments
    p_study_b = sub.add_parser("study-b", parents=[common])
    p_study_b.add_argument("--max-samples", type=int, default=None)
    p_study_b.add_argument(
        "--data-dir",
        default=str(uni_setup_root / "data" / "openr1_psy_splits"),
    )
    p_study_b.add_argument(
        "--output-dir",
        default=str(uni_setup_root / "results"),
    )
    p_study_b.add_argument(
        "--generate-only",
        action="store_true",
        help="Write generations JSONL only (no metrics).",
    )
    p_study_b.add_argument(
        "--from-cache",
        default=None,
        help="Path to cache JSONL (metrics-from-cache mode).",
    )
    p_study_b.add_argument(
        "--cache-out",
        default=None,
        help="Path to write cache JSONL (generation-only mode).",
    )

    # Study C arguments
    p_study_c = sub.add_parser("study-c", parents=[common])
    p_study_c.add_argument("--max-cases", type=int, default=None)
    p_study_c.add_argument(
        "--data-dir",
        default=str(uni_setup_root / "data" / "openr1_psy_splits"),
    )
    p_study_c.add_argument(
        "--output-dir",
        default=str(uni_setup_root / "results"),
    )
    p_study_c.add_argument(
        "--generate-only",
        action="store_true",
        help="Write generations JSONL only (no metrics).",
    )
    p_study_c.add_argument(
        "--from-cache",
        default=None,
        help="Path to cache JSONL (metrics-from-cache mode).",
    )
    p_study_c.add_argument(
        "--cache-out",
        default=None,
        help="Path to write cache JSONL (generation-only mode).",
    )

    args = parser.parse_args()

    # Reuse Qwen3 runner as generic LM Studio runner
    runner = Qwen3LMStudioRunner(
        model_name=args.api_identifier,
        api_base=args.api_base,
        config=GenerationConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
        ),
    )

    if args.cmd == "prompt":
        print(runner.generate(args.prompt, mode=args.mode))
        return

    if args.cmd == "study-a":
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
    elif args.cmd == "study-b":
        cache_out = args.cache_out
        if cache_out is None and args.generate_only:
            cache_out = str(Path(args.output_dir) / args.model_name / "study_b_generations.jsonl")

        run_study_b(
            model=runner,
            data_dir=args.data_dir,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
            model_name=args.model_name,
            generate_only=bool(args.generate_only),
            from_cache=args.from_cache,
            cache_out=cache_out,
        )
    elif args.cmd == "study-c":
        cache_out = args.cache_out
        if cache_out is None and args.generate_only:
            cache_out = str(Path(args.output_dir) / args.model_name / "study_c_generations.jsonl")

        run_study_c(
            model=runner,
            data_dir=args.data_dir,
            max_cases=args.max_cases,
            output_dir=args.output_dir,
            model_name=args.model_name,
            generate_only=bool(args.generate_only),
            cache_out=cache_out,
        )


if __name__ == "__main__":
    main()
