import argparse
import sys
from pathlib import Path

from reliable_clinical_benchmark.models.base import GenerationConfig
from reliable_clinical_benchmark.models.psych_qwen_local import PsychQwen32BLocalRunner
from reliable_clinical_benchmark.pipelines.study_a import run_study_a


def _ensure_src_on_path(uni_setup_root: Path) -> None:
    src_dir = uni_setup_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def main() -> None:
    uni_setup_root = Path(__file__).resolve().parents[1]
    _ensure_src_on_path(uni_setup_root)

    # Allow shared args before/after the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--model",
        default="Compumacy/Psych_Qwen_32B",
        help="HF model id or local directory (e.g. models/Psych_Qwen_32B).",
    )
    common.add_argument("--temperature", type=float, default=0.6)
    common.add_argument("--top-p", type=float, default=0.95)
    common.add_argument("--max-new-tokens", type=int, default=4096)
    common.add_argument(
        "--quantization",
        default="4bit",
        help="Quantization mode (e.g. 4bit, 8bit, none). Default: 4bit",
    )
    common.add_argument(
        "--model-name",
        default="psych-qwen-32b-local",
        help="Folder name under results/ (and model_name field in JSONL).",
    )

    parser = argparse.ArgumentParser(parents=[common])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prompt = sub.add_parser("prompt", parents=[common])
    p_prompt.add_argument("prompt")
    p_prompt.add_argument("--mode", choices=["cot", "direct"], default="cot")

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

    args = parser.parse_args()

    runner = PsychQwen32BLocalRunner(
        model_name=args.model,
        quantization=args.quantization,
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

