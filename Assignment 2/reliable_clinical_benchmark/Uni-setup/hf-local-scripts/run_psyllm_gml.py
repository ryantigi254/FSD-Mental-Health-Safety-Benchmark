import argparse
import os
import sys
from pathlib import Path


def _ensure_src_on_path(uni_setup_root: Path) -> None:
    src_dir = uni_setup_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _ensure_hf_cache_under_models_dir(uni_setup_root: Path) -> None:
    """
    Force Hugging Face + Transformers caches to live under Uni-setup/models/.

    This ensures large checkpoints (e.g., GMLHUHE/PsyLLM) are saved under:
        <Uni-setup>/models/
    rather than the default user cache under C:\\Users\\...

    Env vars are only set if not already defined, so users can override.
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


def _resolve_hf_model_to_local_dir(model: str, uni_setup_root: Path) -> str:
    """
    If `model` is a local path, return it unchanged.

    If `model` looks like a HF repo id (e.g. "GMLHUHE/PsyLLM"), download a snapshot into:
        <Uni-setup>/models/<repo_name>/

    This makes the weights visible under Uni-setup/models (as you requested) and avoids the
    default user cache under C:\\Users\\...
    """

    model_path = Path(model)
    if model_path.exists():
        return str(model_path)

    if "/" not in model:
        return model

    repo_name = model.split("/")[-1].strip()
    if not repo_name:
        return model

    local_dir = uni_setup_root / "models" / repo_name
    local_dir.mkdir(parents=True, exist_ok=True)

    # Import lazily so env vars (HF_HOME / caches) are set before any HF code runs.
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=model,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return str(local_dir)


def main() -> None:
    uni_setup_root = Path(__file__).resolve().parents[1]
    _ensure_src_on_path(uni_setup_root)
    _ensure_hf_cache_under_models_dir(uni_setup_root)

    # Import after sys.path + HF cache env vars are set.
    from reliable_clinical_benchmark.models.base import GenerationConfig
    from reliable_clinical_benchmark.models.psyllm_gml_local import PsyLLMGMLLocalRunner
    from reliable_clinical_benchmark.pipelines.study_a import run_study_a

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--model",
        default="GMLHUHE/PsyLLM",
        help="HF model id or local directory.",
    )
    common.add_argument(
        "--model-name",
        default="psyllm-gml-local",
        help="Folder name under results/ (and model_name field in JSONL).",
    )
    common.add_argument("--temperature", type=float, default=0.6)
    common.add_argument("--top-p", type=float, default=0.95)
    common.add_argument("--max-new-tokens", type=int, default=4096)

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
    p_study_a.add_argument("--from-cache", default=None)
    p_study_a.add_argument("--cache-out", default=None)

    args = parser.parse_args()

    resolved_model = _resolve_hf_model_to_local_dir(args.model, uni_setup_root)
    runner = PsyLLMGMLLocalRunner(
        model_name=resolved_model,
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


