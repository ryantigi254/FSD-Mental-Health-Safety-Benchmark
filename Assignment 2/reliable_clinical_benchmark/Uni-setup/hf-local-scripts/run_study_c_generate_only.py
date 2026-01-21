import argparse
import os
import sys
from pathlib import Path

# Set PyTorch CUDA allocator config to reduce memory fragmentation
# This helps with OOM errors on later turns when conversation history grows
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _ensure_src_on_path(uni_setup_root: Path) -> None:
    src_dir = uni_setup_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Study C generation-only runner (no evaluation.py).")
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
        help="Directory containing study_c_test.json (defaults to Uni-setup/data/openr1_psy_splits).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Results directory (defaults to Uni-setup/results).",
    )
    p.add_argument("--max-cases", type=int, default=None, help="Limit Study C cases.")
    p.add_argument(
        "--max-tokens", 
        type=int, 
        default=7000, 
        help="Max new tokens per generation. Default 7000 accommodates all models including Piaget-8B max (5958). "
             "GPU cache is cleared after each generation to prevent memory buildup. Can be reduced if memory is tight."
    )
    p.add_argument(
        "--cache-out",
        type=str,
        default=None,
        help="Explicit cache path (defaults to results/<model-id>/study_c_generations.jsonl).",
    )
    return p.parse_args()


def _normalize_model_id_for_path(model_id: str, output_dir: Path) -> str:
    """
    Normalize model_id to match existing folder names in results directory.
    
    Checks if a folder exists with the model_id (or variations) and returns
    the existing folder name. Uses partial matching if exact match fails.
    """
    # First, try exact match
    exact_path = output_dir / model_id
    if exact_path.exists() and exact_path.is_dir():
        return model_id
    
    # Try common variations (underscore vs hyphen)
    variations = [
        model_id.replace("_", "-"),
        model_id.replace("-", "_"),
        model_id.lower(),
        model_id.lower().replace("_", "-"),
        model_id.lower().replace("-", "_"),
    ]
    
    for variant in variations:
        variant_path = output_dir / variant
        if variant_path.exists() and variant_path.is_dir():
            return variant
    
    # Try partial matching: if normalized model_id is a prefix of any existing folder
    normalized_base = model_id.replace("_", "-").lower()
    if output_dir.exists():
        for existing_folder in output_dir.iterdir():
            if existing_folder.is_dir():
                folder_name = existing_folder.name.lower()
                # Check if normalized_base is a prefix of folder_name
                # e.g., "gpt-oss" matches "gpt-oss-20b", "piaget-local" matches "piaget-8b-local"
                if folder_name.startswith(normalized_base) or normalized_base.startswith(folder_name.split("-")[0]):
                    # More precise: check if they share the same base components
                    base_parts = set(normalized_base.split("-"))
                    folder_parts = set(folder_name.split("-"))
                    # If most base parts are in folder parts, it's a match
                    if len(base_parts.intersection(folder_parts)) >= min(2, len(base_parts)):
                        return existing_folder.name
    
    # If no existing folder found, return normalized version (use hyphens for consistency)
    return model_id.replace("_", "-").lower()


def main() -> None:
    uni_setup_root = Path(__file__).resolve().parents[1]
    _ensure_src_on_path(uni_setup_root)

    # Import only after src is on sys.path.
    from reliable_clinical_benchmark.eval.runtime_checks import validate_study_c_schema
    from reliable_clinical_benchmark.models.base import GenerationConfig
    from reliable_clinical_benchmark.models.factory import get_model_runner
    from reliable_clinical_benchmark.pipelines.study_c import run_study_c

    args = _parse_args()

    base_data_dir = Path(args.data_dir) if args.data_dir else (uni_setup_root / "data" / "openr1_psy_splits")
    output_dir = Path(args.output_dir) if args.output_dir else (uni_setup_root / "results")

    ok, errors = validate_study_c_schema(str(base_data_dir.parent))
    if not ok:
        raise SystemExit("Study C split validation failed:\n- " + "\n- ".join(errors[:30]))

    config = GenerationConfig(max_tokens=args.max_tokens)
    runner = get_model_runner(args.model_id, config)

    # Normalize model_id to match existing folder structure
    normalized_model_id = _normalize_model_id_for_path(args.model_id, output_dir)

    cache_out = args.cache_out
    if cache_out is None:
        cache_out = str(output_dir / normalized_model_id / "study_c_generations.jsonl")

    run_study_c(
        model=runner,
        data_dir=str(base_data_dir),
        max_cases=args.max_cases,
        output_dir=str(output_dir),
        model_name=normalized_model_id,
        use_nli=False,
        generate_only=True,
        cache_out=cache_out,
    )

    print(cache_out)


if __name__ == "__main__":
    main()


