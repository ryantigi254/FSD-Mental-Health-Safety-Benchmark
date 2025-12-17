import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from reliable_clinical_benchmark.metrics.extraction import (
    is_refusal,
    extract_diagnosis_closed_set,
    compute_complexity_metrics,
    load_valid_diagnoses,
    clean_model_output,
)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract Study A diagnosis predictions using Gold Label Closed-Set Matching."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory containing model subfolders with study_a_generations.jsonl (default: auto-detect)",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Output base directory (default: auto-detect)",
    )
    parser.add_argument(
        "--gold-labels",
        type=Path,
        default=None,
        help="Path to gold_diagnosis_labels.json (default: auto-detect from script location)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="If set, process only this model subfolder name",
    )

    args = parser.parse_args()

    # Auto-detect Uni-setup root: go up from scripts/study_a/metrics/ to Uni-setup root
    script_dir = Path(__file__).parent
    uni_setup_root = script_dir.parent.parent.parent

    # Resolve gold labels path
    if args.gold_labels is None:
        args.gold_labels = uni_setup_root / "data" / "study_a_gold" / "gold_diagnosis_labels.json"
    elif not args.gold_labels.is_absolute():
        args.gold_labels = Path.cwd() / args.gold_labels

    # Resolve results and processed directories
    if args.results_dir is None:
        results_dir = uni_setup_root / "results"
    elif not args.results_dir.is_absolute():
        results_dir = Path.cwd() / args.results_dir
    else:
        results_dir = args.results_dir

    if args.processed_dir is None:
        processed_dir = uni_setup_root / "processed" / "study_a_extracted"
    elif not args.processed_dir.is_absolute():
        processed_dir = Path.cwd() / args.processed_dir
    else:
        processed_dir = args.processed_dir

    # 1. Load Whitelist
    whitelist = load_valid_diagnoses(args.gold_labels)
    if whitelist:
        print(f"Loaded {len(whitelist)} unique diagnosis labels for extraction.")
    else:
        print(f"WARNING: Gold labels not found at {args.gold_labels}. Extraction will use heuristic fallback only.")

    if not results_dir.exists():
        raise FileNotFoundError(f"results dir not found: {results_dir}")

    model_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    if args.model:
        model_dirs = [d for d in model_dirs if d.name == args.model]

    processed_dir.mkdir(parents=True, exist_ok=True)

    for model_dir in model_dirs:
        src_path = model_dir / "study_a_generations.jsonl"
        if not src_path.exists():
            continue

        print(f"Processing {model_dir.name}...")

        out_dir = processed_dir / model_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "study_a_extracted.jsonl"

        with out_path.open("w", encoding="utf-8") as out_f:
            for row in iter_jsonl(src_path):
                raw_text = row.get("output_text", "") or ""
                
                # 2. Clean text ONCE
                clean_text = clean_model_output(raw_text)
                
                # 3. Extract Diagnosis FIRST (using Clean Text + Whitelist)
                # This ensures we find valid diagnoses even if there's a disclaimer at the end
                if whitelist:
                    diagnosis, method = extract_diagnosis_closed_set(clean_text, whitelist)
                else:
                    # Fallback (legacy heuristic) if whitelist missing
                    from reliable_clinical_benchmark.metrics.extraction import extract_diagnosis_with_method
                    diagnosis, method = extract_diagnosis_with_method(clean_text)
                    if method in ("failed", "no_output"):
                        diagnosis = "EXTRACTION_FAILED"
                        method = "missing_whitelist"

                # 4. Refusal Check (ONLY if no valid diagnosis was found)
                # Key insight: If we found a valid diagnosis, the model didn't refuse
                # (even if it added a safety disclaimer at the end)
                refusal_bool = False
                if diagnosis in ("EXTRACTION_FAILED", "NO_WHITELIST", "NO_OUTPUT", "closed_set_no_match"):
                    # Only check for refusal if extraction failed
                    refusal_bool = is_refusal(clean_text)
                    if refusal_bool:
                        diagnosis = "REFUSAL"
                        method = "refusal_detection"
                # If diagnosis was successfully extracted, ignore any disclaimer text

                # 5. Complexity Metrics (using Raw Text to catch bad formatting artifacts)
                # We use raw_text because <think> tags might contain the broken unicode we want to detect.
                verbosity, noise_score, word_count = compute_complexity_metrics(raw_text)

                out_row = {
                    "id": row.get("id"),
                    "mode": row.get("mode"),
                    "model_name": row.get("model_name") or model_dir.name,
                    "status": row.get("status"),
                    "is_refusal": refusal_bool,
                    "extracted_diagnosis": diagnosis,
                    "extraction_method": method,
                    # New Scientific Metrics (split complexity)
                    "response_verbosity": verbosity,
                    "format_noise_score": noise_score,
                    "word_count": word_count,
                }
                out_f.write(json.dumps(out_row, ensure_ascii=False))
                out_f.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
