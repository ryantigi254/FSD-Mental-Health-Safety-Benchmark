import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from reliable_clinical_benchmark.metrics.extraction import (
    is_refusal,
    extract_diagnosis_with_method,
    compute_output_complexity,
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
        description="Extract Study A diagnosis predictions into a separate processed/ tree (no inference rerun)."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing model subfolders with study_a_generations.jsonl",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("processed") / "study_a_extracted",
        help="Output base directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="If set, process only this model subfolder name",
    )

    args = parser.parse_args()

    results_dir: Path = args.results_dir
    processed_dir: Path = args.processed_dir

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

        out_dir = processed_dir / model_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "study_a_extracted.jsonl"

        with out_path.open("w", encoding="utf-8") as out_f:
            for row in iter_jsonl(src_path):
                output_text = row.get("output_text", "") or ""
                refusal = is_refusal(output_text)
                diagnosis, method = extract_diagnosis_with_method(output_text)
                complexity, complexity_features = compute_output_complexity(output_text)

                out_row = {
                    "id": row.get("id"),
                    "mode": row.get("mode"),
                    "model_name": row.get("model_name") or model_dir.name,
                    "status": row.get("status"),
                    "is_refusal": refusal,
                    "extracted_diagnosis": diagnosis,
                    "extraction_method": method,
                    "output_complexity": complexity,
                    "complexity_features": complexity_features,
                }
                out_f.write(json.dumps(out_row, ensure_ascii=False))
                out_f.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
