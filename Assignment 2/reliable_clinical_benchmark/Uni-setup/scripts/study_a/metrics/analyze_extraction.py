import argparse
import json
from pathlib import Path
import sys

from typing import Optional, Dict, Any

import pandas as pd

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


def analyze_model_generations(model_dir: Path) -> Optional[Dict[str, Any]]:
    generations_file = model_dir / "study_a_generations.jsonl"
    if not generations_file.exists():
        return None

    total = 0
    refusals = 0
    extracted = 0
    failed = 0
    complexities = []
    non_latin_ratios = []
    sample_diagnoses = []

    for row in iter_jsonl(generations_file):
        output_text = row.get("output_text", "") or ""
        total += 1

        if is_refusal(output_text):
            refusals += 1
            continue

        diagnosis, _method = extract_diagnosis_with_method(output_text)
        if diagnosis in ("EXTRACTION_FAILED", "NO_OUTPUT"):
            failed += 1
        else:
            extracted += 1
            if len(sample_diagnoses) < 3:
                sample_diagnoses.append(diagnosis)

        complexity, features = compute_output_complexity(output_text)
        complexities.append(complexity)
        non_latin_ratios.append(float(features.get("non_latin_ratio", 0.0)))

    non_refusal = total - refusals
    extraction_success_rate = (extracted / non_refusal) if non_refusal else 0.0
    refusal_rate = (refusals / total) if total else 0.0

    avg_complexity = sum(complexities) / len(complexities) if complexities else 0.0
    avg_non_latin = sum(non_latin_ratios) / len(non_latin_ratios) if non_latin_ratios else 0.0

    return {
        "model": model_dir.name,
        "total": total,
        "refusals": refusals,
        "failed": failed,
        "extracted": extracted,
        "refusal_rate": refusal_rate,
        "extraction_success_rate": extraction_success_rate,
        "avg_output_complexity": avg_complexity,
        "avg_non_latin_ratio": avg_non_latin,
        "sample_diagnoses": sample_diagnoses,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze Study A extractability, refusal rate, and output complexity from existing generations."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing model subfolders with study_a_generations.jsonl",
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    if not results_dir.exists():
        raise FileNotFoundError(f"results dir not found: {results_dir}")

    all_stats = []
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        s = analyze_model_generations(model_dir)
        if s:
            all_stats.append(s)

    df = pd.DataFrame(all_stats)
    if df.empty:
        print("No Study A generations found.")
        return 0

    df = df.sort_values(["refusal_rate", "avg_output_complexity"], ascending=[True, True])

    print(
        df[
            [
                "model",
                "refusal_rate",
                "extraction_success_rate",
                "avg_output_complexity",
                "avg_non_latin_ratio",
                "failed",
                "sample_diagnoses",
            ]
        ].to_string(index=False)
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
