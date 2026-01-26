"""Compare raw vs cleaned data - outputs to file."""
import json
from pathlib import Path
import re

def simple_extract(text):
    if not text or len(text.strip()) < 10:
        return "NO_OUTPUT"
    text_lower = text.lower()
    for pattern in [r"diagnosis:?\s*(.+?)(?:\n|$)", r"final diagnosis:?\s*(.+?)(?:\n|$)"]:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1).strip()[:80]
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return lines[-1][:80] if lines else "EXTRACTION_FAILED"

def main():
    base = Path(__file__).parent.parent
    models = ["deepseek-r1-lmstudio", "gpt-oss-20b", "piaget-8b-local", "psych-qwen-32b-local", 
              "psyche-r1-local", "psyllm-gml-local", "qwen3-lmstudio", "qwq"]

    results = []
    total_entries = 0
    total_diffs = 0
    
    for model in models:
        raw_path = base / "results" / model / "study_a_generations.jsonl"
        cleaned_path = base / "processed" / "study_a_scan_cleaned" / model / "study_a_generations.jsonl"
        if not raw_path.exists() or not cleaned_path.exists():
            continue
        
        raw = [json.loads(l) for l in open(raw_path, encoding="utf-8") if l.strip()]
        cleaned = [json.loads(l) for l in open(cleaned_path, encoding="utf-8") if l.strip()]
        
        diffs = sum(1 for r,c in zip(raw, cleaned) 
                    if simple_extract(r.get("output_text","")) != simple_extract(c.get("output_text","")))
        
        total_entries += len(raw)
        total_diffs += diffs
        pct = diffs/len(raw)*100 if raw else 0
        results.append((model, len(raw), diffs, pct))
    
    # Write to file
    output_path = base / "comparison_report.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("RAW vs CLEANED COMPARISON RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Model':<28} {'Count':>6} {'Diff':>6} {'%':>7}\n")
        f.write("-" * 50 + "\n")
        for model, count, diffs, pct in results:
            f.write(f"{model:<28} {count:>6} {diffs:>6} {pct:>6.1f}%\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'TOTAL':<28} {total_entries:>6} {total_diffs:>6} {total_diffs/total_entries*100:>6.1f}%\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("CONCLUSION:\n")
        if total_diffs/total_entries > 0.1:
            f.write("!! CLEANING SIGNIFICANTLY AFFECTS DIAGNOSIS EXTRACTION !!\n")
            f.write(f"   {total_diffs/total_entries*100:.1f}% of diagnoses differ between raw and cleaned\n")
        else:
            f.write("Cleaning has minimal impact on diagnosis extraction.\n")
    
    print(f"Results written to: {output_path}")
    print(f"\nSUMMARY: {total_diffs}/{total_entries} ({total_diffs/total_entries*100:.1f}%) diagnoses differ")

if __name__ == "__main__":
    main()
