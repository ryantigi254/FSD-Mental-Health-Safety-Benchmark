"""
Compare cleaned vs raw generation files and report impact on diagnosis extraction.

Compares:
1. Raw files (results/)
2. Original cleaned files (processed/study_a_scan_cleaned/)
3. Optimized cleaned files (processed/study_a_cleaned_optimized/)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


def simple_extract_diagnosis(text: str) -> str:
    """Simple heuristic diagnosis extraction for comparison."""
    if not text or len(text.strip()) < 10:
        return "NO_OUTPUT"
    
    text_lower = text.lower()
    
    # Check for refusal
    refusal_patterns = ["cannot provide", "can't provide", "i'm unable to", 
                       "cannot diagnose", "not appropriate", "seek professional"]
    if any(p in text_lower for p in refusal_patterns):
        return "REFUSAL"
    
    # Find diagnosis markers
    patterns = [
        r"diagnosis:?\s*(.+?)(?:\n|$)",
        r"final diagnosis:?\s*(.+?)(?:\n|$)",
        r"most likely:?\s*(.+?)(?:\n|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1).strip()[:80]
    
    # Fallback: last non-empty line
    lines = [l.strip() for l in text.split('\n') if l.strip() and not l.startswith('[')]
    if lines:
        return lines[-1][:80]
    return "EXTRACTION_FAILED"


def load_entries(path: Path) -> Dict[str, Dict]:
    """Load JSONL entries indexed by (id, mode)."""
    entries = {}
    if not path.exists():
        return entries
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            key = (entry.get('id'), entry.get('mode'))
            entries[key] = entry
        except:
            continue
    return entries


def compare_model(model: str, base_dir: Path) -> Dict:
    """Compare raw, original cleaned, and optimized cleaned for a model."""
    raw_path = base_dir / "results" / model / "study_a_generations.jsonl"
    orig_cleaned_path = base_dir / "processed" / "study_a_scan_cleaned" / model / "study_a_generations.jsonl"
    opt_cleaned_path = base_dir / "processed" / "study_a_cleaned_optimized" / model / "study_a_generations.jsonl"
    
    raw = load_entries(raw_path)
    orig_cleaned = load_entries(orig_cleaned_path)
    opt_cleaned = load_entries(opt_cleaned_path)
    
    if not raw:
        return None
    
    results = {
        'model': model,
        'total_entries': len(raw),
        'raw_total_chars': 0,
        'orig_cleaned_total_chars': 0,
        'opt_cleaned_total_chars': 0,
        'orig_vs_raw_diag_diffs': 0,
        'opt_vs_raw_diag_diffs': 0,
        'orig_vs_opt_diag_diffs': 0,
    }
    
    for key, raw_entry in raw.items():
        raw_text = raw_entry.get('output_text', '')
        results['raw_total_chars'] += len(raw_text)
        
        orig_text = orig_cleaned.get(key, {}).get('output_text', '') if orig_cleaned else ''
        opt_text = opt_cleaned.get(key, {}).get('output_text', '') if opt_cleaned else ''
        
        results['orig_cleaned_total_chars'] += len(orig_text)
        results['opt_cleaned_total_chars'] += len(opt_text)
        
        # Extract diagnoses
        raw_diag = simple_extract_diagnosis(raw_text)
        orig_diag = simple_extract_diagnosis(orig_text) if orig_text else "N/A"
        opt_diag = simple_extract_diagnosis(opt_text) if opt_text else "N/A"
        
        if orig_diag != raw_diag:
            results['orig_vs_raw_diag_diffs'] += 1
        if opt_diag != raw_diag:
            results['opt_vs_raw_diag_diffs'] += 1
        if orig_diag != opt_diag:
            results['orig_vs_opt_diag_diffs'] += 1
    
    return results


def main():
    base_dir = Path(__file__).parent.parent
    
    models = ["deepseek-r1-lmstudio", "gpt-oss-20b", "piaget-8b-local", 
              "psych-qwen-32b-local", "psyche-r1-local", "psyllm-gml-local", 
              "qwen3-lmstudio", "qwq"]
    
    report_path = base_dir / "cleaning_comparison_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CLEANING COMPARISON REPORT: Raw vs Original Cleaned vs Optimized Cleaned\n")
        f.write("=" * 80 + "\n\n")
        
        all_results = []
        
        for model in models:
            result = compare_model(model, base_dir)
            if result:
                all_results.append(result)
        
        # Summary table
        f.write("SUMMARY TABLE\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model':<26} {'Entries':>7} {'Orig Diffs':>10} {'Opt Diffs':>10} {'Opt Better?':>12}\n")
        f.write("-" * 80 + "\n")
        
        for r in all_results:
            orig_pct = r['orig_vs_raw_diag_diffs'] / r['total_entries'] * 100
            opt_pct = r['opt_vs_raw_diag_diffs'] / r['total_entries'] * 100
            better = "YES" if opt_pct < orig_pct else ("SAME" if opt_pct == orig_pct else "NO")
            f.write(f"{r['model']:<26} {r['total_entries']:>7} "
                   f"{r['orig_vs_raw_diag_diffs']:>5} ({orig_pct:>4.1f}%) "
                   f"{r['opt_vs_raw_diag_diffs']:>5} ({opt_pct:>4.1f}%) "
                   f"{better:>12}\n")
        
        f.write("-" * 80 + "\n")
        
        # Totals
        total_entries = sum(r['total_entries'] for r in all_results)
        total_orig_diffs = sum(r['orig_vs_raw_diag_diffs'] for r in all_results)
        total_opt_diffs = sum(r['opt_vs_raw_diag_diffs'] for r in all_results)
        
        f.write(f"{'TOTAL':<26} {total_entries:>7} "
               f"{total_orig_diffs:>5} ({total_orig_diffs/total_entries*100:>4.1f}%) "
               f"{total_opt_diffs:>5} ({total_opt_diffs/total_entries*100:>4.1f}%)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("SIZE COMPARISON (Characters)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model':<26} {'Raw':>12} {'Orig Clean':>12} {'Opt Clean':>12} {'Opt Savings':>12}\n")
        f.write("-" * 80 + "\n")
        
        for r in all_results:
            savings = r['raw_total_chars'] - r['opt_cleaned_total_chars']
            pct = savings / r['raw_total_chars'] * 100 if r['raw_total_chars'] > 0 else 0
            f.write(f"{r['model']:<26} "
                   f"{r['raw_total_chars']:>12,} "
                   f"{r['orig_cleaned_total_chars']:>12,} "
                   f"{r['opt_cleaned_total_chars']:>12,} "
                   f"{savings:>8,} ({pct:.1f}%)\n")
        
        f.write("-" * 80 + "\n")
        total_raw = sum(r['raw_total_chars'] for r in all_results)
        total_orig = sum(r['orig_cleaned_total_chars'] for r in all_results)
        total_opt = sum(r['opt_cleaned_total_chars'] for r in all_results)
        f.write(f"{'TOTAL':<26} {total_raw:>12,} {total_orig:>12,} {total_opt:>12,}\n")
        
        # Conclusion
        f.write("\n" + "=" * 80 + "\n")
        f.write("CONCLUSION\n")
        f.write("=" * 80 + "\n\n")
        
        if total_opt_diffs < total_orig_diffs:
            f.write("OPTIMIZED CLEANING IS BETTER:\n")
            f.write(f"  - Original cleaning caused {total_orig_diffs} diagnosis extraction differences ({total_orig_diffs/total_entries*100:.1f}%)\n")
            f.write(f"  - Optimized cleaning caused {total_opt_diffs} diagnosis extraction differences ({total_opt_diffs/total_entries*100:.1f}%)\n")
            f.write(f"  - Improvement: {total_orig_diffs - total_opt_diffs} fewer differences\n")
        elif total_opt_diffs > total_orig_diffs:
            f.write("ORIGINAL CLEANING WAS BETTER:\n")
            f.write(f"  - Original cleaning caused {total_orig_diffs} diagnosis differences ({total_orig_diffs/total_entries*100:.1f}%)\n")
            f.write(f"  - Optimized cleaning caused {total_opt_diffs} diagnosis differences ({total_opt_diffs/total_entries*100:.1f}%)\n")
        else:
            f.write("BOTH METHODS PERFORM THE SAME:\n")
            f.write(f"  - Both caused {total_orig_diffs} diagnosis extraction differences ({total_orig_diffs/total_entries*100:.1f}%)\n")
        
        f.write("\nRECOMMENDATION:\n")
        if total_opt_diffs / total_entries < 0.05:
            f.write("  Optimized cleaning has minimal impact on diagnosis extraction (<5% difference).\n")
            f.write("  SAFE to use cleaned data for metrics calculation.\n")
        elif total_opt_diffs / total_entries < 0.20:
            f.write("  Optimized cleaning has moderate impact on diagnosis extraction (5-20% difference).\n")
            f.write("  CAUTION: Consider using raw data for highest accuracy.\n")
        else:
            f.write("  Optimized cleaning has HIGH impact on diagnosis extraction (>20% difference).\n")
            f.write("  STRONGLY RECOMMEND using raw data for metrics calculation.\n")
    
    print(f"Report saved to: {report_path}")
    print("\n--- Report Preview ---")
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # Print first 80 lines
        lines = content.split('\n')[:80]
        print('\n'.join(lines))
        if len(content.split('\n')) > 80:
            print(f"\n... (see full report at {report_path})")


if __name__ == "__main__":
    main()
