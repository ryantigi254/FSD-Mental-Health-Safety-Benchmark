"""
Compare raw vs cleaned data for ALL studies (A, B, C).

Measures:
- Character reduction from cleaning
- Impact on key extractions (diagnoses for A, agreement for B, entities for C)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


# ============================================================
# STUDY-SPECIFIC EXTRACTION FUNCTIONS
# ============================================================

def extract_study_a_diagnosis(text: str) -> str:
    """Extract diagnosis for Study A."""
    if not text or len(text) < 20:
        return "NO_OUTPUT"
    
    text_lower = text.lower()
    
    # Diagnosis patterns
    for pattern in [r"diagnosis:?\s*(.+?)(?:\n|$)", r"final diagnosis:?\s*(.+?)(?:\n|$)"]:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1).strip()[:80]
    
    # Last line fallback
    lines = [l.strip() for l in text.split('\n') if l.strip() and not l.startswith('[')]
    return lines[-1][:80] if lines else "EXTRACTION_FAILED"


def extract_study_b_agreement(text: str) -> str:
    """Extract agreement/disagreement for Study B."""
    if not text or len(text) < 20:
        return "NO_OUTPUT"
    
    text_lower = text.lower()
    
    # Agreement patterns
    agreement_patterns = [
        "i agree", "you're right", "you are right", "correct",
        "that's accurate", "valid point", "good observation"
    ]
    disagreement_patterns = [
        "i disagree", "i don't agree", "that's not correct", 
        "incorrect", "not accurate", "i must respectfully"
    ]
    
    for p in agreement_patterns:
        if p in text_lower[:500]:
            return "AGREES"
    
    for p in disagreement_patterns:
        if p in text_lower[:500]:
            return "DISAGREES"
    
    return "NEUTRAL"


def extract_study_c_entities(text: str) -> List[str]:
    """Extract key medical entities for Study C."""
    if not text or len(text) < 20:
        return []
    
    # Simple entity patterns (medications, conditions)
    patterns = [
        r'\b(depression|anxiety|ptsd|bipolar|schizophrenia)\b',
        r'\b(ssri|snri|benzodiazepine|antidepressant|antipsychotic)\b',
        r'\b(therapy|cbt|dbt|counseling|medication)\b',
        r'\b(suicidal|self-harm|panic|insomnia|mood)\b',
    ]
    
    entities = set()
    text_lower = text.lower()
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        entities.update(matches)
    
    return list(entities)


# ============================================================
# COMPARISON FUNCTIONS
# ============================================================

def compare_study(study: str, base_dir: Path) -> Dict:
    """Compare raw vs cleaned for a study."""
    raw_dir = base_dir / "results"
    cleaned_dir = base_dir / "processed" / f"{study}_cleaned"
    
    if not cleaned_dir.exists():
        return None
    
    models = [d.name for d in raw_dir.iterdir() if d.is_dir()]
    
    results = {
        'study': study,
        'models': [],
        'total_entries': 0,
        'total_raw_chars': 0,
        'total_cleaned_chars': 0,
        'extraction_diffs': 0,
    }
    
    for model in sorted(models):
        raw_file = raw_dir / model / f"{study}_generations.jsonl"
        cleaned_file = cleaned_dir / model / f"{study}_generations.jsonl"
        
        if not raw_file.exists() or not cleaned_file.exists():
            continue
        
        raw_entries = {
            (json.loads(l).get('id'), json.loads(l).get('turn_idx', 0)): json.loads(l)
            for l in open(raw_file, encoding='utf-8') if l.strip()
        }
        cleaned_entries = {
            (json.loads(l).get('id'), json.loads(l).get('turn_idx', 0)): json.loads(l)
            for l in open(cleaned_file, encoding='utf-8') if l.strip()
        }
        
        model_stats = {
            'model': model,
            'entries': len(raw_entries),
            'raw_chars': 0,
            'cleaned_chars': 0,
            'extraction_diffs': 0,
        }
        
        for key, raw in raw_entries.items():
            cleaned = cleaned_entries.get(key, {})
            
            # Get text field (Study A uses output_text, B/C use response_text)
            raw_text = raw.get('output_text', '') or raw.get('response_text', '')
            cleaned_text = cleaned.get('output_text', '') or cleaned.get('response_text', '')
            
            model_stats['raw_chars'] += len(raw_text)
            model_stats['cleaned_chars'] += len(cleaned_text)
            
            # Compare extractions by study
            if study == 'study_a':
                raw_extract = extract_study_a_diagnosis(raw_text)
                cleaned_extract = extract_study_a_diagnosis(cleaned_text)
            elif study == 'study_b':
                raw_extract = extract_study_b_agreement(raw_text)
                cleaned_extract = extract_study_b_agreement(cleaned_text)
            else:  # study_c
                raw_extract = tuple(sorted(extract_study_c_entities(raw_text)))
                cleaned_extract = tuple(sorted(extract_study_c_entities(cleaned_text)))
            
            if raw_extract != cleaned_extract:
                model_stats['extraction_diffs'] += 1
        
        results['models'].append(model_stats)
        results['total_entries'] += model_stats['entries']
        results['total_raw_chars'] += model_stats['raw_chars']
        results['total_cleaned_chars'] += model_stats['cleaned_chars']
        results['extraction_diffs'] += model_stats['extraction_diffs']
    
    return results


def main():
    base_dir = Path(__file__).parent.parent
    studies = ['study_a', 'study_b', 'study_c']
    
    report_path = base_dir / "all_studies_comparison_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RAW vs CLEANED COMPARISON - ALL STUDIES\n")
        f.write("=" * 80 + "\n\n")
        
        all_results = []
        
        for study in studies:
            result = compare_study(study, base_dir)
            if result and result['total_entries'] > 0:
                all_results.append(result)
        
        # Per-study summaries
        for r in all_results:
            f.write(f"\n{'='*60}\n")
            f.write(f"{r['study'].upper()}\n")
            f.write(f"{'='*60}\n\n")
            
            chars_removed = r['total_raw_chars'] - r['total_cleaned_chars']
            pct_removed = chars_removed / r['total_raw_chars'] * 100 if r['total_raw_chars'] > 0 else 0
            extract_pct = r['extraction_diffs'] / r['total_entries'] * 100 if r['total_entries'] > 0 else 0
            
            f.write(f"{'Model':<26} {'Entries':>8} {'Raw KB':>10} {'Clean KB':>10} {'Removed':>8} {'ExtDiff':>8}\n")
            f.write("-" * 72 + "\n")
            
            for m in r['models']:
                removed = m['raw_chars'] - m['cleaned_chars']
                pct = removed / m['raw_chars'] * 100 if m['raw_chars'] > 0 else 0
                f.write(f"{m['model']:<26} {m['entries']:>8} "
                       f"{m['raw_chars']/1024:>10.1f} {m['cleaned_chars']/1024:>10.1f} "
                       f"{pct:>7.1f}% {m['extraction_diffs']:>8}\n")
            
            f.write("-" * 72 + "\n")
            f.write(f"{'TOTAL':<26} {r['total_entries']:>8} "
                   f"{r['total_raw_chars']/1024:>10.1f} {r['total_cleaned_chars']/1024:>10.1f} "
                   f"{pct_removed:>7.1f}% {r['extraction_diffs']:>8}\n")
            f.write(f"\nExtraction difference rate: {extract_pct:.1f}%\n")
            
            if extract_pct < 5:
                f.write("✅ SAFE: Cleaning has minimal impact on extractions\n")
            elif extract_pct < 15:
                f.write("⚠️ MODERATE: Some extraction differences, review recommended\n")
            else:
                f.write("❌ HIGH IMPACT: Cleaning significantly affects extractions\n")
        
        # Overall summary
        f.write("\n" + "=" * 80 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        for r in all_results:
            chars_removed = r['total_raw_chars'] - r['total_cleaned_chars']
            pct = chars_removed / r['total_raw_chars'] * 100 if r['total_raw_chars'] > 0 else 0
            extract_pct = r['extraction_diffs'] / r['total_entries'] * 100 if r['total_entries'] > 0 else 0
            status = "✅" if extract_pct < 5 else "⚠️" if extract_pct < 15 else "❌"
            f.write(f"{r['study']}: {r['total_entries']} entries, "
                   f"{pct:.1f}% chars removed, {extract_pct:.1f}% extraction diff {status}\n")
    
    print(f"Report saved to: {report_path}")
    
    # Print summary to console
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in all_results:
        chars_removed = r['total_raw_chars'] - r['total_cleaned_chars']
        pct = chars_removed / r['total_raw_chars'] * 100 if r['total_raw_chars'] > 0 else 0
        extract_pct = r['extraction_diffs'] / r['total_entries'] * 100 if r['total_entries'] > 0 else 0
        status = "SAFE" if extract_pct < 5 else "MODERATE" if extract_pct < 15 else "HIGH IMPACT"
        print(f"{r['study']}: {pct:.1f}% cleaned, {extract_pct:.1f}% extraction diff - {status}")


if __name__ == "__main__":
    main()
