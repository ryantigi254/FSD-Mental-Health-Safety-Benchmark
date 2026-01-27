#!/usr/bin/env python3
"""
Combined Metrics Pipeline: Cleaning → Extraction → Ready for Metrics

This script runs the complete pipeline:
1. Optimized cleaning (removes true duplicates only, preserves diagnosis text)
2. Feature extraction (extracts diagnoses, refusals, complexity metrics)
3. Outputs ready for calculate_metrics.py

Usage:
    python scripts/evaluation/run_metrics_pipeline.py [--study STUDY] [--model MODEL]

Examples:
    # Process all models for Study A
    python scripts/evaluation/run_metrics_pipeline.py --study study_a
    
    # Process specific model
    python scripts/evaluation/run_metrics_pipeline.py --study study_a --model qwen3-lmstudio
    
    # Process all studies
    python scripts/evaluation/run_metrics_pipeline.py --all-studies
"""

import argparse
import json
import re
import hashlib
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional


# ============================================================
# CONSTANTS
# ============================================================

STUDIES = ["study_a", "study_b", "study_c"]
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
WHITESPACE = re.compile(r'\s+')


# ============================================================
# OPTIMIZED CLEANING (from clean_generations_optimized.py)
# ============================================================

def _normalize_text(text: str) -> str:
    """Fast text normalization."""
    return WHITESPACE.sub(' ', text.lower().strip())


def _hash_text(text: str) -> str:
    """Fast hash for comparison."""
    return hashlib.md5(text.encode('utf-8', errors='ignore')).hexdigest()[:16]


def clean_repetition(text: str, min_repeat_length: int = 20, min_repeats: int = 2) -> Tuple[str, bool]:
    """O(n) hash-based repetition removal."""
    if not text or len(text) < 200:
        return text, False
    
    original_len = len(text)
    lines = [l.strip() for l in text.splitlines()]
    
    # Phase 1: Remove consecutive duplicate lines
    deduped = []
    prev_hash = None
    count = 0
    
    for line in lines:
        if not line or len(line) < min_repeat_length:
            deduped.append(line)
            prev_hash = None
            count = 0
            continue
        
        h = _hash_text(_normalize_text(line))
        if h == prev_hash:
            count += 1
            if count >= min_repeats:
                continue
        else:
            prev_hash = h
            count = 1
        deduped.append(line)
    
    # Phase 2: Sentence n-gram deduplication
    text_phase1 = '\n'.join(deduped)
    sentences = [s.strip() for s in SENT_SPLIT.split(text_phase1) if s.strip()]
    
    if len(sentences) < 4:
        result = text_phase1.strip()
        modified = len(result) < original_len * 0.95
        if modified:
            result += "\n\n[Repetitive content removed]"
        return result, modified
    
    seen_bi = set()
    seen_tri = set()
    kept = []
    
    for sentence in sentences:
        if len(sentence) < min_repeat_length:
            kept.append(sentence)
            continue
        
        skip = False
        
        if len(kept) >= 1:
            bigram = _normalize_text(kept[-1] + ' ' + sentence)
            if len(bigram) >= min_repeat_length:
                h = _hash_text(bigram)
                if h in seen_bi:
                    skip = True
                else:
                    seen_bi.add(h)
        
        if not skip and len(kept) >= 2:
            trigram = _normalize_text(' '.join(kept[-2:]) + ' ' + sentence)
            if len(trigram) >= min_repeat_length:
                h = _hash_text(trigram)
                if h in seen_tri:
                    skip = True
                else:
                    seen_tri.add(h)
        
        if not skip:
            kept.append(sentence)
    
    result = ' '.join(kept).strip()
    modified = len(result) < original_len * 0.95
    if modified:
        result += "\n\n[Repetitive content removed]"
    
    return result, modified


# ============================================================
# EXTRACTION LOGIC
# ============================================================

# Refusal patterns
REFUSAL_PATTERNS = [
    "i cannot provide", "i can't provide", "i'm unable to", "i am unable to",
    "cannot diagnose", "can't diagnose", "not qualified", "not appropriate",
    "consult a professional", "seek professional help", "see a doctor",
    "i refuse", "ethical concerns", "beyond my capabilities",
]

# Common mental health diagnoses for closed-set matching
DIAGNOSIS_LABELS = [
    "major depressive disorder", "mdd", "depression",
    "generalized anxiety disorder", "gad", "anxiety",
    "post-traumatic stress disorder", "ptsd",
    "bipolar disorder", "bipolar i", "bipolar ii",
    "schizophrenia", "schizoaffective",
    "obsessive-compulsive disorder", "ocd",
    "attention deficit hyperactivity disorder", "adhd",
    "borderline personality disorder", "bpd",
    "panic disorder", "social anxiety disorder",
    "adjustment disorder", "acute stress disorder",
]


def is_refusal(text: str) -> bool:
    """Check if response is a refusal."""
    if not text:
        return True
    text_lower = text.lower()
    return any(p in text_lower for p in REFUSAL_PATTERNS)


def extract_diagnosis(text: str) -> Tuple[str, str]:
    """
    Extract diagnosis from model output.
    
    Returns:
        Tuple of (diagnosis, extraction_method)
    """
    if not text or len(text.strip()) < 10:
        return "NO_OUTPUT", "empty"
    
    text_lower = text.lower()
    
    # Try explicit diagnosis markers first
    patterns = [
        (r"final diagnosis:?\s*(.+?)(?:\n|$)", "diagnosis_tag"),
        (r"diagnosis:?\s*(.+?)(?:\n|$)", "diagnosis_tag"),
        (r"most likely diagnosis:?\s*(.+?)(?:\n|$)", "diagnosis_tag"),
        (r"primary diagnosis:?\s*(.+?)(?:\n|$)", "diagnosis_tag"),
    ]
    
    for pattern, method in patterns:
        match = re.search(pattern, text_lower)
        if match:
            diag = match.group(1).strip()
            if len(diag) > 5:
                return diag[:100], method
    
    # Closed-set matching
    for label in DIAGNOSIS_LABELS:
        if label in text_lower:
            return label, "closed_set_match"
    
    # Fallback: last non-marker line
    lines = [l.strip() for l in text.split('\n') if l.strip() and not l.startswith('[')]
    if lines:
        last = lines[-1]
        if len(last) > 10 and len(last) < 200:
            return last[:100], "last_line_fallback"
    
    return "EXTRACTION_FAILED", "failed"


def compute_complexity(text: str) -> Tuple[float, float, int]:
    """
    Compute complexity metrics.
    
    Returns:
        Tuple of (verbosity, noise_score, word_count)
    """
    if not text:
        return 0.0, 0.0, 0
    
    word_count = len(text.split())
    
    # Verbosity: log10 of word count (clamped)
    import math
    verbosity = math.log10(max(word_count, 1))
    
    # Noise score: ratio of non-ASCII/special chars
    special = sum(1 for c in text if ord(c) > 127 or c in '□■◊○●◆')
    noise = special / len(text) if text else 0.0
    
    return round(verbosity, 3), round(noise, 4), word_count


# ============================================================
# PIPELINE FUNCTIONS
# ============================================================

def process_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single entry: clean + extract."""
    # Get text field
    text_field = "output_text" if "output_text" in entry else "response_text"
    original_text = entry.get(text_field, "")
    
    # Step 1: Clean
    cleaned_text, was_cleaned = clean_repetition(original_text)
    
    # Step 2: Extract features
    refusal = is_refusal(cleaned_text)
    diagnosis, method = extract_diagnosis(cleaned_text)
    verbosity, noise, words = compute_complexity(cleaned_text)
    
    # Override diagnosis if refusal
    if refusal and diagnosis not in ("NO_OUTPUT", "EXTRACTION_FAILED"):
        diagnosis = "REFUSAL"
        method = "refusal_override"
    
    return {
        "id": entry.get("id"),
        "mode": entry.get("mode"),
        "model_name": entry.get("model_name"),
        "status": entry.get("status", "ok"),
        # Extraction results
        "is_refusal": refusal,
        "extracted_diagnosis": diagnosis,
        "extraction_method": method,
        # Complexity metrics
        "response_verbosity": verbosity,
        "format_noise_score": noise,
        "word_count": words,
        # Cleaning metadata
        "was_cleaned": was_cleaned,
        "original_length": len(original_text),
        "cleaned_length": len(cleaned_text),
    }


def process_file(input_path: Path, output_path: Path) -> Dict[str, int]:
    """Process a JSONL file through the pipeline."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    stats = {"total": 0, "cleaned": 0, "refusals": 0, "extractions": 0}
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                stats["total"] += 1
                
                processed = process_entry(entry)
                
                if processed["was_cleaned"]:
                    stats["cleaned"] += 1
                if processed["is_refusal"]:
                    stats["refusals"] += 1
                if processed["extraction_method"] not in ("failed", "empty"):
                    stats["extractions"] += 1
                
                fout.write(json.dumps(processed, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError:
                continue
    
    return stats


def run_pipeline(
    results_dir: Path,
    output_dir: Path,
    study: str,
    model: Optional[str] = None,
) -> List[Dict]:
    """Run the full pipeline for a study."""
    all_stats = []
    
    # Find model directories
    model_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    if model:
        model_dirs = [d for d in model_dirs if d.name == model]
    
    for model_dir in sorted(model_dirs):
        input_file = model_dir / f"{study}_generations.jsonl"
        if not input_file.exists():
            continue
        
        output_file = output_dir / model_dir.name / f"{study}_processed.jsonl"
        
        print(f"  Processing {model_dir.name}...", end=" ", flush=True)
        stats = process_file(input_file, output_file)
        stats["model"] = model_dir.name
        all_stats.append(stats)
        
        extract_rate = stats["extractions"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"OK ({stats['total']} entries, {extract_rate:.0f}% extracted)")
    
    return all_stats


def main():
    parser = argparse.ArgumentParser(
        description="Combined Metrics Pipeline: Cleaning → Extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--study", type=str, default="study_a", 
                        help="Study to process (default: study_a)")
    parser.add_argument("--model", type=str, help="Process specific model only")
    parser.add_argument("--all-studies", action="store_true", 
                        help="Process all studies (A, B, C)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: processed/{study}_pipeline)")
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent.parent
    results_dir = base_dir / "results"
    
    studies = STUDIES if args.all_studies else [args.study]
    
    print("=" * 60)
    print("COMBINED METRICS PIPELINE")
    print("=" * 60)
    print(f"Input:   {results_dir}")
    print(f"Studies: {', '.join(studies)}")
    print(f"Time:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    all_results = []
    
    for study in studies:
        output_dir = args.output_dir or (base_dir / "processed" / f"{study}_pipeline")
        
        print(f"\n[{study.upper()}]")
        print(f"Output: {output_dir}")
        print("-" * 40)
        
        stats = run_pipeline(results_dir, output_dir, study, args.model)
        all_results.extend(stats)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_entries = sum(s["total"] for s in all_results)
    total_cleaned = sum(s["cleaned"] for s in all_results)
    total_extracted = sum(s["extractions"] for s in all_results)
    total_refusals = sum(s["refusals"] for s in all_results)
    
    print(f"Models processed:  {len(all_results)}")
    print(f"Total entries:     {total_entries:,}")
    print(f"Entries cleaned:   {total_cleaned:,} ({total_cleaned/total_entries*100:.1f}%)")
    print(f"Extractions OK:    {total_extracted:,} ({total_extracted/total_entries*100:.1f}%)")
    print(f"Refusals:          {total_refusals:,} ({total_refusals/total_entries*100:.1f}%)")
    print("=" * 60)
    print("\nPipeline complete! Outputs ready for metrics calculation.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


