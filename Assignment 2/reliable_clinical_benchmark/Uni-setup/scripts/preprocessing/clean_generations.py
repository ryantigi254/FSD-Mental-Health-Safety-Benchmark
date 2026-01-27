#!/usr/bin/env python3
"""
STEP 1: Clean Generation Files (Optimized)

Removes repetitive/duplicate content from model outputs while preserving
diagnosis-relevant text. Uses O(n) hash-based deduplication.

Input:  results/{model}/study_a_generations.jsonl
Output: processed/study_a_cleaned/{model}/study_a_generations.jsonl

Usage:
    python scripts/preprocessing/clean_generations.py --study study_a
    python scripts/preprocessing/clean_generations.py --study study_a --model qwen3-lmstudio
    python scripts/preprocessing/clean_generations.py --all-studies

Next Step:
    python scripts/preprocessing/step2_extract_predictions.py --study study_a
"""

import argparse
import json
import re
import hashlib
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional


# ============================================================
# CLEANING ALGORITHM
# ============================================================

SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
WHITESPACE = re.compile(r'\s+')


def _normalize(text: str) -> str:
    """Normalize text for comparison."""
    return WHITESPACE.sub(' ', text.lower().strip())


def _hash(text: str) -> str:
    """Fast hash for deduplication."""
    return hashlib.md5(text.encode('utf-8', errors='ignore')).hexdigest()[:16]


def clean_text(text: str, min_length: int = 20, min_repeats: int = 2) -> Tuple[str, bool]:
    """
    Remove repetitive content from text.
    
    Returns:
        Tuple of (cleaned_text, was_modified)
    """
    if not text or len(text) < 200:
        return text, False
    
    original_len = len(text)
    
    # Phase 1: Remove consecutive duplicate lines
    lines = [l.strip() for l in text.splitlines()]
    deduped = []
    prev_hash = None
    count = 0
    
    for line in lines:
        if not line or len(line) < min_length:
            deduped.append(line)
            prev_hash = None
            count = 0
            continue
        
        h = _hash(_normalize(line))
        if h == prev_hash:
            count += 1
            if count >= min_repeats:
                continue  # Skip duplicate
        else:
            prev_hash = h
            count = 1
        deduped.append(line)
    
    # Phase 2: Sentence n-gram deduplication
    text_p1 = '\n'.join(deduped)
    sentences = [s.strip() for s in SENT_SPLIT.split(text_p1) if s.strip()]
    
    if len(sentences) < 4:
        result = text_p1.strip()
        modified = len(result) < original_len * 0.95
        return result, modified
    
    seen_bi = set()
    seen_tri = set()
    kept = []
    
    for sentence in sentences:
        if len(sentence) < min_length:
            kept.append(sentence)
            continue
        
        skip = False
        
        # Bigram check
        if len(kept) >= 1:
            bigram = _normalize(kept[-1] + ' ' + sentence)
            h = _hash(bigram)
            if h in seen_bi:
                skip = True
            else:
                seen_bi.add(h)
        
        # Trigram check
        if not skip and len(kept) >= 2:
            trigram = _normalize(' '.join(kept[-2:]) + ' ' + sentence)
            h = _hash(trigram)
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


def clean_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Clean a single JSONL entry."""
    cleaned = entry.copy()
    
    # Handle both field names
    text_field = "output_text" if "output_text" in entry else "response_text"
    original = entry.get(text_field, "")
    
    if original:
        cleaned_text, was_modified = clean_text(original)
        cleaned[text_field] = cleaned_text
        
        # Add metadata
        if "meta" not in cleaned:
            cleaned["meta"] = {}
        cleaned["meta"]["cleaned"] = was_modified
        cleaned["meta"]["original_length"] = len(original)
        cleaned["meta"]["cleaned_length"] = len(cleaned_text)
        if was_modified:
            cleaned["meta"]["chars_removed"] = len(original) - len(cleaned_text)
    
    return cleaned


def clean_file(input_path: Path, output_path: Path) -> Dict[str, int]:
    """Clean a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    stats = {"total": 0, "cleaned": 0, "chars_removed": 0}
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                stats["total"] += 1
                
                cleaned = clean_entry(entry)
                
                if cleaned.get("meta", {}).get("cleaned"):
                    stats["cleaned"] += 1
                    stats["chars_removed"] += cleaned["meta"].get("chars_removed", 0)
                
                fout.write(json.dumps(cleaned, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError:
                continue
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Clean generation files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--study", type=str, default="study_a",
                        help="Study to process (default: study_a)")
    parser.add_argument("--model", type=str, help="Process specific model only")
    parser.add_argument("--all-studies", action="store_true",
                        help="Process all studies (A, B, C)")
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent.parent
    results_dir = base_dir / "results"
    
    studies = ["study_a", "study_b", "study_c"] if args.all_studies else [args.study]
    
    print("=" * 60)
    print("STEP 1: CLEAN GENERATION FILES")
    print("=" * 60)
    print(f"Input:  {results_dir}")
    print(f"Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    total_stats = {"files": 0, "entries": 0, "cleaned": 0, "chars": 0}
    
    for study in studies:
        output_dir = base_dir / "processed" / f"{study}_cleaned"
        print(f"\n[{study.upper()}] → {output_dir}")
        print("-" * 40)
        
        models = [d for d in results_dir.iterdir() if d.is_dir()]
        if args.model:
            models = [m for m in models if m.name == args.model]
        
        for model_dir in sorted(models):
            input_file = model_dir / f"{study}_generations.jsonl"
            if not input_file.exists():
                continue
            
            output_file = output_dir / model_dir.name / f"{study}_generations.jsonl"
            
            print(f"  {model_dir.name}...", end=" ", flush=True)
            stats = clean_file(input_file, output_file)
            
            pct = stats["cleaned"] / stats["total"] * 100 if stats["total"] > 0 else 0
            print(f"OK ({stats['total']} entries, {stats['cleaned']} cleaned, {pct:.0f}%)")
            
            total_stats["files"] += 1
            total_stats["entries"] += stats["total"]
            total_stats["cleaned"] += stats["cleaned"]
            total_stats["chars"] += stats["chars_removed"]
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files processed:    {total_stats['files']}")
    print(f"Total entries:      {total_stats['entries']:,}")
    print(f"Entries cleaned:    {total_stats['cleaned']:,} ({total_stats['cleaned']/total_stats['entries']*100:.1f}%)")
    print(f"Characters removed: {total_stats['chars']:,}")
    print("=" * 60)
    print("\nNext step: python scripts/preprocessing/step2_extract_predictions.py --study study_a")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


