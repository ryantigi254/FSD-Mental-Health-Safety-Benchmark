"""
Optimized generation file cleaner with improved efficiency.

Key optimizations:
1. O(n) hash-based n-gram detection instead of O(n²) string comparison
2. Single-pass text processing
3. Pre-computed normalization
4. Streaming file processing (memory efficient)

Usage:
    python scripts/preprocessing/clean_generations_optimized.py [--model MODEL_NAME] [--study STUDY_NAME]
"""

import argparse
import json
import re
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict


# Pre-compiled patterns for efficiency
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
WHITESPACE = re.compile(r'\s+')


def _normalize_text(text: str) -> str:
    """Fast text normalization for comparison."""
    return WHITESPACE.sub(' ', text.lower().strip())


def _hash_text(text: str) -> str:
    """Fast hash for text comparison."""
    return hashlib.md5(text.encode('utf-8', errors='ignore')).hexdigest()[:16]


def _clean_repetition_optimized(
    text: str,
    min_repeat_length: int = 20,
    min_repeats: int = 2,
    max_output_ratio: float = 0.7,
) -> Tuple[str, bool]:
    """
    Optimized repetition removal - O(n) time complexity.
    
    Strategy:
    1. Split into lines, hash each line
    2. Track consecutive duplicates (O(n) single pass)
    3. Track n-gram hashes in sliding window (O(n) with O(1) lookup)
    4. Remove duplicated content
    
    Returns:
        Tuple of (cleaned_text, was_modified)
    """
    if not text or len(text) < 200:
        return text, False
    
    original_len = len(text)
    lines = [l.strip() for l in text.splitlines()]
    
    # === Phase 1: Remove consecutive duplicate lines (O(n)) ===
    deduped_lines = []
    prev_hash = None
    consecutive_count = 0
    
    for line in lines:
        if not line or len(line) < min_repeat_length:
            deduped_lines.append(line)
            prev_hash = None
            consecutive_count = 0
            continue
            
        line_hash = _hash_text(_normalize_text(line))
        
        if line_hash == prev_hash:
            consecutive_count += 1
            if consecutive_count >= min_repeats:
                continue  # Skip duplicate
        else:
            prev_hash = line_hash
            consecutive_count = 1
        
        deduped_lines.append(line)
    
    # === Phase 2: Sentence-level n-gram deduplication (O(n)) ===
    text_after_line_dedup = '\n'.join(deduped_lines)
    sentences = [s.strip() for s in SENT_SPLIT.split(text_after_line_dedup) if s.strip()]
    
    if len(sentences) < 4:
        result = text_after_line_dedup.strip()
        was_modified = len(result) < original_len * 0.95
        if was_modified:
            result += "\n\n[Repetitive content removed]"
        return result, was_modified
    
    # Hash-based n-gram tracking
    seen_bigrams = set()
    seen_trigrams = set()
    kept_sentences = []
    
    for i, sentence in enumerate(sentences):
        if len(sentence) < min_repeat_length:
            kept_sentences.append(sentence)
            continue
        
        skip = False
        
        # Check bigram (current + previous)
        if len(kept_sentences) >= 1:
            bigram_text = _normalize_text(kept_sentences[-1] + ' ' + sentence)
            if len(bigram_text) >= min_repeat_length:
                bigram_hash = _hash_text(bigram_text)
                if bigram_hash in seen_bigrams:
                    skip = True
                else:
                    seen_bigrams.add(bigram_hash)
        
        # Check trigram (current + previous 2)
        if not skip and len(kept_sentences) >= 2:
            trigram_text = _normalize_text(' '.join(kept_sentences[-2:]) + ' ' + sentence)
            if len(trigram_text) >= min_repeat_length:
                trigram_hash = _hash_text(trigram_text)
                if trigram_hash in seen_trigrams:
                    skip = True
                else:
                    seen_trigrams.add(trigram_hash)
        
        if not skip:
            kept_sentences.append(sentence)
    
    # === Phase 3: Detect tail repetition (common in LLM outputs) ===
    if len(kept_sentences) > 10:
        # Check if last few sentences repeat earlier content
        tail_size = min(5, len(kept_sentences) // 4)
        tail_hashes = [_hash_text(_normalize_text(s)) for s in kept_sentences[-tail_size:]]
        
        # Find where repetition starts
        for i in range(len(kept_sentences) - tail_size - 1, -1, -1):
            if _hash_text(_normalize_text(kept_sentences[i])) == tail_hashes[0]:
                # Found potential repeat start - verify sequence
                match = True
                for j, h in enumerate(tail_hashes):
                    if i + j >= len(kept_sentences) - tail_size:
                        break
                    if _hash_text(_normalize_text(kept_sentences[i + j])) != h:
                        match = False
                        break
                if match:
                    kept_sentences = kept_sentences[:i + tail_size]
                    break
    
    # Build result
    result = ' '.join(kept_sentences).strip()
    was_modified = len(result) < original_len * 0.95
    
    if was_modified:
        result += "\n\n[Repetitive content removed]"
    
    return result, was_modified


def clean_entry(entry: Dict[str, Any], min_repeat_length: int = 20) -> Dict[str, Any]:
    """Clean a single generation entry."""
    cleaned = entry.copy()
    
    # Find text field
    text_field = None
    if "output_text" in cleaned and cleaned["output_text"]:
        text_field = "output_text"
    elif "response_text" in cleaned and cleaned["response_text"]:
        text_field = "response_text"
    
    if not text_field:
        return cleaned
    
    original_text = cleaned[text_field]
    cleaned_text, was_modified = _clean_repetition_optimized(
        original_text, 
        min_repeat_length=min_repeat_length
    )
    
    if was_modified:
        cleaned[text_field] = cleaned_text
        if "meta" not in cleaned:
            cleaned["meta"] = {}
        cleaned["meta"]["cleaned"] = True
        cleaned["meta"]["original_length"] = len(original_text)
        cleaned["meta"]["cleaned_length"] = len(cleaned_text)
        cleaned["meta"]["removed_chars"] = len(original_text) - len(cleaned_text)
        cleaned["meta"]["cleaner"] = "clean_generations_optimized.py"
    
    return cleaned


def clean_jsonl_file(
    input_path: Path,
    output_path: Path,
    min_repeat_length: int = 20,
) -> Tuple[int, int, int]:
    """
    Clean a JSONL file using streaming processing.
    
    Returns:
        Tuple of (total_entries, cleaned_entries, chars_removed)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total = 0
    cleaned_count = 0
    chars_removed = 0
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                total += 1
                
                cleaned_entry = clean_entry(entry, min_repeat_length)
                
                if cleaned_entry.get("meta", {}).get("cleaned"):
                    cleaned_count += 1
                    chars_removed += cleaned_entry["meta"]["removed_chars"]
                
                fout.write(json.dumps(cleaned_entry, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError:
                continue
    
    return total, cleaned_count, chars_removed


def main():
    parser = argparse.ArgumentParser(description="Optimized generation file cleaner")
    parser.add_argument("--model", type=str, help="Filter by model name")
    parser.add_argument("--study", type=str, default="study_a", help="Study name (default: study_a)")
    parser.add_argument("--min-repeat-length", type=int, default=20, help="Min repeat length")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent.parent
    results_dir = base_dir / "results"
    output_dir = base_dir / "processed" / "cleaned" / f"{args.study}_cleaned_optimized"
    
    # Find files to process
    models = [d.name for d in results_dir.iterdir() if d.is_dir()]
    if args.model:
        models = [m for m in models if m == args.model]
    
    print(f"Optimized Generation Cleaner")
    print(f"=" * 60)
    print(f"Input:  {results_dir}")
    print(f"Output: {output_dir}")
    print(f"Study:  {args.study}")
    print(f"Models: {len(models)} found")
    print(f"=" * 60)
    
    if args.dry_run:
        for model in models:
            src = results_dir / model / f"{args.study}_generations.jsonl"
            if src.exists():
                print(f"Would process: {model} ({src.stat().st_size / 1024:.1f} KB)")
        return 0
    
    total_files = 0
    total_entries = 0
    total_cleaned = 0
    total_chars = 0
    
    for model in sorted(models):
        src = results_dir / model / f"{args.study}_generations.jsonl"
        if not src.exists():
            continue
        
        dst = output_dir / model / f"{args.study}_generations.jsonl"
        
        print(f"\nProcessing {model}...")
        entries, cleaned, chars = clean_jsonl_file(src, dst, args.min_repeat_length)
        
        print(f"  Entries: {entries}, Cleaned: {cleaned} ({cleaned/entries*100:.1f}%)")
        print(f"  Chars removed: {chars:,}")
        
        total_files += 1
        total_entries += entries
        total_cleaned += cleaned
        total_chars += chars
    
    print(f"\n" + "=" * 60)
    print(f"SUMMARY")
    print(f"=" * 60)
    print(f"Files processed:  {total_files}")
    print(f"Total entries:    {total_entries}")
    print(f"Entries cleaned:  {total_cleaned} ({total_cleaned/total_entries*100:.1f}%)")
    print(f"Chars removed:    {total_chars:,}")
    print(f"Output location:  {output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())


