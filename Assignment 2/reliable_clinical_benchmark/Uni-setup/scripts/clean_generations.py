"""
Clean existing generation files by removing repetitive content.

This script processes all generation JSONL files in the results/ directory,
removes excessive repetition using the same algorithm as Study C pipeline,
and saves cleaned versions to the processed/ directory.

Usage:
    python scripts/clean_generations.py [--model MODEL_NAME] [--study STUDY_NAME]

Examples:
    # Clean all files
    python scripts/clean_generations.py
    
    # Clean specific model
    python scripts/clean_generations.py --model piaget-8b-local
    
    # Clean specific study
    python scripts/clean_generations.py --study study_c
    
    # Clean specific model and study
    python scripts/clean_generations.py --model piaget-8b-local --study study_c
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import re
try:
    from rapidfuzz import fuzz, utils, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _scan_mode_clean(text: str, min_repeat_length: int = 20, min_repeats: int = 2) -> str:
    """Exact-match scan cleaning: remove all repeated segments in one pass."""
    if not text or len(text) < 200:
        return text

    # Remove consecutive duplicate lines beyond min_repeats
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    deduped_lines = []
    prev = None
    repeat_count = 0
    for line in lines:
        if line == prev and len(line) >= min_repeat_length:
            repeat_count += 1
            if repeat_count >= min_repeats:
                continue
        else:
            prev = line
            repeat_count = 1
        deduped_lines.append(line)

    # Sentence n-gram de-duplication (bi/tri/quad-grams)
    sentences = [s.strip() for s in SENT_SPLIT.split(" ".join(deduped_lines)) if s.strip()]
    if len(sentences) < 4:
        return "\n".join(deduped_lines).rstrip()

    kept = []
    seen_ngrams = {2: set(), 3: set(), 4: set()}
    for i, sentence in enumerate(sentences):
        candidate = kept + [sentence]
        skip = False
        for n in (2, 3, 4):
            if len(candidate) < n:
                continue
            ngram = " ".join(candidate[-n:]).strip()
            if len(ngram) < min_repeat_length:
                continue
            if ngram in seen_ngrams[n]:
                skip = True
                break
        if skip:
            continue
        kept.append(sentence)
        for n in (2, 3, 4):
            if len(kept) < n:
                continue
            ngram = " ".join(kept[-n:]).strip()
            if len(ngram) >= min_repeat_length:
                seen_ngrams[n].add(ngram)

    cleaned = " ".join(kept).strip()
    if cleaned != text.strip():
        cleaned += "\n\n[Repetitive content removed]"
    return cleaned


def _standalone_remove_repetition(
    text: str,
    max_repetition_ratio: float = 0.3,
    min_repeat_length: int = 50,
    similarity_cutoff: int = 95,
) -> str:
    """High-accuracy repetition removal using RapidFuzz (fallback to exact matching)."""
    if not text or len(text) < 200:
        return text

    original_text = text

    # Normalize line and sentence units
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    sentences = re.split(r'([.!?]+\s+)', text)
    sentences = [sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '')
                 for i in range(0, len(sentences), 2) if sentences[i].strip()]

    if RAPIDFUZZ_AVAILABLE:
        # 1) Detect near-duplicate tail lines (common looping)
        if len(lines) > 5:
            recent = lines[-30:] if len(lines) > 30 else lines
            base = recent[-1]
            if len(base) >= min_repeat_length:
                repeats = []
                for idx in range(len(recent) - 1, -1, -1):
                    sim = fuzz.ratio(base, recent[idx], score_cutoff=similarity_cutoff)
                    if sim >= similarity_cutoff:
                        repeats.append(idx)
                    else:
                        break
                if len(repeats) >= 3:
                    cut_idx = len(lines) - len(recent) + min(repeats)
                    kept = lines[:cut_idx]
                    trimmed = '\n'.join(kept).rstrip()
                    return trimmed + "\n[Repetitive content removed]"

        # 2) Detect repeated sentence windows using token_set_ratio
        if len(sentences) >= 6:
            max_window = min(6, len(sentences) // 3)
            for window in range(max_window, 1, -1):
                window_texts = [
                    ' '.join(s.strip() for s in sentences[i:i + window])
                    for i in range(0, len(sentences) - window + 1)
                ]
                window_norm = [utils.default_process(w) for w in window_texts]
                for i, w in enumerate(window_norm[:-window]):
                    if len(w) < min_repeat_length:
                        continue
                    matches = process.extract(
                        w,
                        window_norm[i + window:],
                        scorer=fuzz.token_set_ratio,
                        score_cutoff=similarity_cutoff,
                    )
                    if len(matches) >= 2:
                        total_after = len(sentences) - i - window
                        ratio = (len(matches) * window) / total_after if total_after > 0 else 0
                        if ratio > max_repetition_ratio:
                            kept = sentences[:i + window]
                            trimmed = ''.join(kept).strip()
                            return trimmed + "\n\n[Repetitive content removed]"

        # 3) Exact duplicate sentences (fast path)
        if len(sentences) >= 6:
            seen = {}
            for i, s in enumerate(sentences):
                key = utils.default_process(s)
                if len(key) < min_repeat_length:
                    continue
                seen.setdefault(key, []).append(i)
            for idxs in seen.values():
                if len(idxs) >= 3:
                    cut = idxs[0] + 1
                    trimmed = ''.join(sentences[:cut]).strip()
                    return trimmed + "\n\n[Repetitive content removed]"

    # Fallback: exact matching without RapidFuzz
    if len(lines) > 5:
        current = lines[-1]
        if len(current) >= min_repeat_length:
            count = 1
            for line in reversed(lines[:-1]):
                if line == current:
                    count += 1
                else:
                    break
            if count >= 3:
                kept = lines[:-count + 1]
                trimmed = '\n'.join(kept).rstrip()
                return trimmed + "\n[Repetitive content removed]"

    return original_text

# Import cleaning function - try to import from pipeline, fallback to standalone version
try:
    # Add src to path to import the cleaning function
    script_dir = Path(__file__).parent
    uni_setup_dir = script_dir.parent
    src_dir = uni_setup_dir / "src"
    sys.path.insert(0, str(src_dir))
    sys.path.insert(0, str(uni_setup_dir))
    
    from reliable_clinical_benchmark.pipelines.study_c import _remove_repetition as _pipeline_remove_repetition
except Exception:
    def _remove_repetition(
        text: str,
        max_repetition_ratio: float = 0.3,
        min_repeat_length: int = 50,
        similarity_cutoff: int = 95,
    ) -> str:
        return _standalone_remove_repetition(
            text,
            max_repetition_ratio=max_repetition_ratio,
            min_repeat_length=min_repeat_length,
            similarity_cutoff=similarity_cutoff,
        )

else:
    def _remove_repetition(
        text: str,
        max_repetition_ratio: float = 0.3,
        min_repeat_length: int = 50,
        similarity_cutoff: int = 95,
    ) -> str:
        if similarity_cutoff != 95:
            # Fallback to standalone to honor relaxed similarity thresholds
            return _standalone_remove_repetition(
                text,
                max_repetition_ratio=max_repetition_ratio,
                min_repeat_length=min_repeat_length,
                similarity_cutoff=similarity_cutoff,
            )
        try:
            return _pipeline_remove_repetition(
                text,
                max_repetition_ratio=max_repetition_ratio,
                min_repeat_length=min_repeat_length,
            )
        except TypeError:
            return _pipeline_remove_repetition(text)


def clean_generation_entry(
    entry: Dict[str, Any],
    *,
    max_repetition_ratio: float,
    min_repeat_length: int,
    similarity_cutoff: int,
    scan_mode: bool,
    scan_min_repeats: int,
) -> Dict[str, Any]:
    """
    Clean a single generation entry by removing repetition from response_text or output_text.
    
    Handles both Study A format (output_text) and Study B/C format (response_text).
    
    Args:
        entry: Dictionary containing generation data
    
    Returns:
        Copy of entry with cleaned text field (if cleaning occurred)
    """
    cleaned_entry = entry.copy()
    
    # Check which field name is used (Study A uses 'output_text', Study B/C use 'response_text')
    text_field = None
    if "response_text" in cleaned_entry and cleaned_entry["response_text"]:
        text_field = "response_text"
    elif "output_text" in cleaned_entry and cleaned_entry["output_text"]:
        text_field = "output_text"
    
    if text_field:
        original_text = cleaned_entry[text_field]
        if scan_mode:
            cleaned_text = _scan_mode_clean(
                original_text,
                min_repeat_length=min_repeat_length,
                min_repeats=scan_min_repeats,
            )
        else:
            cleaned_text = _remove_repetition(
                original_text,
                max_repetition_ratio=max_repetition_ratio,
                min_repeat_length=min_repeat_length,
                similarity_cutoff=similarity_cutoff,
            )
        
        if cleaned_text != original_text:
            cleaned_entry[text_field] = cleaned_text
            # Add metadata about cleaning
            if "meta" not in cleaned_entry:
                cleaned_entry["meta"] = {}
            cleaned_entry["meta"]["cleaned"] = True
            cleaned_entry["meta"]["original_length"] = len(original_text)
            cleaned_entry["meta"]["cleaned_length"] = len(cleaned_text)
            cleaned_entry["meta"]["removed_chars"] = len(original_text) - len(cleaned_text)
            cleaned_entry["meta"]["cleaned_by"] = "clean_generations.py"
        else:
            # No cleaning occurred
            if "meta" not in cleaned_entry:
                cleaned_entry["meta"] = {}
            cleaned_entry["meta"]["cleaned"] = False
    
    return cleaned_entry


def clean_jsonl_file(
    input_path: Path,
    output_path: Path,
    *,
    max_repetition_ratio: float,
    min_repeat_length: int,
    similarity_cutoff: int,
    scan_mode: bool,
    scan_min_repeats: int,
) -> tuple[int, int]:
    """
    Clean a JSONL file by removing repetition from all response_text fields.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
    
    Returns:
        Tuple of (total_entries, cleaned_entries)
    """
    total_entries = 0
    cleaned_entries = 0
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                total_entries += 1
                
                # Clean the entry
                cleaned_entry = clean_generation_entry(
                    entry,
                    max_repetition_ratio=max_repetition_ratio,
                    min_repeat_length=min_repeat_length,
                    similarity_cutoff=similarity_cutoff,
                    scan_mode=scan_mode,
                    scan_min_repeats=scan_min_repeats,
                )
                
                # Check if cleaning occurred (check both field names)
                original_text = entry.get("response_text") or entry.get("output_text", "")
                cleaned_text = cleaned_entry.get("response_text") or cleaned_entry.get("output_text", "")
                
                if cleaned_text != original_text:
                    cleaned_entries += 1
                
                # Write cleaned entry
                f_out.write(json.dumps(cleaned_entry, ensure_ascii=False) + "\n")
                
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num} of {input_path}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num} of {input_path}: {e}")
                continue
    
    return total_entries, cleaned_entries


def find_generation_files(results_dir: Path, model_name: Optional[str] = None, study_name: Optional[str] = None) -> list[tuple[Path, Path]]:
    """
    Find all generation JSONL files to process.
    
    Args:
        results_dir: Root results directory
        model_name: Optional model name filter (e.g., 'piaget-8b-local')
        study_name: Optional study name filter (e.g., 'study_c')
    
    Returns:
        List of tuples: (input_path, output_path)
    """
    files_to_process = []
    
    # Look for model directories
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        # Filter by model name if specified
        if model_name and model_dir.name != model_name:
            continue
        
        # Look for generation files
        for gen_file in model_dir.glob("*_generations.jsonl"):
            # Extract study name from filename (e.g., 'study_c_generations.jsonl' -> 'study_c')
            file_study = gen_file.stem.replace("_generations", "")
            
            # Filter by study name if specified
            if study_name and file_study != study_name:
                continue
            
            # Determine output path: processed/{model}/{study}_generations.jsonl
            output_path = results_dir.parent / "processed" / model_dir.name / gen_file.name
            
            files_to_process.append((gen_file, output_path))
    
    return files_to_process


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Clean generation files by removing repetitive content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Filter by model name (e.g., 'piaget-8b-local')"
    )
    parser.add_argument(
        "--study",
        type=str,
        help="Filter by study name (e.g., 'study_c', 'study_a', 'study_b')"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent.parent / "results",
        help="Results directory (default: results/)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually cleaning files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Custom output directory (default: processed/{model}/)"
    )
    parser.add_argument(
        "--max-repetition-ratio",
        type=float,
        default=0.3,
        help="Threshold for repetition ratio (lower is stricter; default: 0.3)"
    )
    parser.add_argument(
        "--min-repeat-length",
        type=int,
        default=50,
        help="Minimum repeat length to consider (default: 50)"
    )
    parser.add_argument(
        "--similarity-cutoff",
        type=int,
        default=95,
        help="Fuzzy similarity cutoff for repeats (default: 95)"
    )
    parser.add_argument(
        "--scan-mode",
        action="store_true",
        help="Use exact-match scan mode for repetition removal"
    )
    parser.add_argument(
        "--scan-min-repeats",
        type=int,
        default=2,
        help="Minimum repeat count for scan mode (default: 2)"
    )
    
    args = parser.parse_args()
    
    # Find files to process
    files_to_process = find_generation_files(args.results_dir, args.model, args.study)
    
    # Override output paths if custom output directory specified
    if args.output_dir:
        custom_output_dir = Path(args.output_dir)
        files_to_process = [
            (input_path, custom_output_dir / input_path.parent.name / input_path.name)
            for input_path, _ in files_to_process
        ]
    
    if not files_to_process:
        print(f"No generation files found matching criteria.")
        if args.model:
            print(f"  Model filter: {args.model}")
        if args.study:
            print(f"  Study filter: {args.study}")
        return 1
    
    print(f"Found {len(files_to_process)} file(s) to process:")
    for input_path, output_path in files_to_process:
        try:
            input_rel = input_path.relative_to(args.results_dir.parent)
        except ValueError:
            input_rel = input_path
        try:
            output_rel = output_path.relative_to(args.results_dir.parent)
        except ValueError:
            output_rel = output_path
        print(f"  {input_rel}")
        print(f"    -> {output_rel}")
    
    if args.dry_run:
        print("\nDry run mode - no files were modified.")
        return 0
    
    print("\nProcessing files...")
    
    total_files = len(files_to_process)
    total_entries_processed = 0
    total_entries_cleaned = 0
    
    for input_path, output_path in files_to_process:
        print(f"\nProcessing: {input_path.name}")
        
        try:
            entries, cleaned = clean_jsonl_file(
                input_path,
                output_path,
                max_repetition_ratio=args.max_repetition_ratio,
                min_repeat_length=args.min_repeat_length,
                similarity_cutoff=args.similarity_cutoff,
                scan_mode=args.scan_mode,
                scan_min_repeats=args.scan_min_repeats,
            )
            total_entries_processed += entries
            total_entries_cleaned += cleaned
            
            print(f"  Processed: {entries} entries")
            print(f"  Cleaned: {cleaned} entries ({cleaned/entries*100:.1f}%)" if entries > 0 else "  Cleaned: 0 entries")
            print(f"  Output: {output_path}")
            
        except Exception as e:
            print(f"  ERROR: Failed to process {input_path}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Files processed: {total_files}")
    print(f"  Total entries: {total_entries_processed}")
    print(f"  Entries cleaned: {total_entries_cleaned}")
    if total_entries_processed > 0:
        print(f"  Cleaning rate: {total_entries_cleaned/total_entries_processed*100:.1f}%")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
