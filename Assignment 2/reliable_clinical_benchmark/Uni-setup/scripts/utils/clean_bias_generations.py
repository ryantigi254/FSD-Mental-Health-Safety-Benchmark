"""
Utility script to clean existing study_a_bias_generations.jsonl files.

IMPORTANT: This script is for READABILITY ONLY, not for evaluation.
- Metrics MUST be calculated on raw, unmodified model outputs
- This script creates cleaned versions for human review/display
- Never use cleaned outputs for metric calculation

Removes excessive repetition and truncates overly long outputs from existing generation files.
This is useful when models have generated outputs with repetition loops that make files hard to read.
"""

import argparse
import json
import sys
from pathlib import Path


def _clean_output_text(text: str, max_chars: int = 20000, max_repetition_ratio: float = 0.3) -> str:
    """
    Clean up model output to prevent excessive repetition and overly long responses.
    
    Args:
        text: Raw model output
        max_chars: Maximum character length before truncation (default: 20000)
        max_repetition_ratio: Maximum ratio of repeated content (default: 0.3 = 30%)
    
    Returns:
        Cleaned text with repetition removed and length limited
    """
    if not text:
        return text
    
    # First, truncate if too long (before repetition detection to save processing)
    if len(text) > max_chars:
        # Try to truncate at a sentence boundary near the limit
        truncated = text[:max_chars]
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        cut_point = max(last_period, last_newline)
        
        if cut_point > max_chars * 0.8:  # Only use boundary if it's reasonably close
            text = text[:cut_point + 1] + "\n\n[Output truncated due to excessive length]"
        else:
            text = text[:max_chars] + "\n\n[Output truncated due to excessive length]"
    
    # Detect excessive repetition by checking for repeated phrases
    # Split into sentences for better detection
    sentences = text.split('.')
    if len(sentences) < 3:
        # Not enough sentences to detect repetition, return as-is
        return text
    
    # Check for repeated sentences (same sentence appearing many times)
    sentence_counts = {}
    for sent in sentences:
        sent_clean = sent.strip().lower()
        if len(sent_clean) > 20:  # Only count substantial sentences
            sentence_counts[sent_clean] = sentence_counts.get(sent_clean, 0) + 1
    
    # Find sentences that appear too frequently
    total_sentences = len([s for s in sentences if len(s.strip()) > 20])
    if total_sentences > 0:
        for sent_clean, count in sentence_counts.items():
            if count / total_sentences > max_repetition_ratio:
                # This sentence is repeated too much - truncate at first occurrence
                # Find where this repetition starts
                first_occurrence_idx = None
                for i, sent in enumerate(sentences):
                    if sent.strip().lower() == sent_clean:
                        first_occurrence_idx = i
                        break
                
                if first_occurrence_idx is not None:
                    # Keep content up to first occurrence, then check if there's more unique content
                    kept_sentences = sentences[:first_occurrence_idx + 1]
                    remaining = sentences[first_occurrence_idx + 1:]
                    
                    # Check if remaining has unique content
                    unique_remaining = [s for s in remaining if s.strip().lower() != sent_clean]
                    if len(unique_remaining) < len(remaining) * 0.5:
                        # More than 50% is repetition, truncate here
                        text = '.'.join(kept_sentences)
                        if not text.endswith('.'):
                            text += '.'
                        text += "\n\n[Output truncated due to excessive repetition]"
                        break
    
    # Additional check: detect very long repeated phrases (e.g., "Diagnosis: X" repeated many times)
    # Look for patterns that repeat more than 5 times consecutively
    lines = text.split('\n')
    if len(lines) > 10:
        # Check last 20 lines for repetition
        recent_lines = [l.strip() for l in lines[-20:] if l.strip()]
        if len(recent_lines) > 5:
            # Check if last few lines are identical
            last_line = recent_lines[-1]
            if len(last_line) > 30:  # Only check substantial lines
                repeat_count = sum(1 for line in recent_lines[-10:] if line == last_line)
                if repeat_count >= 5:
                    # Truncate before the repetition starts
                    for i in range(len(lines) - 1, max(0, len(lines) - 20), -1):
                        if lines[i].strip() != last_line:
                            text = '\n'.join(lines[:i+1])
                            text += "\n\n[Output truncated due to excessive repetition]"
                            break
    
    return text


def clean_jsonl_file(input_path: Path, output_path: Path, max_chars: int = 20000, max_repetition_ratio: float = 0.3) -> None:
    """
    Clean all entries in a JSONL file by removing excessive repetition.
    
    WARNING: This is for READABILITY ONLY. Never use cleaned outputs for metric calculation.
    Metrics must be calculated on raw, unmodified model outputs to maintain objectivity.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file (can be same as input for in-place)
        max_chars: Maximum characters per output
        max_repetition_ratio: Maximum repetition ratio before truncation
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    cleaned_count = 0
    total_count = 0
    
    with open(input_path, "r", encoding="utf-8") as f_in:
        with open(output_path, "w", encoding="utf-8") as f_out:
            for line in f_in:
                if not line.strip():
                    continue
                
                try:
                    entry = json.loads(line)
                    total_count += 1
                    
                    original_text = entry.get("output_text", "")
                    if original_text:
                        original_length = len(original_text)
                        cleaned_text = _clean_output_text(
                            original_text,
                            max_chars=max_chars,
                            max_repetition_ratio=max_repetition_ratio
                        )
                        
                        if len(cleaned_text) < original_length:
                            cleaned_count += 1
                            entry["output_text"] = cleaned_text
                            
                            # Update metadata
                            if "meta" not in entry:
                                entry["meta"] = {}
                            entry["meta"]["output_was_truncated"] = True
                            entry["meta"]["original_output_length"] = original_length
                            entry["meta"]["cleaned_by"] = "clean_bias_generations.py"
                    
                    f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}", file=sys.stderr)
                    continue
    
    print(f"Cleaned {cleaned_count} out of {total_count} entries")
    print(f"Output written to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean study_a_bias_generations.jsonl files by removing excessive repetition"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to input JSONL file (e.g., processed/study_a_bias/piaget_local/study_a_bias_generations.jsonl)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Path to output JSONL file (default: overwrites input file)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=20000,
        help="Maximum output characters before truncation (default: 20000)",
    )
    parser.add_argument(
        "--max-repetition-ratio",
        type=float,
        default=0.3,
        help="Maximum ratio of repeated content before truncation (default: 0.3 = 30%%)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create a backup of the input file before cleaning",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("WARNING: This script is for READABILITY ONLY, not for evaluation!")
    print("Metrics MUST be calculated on raw, unmodified model outputs.")
    print("This creates cleaned versions for human review/display purposes.")
    print("=" * 70)
    print()
    
    input_path = args.input_file.resolve()
    output_path = args.output.resolve() if args.output else input_path
    
    if args.backup and input_path == output_path:
        backup_path = input_path.with_suffix(input_path.suffix + ".backup")
        print(f"Creating backup: {backup_path}")
        import shutil
        shutil.copy2(input_path, backup_path)
    
    clean_jsonl_file(
        input_path,
        output_path,
        max_chars=args.max_chars,
        max_repetition_ratio=args.max_repetition_ratio
    )


if __name__ == "__main__":
    main()

