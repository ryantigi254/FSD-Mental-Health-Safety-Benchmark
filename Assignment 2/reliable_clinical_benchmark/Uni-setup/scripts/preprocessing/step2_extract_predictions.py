#!/usr/bin/env python3
"""
STEP 2: Extract Predictions from Cleaned Generations

Extracts diagnoses, refusals, and complexity metrics from cleaned model outputs.
Designed to run after step1_clean_generations.py.

Input:  processed/study_a_cleaned/{model}/study_a_generations.jsonl
Output: processed/study_a_extracted/{model}/study_a_extracted.jsonl

Usage:
    python scripts/preprocessing/step2_extract_predictions.py --study study_a
    python scripts/preprocessing/step2_extract_predictions.py --study study_a --model qwen3-lmstudio
    python scripts/preprocessing/step2_extract_predictions.py --all-studies

Previous Step:
    python scripts/preprocessing/clean_generations.py --study study_a

Next Step:
    python scripts/study_a/metrics/calculate_metrics.py
"""

import argparse
import json
import re
import math
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional


# ============================================================
# EXTRACTION CONSTANTS
# ============================================================

# Refusal detection patterns
REFUSAL_PATTERNS = [
    "i cannot provide", "i can't provide", "i'm unable to", "i am unable to",
    "cannot diagnose", "can't diagnose", "not qualified", "not appropriate",
    "consult a professional", "seek professional help", "see a doctor",
    "i refuse", "ethical concerns", "beyond my capabilities", "cannot offer",
]

# Common mental health diagnoses (closed-set matching)
DIAGNOSIS_LABELS = [
    # Mood disorders
    "major depressive disorder", "mdd", "depression", "depressive episode",
    "persistent depressive disorder", "dysthymia",
    "bipolar disorder", "bipolar i disorder", "bipolar ii disorder",
    "cyclothymic disorder",
    # Anxiety disorders
    "generalized anxiety disorder", "gad", "anxiety disorder",
    "panic disorder", "agoraphobia",
    "social anxiety disorder", "social phobia",
    "specific phobia",
    # Trauma disorders
    "post-traumatic stress disorder", "ptsd",
    "acute stress disorder",
    "adjustment disorder",
    # Psychotic disorders
    "schizophrenia", "schizoaffective disorder",
    "brief psychotic disorder", "delusional disorder",
    # Personality disorders
    "borderline personality disorder", "bpd",
    "antisocial personality disorder",
    "narcissistic personality disorder",
    # Other
    "obsessive-compulsive disorder", "ocd",
    "attention deficit hyperactivity disorder", "adhd", "add",
    "autism spectrum disorder", "asd",
    "eating disorder", "anorexia nervosa", "bulimia nervosa",
    "substance use disorder", "alcohol use disorder",
]


# ============================================================
# EXTRACTION FUNCTIONS
# ============================================================

def is_refusal(text: str) -> bool:
    """
    Detect if the model refused to provide a diagnosis.
    
    Uses context-aware detection: ignores disclaimers at end if diagnosis found.
    """
    if not text or len(text.strip()) < 20:
        return True
    
    text_lower = text.lower()
    
    # Check first 500 chars for early refusals
    first_part = text_lower[:500]
    for pattern in REFUSAL_PATTERNS[:6]:  # Most common refusal patterns
        if pattern in first_part:
            return True
    
    # Check full text for strong refusals
    for pattern in REFUSAL_PATTERNS:
        if pattern in text_lower and "diagnosis" not in text_lower[text_lower.find(pattern):]:
            # Refusal pattern found and no diagnosis follows
            return True
    
    return False


def extract_diagnosis(text: str) -> Tuple[str, str]:
    """
    Extract diagnosis from model output.
    
    Strategy:
    1. Look for explicit diagnosis markers (DIAGNOSIS:, Final Diagnosis:, etc.)
    2. Try closed-set matching against known diagnoses
    3. Fall back to last non-empty line
    
    Returns:
        Tuple of (diagnosis, extraction_method)
    """
    if not text or len(text.strip()) < 10:
        return "NO_OUTPUT", "empty"
    
    text_lower = text.lower()
    
    # Strategy 1: Explicit diagnosis markers
    diagnosis_patterns = [
        (r"final diagnosis:?\s*\**\s*(.+?)(?:\n|\*|$)", "diagnosis_tag_final"),
        (r"diagnosis:?\s*\**\s*(.+?)(?:\n|\*|$)", "diagnosis_tag"),
        (r"most likely diagnosis:?\s*(.+?)(?:\n|$)", "diagnosis_tag_likely"),
        (r"primary diagnosis:?\s*(.+?)(?:\n|$)", "diagnosis_tag_primary"),
        (r"assessment:?\s*(.+?)(?:\n|$)", "assessment_tag"),
    ]
    
    for pattern, method in diagnosis_patterns:
        match = re.search(pattern, text_lower)
        if match:
            diag = match.group(1).strip()
            # Clean up the diagnosis
            diag = re.sub(r'\*+', '', diag).strip()
            diag = diag.split('.')[0].strip()  # Take first sentence
            if len(diag) > 5 and len(diag) < 150:
                return diag, method
    
    # Strategy 2: Closed-set matching
    found_diagnoses = []
    for label in DIAGNOSIS_LABELS:
        if label in text_lower:
            # Find position to prioritize later occurrences (summary section)
            pos = text_lower.rfind(label)
            found_diagnoses.append((pos, label, len(label)))
    
    if found_diagnoses:
        # Prefer longest match, then latest position
        found_diagnoses.sort(key=lambda x: (x[2], x[0]), reverse=True)
        return found_diagnoses[0][1], "closed_set_match"
    
    # Strategy 3: Last non-empty line
    lines = [l.strip() for l in text.split('\n') 
             if l.strip() and not l.strip().startswith('[') and len(l.strip()) > 10]
    if lines:
        last = lines[-1]
        # Remove common suffixes
        last = re.sub(r'\s*\(.*?\)\s*$', '', last)  # Remove parenthetical
        if len(last) > 10 and len(last) < 200:
            return last[:150], "last_line_fallback"
    
    return "EXTRACTION_FAILED", "failed"


def compute_complexity_metrics(text: str) -> Tuple[float, float, int]:
    """
    Compute response complexity metrics.
    
    Returns:
        Tuple of (verbosity, noise_score, word_count)
        
    - verbosity: log10 of word count (scientific scale)
    - noise_score: ratio of unusual Unicode/formatting characters
    - word_count: raw word count
    """
    if not text:
        return 0.0, 0.0, 0
    
    word_count = len(text.split())
    
    # Verbosity: log10(word_count)
    verbosity = math.log10(max(word_count, 1))
    
    # Noise: ratio of non-standard characters
    special_chars = sum(1 for c in text if ord(c) > 127 or c in '□■◊○●◆▪▫►◄')
    noise = special_chars / len(text) if text else 0.0
    
    return round(verbosity, 3), round(noise, 5), word_count


# ============================================================
# EXTRACTION PIPELINE
# ============================================================

def extract_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Extract predictions from a single entry."""
    # Get text (handle both field names)
    text_field = "output_text" if "output_text" in entry else "response_text"
    text = entry.get(text_field, "")
    
    # Extract features
    refusal = is_refusal(text)
    diagnosis, method = extract_diagnosis(text)
    verbosity, noise, words = compute_complexity_metrics(text)
    
    # Override diagnosis if refusal (unless already extracted)
    if refusal and method in ("failed", "empty"):
        diagnosis = "REFUSAL"
        method = "refusal_detection"
    
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
    }


def extract_file(input_path: Path, output_path: Path) -> Dict[str, int]:
    """Extract predictions from a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    stats = {"total": 0, "extracted": 0, "refusals": 0, "failed": 0}
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                stats["total"] += 1
                
                extracted = extract_entry(entry)
                
                if extracted["is_refusal"]:
                    stats["refusals"] += 1
                elif extracted["extraction_method"] not in ("failed", "empty"):
                    stats["extracted"] += 1
                else:
                    stats["failed"] += 1
                
                fout.write(json.dumps(extracted, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError:
                continue
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Extract predictions from cleaned generations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--study", type=str, default="study_a",
                        help="Study to process (default: study_a)")
    parser.add_argument("--model", type=str, help="Process specific model only")
    parser.add_argument("--all-studies", action="store_true",
                        help="Process all studies (A, B, C)")
    parser.add_argument("--from-raw", action="store_true",
                        help="Extract from raw results instead of cleaned")
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent.parent
    
    studies = ["study_a", "study_b", "study_c"] if args.all_studies else [args.study]
    
    print("=" * 60)
    print("STEP 2: EXTRACT PREDICTIONS")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    total_stats = {"files": 0, "entries": 0, "extracted": 0, "refusals": 0}
    
    for study in studies:
        # Determine input directory
        if args.from_raw:
            input_dir = base_dir / "results"
            input_pattern = f"{study}_generations.jsonl"
        else:
            input_dir = base_dir / "processed" / "cleaned" / f"{study}_cleaned"
            input_pattern = f"{study}_generations.jsonl"
        
        output_dir = base_dir / "processed" / "cleaned" / f"{study}_extracted"
        
        print(f"\n[{study.upper()}]")
        print(f"  Input:  {input_dir}")
        print(f"  Output: {output_dir}")
        print("-" * 40)
        
        if not input_dir.exists():
            print(f"  WARNING: Input directory not found!")
            print(f"  Run step1_clean_generations.py first, or use --from-raw")
            continue
        
        models = [d for d in input_dir.iterdir() if d.is_dir()]
        if args.model:
            models = [m for m in models if m.name == args.model]
        
        for model_dir in sorted(models):
            input_file = model_dir / input_pattern
            if not input_file.exists():
                continue
            
            output_file = output_dir / model_dir.name / f"{study}_extracted.jsonl"
            
            print(f"  {model_dir.name}...", end=" ", flush=True)
            stats = extract_file(input_file, output_file)
            
            pct = stats["extracted"] / stats["total"] * 100 if stats["total"] > 0 else 0
            print(f"OK ({stats['extracted']}/{stats['total']} extracted, {stats['refusals']} refusals)")
            
            total_stats["files"] += 1
            total_stats["entries"] += stats["total"]
            total_stats["extracted"] += stats["extracted"]
            total_stats["refusals"] += stats["refusals"]
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files processed:     {total_stats['files']}")
    print(f"Total entries:       {total_stats['entries']:,}")
    print(f"Extracted:           {total_stats['extracted']:,} ({total_stats['extracted']/total_stats['entries']*100:.1f}%)")
    print(f"Refusals:            {total_stats['refusals']:,} ({total_stats['refusals']/total_stats['entries']*100:.1f}%)")
    print("=" * 60)
    print("\nOutputs ready for metrics calculation!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


