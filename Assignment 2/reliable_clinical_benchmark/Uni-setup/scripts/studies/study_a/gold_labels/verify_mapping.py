"""Verify and map study_a gold diagnosis labels back to original OpenR1-Psy dataset.

This script:
1. Loads study_a_test.json and study_a_gold_diagnosis_labels.json
2. Maps each study_a ID to its corresponding OpenR1-Psy test split row
3. Verifies the mapping is correct by comparing prompt text
4. Outputs a verification report with mappings and any discrepancies
"""

import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from datasets import load_dataset
from reliable_clinical_benchmark.data.study_a_loader import load_study_a_data


def build_mapping_report() -> Dict:
    """Build a comprehensive mapping report between study_a IDs and OpenR1-Psy rows."""
    print("Loading OpenR1-Psy test split...")
    ds = load_dataset("GMLHUHE/OpenR1-Psy", split="test")
    
    print("Loading study_a_test.json...")
    study_a_path = Path("data/openr1_psy_splits/study_a_test.json")
    vignettes = load_study_a_data(str(study_a_path))
    
    print("Loading study_a_gold_diagnosis_labels.json...")
    labels_path = Path("data/study_a_gold/gold_diagnosis_labels.json")
    with labels_path.open("r", encoding="utf-8") as f:
        labels_data = json.load(f)
    labels = labels_data.get("labels", {})
    
    # Build mapping: study_a ID -> OpenR1-Psy row
    mapping = {}
    sample_idx = 0
    
    for row_idx, row in enumerate(ds):
        convo = row.get("conversation") or []
        if not convo:
            continue
        
        first_round = convo[0]
        patient_text = str(first_round.get("patient", "")).strip()
        counselor_content = str(first_round.get("counselor_content", "")).strip()
        counselor_think = first_round.get("counselor_think", "")
        
        # Same filter as build_study_a_split
        if not patient_text or not counselor_content:
            continue
        
        sample_idx += 1
        study_a_id = f"a_{sample_idx:03d}"
        
        if sample_idx > 300:
            break
        
        # Find corresponding vignette
        vignette = None
        for v in vignettes:
            if v["id"] == study_a_id:
                vignette = v
                break
        
        if not vignette:
            mapping[study_a_id] = {
                "status": "missing_vignette",
                "openr1_row": row_idx,
                "openr1_post_id": row.get("post_id"),
            }
            continue
        
        # Verify prompt text matches
        prompt_match = False
        prompt_similarity = 0.0
        
        study_a_prompt = vignette.get("prompt", "").strip()
        openr1_prompt = patient_text.strip()
        
        # Normalise for comparison
        study_a_norm = " ".join(study_a_prompt.split())
        openr1_norm = " ".join(openr1_prompt.split())
        
        if study_a_norm == openr1_norm:
            prompt_match = True
            prompt_similarity = 1.0
        elif study_a_norm.lower() == openr1_norm.lower():
            prompt_match = True
            prompt_similarity = 0.95
        elif study_a_norm in openr1_norm or openr1_norm in study_a_norm:
            prompt_match = True
            prompt_similarity = 0.8
        else:
            # Calculate simple word overlap
            study_a_words = set(study_a_norm.lower().split())
            openr1_words = set(openr1_norm.lower().split())
            if study_a_words and openr1_words:
                overlap = len(study_a_words & openr1_words)
                total = len(study_a_words | openr1_words)
                prompt_similarity = overlap / total if total > 0 else 0.0
                prompt_match = prompt_similarity > 0.7
        
        gold_label = labels.get(study_a_id, "")
        
        mapping[study_a_id] = {
            "status": "matched" if prompt_match else "mismatch",
            "openr1_row": row_idx,
            "openr1_post_id": row.get("post_id"),
            "prompt_match": prompt_match,
            "prompt_similarity": prompt_similarity,
            "gold_label": gold_label,
            "study_a_prompt_preview": study_a_prompt[:100] + "..." if len(study_a_prompt) > 100 else study_a_prompt,
            "openr1_prompt_preview": openr1_prompt[:100] + "..." if len(openr1_prompt) > 100 else openr1_prompt,
            "counselor_think_preview": counselor_think[:200] + "..." if len(counselor_think) > 200 else counselor_think,
        }
    
    return mapping


def generate_report(mapping: Dict) -> None:
    """Generate a human-readable verification report."""
    total = len(mapping)
    matched = sum(1 for m in mapping.values() if m.get("status") == "matched")
    mismatched = sum(1 for m in mapping.values() if m.get("status") == "mismatch")
    missing = sum(1 for m in mapping.values() if m.get("status") == "missing_vignette")
    labeled = sum(1 for m in mapping.values() if m.get("gold_label"))
    
    print("\n" + "="*80)
    print("GOLD LABELS MAPPING VERIFICATION REPORT")
    print("="*80)
    print(f"\nTotal study_a IDs: {total}")
    print(f"  [OK] Matched to OpenR1-Psy: {matched} ({matched/total*100:.1f}%)")
    print(f"  [X] Mismatched prompts: {mismatched} ({mismatched/total*100:.1f}%)")
    print(f"  [?] Missing vignettes: {missing} ({missing/total*100:.1f}%)")
    print(f"  [*] Labeled: {labeled} ({labeled/total*100:.1f}%)")
    
    # Show mismatches
    if mismatched > 0:
        print(f"\n[WARNING] MISMATCHED PROMPTS ({mismatched}):")
        for sid, info in sorted(mapping.items()):
            if info.get("status") == "mismatch":
                print(f"\n  {sid} (similarity: {info['prompt_similarity']:.2f})")
                print(f"    Study A: {info['study_a_prompt_preview']}")
                print(f"    OpenR1:  {info['openr1_prompt_preview']}")
    
    # Show unlabeled
    unlabeled = [sid for sid, info in mapping.items() if not info.get("gold_label")]
    if unlabeled:
        print(f"\n[UNLABELED] CASES ({len(unlabeled)}):")
        print(f"  IDs: {', '.join(unlabeled[:20])}" + ("..." if len(unlabeled) > 20 else ""))
    
    # Label distribution
    from collections import Counter
    labels_only = [info.get("gold_label") for info in mapping.values() if info.get("gold_label")]
    if labels_only:
        counts = Counter(labels_only)
        print(f"\n[STATS] LABEL DISTRIBUTION:")
        for label, count in counts.most_common():
            print(f"  {label}: {count}")
    
    print("\n" + "="*80)


def save_mapping_json(mapping: Dict, output_path: Path) -> None:
    """Save the full mapping to JSON for reference."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({
            "mapping": mapping,
            "summary": {
                "total": len(mapping),
                "matched": sum(1 for m in mapping.values() if m.get("status") == "matched"),
                "mismatched": sum(1 for m in mapping.values() if m.get("status") == "mismatch"),
                "labeled": sum(1 for m in mapping.values() if m.get("gold_label")),
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nFull mapping saved to: {output_path}")


def main() -> int:
    """Generate verification report."""
    mapping = build_mapping_report()
    
    generate_report(mapping)
    
    output_path = Path("data/study_a_gold/gold_labels_mapping.json")
    save_mapping_json(mapping, output_path)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



