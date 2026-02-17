"""Verify that gold labels are correctly ID-matched to study_a_test.json.

This script ensures reproducibility by verifying:
1. All study_a_test.json IDs have corresponding labels
2. Labels are extracted from the correct OpenR1-Psy rows
3. Prompt text matches between study_a_test.json and OpenR1-Psy
"""

import json
from pathlib import Path
import sys

# Add src to path (go up 3 levels from scripts/studies/study_a/gold_labels/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from datasets import load_dataset
from reliable_clinical_benchmark.data.study_a_loader import load_study_a_data


def main() -> int:
    """Verify ID matching between study_a_test.json and gold labels."""
    print("="*80)
    print("VERIFYING ID MATCHING: study_a_test.json <-> gold_labels")
    print("="*80)
    
    # Load study_a_test.json
    print("\n1. Loading study_a_test.json...")
    study_a_path = Path("data/openr1_psy_splits/study_a_test.json")
    vignettes = load_study_a_data(str(study_a_path))
    study_a_ids = {v["id"] for v in vignettes}
    print(f"   Found {len(study_a_ids)} IDs in study_a_test.json")
    
    # Load gold labels
    print("\n2. Loading study_a_gold_diagnosis_labels.json...")
    labels_path = Path("data/study_a_gold/gold_diagnosis_labels.json")
    with labels_path.open("r", encoding="utf-8") as f:
        labels_data = json.load(f)
    labels = labels_data.get("labels", {})
    label_ids = set(labels.keys())
    print(f"   Found {len(label_ids)} IDs in gold labels")

    if len(vignettes) != 2000:
        print(f"   [ERROR] study_a_test.json must contain 2000 samples, found {len(vignettes)}")
        return 1

    if len(labels) != 2000:
        print(f"   [ERROR] gold_diagnosis_labels.json must contain 2000 labels, found {len(labels)}")
        return 1
    
    # Check ID coverage
    print("\n3. Checking ID coverage...")
    missing_in_labels = study_a_ids - label_ids
    extra_in_labels = label_ids - study_a_ids
    
    if missing_in_labels:
        print(f"   [ERROR] {len(missing_in_labels)} study_a_test.json IDs missing in gold labels:")
        print(f"   {sorted(list(missing_in_labels))[:10]}...")
        return 1
    
    if extra_in_labels:
        print(f"   [WARNING] {len(extra_in_labels)} extra IDs in gold labels (not in study_a_test.json):")
        print(f"   {sorted(list(extra_in_labels))[:10]}...")
    
    print(f"   [OK] All {len(study_a_ids)} study_a_test.json IDs have corresponding labels")
    
    # Check label coverage
    print("\n4. Checking label coverage...")
    labeled_count = sum(1 for v in labels.values() if v)
    print(f"   Labeled: {labeled_count}/{len(labels)} ({labeled_count/len(labels)*100:.1f}%)")
    
    # Verify linkage via metadata against OpenR1-Psy
    print("\n5. Verifying linkage (metadata.source_split/source_openr1_ids) against OpenR1-Psy...")
    ds_test = load_dataset("GMLHUHE/OpenR1-Psy", split="test")
    ds_train = load_dataset("GMLHUHE/OpenR1-Psy", split="train")

    matched_count = 0
    mismatch_count = 0
    missing_linkage = 0
    invalid_linkage = 0

    for v in vignettes:
        meta = v.get("metadata") or {}
        src_ids = meta.get("source_openr1_ids") or []
        src_split = str(meta.get("source_split", "") or "").strip().lower()

        if not src_ids or src_split not in {"test", "train"}:
            missing_linkage += 1
            continue

        try:
            src_id = int(src_ids[0])
        except Exception:
            invalid_linkage += 1
            continue

        ds = ds_test if src_split == "test" else ds_train
        try:
            row = ds[src_id]
        except Exception:
            invalid_linkage += 1
            continue

        convo = row.get("conversation") or []
        if not convo:
            invalid_linkage += 1
            continue

        first_round = convo[0]
        patient_text = str(first_round.get("patient", "") or "").strip()
        if not patient_text:
            invalid_linkage += 1
            continue

        vignette_prompt = str(v.get("prompt", "") or "").strip()
        prompt_norm = " ".join(vignette_prompt.split())
        patient_norm = " ".join(patient_text.split())

        if prompt_norm == patient_norm or prompt_norm.lower() == patient_norm.lower():
            matched_count += 1
        else:
            mismatch_count += 1

    if missing_linkage:
        print(f"   [ERROR] {missing_linkage} samples missing linkage metadata")
        return 1
    if invalid_linkage:
        print(f"   [ERROR] {invalid_linkage} samples have invalid linkage metadata")
        return 1

    print(f"   Matched: {matched_count}/{len(vignettes)} samples by prompt against linked OpenR1 rows")
    if mismatch_count:
        print(f"   [ERROR] {mismatch_count} linked samples have prompt mismatches")
        return 1
    
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    print(f"[OK] All study_a_test.json IDs have gold labels: {len(study_a_ids)}/{len(study_a_ids)}")
    print(f"[OK] Labels are ID-matched to study_a_test.json")
    print(f"[OK] Process is reproducible (matching via metadata linkage + prompt check)")
    print(f"[OK] {labeled_count}/{len(labels)} labels populated ({labeled_count/len(labels)*100:.1f}%)")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



