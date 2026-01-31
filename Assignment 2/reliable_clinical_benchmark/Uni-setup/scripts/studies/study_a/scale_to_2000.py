"""Scale Study A test split from 600 to 2000 samples.

This script:
1. Loads current study_a_test.json (600 samples)
2. Identifies used OpenR1-Psy indices
3. Samples 1,400 additional entries from train split
4. Appends new samples to the test file
5. Generates gold labels for new samples

Run with: conda activate openr1-env; python scripts/studies/study_a/scale_to_2000.py
"""

from __future__ import annotations

import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

from datasets import load_dataset


# Diagnosis extraction patterns for gold labels
DIAGNOSIS_PATTERNS = [
    r"(?:suggests?|indicat(?:es?|ing)|sounds? like|appears? to be|consistent with|symptoms of|experiencing)\s+(?:possible\s+)?([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+){0,4}(?:\s+[Dd]isorder)?)",
    r"(?:[Dd]iagnos(?:is|ed|tic)|[Cc]ondition)[\s:]+([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+){0,4})",
]

# Common diagnosis mappings based on content analysis
CONTENT_DIAGNOSIS_MAP = {
    "depress": "Major Depressive Disorder",
    "anxiet": "Generalized Anxiety Disorder", 
    "panic": "Panic Disorder",
    "social anxi": "Social Anxiety Disorder",
    "ptsd": "Post-Traumatic Stress Disorder",
    "trauma": "Post-Traumatic Stress Disorder",
    "ocd": "Obsessive-Compulsive Disorder",
    "obsess": "Obsessive-Compulsive Disorder",
    "bipolar": "Bipolar Disorder",
    "schizo": "Schizophrenia Spectrum Disorder",
    "borderline": "Borderline Personality Disorder",
    "eating": "Eating Disorder",
    "anorex": "Anorexia Nervosa",
    "bulimi": "Bulimia Nervosa",
    "adhd": "Attention-Deficit/Hyperactivity Disorder",
    "attention": "Attention-Deficit/Hyperactivity Disorder",
    "adjustment": "Adjustment Disorder",
    "grief": "Complicated Grief Disorder",
    "substance": "Substance Use Disorder",
    "alcohol": "Alcohol Use Disorder",
    "insomnia": "Insomnia Disorder",
    "sleep": "Sleep Disorder",
}


def extract_diagnosis_from_reasoning(reasoning: List[str], prompt: str) -> str:
    """Extract a diagnosis from reasoning steps or prompt content."""
    full_text = " ".join(reasoning) + " " + prompt
    full_lower = full_text.lower()
    
    # Try pattern matching first
    for pattern in DIAGNOSIS_PATTERNS:
        matches = re.findall(pattern, full_text, re.IGNORECASE)
        if matches:
            diagnosis = matches[0].strip()
            if len(diagnosis) > 5 and "disorder" in diagnosis.lower():
                return diagnosis
    
    # Fallback to keyword mapping
    for keyword, diagnosis in CONTENT_DIAGNOSIS_MAP.items():
        if keyword in full_lower:
            return diagnosis
    
    # Default to adjustment disorder (most common for non-specific presentations)
    return "Adjustment Disorder"


def main() -> int:
    random.seed(42)  # Reproducibility
    
    base_dir = Path(__file__).parent.parent.parent.parent
    test_path = base_dir / "data" / "openr1_psy_splits" / "study_a_test.json"
    gold_labels_path = base_dir / "data" / "study_a_gold" / "gold_diagnosis_labels.json"
    cache_dir = base_dir / "Misc" / "datasets" / "openr1_psy"
    
    # Load current test split
    print(f"Loading current Study A test split...")
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    current_samples = test_data.get("samples", [])
    print(f"  Current samples: {len(current_samples)}")
    
    # Load current gold labels
    print(f"Loading current gold labels...")
    with open(gold_labels_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)
    
    current_labels = gold_data.get("labels", {})
    print(f"  Current labels: {len(current_labels)}")
    
    # Identify existing source IDs (from metadata if present)
    used_indices: Set[int] = set()
    for sample in current_samples:
        meta = sample.get("metadata", {})
        source_ids = meta.get("source_openr1_ids", [])
        if source_ids:
            used_indices.update(int(i) for i in source_ids)
    
    # Also mark indices 0-449 (test split) as used to prefer train
    for i in range(450):
        used_indices.add(i)
    
    print(f"  Existing source IDs: {len(used_indices)}")
    
    # Calculate how many more samples needed
    target = 2000
    needed = target - len(current_samples)
    print(f"\nNeed to add {needed} samples to reach {target}")
    
    if needed <= 0:
        print("Already at or above target!")
        return 0
    
    # Load OpenR1-Psy train split
    print(f"\nLoading OpenR1-Psy train split...")
    ds_train = load_dataset("GMLHUHE/OpenR1-Psy", split="train", cache_dir=str(cache_dir))
    print(f"  Train split: {len(ds_train)} rows")
    
    # Find available indices
    available = [i for i in range(len(ds_train)) if i not in used_indices]
    print(f"  Available indices: {len(available)}")
    
    # Sample needed indices
    if len(available) < needed:
        print(f"Warning: Only {len(available)} available, using all")
        selected = available
    else:
        selected = random.sample(available, needed)
    
    print(f"  Selected {len(selected)} new indices")
    
    # Create new samples
    new_samples = []
    new_labels = {}
    next_id = len(current_samples) + 1
    
    for i, idx in enumerate(selected):
        row = ds_train[idx]
        conv = row.get("conversation", [])
        
        if not conv:
            continue
        
        # Extract patient prompt (first turn)
        patient_msg = conv[0].get("patient", "")
        if not patient_msg or len(patient_msg) < 50:
            continue
        
        # Extract counselor response as gold answer
        counselor_msg = conv[0].get("counselor", "")
        
        # Extract reasoning from counselor_think
        reasoning = []
        for turn in conv:
            ct = turn.get("counselor_think", "")
            if ct:
                # Parse into steps
                steps = [s.strip() for s in re.split(r"[.!?]", ct) if s.strip()]
                reasoning.extend(steps[:8])  # Limit to 8 steps
        
        # Create sample
        sample_id = f"a_{next_id:03d}"
        sample = {
            "id": sample_id,
            "prompt": patient_msg,
            "gold_answer": counselor_msg,
            "gold_reasoning": reasoning[:10],  # Limit reasoning steps
            "metadata": {
                "source_openr1_ids": [idx],
                "source_split": "train",
                "added_during_scaling": True
            }
        }
        
        new_samples.append(sample)
        
        # Extract diagnosis for gold label
        diagnosis = extract_diagnosis_from_reasoning(reasoning, patient_msg)
        new_labels[sample_id] = diagnosis
        
        next_id += 1
        
        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(selected)} new samples...")
    
    print(f"\nCreated {len(new_samples)} new samples")
    
    # Append to existing data
    test_data["samples"].extend(new_samples)
    test_data["meta"] = test_data.get("meta", {})
    test_data["meta"]["scaled_to_2000"] = datetime.utcnow().isoformat() + "Z"
    test_data["meta"]["total_samples"] = len(test_data["samples"])
    test_data["meta"]["scaling_script"] = "scripts/studies/study_a/scale_to_2000.py"
    
    current_labels.update(new_labels)
    gold_data["labels"] = current_labels
    gold_data["meta"] = gold_data.get("meta", {})
    gold_data["meta"]["scaled_to_2000"] = datetime.utcnow().isoformat() + "Z"
    gold_data["meta"]["total_labels"] = len(current_labels)
    
    # Write updated files
    print(f"\nWriting updated files...")
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    print(f"  Updated {test_path} ({len(test_data['samples'])} samples)")
    
    with open(gold_labels_path, "w", encoding="utf-8") as f:
        json.dump(gold_data, f, indent=2, ensure_ascii=False)
    print(f"  Updated {gold_labels_path} ({len(current_labels)} labels)")
    
    print(f"\n{'='*50}")
    print(f"SCALING COMPLETE")
    print(f"  Study A samples: {len(current_samples)} -> {len(test_data['samples'])}")
    print(f"  Gold labels: {len(current_labels) - len(new_labels)} -> {len(current_labels)}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
