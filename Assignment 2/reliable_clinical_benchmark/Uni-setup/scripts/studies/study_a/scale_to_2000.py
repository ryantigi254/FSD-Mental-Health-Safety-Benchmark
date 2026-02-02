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
from typing import Any, Dict, List, Set, Tuple, Optional

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


def _normalise_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _extract_first_turn_gold_answer(conversation: List[Dict[str, Any]]) -> str:
    if not conversation:
        return ""
    first = conversation[0] if isinstance(conversation[0], dict) else {}
    ans = str(first.get("counselor_content", "") or "").strip()
    if ans:
        return ans
    return str(first.get("counselor", "") or "").strip()


def _extract_reasoning_steps(conversation: List[Dict[str, Any]], max_steps: int = 10) -> List[str]:
    steps: List[str] = []
    for turn in conversation:
        if not isinstance(turn, dict):
            continue
        ct = str(turn.get("counselor_think", "") or "").strip()
        if not ct:
            continue
        parts = [s.strip() for s in re.split(r"[.!?]", ct) if s.strip()]
        steps.extend(parts)
        if len(steps) >= max_steps:
            break
    return steps[:max_steps]


def _build_openr1_prompt_index(ds: Any) -> Dict[str, List[int]]:
    index: Dict[str, List[int]] = {}
    for i, row in enumerate(ds):
        conv = row.get("conversation", []) if isinstance(row, dict) else []
        if not conv:
            continue
        first = conv[0] if isinstance(conv[0], dict) else {}
        patient_msg = _normalise_prompt(first.get("patient", ""))
        if not patient_msg:
            continue
        index.setdefault(patient_msg, []).append(int(i))
    return index


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

    target_total_samples = 2000

    print(f"\nLoading OpenR1-Psy (test + train) splits...")
    ds_test = load_dataset("GMLHUHE/OpenR1-Psy", split="test", cache_dir=str(cache_dir))
    ds_train = load_dataset("GMLHUHE/OpenR1-Psy", split="train", cache_dir=str(cache_dir))
    print(f"  Test split: {len(ds_test)} rows")
    print(f"  Train split: {len(ds_train)} rows")

    print(f"Building OpenR1 prompt index (exact match)...")
    test_prompt_index = _build_openr1_prompt_index(ds_test)
    train_prompt_index = _build_openr1_prompt_index(ds_train)

    used_test_indices: Set[int] = set()
    used_train_indices: Set[int] = set()
    for sample in current_samples:
        meta = sample.get("metadata", {})
        source_ids = meta.get("source_openr1_ids", [])
        source_split = str(meta.get("source_split", "") or "").strip().lower()
        if not source_ids:
            continue
        if source_split == "test":
            used_test_indices.update(int(i) for i in source_ids)
        elif source_split == "train":
            used_train_indices.update(int(i) for i in source_ids)
        else:
            used_train_indices.update(int(i) for i in source_ids)

    legacy_needing_linkage = [
        s
        for s in current_samples
        if not ((s.get("metadata") or {}).get("source_openr1_ids"))
    ]
    print(f"  Legacy samples needing linkage review: {len(legacy_needing_linkage)}")

    min_test_total = min(len(ds_test), max(1, int(0.2 * target_total_samples)))

    replaced_legacy_test = 0
    replaced_legacy_train = 0

    for sample in legacy_needing_linkage:
        sample_id = str(sample.get("id", "") or "")
        prompt = _normalise_prompt(sample.get("prompt", ""))
        linked: Optional[Tuple[str, int]] = None

        if prompt:
            if prompt in test_prompt_index:
                for idx in test_prompt_index[prompt]:
                    if idx in used_test_indices:
                        continue
                    row = ds_test[int(idx)]
                    conv = row.get("conversation", []) if isinstance(row, dict) else []
                    if not conv:
                        continue
                    if not _extract_first_turn_gold_answer(conv):
                        continue
                    linked = ("test", int(idx))
                    break

            if linked is None and prompt in train_prompt_index:
                for idx in train_prompt_index[prompt]:
                    if idx in used_train_indices:
                        continue
                    row = ds_train[int(idx)]
                    conv = row.get("conversation", []) if isinstance(row, dict) else []
                    if not conv:
                        continue
                    if not _extract_first_turn_gold_answer(conv):
                        continue
                    linked = ("train", int(idx))
                    break

        if linked is not None:
            split_name, idx = linked
            row = ds_test[idx] if split_name == "test" else ds_train[idx]
            conv = row.get("conversation", []) if isinstance(row, dict) else []

            gold_answer = str(sample.get("gold_answer", "") or "").strip()
            if not gold_answer:
                sample["gold_answer"] = _extract_first_turn_gold_answer(conv)

            gold_reasoning = sample.get("gold_reasoning", [])
            if not isinstance(gold_reasoning, list) or not gold_reasoning:
                sample["gold_reasoning"] = _extract_reasoning_steps(conv)

            sample["metadata"] = dict(sample.get("metadata", {}) or {})
            sample["metadata"]["source_openr1_ids"] = [int(idx)]
            sample["metadata"]["source_split"] = split_name

            if split_name == "test":
                used_test_indices.add(int(idx))
            else:
                used_train_indices.add(int(idx))

            diagnosis = extract_diagnosis_from_reasoning(
                reasoning=sample.get("gold_reasoning", []) or [],
                prompt=sample.get("prompt", "") or "",
            )
            if sample_id:
                current_labels[sample_id] = diagnosis
            continue

        current_test_count = sum(
            1
            for s in current_samples
            if str((s.get("metadata") or {}).get("source_split", "")).lower() == "test"
        )
        prefer_test = current_test_count < min_test_total

        max_replacement_attempts = 50
        replacement: Optional[Tuple[str, int]] = None
        for _ in range(max_replacement_attempts):
            preferred = "test" if prefer_test else "train"
            if preferred == "test":
                available_test = [i for i in range(len(ds_test)) if i not in used_test_indices]
                if available_test:
                    replacement = ("test", int(random.choice(available_test)))
                else:
                    replacement = None
            else:
                available_train = [i for i in range(len(ds_train)) if i not in used_train_indices]
                if available_train:
                    replacement = ("train", int(random.choice(available_train)))
                else:
                    replacement = None

            if replacement is None:
                fallback = "train" if preferred == "test" else "test"
                if fallback == "test":
                    available_test = [i for i in range(len(ds_test)) if i not in used_test_indices]
                    if available_test:
                        replacement = ("test", int(random.choice(available_test)))
                else:
                    available_train = [i for i in range(len(ds_train)) if i not in used_train_indices]
                    if available_train:
                        replacement = ("train", int(random.choice(available_train)))

            if replacement is None:
                continue

            split_name, idx = replacement
            row = ds_test[idx] if split_name == "test" else ds_train[idx]
            conv = row.get("conversation", []) if isinstance(row, dict) else []
            if not conv:
                replacement = None
                continue

            first = conv[0] if isinstance(conv[0], dict) else {}
            patient_msg = str(first.get("patient", "") or "").strip()
            if not patient_msg or len(patient_msg) < 50:
                replacement = None
                continue

            gold_answer = _extract_first_turn_gold_answer(conv)
            if not gold_answer:
                replacement = None
                continue

            sample["prompt"] = patient_msg
            sample["gold_answer"] = gold_answer
            sample["gold_reasoning"] = _extract_reasoning_steps(conv)
            sample["metadata"] = {
                "source_openr1_ids": [int(idx)],
                "source_split": split_name,
                "added_during_scaling": True,
            }

            if split_name == "test":
                used_test_indices.add(int(idx))
                replaced_legacy_test += 1
            else:
                used_train_indices.add(int(idx))
                replaced_legacy_train += 1

            diagnosis = extract_diagnosis_from_reasoning(
                reasoning=sample.get("gold_reasoning", []) or [],
                prompt=sample.get("prompt", "") or "",
            )
            if sample_id:
                current_labels[sample_id] = diagnosis
            break

        assert (
            (sample.get("metadata") or {}).get("source_openr1_ids")
        ), f"Failed to link/replace legacy sample {sample_id}"

    if legacy_needing_linkage:
        print(
            f"Replaced legacy samples: {replaced_legacy_test + replaced_legacy_train} (test={replaced_legacy_test}, train={replaced_legacy_train})"
        )
    
    # Calculate how many more samples needed
    assert len(current_samples) <= target_total_samples, (
        f"study_a_test.json has {len(current_samples)} samples; expected <= {target_total_samples}"
    )

    needed = target_total_samples - len(current_samples)
    print(f"\nNeed to add {needed} samples to reach {target_total_samples}")
    
    new_samples: List[Dict[str, Any]] = []
    new_labels: Dict[str, str] = {}

    numeric_ids: List[int] = []
    for s in current_samples:
        sid = str(s.get("id", "") or "")
        m = re.match(r"^a_(\d{3,})$", sid)
        if m:
            try:
                numeric_ids.append(int(m.group(1)))
            except Exception:
                continue
    next_id_num = (max(numeric_ids) + 1) if numeric_ids else (len(current_samples) + 1)

    while needed > 0:
        current_test_count = sum(
            1
            for s in (current_samples + new_samples)
            if str((s.get("metadata") or {}).get("source_split", "")).lower() == "test"
        )
        need_more_test = current_test_count < min_test_total

        pick_split = "test" if need_more_test else "train"
        if pick_split == "test":
            candidates = [i for i in range(len(ds_test)) if i not in used_test_indices]
        else:
            candidates = [i for i in range(len(ds_train)) if i not in used_train_indices]

        if not candidates:
            if pick_split == "test":
                pick_split = "train"
                candidates = [i for i in range(len(ds_train)) if i not in used_train_indices]
            else:
                pick_split = "test"
                candidates = [i for i in range(len(ds_test)) if i not in used_test_indices]

        if not candidates:
            break

        idx = int(random.choice(candidates))
        row = ds_test[idx] if pick_split == "test" else ds_train[idx]
        conv = row.get("conversation", []) if isinstance(row, dict) else []
        if not conv:
            continue

        first = conv[0] if isinstance(conv[0], dict) else {}
        patient_msg = str(first.get("patient", "") or "").strip()
        if not patient_msg or len(patient_msg) < 50:
            continue

        sample_id = f"a_{next_id_num:03d}"
        gold_answer = _extract_first_turn_gold_answer(conv)
        if not gold_answer:
            continue
        reasoning = _extract_reasoning_steps(conv)

        sample = {
            "id": sample_id,
            "prompt": patient_msg,
            "gold_answer": gold_answer,
            "gold_reasoning": reasoning,
            "metadata": {
                "source_openr1_ids": [idx],
                "source_split": pick_split,
                "added_during_scaling": True,
            },
        }

        new_samples.append(sample)
        new_labels[sample_id] = extract_diagnosis_from_reasoning(reasoning, patient_msg)

        if pick_split == "test":
            used_test_indices.add(idx)
        else:
            used_train_indices.add(idx)

        next_id_num += 1
        needed -= 1

        if len(new_samples) % 200 == 0:
            print(f"  Created {len(new_samples)} new samples so far...")

    print(f"\nCreated {len(new_samples)} new samples")

    assert needed == 0, f"Failed to reach 2000 samples; remaining_needed={needed}"
    
    # Append to existing data
    test_data["samples"].extend(new_samples)
    test_data["meta"] = test_data.get("meta", {})
    test_data["meta"]["scaled_to_2000"] = datetime.utcnow().isoformat() + "Z"
    test_data["meta"]["total_samples"] = len(test_data["samples"])
    test_data["meta"]["scaling_script"] = "scripts/studies/study_a/scale_to_2000.py"
    
    current_labels.update(new_labels)

    sample_ids = {
        str(s.get("id", "") or "") for s in test_data.get("samples", []) if str(s.get("id", "") or "")
    }
    for sid in sample_ids:
        if sid in current_labels:
            continue
        s = next((x for x in test_data.get("samples", []) if str(x.get("id", "") or "") == sid), None)
        if s is None:
            continue
        current_labels[sid] = extract_diagnosis_from_reasoning(
            reasoning=s.get("gold_reasoning", []) or [],
            prompt=s.get("prompt", "") or "",
        )

    extra_label_ids = [sid for sid in list(current_labels.keys()) if sid not in sample_ids]
    for sid in extra_label_ids:
        del current_labels[sid]

    for s in test_data.get("samples", []):
        assert str(s.get("gold_answer", "") or "").strip(), f"Empty gold_answer for sample_id={s.get('id')}"
        meta = s.get("metadata") or {}
        assert meta.get("source_openr1_ids"), f"Missing source_openr1_ids for sample_id={s.get('id')}"
        assert str(meta.get("source_split", "") or "").strip(), f"Missing source_split for sample_id={s.get('id')}"

    assert len(test_data.get("samples", [])) == 2000, "study_a_test.json must contain exactly 2000 samples"
    assert len(current_labels) == 2000, "gold_diagnosis_labels.json must contain exactly 2000 labels"
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
