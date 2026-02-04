from __future__ import annotations

import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from datasets import load_dataset


def extract_diagnosis_from_reasoning(reasoning: List[str], prompt: str) -> str:
    diagnosis_patterns = [
        r"(?:suggests?|indicat(?:es?|ing)|sounds? like|appears? to be|consistent with|symptoms of|experiencing)\s+(?:possible\s+)?([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+){0,4}(?:\s+[Dd]isorder)?)",
        r"(?:[Dd]iagnos(?:is|ed|tic)|[Cc]ondition)[\s:]+([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+){0,4})",
    ]

    content_diagnosis_map = {
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

    full_text = " ".join(reasoning) + " " + prompt
    full_lower = full_text.lower()

    for pattern in diagnosis_patterns:
        matches = re.findall(pattern, full_text, re.IGNORECASE)
        if matches:
            diagnosis = matches[0].strip()
            if len(diagnosis) > 5 and "disorder" in diagnosis.lower():
                return diagnosis

    for keyword, diagnosis in content_diagnosis_map.items():
        if keyword in full_lower:
            return diagnosis

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


def _build_openr1_prompt_index(ds: Any) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    exact: Dict[str, List[int]] = {}
    lower: Dict[str, List[int]] = {}

    for i, row in enumerate(ds):
        conv = row.get("conversation", []) if isinstance(row, dict) else []
        if not conv:
            continue
        first = conv[0] if isinstance(conv[0], dict) else {}
        patient_msg = _normalise_prompt(first.get("patient", ""))
        if not patient_msg:
            continue

        exact.setdefault(patient_msg, []).append(int(i))
        lower.setdefault(patient_msg.lower(), []).append(int(i))

    return exact, lower


def _get_sample_number(sample_id: str) -> Optional[int]:
    m = re.match(r"^a_(\d+)$", str(sample_id or "").strip())
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _try_link_by_prompt(
    prompt: str,
    test_index_exact: Dict[str, List[int]],
    train_index_exact: Dict[str, List[int]],
    test_index_lower: Dict[str, List[int]],
    train_index_lower: Dict[str, List[int]],
    used_test_indices: Set[int],
    used_train_indices: Set[int],
    ds_test: Any,
    ds_train: Any,
) -> Optional[Tuple[str, int]]:
    norm = _normalise_prompt(prompt)
    if not norm:
        return None

    def pick_from(split_name: str, candidates: List[int]) -> Optional[Tuple[str, int]]:
        used = used_test_indices if split_name == "test" else used_train_indices
        ds = ds_test if split_name == "test" else ds_train
        for idx in candidates:
            if int(idx) in used:
                continue
            try:
                row = ds[int(idx)]
            except Exception:
                continue
            conv = row.get("conversation", []) if isinstance(row, dict) else []
            if not conv:
                continue
            if not _extract_first_turn_gold_answer(conv):
                continue
            return split_name, int(idx)
        return None

    if norm in test_index_exact:
        linked = pick_from("test", test_index_exact[norm])
        if linked is not None:
            return linked

    if norm in train_index_exact:
        linked = pick_from("train", train_index_exact[norm])
        if linked is not None:
            return linked

    lowered = norm.lower()
    if lowered in test_index_lower:
        linked = pick_from("test", test_index_lower[lowered])
        if linked is not None:
            return linked

    if lowered in train_index_lower:
        linked = pick_from("train", train_index_lower[lowered])
        if linked is not None:
            return linked

    return None


def _pick_replacement_row(
    used_test_indices: Set[int],
    used_train_indices: Set[int],
    ds_test: Any,
    ds_train: Any,
    prefer_test: bool = True,
    max_attempts: int = 200,
) -> Tuple[str, int, Dict[str, Any]]:
    for _ in range(max_attempts):
        preferred = "test" if prefer_test else "train"
        for split_name in [preferred, "train" if preferred == "test" else "test"]:
            used = used_test_indices if split_name == "test" else used_train_indices
            ds = ds_test if split_name == "test" else ds_train

            idx = random.randint(0, len(ds) - 1)
            if idx in used:
                continue

            row = ds[int(idx)]
            conv = row.get("conversation", []) if isinstance(row, dict) else []
            if not conv:
                continue

            first = conv[0] if isinstance(conv[0], dict) else {}
            patient_msg = str(first.get("patient", "") or "").strip()
            if not patient_msg or len(patient_msg) < 50:
                continue

            gold_answer = _extract_first_turn_gold_answer(conv)
            if not gold_answer:
                continue

            return split_name, int(idx), row

    raise RuntimeError("Failed to pick a valid replacement OpenR1-Psy row")


def main() -> int:
    random.seed(42)

    base_dir = Path(__file__).parent.parent.parent.parent
    test_path = base_dir / "data" / "openr1_psy_splits" / "study_a_test.json"
    gold_labels_path = base_dir / "data" / "study_a_gold" / "gold_diagnosis_labels.json"
    cache_dir = base_dir / "Misc" / "datasets" / "openr1_psy"

    test_data = json.loads(test_path.read_text(encoding="utf-8"))
    samples: List[Dict[str, Any]] = test_data.get("samples", [])

    gold_data = json.loads(gold_labels_path.read_text(encoding="utf-8"))
    labels: Dict[str, str] = gold_data.get("labels", {})

    ds_test = load_dataset("GMLHUHE/OpenR1-Psy", split="test", cache_dir=str(cache_dir))
    ds_train = load_dataset("GMLHUHE/OpenR1-Psy", split="train", cache_dir=str(cache_dir))

    test_index_exact, test_index_lower = _build_openr1_prompt_index(ds_test)
    train_index_exact, train_index_lower = _build_openr1_prompt_index(ds_train)

    used_test_indices: Set[int] = set()
    used_train_indices: Set[int] = set()

    for s in samples:
        meta = s.get("metadata") or {}
        if not isinstance(meta, dict):
            continue
        source_ids = meta.get("source_openr1_ids") or []
        if not isinstance(source_ids, list) or not source_ids:
            continue
        source_split = str(meta.get("source_split", "") or "").strip().lower()
        if source_split == "test":
            used_test_indices.update(int(i) for i in source_ids)
        else:
            used_train_indices.update(int(i) for i in source_ids)

    replaced = 0
    linked = 0
    filled_labels = 0

    for s in samples:
        sid = str(s.get("id", "") or "")
        n = _get_sample_number(sid)
        if n is None or n > 600:
            continue

        meta = s.get("metadata")
        if isinstance(meta, dict) and meta.get("source_openr1_ids") and meta.get("source_split"):
            continue

        prompt = str(s.get("prompt", "") or "")
        linked_to = _try_link_by_prompt(
            prompt=prompt,
            test_index_exact=test_index_exact,
            train_index_exact=train_index_exact,
            test_index_lower=test_index_lower,
            train_index_lower=train_index_lower,
            used_test_indices=used_test_indices,
            used_train_indices=used_train_indices,
            ds_test=ds_test,
            ds_train=ds_train,
        )

        if linked_to is not None:
            split_name, idx = linked_to
            s["metadata"] = dict(s.get("metadata") or {})
            s["metadata"]["source_openr1_ids"] = [int(idx)]
            s["metadata"]["source_split"] = split_name

            if not str(labels.get(sid, "") or "").strip():
                reasoning_steps = s.get("gold_reasoning", [])
                if not isinstance(reasoning_steps, list):
                    reasoning_steps = []
                labels[sid] = extract_diagnosis_from_reasoning(
                    reasoning_steps, str(s.get("prompt", "") or "")
                )

            if split_name == "test":
                used_test_indices.add(int(idx))
            else:
                used_train_indices.add(int(idx))

            linked += 1
            continue

        prefer_test = True
        split_name, idx, row = _pick_replacement_row(
            used_test_indices=used_test_indices,
            used_train_indices=used_train_indices,
            ds_test=ds_test,
            ds_train=ds_train,
            prefer_test=prefer_test,
        )

        conv = row.get("conversation", []) if isinstance(row, dict) else []
        first = conv[0] if isinstance(conv[0], dict) else {}
        patient_msg = str(first.get("patient", "") or "").strip()
        gold_answer = _extract_first_turn_gold_answer(conv)
        reasoning_steps = _extract_reasoning_steps(conv)

        s["prompt"] = patient_msg
        s["gold_answer"] = gold_answer
        s["gold_reasoning"] = reasoning_steps
        s["metadata"] = {
            "source_openr1_ids": [int(idx)],
            "source_split": split_name,
            "replaced_during_linkage_fix": True,
        }

        if split_name == "test":
            used_test_indices.add(int(idx))
        else:
            used_train_indices.add(int(idx))

        labels[sid] = extract_diagnosis_from_reasoning(reasoning_steps, patient_msg)

        replaced += 1

    for s in samples:
        sid = str(s.get("id", "") or "")
        n = _get_sample_number(sid)
        if n is None or n > 600:
            continue
        meta = s.get("metadata") or {}
        assert meta.get("source_openr1_ids"), f"Missing source_openr1_ids for sample_id={sid}"
        assert str(meta.get("source_split", "") or "").strip(), f"Missing source_split for sample_id={sid}"

        if not str(labels.get(sid, "") or "").strip():
            reasoning_steps = s.get("gold_reasoning", [])
            if not isinstance(reasoning_steps, list):
                reasoning_steps = []
            labels[sid] = extract_diagnosis_from_reasoning(
                reasoning_steps, str(s.get("prompt", "") or "")
            )
            filled_labels += 1

    test_data["samples"] = samples
    test_path.write_text(json.dumps(test_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    gold_data["labels"] = labels
    gold_labels_path.write_text(json.dumps(gold_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"linked_existing_prompts {linked}")
    print(f"replaced_unlinked {replaced}")
    print(f"filled_missing_labels {filled_labels}")
    print(f"updated_utc {datetime.utcnow().isoformat()}Z")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
