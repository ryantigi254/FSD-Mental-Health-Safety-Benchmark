from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset


PLAN_KEYWORDS = [
    "recommend",
    "suggest",
    "consider",
    "plan",
    "next step",
    "we can",
    "you could",
    "encourage",
    "try",
    "practice",
    "skills",
    "coping",
    "strategy",
    "technique",
    "exercise",
    "breathing",
    "relax",
    "mindfulness",
    "ground",
    "cbt",
    "dbt",
    "act",
    "therapy",
    "counselling",
    "medication",
    "ssri",
    "psychiat",
    "follow up",
    "monitor",
    "track",
    "journal",
    "schedule",
    "routine",
    "sleep hygiene",
    "safety plan",
]


def _normalise_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def extract_plan_from_reasoning(reasoning_text: str, max_sentences: int = 4) -> str:
    raw = _normalise_ws(reasoning_text)
    if not raw:
        return ""

    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", raw)
        if s and s.strip()
    ]

    chosen: List[str] = []
    seen = set()

    for s in sentences:
        lower = s.lower()
        if any(k in lower for k in PLAN_KEYWORDS):
            if 30 <= len(s) <= 350:
                key = lower
                if key in seen:
                    continue
                seen.add(key)
                chosen.append(s)
        if len(chosen) >= max_sentences:
            break

    if not chosen:
        for s in sentences[:max_sentences]:
            if 30 <= len(s) <= 350:
                chosen.append(s)

    plan = " ".join(chosen).strip()
    if plan and not plan.endswith((".", "!", "?")):
        plan += "."
    return plan


def _collect_counselor_think(conversation: Any) -> str:
    if not isinstance(conversation, list) or not conversation:
        return ""
    parts: List[str] = []
    for turn in conversation:
        if not isinstance(turn, dict):
            continue
        ct = str(turn.get("counselor_think", "") or "").strip()
        if ct:
            parts.append(ct)
    return "\n".join(parts).strip()


def main() -> int:
    p = argparse.ArgumentParser(
        description="Generate Study A gold treatment plans from linked OpenR1-Psy counselor_think"
    )
    p.add_argument(
        "--study-a",
        type=str,
        default="data/openr1_psy_splits/study_a_test.json",
        help="Path to study_a_test.json",
    )
    p.add_argument(
        "--out",
        type=str,
        default="data/study_a_gold/target_plans.json",
        help="Output path (JSON)",
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default="Misc/datasets/openr1_psy",
        help="HuggingFace datasets cache dir",
    )
    p.add_argument(
        "--max-sentences",
        type=int,
        default=4,
        help="Max sentences to keep in extracted plan",
    )

    args = p.parse_args()

    study_a_path = Path(args.study_a)
    out_path = Path(args.out)

    payload = json.loads(study_a_path.read_text(encoding="utf-8"))
    samples = payload.get("samples") or []

    ds_test = load_dataset(
        "GMLHUHE/OpenR1-Psy", split="test", cache_dir=str(Path(args.cache_dir))
    )
    ds_train = load_dataset(
        "GMLHUHE/OpenR1-Psy", split="train", cache_dir=str(Path(args.cache_dir))
    )

    plans: Dict[str, Dict[str, Any]] = {}
    split_counts: Dict[str, int] = {"test": 0, "train": 0}
    empty = 0

    for sample in samples:
        sid = str(sample.get("id", "") or "")
        meta = sample.get("metadata") or {}
        if not isinstance(meta, dict):
            raise TypeError(f"Sample {sid}: metadata missing/invalid")

        source_ids = meta.get("source_openr1_ids")
        if not isinstance(source_ids, list) or not source_ids:
            raise ValueError(f"Sample {sid}: metadata.source_openr1_ids missing")

        source_split = str(meta.get("source_split", "") or "").strip().lower()
        if source_split not in {"test", "train"}:
            raise ValueError(f"Sample {sid}: metadata.source_split must be test/train")

        source_id = int(source_ids[0])
        row = ds_test[source_id] if source_split == "test" else ds_train[source_id]
        conversation = row.get("conversation", []) if isinstance(row, dict) else []
        reasoning_text = _collect_counselor_think(conversation)

        plan_text = extract_plan_from_reasoning(
            reasoning_text=reasoning_text, max_sentences=int(args.max_sentences)
        )

        if not plan_text:
            empty += 1

        plans[sid] = {
            "plan": plan_text,
            "source_openr1_id": source_id,
            "source_split": source_split,
        }

        split_counts[source_split] = split_counts.get(source_split, 0) + 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_payload = {
        "meta": {
            "dataset": "GMLHUHE/OpenR1-Psy",
            "split": "mixed",
            "extraction": "generate_gold_plans.py",
            "notes": "Gold treatment plan snippets for Study A extracted heuristically from OpenR1-Psy counselor_think.",
            "updated_utc": datetime.utcnow().isoformat() + "Z",
            "script": "scripts/studies/study_a/gold_plans/generate_gold_plans.py",
            "source_split_counts": {
                "test": split_counts.get("test", 0),
                "train": split_counts.get("train", 0),
                "total": len(plans),
                "empty_plan": empty,
            },
        },
        "plans": plans,
    }

    out_path.write_text(
        json.dumps(out_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"wrote {out_path}")
    print(json.dumps(out_payload["meta"]["source_split_counts"], indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
