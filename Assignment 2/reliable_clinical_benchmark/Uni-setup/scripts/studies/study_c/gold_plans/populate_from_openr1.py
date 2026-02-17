"""Populate Study C gold target plans from OpenR1-Psy (deterministic, no API).

This script builds plan-of-care target plans for Study C (Longitudinal Drift)
by extracting them from OpenR1-Psy gold therapist reasoning.

Key idea:
- Study C cases include `metadata.source_openr1_ids`, which map back to row indices
  in the OpenR1-Psy dataset. Most indices are in the test split, but a small subset
  may reference the train split (out-of-bounds for test).
- We extract plan-of-care summaries from OpenR1-Psy `counselor_think` across the
  full conversation.

Output:
- data/study_c_gold/target_plans.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from datasets import load_dataset

from reliable_clinical_benchmark.data.study_c_loader import load_study_c_data


def _extract_plan_from_reasoning(text: str) -> str:
    if not text:
        return ""

    plan_parts: List[str] = []

    medication_patterns = [
        r"(?:recommend|suggest|prescribe|consider|start|continue|adjust|increase|decrease|change)\s+([^\.]{0,180}(?:medication|antidepressant|antianxiety|ssri|snri)[^\.]*)",
        r"(?:medication|antidepressant|antianxiety|ssri|snri)\s+(?:such as|like|including)?\s*([^\.]{10,180})",
    ]

    for pattern in medication_patterns:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            chunk = (m.group(1) or "").strip()
            if 10 <= len(chunk) <= 200:
                plan_parts.append(f"Medication: {chunk}")

    therapy_patterns = [
        r"(?:recommend|suggest|consider|refer|engage in|start|continue)\s+([^\.]{0,260}(?:therapy|counseling|cbt|act|psychotherapy|intervention|support)[^\.]*)",
        r"(?:therapy|counseling|cbt|act|psychotherapy|intervention)\s+(?:such as|like|including|focusing on)?\s*([^\.]{10,260})",
    ]

    for pattern in therapy_patterns:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            chunk = (m.group(1) or "").strip()
            if 15 <= len(chunk) <= 300:
                plan_parts.append(f"Therapy: {chunk}")

    coping_patterns = [
        r"(?:practice|try|use|learn|develop|implement)\s+([^\.]{0,180}(?:breathing|relaxation|mindfulness|grounding|exposure|skills|techniques|strategies)[^\.]*)",
        r"(?:coping|self-care|skills|techniques|strategies)\s+(?:such as|like|including)?\s*([^\.]{10,180})",
    ]

    for pattern in coping_patterns:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            chunk = (m.group(1) or "").strip()
            if 10 <= len(chunk) <= 200:
                plan_parts.append(f"Skills: {chunk}")

    followup_patterns = [
        r"(?:follow-up|follow up|monitor|track|check in|reassess|review)\s+([^\.]{10,180})",
        r"(?:schedule|plan|arrange)\s+(?:a|an)?\s*([^\.]{0,180}(?:follow-up|appointment|session|check-in)[^\.]*)",
    ]

    for pattern in followup_patterns:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            chunk = (m.group(1) or "").strip()
            if 10 <= len(chunk) <= 200:
                plan_parts.append(f"Follow-up: {chunk}")

    if plan_parts:
        return ". ".join(plan_parts[:5])

    sentences = [s.strip() for s in re.split(r"[.!?]\s+", text) if s.strip()]
    action_verbs = (
        "recommend",
        "suggest",
        "consider",
        "start",
        "continue",
        "try",
        "practice",
        "focus on",
        "work on",
        "develop",
        "build",
        "maintain",
        "monitor",
        "track",
    )

    action_sentences: List[str] = []
    for sent in sentences:
        sent_lower = sent.lower()
        if any(v in sent_lower for v in action_verbs) and 20 <= len(sent) <= 300:
            action_sentences.append(sent)

    # Fallback collectors
    substantial = [s for s in sentences[:3] if len(s) > 30]
    
    therapeutic_keywords = [
        "help", "support", "treatment", "therapy", "counseling", "patient", "client",
        "feel", "emotion", "thought", "behavior", "pattern", "skill", "strategy",
        "medication", "medication", "session", "progress", "goal", "plan"
    ]
    keyword_sentences = []
    for sent in sentences[:5]:
        sent_lower = sent.lower()
        if any(keyword in sent_lower for keyword in therapeutic_keywords) and len(sent.strip()) > 20:
            keyword_sentences.append(sent.strip())

    # Post-extraction selection to remove meta-talk and artifacts
    final_output = ""
    if plan_parts:
        final_output = ". ".join(plan_parts[:5])
    elif action_sentences:
        final_output = ". ".join(action_sentences[:3])
    elif substantial:
        final_output = ". ".join(substantial)
    elif keyword_sentences:
        final_output = ". ".join(keyword_sentences[:3])
    else:
        fallback = [s.strip() for s in sentences[:3] if len(s.strip()) > 25]
        final_output = ". ".join(fallback) if fallback else ""

    if not final_output:
        return ""

    # Function to check if a sentence is meta-talk
    def is_meta(s: str) -> bool:
        s = s.strip().lower()
        if not s: return True
        # Remove patterns that are explicitly meta-reasoning
        patterns = [
            r"^(?:okay|alright|so|now|first|then|finally|next),?\s*(?:let's|let (?:us|me)|i (?:will|should|need to|must|want to)|we (?:will|should))",
            r"^(?:next|then|firstly|secondly),?\s+(?:considering|regarding|following|based on)\b",
            r"^(?:considering|based on|given|following|regarding|considering)\b",
            r"^(?:the|my) (?:goal|plan|approach|strategy|process|structure) is\b",
            r"^(?:let me|i will|i should|i need to) (?:check|ensure|make sure|start|begin|try|use|incorporate|focus on|explore|suggest|unpack|provide)\b",
            r"^i (?:think|believe|feel) (?:that )?",
            r"^(?:instead|maybe|perhaps|additionally),?\s+(?:use|try|suggest|focus on|incorporate)\b",
            r"^let me (?:start|begin|unpack)\b",
            r"^i (?:should|need to|will) (?:gently|simply|just)\b",
        ]
        return any(re.search(p, s) for p in patterns)

    # Split into discrete sentences for cleaning
    sentences_to_clean = [s.strip() for s in re.split(r"(?<=[.!?])\s+", final_output) if s.strip()]
    
    # Filter out any meta-sentences
    cleaned_sentences = [s for s in sentences_to_clean if not is_meta(s)]
    
    if not cleaned_sentences:
        return ""

    final_output = " ".join(cleaned_sentences)
    
    # Capitalize first letter if it was lowercased by sub
    if final_output and final_output[0].islower():
        final_output = final_output[0].upper() + final_output[1:]

    # 2. Remove filler phrases
    fillers = [
        r"\b(?:gently|simply|just|really)\b",
        r"\b(?:try to|attempt to|look at)\b",
    ]
    for pattern in fillers:
        final_output = re.sub(pattern, "", final_output, flags=re.IGNORECASE).strip()

    # 3. Final polish: remove double spaces, fix punctuation
    final_output = re.sub(r'\s+', ' ', final_output)
    final_output = re.sub(r'\s+\.', '.', final_output)
    
    return final_output


def _collect_full_counselor_think(conversation: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for turn in conversation:
        ct = str(turn.get("counselor_think", "") or "").strip()
        if ct:
            parts.append(ct)
    return " ".join(parts).strip()


def build_plans(
    *,
    study_c_path: Path,
    openr1_revision: str,
    preferred_split: str,
) -> Dict[str, Dict[str, Any]]:
    cases = load_study_c_data(str(study_c_path))

    # Load both test and train splits (some cases may reference train split)
    ds_test = load_dataset("GMLHUHE/OpenR1-Psy", split="test", revision=openr1_revision)
    ds_train = load_dataset("GMLHUHE/OpenR1-Psy", split="train", revision=openr1_revision)

    preferred_split = (preferred_split or "test").lower().strip()
    if preferred_split not in ("test", "train"):
        preferred_split = "test"

    out: Dict[str, Dict[str, Any]] = {}

    for case in cases:
        source_ids = (case.metadata or {}).get("source_openr1_ids", [])
        if not isinstance(source_ids, list):
            source_ids = []

        chosen_idx: Optional[int] = None
        plan_text: str = ""
        used_split: Optional[str] = None

        for idx in source_ids:
            attempts = [preferred_split, "train" if preferred_split == "test" else "test"]
            row = None
            current_split = None

            for split_name in attempts:
                try:
                    ds = ds_test if split_name == "test" else ds_train
                    row = ds[int(idx)]
                    current_split = split_name
                    break
                except (IndexError, ValueError, KeyError):
                    continue

            if row is None or current_split is None:
                print(
                    f"  WARNING: {case.id} - OpenR1-Psy index {idx} invalid in both test and train"
                )
                continue

            convo = row.get("conversation") or []
            if not isinstance(convo, list) or not convo:
                print(f"  WARNING: {case.id} - OpenR1-Psy index {idx} ({current_split}) has no conversation")
                continue

            reasoning = _collect_full_counselor_think(convo)
            if not reasoning:
                print(f"  WARNING: {case.id} - OpenR1-Psy index {idx} ({current_split}) has no counselor_think")
                continue

            candidate = _extract_plan_from_reasoning(reasoning)
            if candidate and len(candidate.strip()) > 10:  # Ensure we have meaningful content
                chosen_idx = int(idx)
                plan_text = candidate
                used_split = current_split
                break
            elif not candidate:
                # If extraction failed, try to create a minimal plan from first few sentences
                sentences = [s.strip() for s in re.split(r"[.!?]\s+", reasoning) if s.strip()]
                if sentences:
                    # Take first 2-3 sentences that are substantial
                    fallback_sentences = [s for s in sentences[:3] if len(s) > 25]
                    if fallback_sentences:
                        chosen_idx = int(idx)
                        plan_text = ". ".join(fallback_sentences[:2])
                        used_split = current_split
                        print(f"  NOTE: {case.id} - Using fallback extraction from OpenR1-Psy index {idx} ({current_split})")
                        break

        out[case.id] = {
            "plan": plan_text,
            "source_openr1_id": chosen_idx,
            "source_split": used_split,  # Track which split was used
        }

    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Populate Study C gold target plans from OpenR1-Psy")
    p.add_argument(
        "--data-dir",
        type=str,
        default="data/openr1_psy_splits",
        help="Directory containing study_c_test.json",
    )
    p.add_argument(
        "--out",
        type=str,
        default="data/study_c_gold/target_plans.json",
        help="Output path for target_plans.json",
    )
    p.add_argument(
        "--revision",
        type=str,
        default="main",
        help="OpenR1-Psy dataset revision (commit hash, tag, or branch).",
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        help="Preferred OpenR1-Psy split to try first: test or train (default: test)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing plans even if non-empty",
    )
    args = p.parse_args()

    study_c_path = Path(args.data_dir) / "study_c_test.json"
    if not study_c_path.exists():
        raise SystemExit(f"Study C split not found: {study_c_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    new_plans = build_plans(
        study_c_path=study_c_path,
        openr1_revision=args.revision,
        preferred_split=args.split,
    )

    split_counts: Dict[str, int] = {"test": 0, "train": 0, "none": 0}
    for v in new_plans.values():
        if not isinstance(v, dict):
            continue
        s = v.get("source_split")
        if s in ("test", "train"):
            split_counts[str(s)] += 1
        else:
            split_counts["none"] += 1

    existing: Dict[str, Any] = {
        "meta": {},
        "plans": {},
    }
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {"meta": {}, "plans": {}}

    existing_meta = existing.get("meta", {}) if isinstance(existing.get("meta"), dict) else {}
    existing_plans = existing.get("plans", {}) if isinstance(existing.get("plans"), dict) else {}

    updated = 0
    for cid, entry in new_plans.items():
        if not isinstance(entry, dict):
            continue

        plan = str(entry.get("plan") or "")

        if cid not in existing_plans:
            existing_plans[cid] = entry
            updated += 1
            continue

        if args.force:
            if existing_plans.get(cid) != entry:
                existing_plans[cid] = entry
                updated += 1
        else:
            current = existing_plans.get(cid)
            current_plan = ""
            if isinstance(current, dict):
                current_plan = str(current.get("plan") or "")
            elif isinstance(current, str):
                current_plan = current

            if not current_plan and plan:
                existing_plans[cid] = entry
                updated += 1

    existing_meta.update(
        {
            "dataset": "GMLHUHE/OpenR1-Psy",
            "split": "mixed",
            "preferred_split": args.split,
            "revision": args.revision,
            "updated_utc": datetime.utcnow().isoformat() + "Z",
            "script": "scripts/studies/study_c/gold_plans/populate_from_openr1.py",
            "source_split_counts": split_counts,
        }
    )

    payload = {
        "meta": existing_meta,
        "plans": existing_plans,
    }

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    filled = sum(1 for v in existing_plans.values() if isinstance(v, dict) and v.get("plan"))
    total = len(existing_plans)
    print(f"Wrote {out_path} (updated={updated}; filled={filled}/{total})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


