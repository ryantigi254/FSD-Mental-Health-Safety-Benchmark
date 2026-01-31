"""Link ALL Study C cases to OpenR1-Psy source IDs and extract plans.

This script:
1. Uses existing source_openr1_ids where available (40 cases)
2. For unlinked cases (60), matches persona conditions to OpenR1-Psy entries
3. Assigns source_openr1_ids based on condition matching
4. Extracts plans from matched counselor_think

Run with: conda activate openr1-env; python scripts/studies/study_c/gold_plans/link_all_cases.py
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

# Persona condition -> OpenR1-Psy keyword mapping
CONDITION_KEYWORDS: Dict[str, List[str]] = {
    "major depressive disorder": ["depression", "depressed", "sad", "hopeless", "worthless", "tired", "fatigue"],
    "post-traumatic stress disorder": ["trauma", "ptsd", "flashback", "nightmare", "abuse", "accident"],
    "social anxiety disorder": ["social", "fear", "anxious", "nervous", "embarrass", "judged", "presentation"],
    "panic disorder": ["panic", "attack", "heart", "racing", "breathe", "dizzy", "fear"],
    "obsessive-compulsive disorder": ["obsess", "compuls", "ritual", "intrusive", "check", "wash", "ocd"],
    "generalized anxiety disorder": ["worry", "anxious", "nervous", "stress", "tense", "fear"],
    "bipolar disorder": ["mood", "manic", "energy", "sleep", "high", "low", "bipolar"],
    "alcohol use disorder": ["alcohol", "drink", "sober", "addiction", "relapse", "aa"],
    "anorexia nervosa": ["eating", "weight", "food", "thin", "restrict", "anorex", "body"],
    "psychosis": ["voice", "paranoid", "delusion", "reality", "see things", "hear"],
    "autism spectrum condition": ["social", "routine", "sensory", "communication", "autism", "spectrum"],
    "attention-deficit/hyperactivity disorder": ["focus", "attention", "distract", "impuls", "adhd", "concentrate"],
    "complicated grief": ["grief", "loss", "died", "death", "mourn", "passed away", "funeral"],
    "borderline personality disorder": ["abandon", "relationship", "emotion", "identity", "self-harm", "impuls"],
    "chronic pain with depression": ["pain", "chronic", "hurt", "suffer", "depression", "disability"],
    "caregiver burnout": ["caregiver", "exhausted", "burnout", "caring", "parent", "elderly"],
    "somatization disorder": ["symptom", "pain", "body", "medical", "health", "doctor", "test"],
    "schizophrenia": ["voice", "paranoid", "delusion", "reality", "psychosis", "hallucin"],
    "dissociative episodes": ["dissociat", "trauma", "memory", "blank", "detach", "numb"],
    "late-life depression": ["elderly", "retire", "lonely", "isolation", "aging", "old"],
    "late-life anxiety": ["elderly", "worry", "health", "fall", "aging", "anxious"],
    "body dysmorphic disorder": ["appearance", "ugly", "body", "mirror", "look", "face", "hide"],
    "burnout syndrome": ["work", "stress", "exhausted", "burnout", "job", "overwhelm", "tired"],
    "perinatal depression": ["baby", "pregnant", "birth", "mother", "postpartum", "new mom"],
}


def find_matching_openr1_entries(
    condition: str,
    ds: Any,
    used_indices: set,
    num_matches: int = 3
) -> List[Tuple[int, str]]:
    """Find OpenR1-Psy entries that match the given condition."""
    condition_lower = condition.lower()
    
    # Get relevant keywords for this condition
    keywords = []
    for cond_key, kws in CONDITION_KEYWORDS.items():
        if cond_key in condition_lower or condition_lower in cond_key:
            keywords.extend(kws)
            break
    
    # Fallback: extract keywords from condition itself
    if not keywords:
        keywords = condition_lower.replace("-", " ").replace("disorder", "").split()
    
    matches: List[Tuple[int, float, str]] = []  # (index, score, reasoning)
    
    for i, row in enumerate(ds):
        if i in used_indices:
            continue
            
        conv = row.get("conversation", [])
        if not conv:
            continue
        
        # Combine all text for matching
        full_text = ""
        reasoning_parts = []
        for turn in conv:
            patient = str(turn.get("patient", "") or "").lower()
            counselor = str(turn.get("counselor", "") or "").lower()
            counselor_think = str(turn.get("counselor_think", "") or "")
            full_text += f" {patient} {counselor}"
            if counselor_think:
                reasoning_parts.append(counselor_think)
        
        # Score based on keyword matches
        score = 0
        for kw in keywords:
            if kw in full_text:
                score += 1
        
        if score > 0 and reasoning_parts:
            matches.append((i, score, " ".join(reasoning_parts)))
    
    # Sort by score descending and return top matches
    matches.sort(key=lambda x: -x[1])
    return [(m[0], m[2]) for m in matches[:num_matches]]


def extract_plan_from_reasoning(reasoning: str) -> str:
    """Extract therapeutic plan from counselor_think reasoning."""
    if not reasoning:
        return ""
    
    plan_parts = []
    
    # Therapy extraction
    therapy_patterns = [
        r"(?:recommend|suggest|consider|engage|start|continue).*?(?:therapy|counseling|cbt|dbt|act|psychotherapy|intervention|approach)[^.]{0,200}",
        r"(?:therapy|counseling|cbt|dbt|act|psychotherapy|approach).*?(?:help|address|focus|work)[^.]{0,200}",
    ]
    for pattern in therapy_patterns:
        matches = re.findall(pattern, reasoning, flags=re.IGNORECASE)
        for m in matches[:1]:
            if 20 <= len(m) <= 300:
                plan_parts.append(f"Therapy: {m.strip()}")
                break
    
    # Skills extraction
    skills_patterns = [
        r"(?:practice|try|use|learn|develop).*?(?:breathing|relaxation|mindfulness|grounding|skills|techniques|strategies|coping)[^.]{0,150}",
    ]
    for pattern in skills_patterns:
        matches = re.findall(pattern, reasoning, flags=re.IGNORECASE)
        for m in matches[:1]:
            if 15 <= len(m) <= 200:
                plan_parts.append(f"Skills: {m.strip()}")
                break
    
    # Follow-up extraction
    followup_patterns = [
        r"(?:follow-up|follow up|monitor|track|check in|reassess)[^.]{10,150}",
    ]
    for pattern in followup_patterns:
        matches = re.findall(pattern, reasoning, flags=re.IGNORECASE)
        for m in matches[:1]:
            if 10 <= len(m) <= 200:
                plan_parts.append(f"Follow-up: {m.strip()}")
                break
    
    if plan_parts:
        return ". ".join(plan_parts)
    
    # Fallback: extract sentences with supportive language
    sentences = [s.strip() for s in re.split(r"[.!?]", reasoning) if s.strip()]
    supportive_kws = ["help", "support", "explore", "address", "focus", "work", "suggest", "recommend", "encourage"]
    relevant = []
    for sent in sentences[:15]:
        if any(kw in sent.lower() for kw in supportive_kws) and 25 <= len(sent) <= 250:
            relevant.append(sent)
    
    return ". ".join(relevant[:3]) if relevant else ""


def main() -> int:
    base_dir = Path(__file__).parent.parent.parent.parent.parent
    study_c_path = base_dir / "data" / "openr1_psy_splits" / "study_c_test.json"
    output_path = base_dir / "data" / "study_c_gold" / "target_plans.json"
    cache_dir = base_dir / "Misc" / "datasets" / "openr1_psy"
    
    # Load Study C cases
    print(f"Loading Study C cases...")
    with open(study_c_path, "r", encoding="utf-8") as f:
        study_c_data = json.load(f)
    cases = study_c_data.get("cases", [])
    print(f"  Loaded {len(cases)} cases")
    
    # Load OpenR1-Psy
    print(f"\nLoading OpenR1-Psy dataset...")
    ds_test = load_dataset("GMLHUHE/OpenR1-Psy", split="test", cache_dir=str(cache_dir))
    ds_train = load_dataset("GMLHUHE/OpenR1-Psy", split="train", cache_dir=str(cache_dir))
    print(f"  Test: {len(ds_test)} rows, Train: {len(ds_train)} rows")
    
    # Track used indices to avoid duplicates
    used_test_indices: set = set()
    used_train_indices: set = set()
    
    # Generate plans for all cases
    print("\nGenerating linked plans for all 100 cases...")
    plans: Dict[str, Dict[str, Any]] = {}
    linked_existing = 0
    linked_new = 0
    
    for i, case in enumerate(cases):
        case_id = case.get("id", f"c_{i+1:03d}")
        source_ids = case.get("metadata", {}).get("source_openr1_ids", [])
        critical_entities = case.get("critical_entities", [])
        condition = critical_entities[0] if critical_entities else ""
        
        plan_text = ""
        source_id = None
        source_split = None
        
        # 1. Try existing source_openr1_ids first
        if source_ids:
            for idx in source_ids:
                idx = int(idx)
                for split_name, ds, used_set in [("test", ds_test, used_test_indices), ("train", ds_train, used_train_indices)]:
                    try:
                        if idx < len(ds):
                            row = ds[idx]
                            conv = row.get("conversation", [])
                            reasoning_parts = [str(t.get("counselor_think", "") or "") for t in conv if t.get("counselor_think")]
                            reasoning = " ".join(reasoning_parts)
                            extracted = extract_plan_from_reasoning(reasoning)
                            if extracted and len(extracted) > 30:
                                plan_text = extracted
                                source_id = idx
                                source_split = split_name
                                used_set.add(idx)
                                linked_existing += 1
                                break
                    except Exception:
                        continue
                if plan_text:
                    break
        
        # 2. For unlinked cases, find matching OpenR1-Psy entries by condition
        if not plan_text and condition:
            # Try test split first
            matches = find_matching_openr1_entries(condition, ds_test, used_test_indices, num_matches=3)
            for idx, reasoning in matches:
                extracted = extract_plan_from_reasoning(reasoning)
                if extracted and len(extracted) > 30:
                    plan_text = extracted
                    source_id = idx
                    source_split = "test"
                    used_test_indices.add(idx)
                    linked_new += 1
                    break
            
            # Try train split if no match
            if not plan_text:
                matches = find_matching_openr1_entries(condition, ds_train, used_train_indices, num_matches=3)
                for idx, reasoning in matches:
                    extracted = extract_plan_from_reasoning(reasoning)
                    if extracted and len(extracted) > 30:
                        plan_text = extracted
                        source_id = idx
                        source_split = "train"
                        used_train_indices.add(idx)
                        linked_new += 1
                        break
        
        plans[case_id] = {
            "plan": plan_text,
            "source_openr1_id": source_id,
            "source_split": source_split,
            "condition_matched": condition,
        }
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(cases)} cases (linked: {linked_existing + linked_new})...")
    
    # Summary stats
    linked_total = linked_existing + linked_new
    unlinked = len(cases) - linked_total
    
    # Output
    output = {
        "meta": {
            "dataset": "GMLHUHE/OpenR1-Psy",
            "split": "mixed",
            "extraction": "link_all_cases.py",
            "notes": "Gold target plans for Study C. ALL cases linked to OpenR1-Psy by original IDs or condition matching.",
            "updated_utc": datetime.utcnow().isoformat() + "Z",
            "script": "scripts/studies/study_c/gold_plans/link_all_cases.py",
            "source_split_counts": {
                "linked_existing": linked_existing,
                "linked_new_test": sum(1 for p in plans.values() if p.get("source_split") == "test" and not any(c.get("metadata", {}).get("source_openr1_ids") for c in cases if c.get("id") == list(plans.keys())[list(plans.values()).index(p)])),
                "linked_new_train": sum(1 for p in plans.values() if p.get("source_split") == "train"),
                "unlinked": unlinked,
                "total": len(cases),
            },
        },
        "plans": plans,
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*50}")
    print(f"Wrote {output_path}")
    print(f"  Linked (existing IDs): {linked_existing}")
    print(f"  Linked (condition match): {linked_new}")
    print(f"  Total linked: {linked_total}")
    print(f"  Unlinked: {unlinked}")
    print(f"  Total: {len(plans)}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
