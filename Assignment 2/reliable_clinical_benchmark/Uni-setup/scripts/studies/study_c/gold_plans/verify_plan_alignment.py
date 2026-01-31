"""Verify that extracted plans match their OpenR1-Psy source conversations.

Run with: conda activate openr1-env; python scripts/studies/study_c/gold_plans/verify_plan_alignment.py
"""

from __future__ import annotations

import json
from pathlib import Path
from datasets import load_dataset


def main():
    base_dir = Path(__file__).parent.parent.parent.parent.parent
    plans_path = base_dir / "data" / "study_c_gold" / "target_plans.json"
    cache_dir = base_dir / "Misc" / "datasets" / "openr1_psy"
    
    # Load plans
    with open(plans_path, "r", encoding="utf-8") as f:
        plans_data = json.load(f)
    plans = plans_data.get("plans", {})
    
    # Load dataset
    print("Loading OpenR1-Psy dataset...")
    ds_test = load_dataset("GMLHUHE/OpenR1-Psy", split="test", cache_dir=str(cache_dir))
    ds_train = load_dataset("GMLHUHE/OpenR1-Psy", split="train", cache_dir=str(cache_dir))
    
    # Sample cases to verify (spread across conditions)
    sample_cases = ["c_001", "c_011", "c_018", "c_020", "c_025", "c_050", "c_075"]
    
    print("\n" + "="*80)
    print("PLAN vs SOURCE CONVERSATION VERIFICATION")
    print("="*80)
    
    mismatches = []
    
    for case_id in sample_cases:
        if case_id not in plans:
            print(f"\n[SKIP] {case_id} not in plans")
            continue
            
        plan_entry = plans[case_id]
        source_id = plan_entry.get("source_openr1_id")
        source_split = plan_entry.get("source_split")
        plan_text = plan_entry.get("plan", "")
        condition = plan_entry.get("condition_matched", "")
        
        if source_id is None or source_split == "generated":
            print(f"\n[SKIP] {case_id} has no source_openr1_id (generated)")
            continue
        
        # Get source conversation
        ds = ds_test if source_split == "test" else ds_train
        try:
            row = ds[int(source_id)]
        except (IndexError, ValueError):
            print(f"\n[ERROR] {case_id}: Invalid source_id {source_id} in {source_split}")
            mismatches.append(case_id)
            continue
        
        conv = row.get("conversation", [])
        
        # Extract all counselor_think
        all_think = []
        for turn in conv:
            ct = turn.get("counselor_think", "")
            if ct:
                all_think.append(ct)
        
        full_reasoning = " ".join(all_think)
        
        # Check if plan text appears in source reasoning
        # Normalize for comparison
        plan_lower = plan_text.lower().replace("therapy:", "").replace("skills:", "").strip()
        reasoning_lower = full_reasoning.lower()
        
        # Extract key phrases from plan (first 50 chars of each sentence)
        plan_phrases = [s.strip()[:50].lower() for s in plan_text.split(".") if s.strip()]
        
        matches_found = 0
        for phrase in plan_phrases:
            if len(phrase) > 15 and phrase in reasoning_lower:
                matches_found += 1
        
        match_ratio = matches_found / max(len(plan_phrases), 1)
        
        print(f"\n{'='*60}")
        print(f"CASE: {case_id} | source_openr1_id: {source_id} ({source_split})")
        print(f"Condition: {condition}")
        print("-"*60)
        
        # Show patient context
        if conv:
            patient_msg = conv[0].get("patient", "")[:150]
            print(f"Patient: {patient_msg}...")
        
        print(f"\nExtracted Plan:")
        print(f"  {plan_text[:300]}...")
        
        print(f"\nSource counselor_think (excerpt):")
        print(f"  {full_reasoning[:400]}...")
        
        print(f"\nMatch Analysis:")
        print(f"  Phrases checked: {len(plan_phrases)}")
        print(f"  Phrases found in source: {matches_found}")
        print(f"  Match ratio: {match_ratio:.1%}")
        
        if match_ratio < 0.3:
            print(f"  ⚠️  LOW MATCH - may need review")
            mismatches.append(case_id)
        else:
            print(f"  ✓ GOOD MATCH")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Cases verified: {len(sample_cases)}")
    print(f"Low match cases: {len(mismatches)}")
    if mismatches:
        print(f"  Cases needing review: {mismatches}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
