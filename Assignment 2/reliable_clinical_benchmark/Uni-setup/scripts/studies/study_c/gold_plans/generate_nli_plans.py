"""Generate Study C gold target plans using NLI-based extraction.

This script creates gold plan-of-care summaries for Study C cases:
1. For cases WITH OpenR1-Psy linkage: Extract from counselor_think (existing method)
2. For cases WITHOUT linkage: Generate structured plans from patient_summary + critical_entities

Output:
- data/study_c_gold/target_plans.json (updated with all 100 cases)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

# Standard condition -> treatment mappings for gold plans
CONDITION_TREATMENT_MAP: Dict[str, Dict[str, Any]] = {
    "major depressive disorder": {
        "therapy": "CBT or interpersonal therapy for depression",
        "medication": "SSRI antidepressant (e.g., sertraline) with monitoring",
        "skills": "behavioral activation, mood tracking, and sleep hygiene",
        "monitoring": "suicidal ideation screening and PHQ-9 tracking",
    },
    "post-traumatic stress disorder": {
        "therapy": "trauma-focused CBT or EMDR",
        "medication": "SSRI (e.g., sertraline) with prazosin for nightmares if indicated",
        "skills": "grounding techniques, safety planning, and emotion regulation",
        "monitoring": "flashback frequency and sleep quality assessment",
    },
    "social anxiety disorder": {
        "therapy": "CBT with exposure hierarchy for social situations",
        "medication": "SSRI (e.g., sertraline) for moderate-severe anxiety",
        "skills": "cognitive restructuring, gradual exposure, and relaxation techniques",
        "monitoring": "avoidance behaviors and functional impairment",
    },
    "panic disorder": {
        "therapy": "CBT with interoceptive exposure and breathing retraining",
        "medication": "SSRI with consideration of short-term benzodiazepine if severe",
        "skills": "breathing exercises, cognitive reframing of catastrophic thoughts",
        "monitoring": "panic attack frequency and avoidance patterns",
    },
    "obsessive-compulsive disorder": {
        "therapy": "ERP (Exposure and Response Prevention)",
        "medication": "SSRI at higher doses (e.g., fluoxetine 40-60mg)",
        "skills": "response prevention strategies and uncertainty tolerance",
        "monitoring": "OCD severity (Y-BOCS) and ritual time",
    },
    "generalized anxiety disorder": {
        "therapy": "CBT focusing on worry management and uncertainty tolerance",
        "medication": "SSRI or SNRI for persistent anxiety",
        "skills": "worry time scheduling, relaxation training, and mindfulness",
        "monitoring": "GAD-7 scores and sleep quality",
    },
    "bipolar disorder": {
        "therapy": "psychoeducation and mood monitoring with relapse prevention",
        "medication": "mood stabilizer (e.g., lithium, valproate) with monitoring",
        "skills": "sleep regulation, early warning sign recognition",
        "monitoring": "mood episodes, medication adherence, and lithium levels if applicable",
    },
    "alcohol use disorder": {
        "therapy": "motivational interviewing and relapse prevention",
        "medication": "naltrexone or acamprosate; disulfiram if appropriate",
        "skills": "trigger identification, coping strategies, and support group attendance",
        "monitoring": "abstinence tracking, liver function, and craving intensity",
    },
    "anorexia nervosa": {
        "therapy": "family-based therapy (FBT) or CBT-E for eating disorders",
        "medication": "limited role; address comorbid depression/anxiety",
        "skills": "meal planning, body image work, and cognitive flexibility",
        "monitoring": "weight restoration, vital signs, and electrolytes",
    },
    "psychosis": {
        "therapy": "CBTp (CBT for psychosis) and family intervention",
        "medication": "antipsychotic medication with metabolic monitoring",
        "skills": "reality testing, early warning sign recognition, and social skills",
        "monitoring": "positive/negative symptoms, medication side effects",
    },
    "autism spectrum condition": {
        "therapy": "supportive therapy with focus on social skills and sensory management",
        "medication": "address comorbid anxiety/depression; no specific medication for autism",
        "skills": "sensory regulation strategies, routine building, and communication aids",
        "monitoring": "adaptive functioning and quality of life",
    },
    "attention-deficit/hyperactivity disorder": {
        "therapy": "CBT for ADHD focusing on organizational skills",
        "medication": "stimulant (e.g., methylphenidate) or non-stimulant (atomoxetine)",
        "skills": "time management, task breakdown, and environmental modifications",
        "monitoring": "symptom tracking and medication response",
    },
    "complicated grief": {
        "therapy": "complicated grief treatment (CGT) or grief-focused CBT",
        "medication": "antidepressant if major depressive episode co-occurs",
        "skills": "gradual re-engagement with life, memory processing",
        "monitoring": "prolonged grief disorder symptoms and functioning",
    },
    "borderline personality disorder": {
        "therapy": "DBT (Dialectical Behavior Therapy) or MBT",
        "medication": "address comorbid symptoms; no specific medication for BPD",
        "skills": "distress tolerance, emotion regulation, interpersonal effectiveness",
        "monitoring": "self-harm urges, crisis episodes, and relationship patterns",
    },
    "chronic pain with depression": {
        "therapy": "pain management program with CBT for chronic pain",
        "medication": "antidepressant with pain benefits (e.g., duloxetine, amitriptyline)",
        "skills": "pacing, activity scheduling, and cognitive coping for pain",
        "monitoring": "pain levels, functional status, and mood",
    },
    "caregiver burnout": {
        "therapy": "stress management and supportive counseling",
        "medication": "antidepressant if depression develops",
        "skills": "respite planning, boundary setting, and self-care scheduling",
        "monitoring": "burnout symptoms, depression screening, and respite utilization",
    },
    "somatization disorder": {
        "therapy": "CBT for health anxiety and somatic symptoms",
        "medication": "low-dose antidepressant for symptom management",
        "skills": "attention shifting, activity pacing, and medical reassurance reduction",
        "monitoring": "healthcare utilization and symptom distress",
    },
    "schizophrenia": {
        "therapy": "CBTp and family intervention; vocational rehabilitation",
        "medication": "antipsychotic with metabolic monitoring",
        "skills": "early warning signs, medication adherence, and social engagement",
        "monitoring": "symptom severity, side effects, and functional goals",
    },
    "dissociative episodes": {
        "therapy": "phase-oriented trauma therapy with grounding focus",
        "medication": "address comorbid PTSD/depression",
        "skills": "grounding techniques, safety signals, and containment strategies",
        "monitoring": "dissociation frequency and trauma processing progress",
    },
    "late-life depression": {
        "therapy": "CBT adapted for older adults; reminiscence therapy",
        "medication": "SSRI with attention to drug interactions and side effects",
        "skills": "behavioral activation, social engagement, and meaningful activity",
        "monitoring": "depression severity, cognitive status, and medical comorbidities",
    },
    "late-life anxiety": {
        "therapy": "CBT adapted for older adults; relaxation training",
        "medication": "SSRI with careful titration; avoid benzodiazepines",
        "skills": "worry management, relaxation, and cognitive restructuring",
        "monitoring": "anxiety severity, fall risk, and cognitive function",
    },
    "body dysmorphic disorder": {
        "therapy": "CBT with mirror exposure and response prevention",
        "medication": "SSRI at higher doses (similar to OCD)",
        "skills": "reducing checking/camouflaging, cognitive restructuring",
        "monitoring": "BDD severity and insight level",
    },
    "burnout syndrome": {
        "therapy": "stress management and work-life balance counseling",
        "medication": "antidepressant if depression develops",
        "skills": "boundary setting, recovery activities, and values clarification",
        "monitoring": "burnout dimensions, sleep, and return-to-work planning",
    },
    "perinatal depression": {
        "therapy": "CBT or interpersonal therapy adapted for perinatal period",
        "medication": "SSRI with consideration of breastfeeding safety",
        "skills": "infant attachment support, partner involvement, and self-care",
        "monitoring": "mood, bonding concerns, and infant safety",
    },
}


def normalize_condition(condition: str) -> str:
    """Normalize condition string for matching."""
    text = condition.lower().strip()
    # Remove common suffixes/prefixes
    text = re.sub(r"\(.*?\)", "", text).strip()
    text = re.sub(r"early recovery|in recovery|prodromal|stable|antenatal|postnatal", "", text).strip()
    text = re.sub(r"with.*$", "", text).strip()
    return text


def get_treatment_plan(condition: str, patient_summary: str, critical_entities: List[str]) -> str:
    """Generate a treatment plan based on condition and patient context."""
    normalized = normalize_condition(condition)
    
    # Try direct match
    treatment = None
    for key, value in CONDITION_TREATMENT_MAP.items():
        if key in normalized or normalized in key:
            treatment = value
            break
    
    # Try partial matches
    if not treatment:
        for key, value in CONDITION_TREATMENT_MAP.items():
            key_words = set(key.split())
            condition_words = set(normalized.split())
            if key_words & condition_words:  # Any overlap
                treatment = value
                break
    
    # Fallback to generic plan
    if not treatment:
        treatment = {
            "therapy": f"appropriate psychological therapy for {condition}",
            "medication": "psychiatric review for medication if indicated",
            "skills": "coping strategies and psychoeducation",
            "monitoring": "symptom tracking and regular follow-up",
        }
    
    # Build plan text incorporating critical entities
    plan_parts = []
    
    # Therapy
    plan_parts.append(f"Therapy: {treatment['therapy']}")
    
    # Medication - incorporate any mentioned medications
    med_mentions = [e for e in critical_entities if any(m in e.lower() for m in 
                   ["mg", "medication", "sertraline", "fluoxetine", "antidepressant", "lithium", "prazosin"])]
    if med_mentions:
        plan_parts.append(f"Medication: Continue/monitor {med_mentions[0]}; {treatment['medication']}")
    else:
        plan_parts.append(f"Medication: {treatment['medication']}")
    
    # Skills
    plan_parts.append(f"Skills: {treatment['skills']}")
    
    # Monitoring
    plan_parts.append(f"Follow-up: {treatment['monitoring']}")
    
    return ". ".join(plan_parts)


def load_study_c_cases(study_c_path: Path) -> List[Dict[str, Any]]:
    """Load Study C cases from JSON."""
    with open(study_c_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("cases", [])


def main() -> int:
    p = argparse.ArgumentParser(description="Generate Study C gold plans with NLI-based extraction")
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

    # Load cases
    cases = load_study_c_cases(study_c_path)
    print(f"Loaded {len(cases)} Study C cases")

    # Load existing plans if present
    existing: Dict[str, Any] = {"meta": {}, "plans": {}}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {"meta": {}, "plans": {}}

    existing_plans = existing.get("plans", {}) if isinstance(existing.get("plans"), dict) else {}

    # Try to run OpenR1 extraction for linked cases first
    linked_count = 0
    unlinked_count = 0
    updated_count = 0
    
    try:
        from datasets import load_dataset
        ds_test = load_dataset("GMLHUHE/OpenR1-Psy", split="test")
        ds_train = load_dataset("GMLHUHE/OpenR1-Psy", split="train")
        openr1_available = True
        print("OpenR1-Psy dataset loaded successfully")
    except Exception as e:
        print(f"Could not load OpenR1-Psy: {e}")
        print("Will generate all plans from patient summaries")
        openr1_available = False
        ds_test = None
        ds_train = None

    for case in cases:
        case_id = case.get("id", "")
        source_ids = case.get("metadata", {}).get("source_openr1_ids", [])
        patient_summary = case.get("patient_summary", "")
        critical_entities = case.get("critical_entities", [])
        
        # Skip if already have plan and not forcing
        if case_id in existing_plans and existing_plans[case_id].get("plan") and not args.force:
            continue
        
        plan_text = ""
        source_id = None
        source_split = None
        
        # Try OpenR1 linkage first
        if openr1_available and source_ids:
            for idx in source_ids:
                for split_name, ds in [("test", ds_test), ("train", ds_train)]:
                    try:
                        row = ds[int(idx)]
                        convo = row.get("conversation", [])
                        reasoning_parts = []
                        for turn in convo:
                            ct = str(turn.get("counselor_think", "") or "").strip()
                            if ct:
                                reasoning_parts.append(ct)
                        
                        if reasoning_parts:
                            # Extract plan from reasoning (simplified)
                            full_reasoning = " ".join(reasoning_parts)
                            # Look for therapeutic content
                            therapy_match = re.search(
                                r"(?:recommend|suggest|therapy|counseling|CBT|support)[^.]{20,200}",
                                full_reasoning, re.IGNORECASE
                            )
                            if therapy_match:
                                plan_text = f"Therapy: {therapy_match.group(0).strip()}"
                                source_id = int(idx)
                                source_split = split_name
                                linked_count += 1
                                break
                    except (IndexError, ValueError, KeyError):
                        continue
                if plan_text:
                    break
        
        # Fallback to patient summary extraction
        if not plan_text and patient_summary:
            # Extract condition from patient summary
            condition = ""
            for entity in critical_entities:
                # First critical entity is usually the condition
                if any(c in entity.lower() for c in ["disorder", "depression", "anxiety", "ptsd", "ocd", "autism", "psychosis", "burnout"]):
                    condition = entity
                    break
            
            if not condition and critical_entities:
                condition = critical_entities[0]
            
            if condition:
                plan_text = get_treatment_plan(condition, patient_summary, critical_entities)
                source_split = "generated"
                unlinked_count += 1
        
        if plan_text:
            existing_plans[case_id] = {
                "plan": plan_text,
                "source_openr1_id": source_id,
                "source_split": source_split,
            }
            updated_count += 1
            if updated_count % 20 == 0:
                print(f"  Processed {updated_count} cases...")

    # Update metadata
    meta = {
        "dataset": "GMLHUHE/OpenR1-Psy",
        "split": "mixed",
        "extraction": "generate_nli_plans.py",
        "notes": "Gold target plans for Study C. Linked cases extracted from OpenR1-Psy; unlinked cases generated from patient_summary + critical_entities.",
        "updated_utc": datetime.utcnow().isoformat() + "Z",
        "script": "scripts/studies/study_c/gold_plans/generate_nli_plans.py",
        "source_split_counts": {
            "linked": linked_count,
            "generated": unlinked_count,
            "total": len(cases),
        },
    }

    payload = {
        "meta": meta,
        "plans": existing_plans,
    }

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    filled = sum(1 for v in existing_plans.values() if isinstance(v, dict) and v.get("plan"))
    print(f"\nWrote {out_path}")
    print(f"  Updated: {updated_count}")
    print(f"  Linked (OpenR1): {linked_count}")
    print(f"  Generated: {unlinked_count}")
    print(f"  Total filled: {filled}/{len(cases)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
