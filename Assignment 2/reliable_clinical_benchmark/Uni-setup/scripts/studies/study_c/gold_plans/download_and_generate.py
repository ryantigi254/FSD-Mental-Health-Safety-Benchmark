"""Download OpenR1-Psy dataset and generate Study C gold plans with proper linkage.

This script:
1. Downloads OpenR1-Psy to Misc/datasets/openr1_psy/
2. Generates gold plans for all 100 Study C cases
3. Links plans to source_openr1_ids where available
4. Uses condition-treatment mapping as fallback

Run with: conda activate openr1-env; python scripts/studies/study_c/gold_plans/download_and_generate.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure we can import from parent
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from datasets import load_dataset

# Condition -> treatment mappings
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


def extract_plan_from_counselor_think(conversation: List[Dict[str, Any]]) -> str:
    """Extract therapeutic plan from counselor_think fields."""
    all_reasoning = []
    for turn in conversation:
        ct = str(turn.get("counselor_think", "") or "").strip()
        if ct:
            all_reasoning.append(ct)
    
    if not all_reasoning:
        return ""
    
    full_text = " ".join(all_reasoning)
    
    plan_parts = []
    
    # Extract therapy recommendations
    therapy_patterns = [
        r"(?:recommend|suggest|consider|refer|engage in|start|continue).*?(?:therapy|counseling|cbt|act|dbt|psychotherapy|intervention|support)[^.]{0,200}",
        r"(?:therapy|counseling|cbt|act|dbt|psychotherapy|intervention).*?(?:help|address|focus|work on)[^.]{0,200}",
    ]
    for pattern in therapy_patterns:
        matches = re.findall(pattern, full_text, flags=re.IGNORECASE)
        for m in matches[:2]:
            if 20 <= len(m) <= 300:
                plan_parts.append(f"Therapy: {m.strip()}")
                break
    
    # Extract coping/skills recommendations
    skills_patterns = [
        r"(?:practice|try|use|learn|develop).*?(?:breathing|relaxation|mindfulness|grounding|skills|techniques|strategies)[^.]{0,150}",
        r"(?:coping|self-care|skills|techniques).*?(?:help|useful|effective)[^.]{0,150}",
    ]
    for pattern in skills_patterns:
        matches = re.findall(pattern, full_text, flags=re.IGNORECASE)
        for m in matches[:1]:
            if 15 <= len(m) <= 200:
                plan_parts.append(f"Skills: {m.strip()}")
                break
    
    # Extract follow-up/monitoring
    followup_patterns = [
        r"(?:follow-up|follow up|monitor|track|check in|reassess)[^.]{10,150}",
        r"(?:schedule|plan|arrange).*?(?:appointment|session|check-in)[^.]{0,150}",
    ]
    for pattern in followup_patterns:
        matches = re.findall(pattern, full_text, flags=re.IGNORECASE)
        for m in matches[:1]:
            if 10 <= len(m) <= 200:
                plan_parts.append(f"Follow-up: {m.strip()}")
                break
    
    if plan_parts:
        return ". ".join(plan_parts)
    
    # Fallback: extract sentences with action verbs
    sentences = [s.strip() for s in re.split(r"[.!?]", full_text) if s.strip()]
    action_keywords = ["recommend", "suggest", "help", "support", "explore", "address", "focus", "work"]
    relevant = []
    for sent in sentences[:10]:
        if any(kw in sent.lower() for kw in action_keywords) and 25 <= len(sent) <= 250:
            relevant.append(sent)
    
    return ". ".join(relevant[:3]) if relevant else ""


def normalize_condition(condition: str) -> str:
    """Normalize condition string for matching."""
    text = condition.lower().strip()
    text = re.sub(r"\(.*?\)", "", text).strip()
    text = re.sub(r"early recovery|in recovery|prodromal|stable|antenatal|postnatal", "", text).strip()
    text = re.sub(r"with.*$", "", text).strip()
    return text


def get_treatment_plan(condition: str, critical_entities: List[str]) -> str:
    """Generate a treatment plan based on condition and patient context."""
    normalized = normalize_condition(condition)
    
    treatment = None
    for key, value in CONDITION_TREATMENT_MAP.items():
        if key in normalized or normalized in key:
            treatment = value
            break
    
    if not treatment:
        for key, value in CONDITION_TREATMENT_MAP.items():
            key_words = set(key.split())
            condition_words = set(normalized.split())
            if key_words & condition_words:
                treatment = value
                break
    
    if not treatment:
        treatment = {
            "therapy": f"appropriate psychological therapy for {condition}",
            "medication": "psychiatric review for medication if indicated",
            "skills": "coping strategies and psychoeducation",
            "monitoring": "symptom tracking and regular follow-up",
        }
    
    plan_parts = []
    plan_parts.append(f"Therapy: {treatment['therapy']}")
    
    med_mentions = [e for e in critical_entities if any(m in e.lower() for m in 
                   ["mg", "medication", "sertraline", "fluoxetine", "antidepressant", "lithium", "prazosin"])]
    if med_mentions:
        plan_parts.append(f"Medication: Continue/monitor {med_mentions[0]}; {treatment['medication']}")
    else:
        plan_parts.append(f"Medication: {treatment['medication']}")
    
    plan_parts.append(f"Skills: {treatment['skills']}")
    plan_parts.append(f"Follow-up: {treatment['monitoring']}")
    
    return ". ".join(plan_parts)


def main() -> int:
    # Paths
    base_dir = Path(__file__).parent.parent.parent.parent.parent
    study_c_path = base_dir / "data" / "openr1_psy_splits" / "study_c_test.json"
    output_path = base_dir / "data" / "study_c_gold" / "target_plans.json"
    cache_dir = base_dir / "Misc" / "datasets" / "openr1_psy"
    
    # Ensure directories exist
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load Study C cases
    print(f"Loading Study C cases from {study_c_path}...")
    with open(study_c_path, "r", encoding="utf-8") as f:
        study_c_data = json.load(f)
    cases = study_c_data.get("cases", [])
    print(f"  Loaded {len(cases)} cases")
    
    # Load OpenR1-Psy dataset
    print(f"\nDownloading/loading OpenR1-Psy dataset to {cache_dir}...")
    ds_test = load_dataset("GMLHUHE/OpenR1-Psy", split="test", cache_dir=str(cache_dir))
    ds_train = load_dataset("GMLHUHE/OpenR1-Psy", split="train", cache_dir=str(cache_dir))
    print(f"  Test split: {len(ds_test)} rows")
    print(f"  Train split: {len(ds_train)} rows")
    
    # Save dataset info locally for reference
    dataset_info = {
        "test_count": len(ds_test),
        "train_count": len(ds_train),
        "downloaded_at": datetime.utcnow().isoformat() + "Z",
        "cache_dir": str(cache_dir),
    }
    info_path = cache_dir / "dataset_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)
    print(f"  Saved dataset info to {info_path}")
    
    # Generate plans
    print("\nGenerating gold plans...")
    plans: Dict[str, Dict[str, Any]] = {}
    linked_count = 0
    generated_count = 0
    
    for i, case in enumerate(cases):
        case_id = case.get("id", f"c_{i+1:03d}")
        source_ids = case.get("metadata", {}).get("source_openr1_ids", [])
        patient_summary = case.get("patient_summary", "")
        critical_entities = case.get("critical_entities", [])
        
        plan_text = ""
        source_id = None
        source_split = None
        
        # Try OpenR1-Psy linkage
        if source_ids:
            for idx in source_ids:
                idx = int(idx)
                # Try test split first
                for split_name, ds in [("test", ds_test), ("train", ds_train)]:
                    try:
                        if idx < len(ds):
                            row = ds[idx]
                            convo = row.get("conversation", [])
                            extracted = extract_plan_from_counselor_think(convo)
                            if extracted and len(extracted) > 30:
                                plan_text = extracted
                                source_id = idx
                                source_split = split_name
                                linked_count += 1
                                break
                    except Exception:
                        continue
                if plan_text:
                    break
        
        # Fallback to condition-based generation
        if not plan_text and critical_entities:
            condition = critical_entities[0] if critical_entities else ""
            plan_text = get_treatment_plan(condition, critical_entities)
            source_split = "generated"
            generated_count += 1
        
        plans[case_id] = {
            "plan": plan_text,
            "source_openr1_id": source_id,
            "source_split": source_split,
        }
        
        if (i + 1) % 25 == 0:
            print(f"  Processed {i + 1}/{len(cases)} cases...")
    
    # Build output
    output = {
        "meta": {
            "dataset": "GMLHUHE/OpenR1-Psy",
            "split": "mixed",
            "extraction": "download_and_generate.py",
            "notes": "Gold target plans for Study C. Linked cases extracted from OpenR1-Psy counselor_think; unlinked cases generated from condition-treatment mappings.",
            "updated_utc": datetime.utcnow().isoformat() + "Z",
            "script": "scripts/studies/study_c/gold_plans/download_and_generate.py",
            "source_split_counts": {
                "linked_test": sum(1 for p in plans.values() if p.get("source_split") == "test"),
                "linked_train": sum(1 for p in plans.values() if p.get("source_split") == "train"),
                "generated": generated_count,
                "total": len(cases),
            },
        },
        "plans": plans,
    }
    
    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nWrote {output_path}")
    print(f"  Linked (OpenR1): {linked_count}")
    print(f"  Generated: {generated_count}")
    print(f"  Total: {len(plans)}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
