"""Generate Study C gold target plans.

This script creates plan-of-care targets for Study C cases:
1. For linked cases: classify a fixed set of plan components using NLI as a verifier over OpenR1-Psy `counselor_think`, then render a deterministic structured plan.
2. For unlinked/synthetic cases: generate a rule-based guideline plan from patient_summary + critical_entities.

Output:
- data/study_c_gold/target_plans.json (updated with all cases)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from reliable_clinical_benchmark.utils.nli import NLIModel
from reliable_clinical_benchmark.utils.plan_components import (
    DEFAULT_PLAN_COMPONENTS,
    classify_plan_components,
    extract_recommendation_candidates,
    nli_filter_candidates,
    render_plan_from_components,
)

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


def _collect_full_counselor_think(conversation: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for turn in conversation:
        ct = str(turn.get("counselor_think", "") or "").strip()
        if ct:
            parts.append(ct)
    return " ".join(parts).strip()


def _looks_like_actionable_plan(text: str) -> bool:
    t = str(text or "").strip()
    if not t:
        return False

    if any(k in t for k in ("Therapy:", "Medication:", "Skills:", "Follow-up:")):
        return True

    if len(t) < 120:
        return False

    t_lower = t.lower()
    action_terms = (
        "recommend",
        "suggest",
        "consider",
        "plan",
        "review",
        "assess",
        "follow-up",
        "therapy",
        "medication",
        "refer",
        "monitor",
        "skills",
    )
    return sum(1 for term in action_terms if term in t_lower) >= 2


_MED_KEYWORDS = {
    "sertraline",
    "fluoxetine",
    "citalopram",
    "escitalopram",
    "paroxetine",
    "venlafaxine",
    "duloxetine",
    "mirtazapine",
    "bupropion",
    "lisdexamfetamine",
    "methylphenidate",
    "atomoxetine",
    "naltrexone",
    "acamprosate",
    "disulfiram",
    "prazosin",
    "lithium",
    "valproate",
    "lamotrigine",
    "quetiapine",
    "olanzapine",
    "risperidone",
    "aripiprazole",
    "clozapine",
    "amlodipine",
}


_CONSTRAINT_KEYWORDS = {
    "allergy",
    "allergic",
    "pregnan",
    "postpartum",
    "breastfeed",
    "breast-feeding",
    "contraindicat",
    "intolerance",
    "hypertension",
    "diabetes",
    "liver",
    "renal",
    "kidney",
}


_DIAGNOSIS_MARKERS = (
    "disorder",
    "depression",
    "anxiety",
    "ptsd",
    "ocd",
    "psychosis",
    "schizophrenia",
    "bipolar",
    "autism",
    "attention-deficit",
    "adhd",
    "grief",
    "pain",
    "burnout",
    "somat",
    "personality",
    "substance",
    "alcohol",
 )


def _split_case_anchors(critical_entities: List[str]) -> Dict[str, List[str]]:
    problem: List[str] = []
    other_context: List[str] = []
    meds_constraints: List[str] = []

    seen_problem = set()
    seen_other = set()
    seen_meds_constraints = set()

    for raw in critical_entities or []:
        text = str(raw or "").strip()
        if not text:
            continue
        lower = text.lower()

        is_med = (
            "mg" in lower
            or any(k in lower for k in _MED_KEYWORDS)
            or lower.startswith("medication")
        )
        is_constraint = any(k in lower for k in _CONSTRAINT_KEYWORDS)
        is_problem = any(m in lower for m in _DIAGNOSIS_MARKERS)

        if is_med or is_constraint:
            if lower not in seen_meds_constraints:
                seen_meds_constraints.add(lower)
                meds_constraints.append(text)
        elif is_problem:
            if lower not in seen_problem:
                seen_problem.add(lower)
                problem.append(text)
        else:
            if lower not in seen_other:
                seen_other.add(lower)
                other_context.append(text)

    if not problem and other_context:
        problem = other_context[:1]

    return {"problem": problem, "meds_constraints": meds_constraints}


def _enrich_plan_for_alignment(plan_text: str, critical_entities: List[str]) -> str:
    text = str(plan_text or "").strip()
    if not text:
        return text

    lower = text.lower()

    if not any(m in lower for m in ("follow-up:", "monitor", "homework", "review", "tracking")):
        if text.endswith("."):
            text = text + " Follow-up: monitor symptoms, medication effects if relevant, and review/practise homework."
        else:
            text = text + ". Follow-up: monitor symptoms, medication effects if relevant, and review/practise homework."

    anchors = _split_case_anchors(critical_entities)
    problem_entities = anchors.get("problem") or []
    meds_constraints = anchors.get("meds_constraints") or []

    if "case anchors:" not in text.lower():
        problem = "; ".join(problem_entities) if problem_entities else "unspecified"
        constraints = "; ".join(meds_constraints) if meds_constraints else "none noted"
        if text.endswith("."):
            text = text + f" Case anchors: Problem: {problem}. Constraints/Meds: {constraints}."
        else:
            text = text + f". Case anchors: Problem: {problem}. Constraints/Meds: {constraints}."

    return text


def main() -> int:
    p = argparse.ArgumentParser(description="Generate Study C gold plans (NLI-verified components for linked cases)")
    p.add_argument(
        "--data-dir",
        type=str,
        default="data/openr1_psy_splits",
        help="Directory containing study_c_test.json",
    )
    p.add_argument(
        "--revision",
        type=str,
        default="main",
        help="OpenR1-Psy dataset revision (commit hash, tag, or branch).",
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional Hugging Face datasets cache directory for OpenR1-Psy.",
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
    linked_fallback_count = 0
    updated_count = 0
    
    try:
        from datasets import load_dataset
        ds_kwargs = {"revision": args.revision}
        if args.cache_dir:
            ds_kwargs["cache_dir"] = args.cache_dir
        ds_test = load_dataset("GMLHUHE/OpenR1-Psy", split="test", **ds_kwargs)
        ds_train = load_dataset("GMLHUHE/OpenR1-Psy", split="train", **ds_kwargs)
        openr1_available = True
        print("OpenR1-Psy dataset loaded successfully")
    except Exception as e:
        print(f"Could not load OpenR1-Psy: {e}")
        print("Will generate all plans from patient summaries")
        openr1_available = False
        ds_test = None
        ds_train = None

    nli_model: Optional[NLIModel] = None
    if openr1_available:
        try:
            nli_model = NLIModel()
        except Exception as e:
            print(f"Could not load NLI model: {e}")
            print("Will generate all plans from patient summaries")
            nli_model = None

    for case in cases:
        case_id = case.get("id", "")
        case_metadata = case.get("metadata", {}) or {}
        source_ids = case_metadata.get("source_openr1_ids", [])
        preferred_source_split = str(case_metadata.get("source_split", "") or "").strip().lower()
        patient_summary = case.get("patient_summary", "")
        critical_entities = case.get("critical_entities", [])
        
        if args.force and case_id:
            existing_plans.pop(case_id, None)
        
        # Skip if already have plan and not forcing
        if not args.force and case_id in existing_plans:
            existing_entry = existing_plans.get(case_id)
            if isinstance(existing_entry, dict) and existing_entry.get("plan"):
                continue
        
        plan_text = ""
        source_id = None
        source_split = None
        has_source_link = bool(source_ids) and preferred_source_split in {"test", "train"}
        
        # Try OpenR1 linkage first (NLI plan component classification)
        if (
            openr1_available
            and nli_model is not None
            and source_ids
            and preferred_source_split != "generated"
        ):
            split_order = [("test", ds_test), ("train", ds_train)]
            if preferred_source_split == "train":
                split_order = [("train", ds_train), ("test", ds_test)]
            elif preferred_source_split == "test":
                split_order = [("test", ds_test), ("train", ds_train)]

            for idx in source_ids:
                for split_name, ds in split_order:
                    if ds is None:
                        continue
                    try:
                        row = ds[int(idx)]
                        convo = row.get("conversation", [])
                        reasoning_text = _collect_full_counselor_think(convo)

                        if reasoning_text:
                            entailed, evidence = classify_plan_components(
                                premise=reasoning_text,
                                nli_model=nli_model,
                                components=DEFAULT_PLAN_COMPONENTS,
                            )
                            plan_text = render_plan_from_components(
                                entailed_by_component_id=entailed,
                                components=DEFAULT_PLAN_COMPONENTS,
                            )

                            if plan_text and not _looks_like_actionable_plan(plan_text):
                                plan_text = ""

                            if not plan_text:
                                candidates = extract_recommendation_candidates(
                                    reasoning_text=reasoning_text
                                )
                                kept = nli_filter_candidates(
                                    premise=reasoning_text,
                                    candidates=candidates,
                                    nli_model=nli_model,
                                    max_keep=3,
                                )
                                if kept:
                                    plan_text = " ".join(
                                        [s if s.endswith(".") else (s + ".") for s in kept]
                                    ).strip()

                                    if _looks_like_actionable_plan(plan_text):
                                        linked_fallback_count += 1
                                    else:
                                        plan_text = ""

                            if plan_text:
                                plan_text = _enrich_plan_for_alignment(
                                    plan_text, critical_entities
                                )
                                source_id = int(idx)
                                source_split = split_name
                                linked_count += 1
                                existing_plans[case_id] = {
                                    "plan": plan_text,
                                    "source_openr1_id": source_id,
                                    "source_split": source_split,
                                    "plan_components": [
                                        cid
                                        for cid, ok in entailed.items()
                                        if ok
                                    ],
                                    "plan_component_evidence": evidence,
                                }
                                updated_count += 1
                                break
                    except (IndexError, ValueError, KeyError, TypeError):
                        continue
                if plan_text:
                    break
        
        # Fallback to patient summary synthesis
        if case_id not in existing_plans and patient_summary:
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
                if has_source_link:
                    source_id = int(source_ids[0])
                    source_split = "linked_fallback"
                    linked_fallback_count += 1
                else:
                    source_split = "generated"
                    unlinked_count += 1

        if plan_text and case_id not in existing_plans:
            plan_text = _enrich_plan_for_alignment(plan_text, critical_entities)
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
        "notes": "Gold target plans for Study C. Linked cases use NLI-verified plan component classification over OpenR1-Psy counselor_think; unlinked cases are generated from patient_summary + critical_entities.",
        "updated_utc": datetime.utcnow().isoformat() + "Z",
        "script": "scripts/studies/study_c/gold_plans/generate_nli_plans.py",
        "nli_model": "cross-encoder/nli-deberta-v3-base",
        "revision": args.revision,
        "cache_dir": args.cache_dir,
        "source_split_counts": {
            "linked": linked_count,
            "generated": unlinked_count,
            "linked_fallback": linked_fallback_count,
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
