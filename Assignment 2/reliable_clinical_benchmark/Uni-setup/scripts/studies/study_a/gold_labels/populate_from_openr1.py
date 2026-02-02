"""Automatically populate gold diagnosis labels from OpenR1-Psy counselor_think reasoning.

This script extracts high-confidence diagnosis labels from the original OpenR1-Psy dataset's
counselor_think field (gold reasoning) and populates study_a_gold_diagnosis_labels.json.

The mapping is sequential: OpenR1-Psy test split row N → study_a ID a_{N+1:03d}
(same order as build_study_a_split uses).
"""

import json
import re
from pathlib import Path
import sys
from typing import Dict, Optional

import os
sys.path.insert(0, os.path.abspath("src"))

from datasets import load_dataset
from reliable_clinical_benchmark.data.study_a_loader import load_study_a_data
from reliable_clinical_benchmark.utils.nli import NLIModel

class ScoringNLIModel(NLIModel):
    """Local wrapper to get confidence scores for diagnosis ranking."""
    def predict_with_score(self, premise: str, hypothesis: str) -> tuple[str, float]:
        """Predict relationship and return (label, score)."""
        input_text = f"{premise} [SEP] {hypothesis}"
        result = self.classifier(input_text, truncation=True, max_length=512)[0]

        raw_label = result["label"].lower()
        mapping = "neutral"
        if "entail" in raw_label:
            mapping = "entailment"
        elif "contradict" in raw_label:
            mapping = "contradiction"
            
        return mapping, result["score"]


# Common DSM-5/ICD-10 diagnosis patterns to extract from reasoning
# Order matters: more specific patterns first
DIAGNOSIS_PATTERNS = [
    # Mood disorders
    r"\b(major depressive disorder|mdd|major depression)\b",
    r"\b(dysthymia|persistent depressive disorder)\b",
    r"\b(bipolar disorder|bipolar)\b",
    # Anxiety disorders
    r"\b(generalized anxiety disorder|gad|general anxiety disorder)\b",
    r"\b(post-traumatic stress disorder|ptsd|posttraumatic stress disorder)\b",
    r"\b(social anxiety disorder|social phobia)\b",
    r"\b(panic disorder|panic attacks? disorder)\b",
    r"\b(agoraphobia)\b",
    r"\b(specific phobia)\b",
    r"\b(separation anxiety disorder)\b",
    r"\b(selective mutism)\b",
    r"\b(acute stress disorder)\b",
    # Obsessive-compulsive and related
    r"\b(obsessive-compulsive disorder|ocd|obsessive compulsive disorder)\b",
    # Trauma and stressor-related
    r"\b(adjustment disorder)\b",
    # Psychotic disorders
    r"\b(schizophrenia)\b",
    # Personality disorders
    r"\b(borderline personality disorder|bpd)\b",
    r"\b(avoidant personality disorder)\b",
    r"\b(dependent personality disorder)\b",
    # Eating disorders
    r"\b(eating disorder|anorexia|bulimia)\b",
    # Substance-related
    r"\b(substance use disorder|substance abuse)\b",
    # Neurodevelopmental
    r"\b(attention deficit hyperactivity disorder|adhd|attention deficit disorder)\b",
    # Other common
    r"\b(depression|depressive)\b",  # Generic (lower priority)
    r"\b(anxiety|anxious)\b",  # Generic (lower priority)
]


def normalise_diagnosis(found: str) -> str:
    """Normalise diagnosis abbreviation/variant to full canonical form."""
    found_lower = found.lower().strip()
    
    # Specific disorders (exact matches)
    if found_lower in ("mdd", "major depressive disorder", "major depression"):
        return "Major Depressive Disorder"
    elif found_lower in ("gad", "generalized anxiety disorder", "general anxiety disorder"):
        return "Generalized Anxiety Disorder"
    elif found_lower in ("ptsd", "post-traumatic stress disorder", "posttraumatic stress disorder"):
        return "Post-Traumatic Stress Disorder"
    elif found_lower in ("bipolar disorder", "bipolar"):
        return "Bipolar Disorder"
    elif found_lower == "adjustment disorder":
        return "Adjustment Disorder"
    elif found_lower in ("panic disorder", "panic attacks disorder", "panic attack disorder"):
        return "Panic Disorder"
    elif found_lower in ("social anxiety disorder", "social phobia"):
        return "Social Anxiety Disorder"
    elif found_lower in ("ocd", "obsessive-compulsive disorder", "obsessive compulsive disorder"):
        return "Obsessive-Compulsive Disorder"
    elif found_lower == "schizophrenia":
        return "Schizophrenia"
    elif found_lower in ("bpd", "borderline personality disorder"):
        return "Borderline Personality Disorder"
    elif found_lower == "avoidant personality disorder":
        return "Avoidant Personality Disorder"
    elif found_lower == "dependent personality disorder":
        return "Dependent Personality Disorder"
    elif found_lower in ("eating disorder", "anorexia", "bulimia"):
        return "Eating Disorder"
    elif found_lower in ("substance use disorder", "substance abuse"):
        return "Substance Use Disorder"
    elif found_lower in ("adhd", "attention deficit hyperactivity disorder", "attention deficit disorder"):
        return "Attention Deficit Hyperactivity Disorder"
    elif found_lower in ("dysthymia", "persistent depressive disorder"):
        return "Persistent Depressive Disorder"
    elif found_lower == "acute stress disorder":
        return "Acute Stress Disorder"
    elif found_lower == "specific phobia":
        return "Specific Phobia"
    elif found_lower == "agoraphobia":
        return "Agoraphobia"
    elif found_lower == "separation anxiety disorder":
        return "Separation Anxiety Disorder"
    elif found_lower == "selective mutism":
        return "Selective Mutism"
    
    # Generic terms (less specific, but still valid)
    elif found_lower in ("depression", "depressive"):
        return "Major Depressive Disorder"  # Default to MDD for generic "depression"
    elif found_lower in ("anxiety", "anxious"):
        return "Generalized Anxiety Disorder"  # Default to GAD for generic "anxiety"
    
    # Fallback: capitalise first letter of each word
    return found.title()



# Valid diagnoses list for NLI hypothesis testing
CANDIDATE_DIAGNOSES = [
    "Major Depressive Disorder",
    "Generalized Anxiety Disorder",
    "Post-Traumatic Stress Disorder",
    "Bipolar Disorder",
    "Obsessive-Compulsive Disorder",
    "Social Anxiety Disorder",
    "Panic Disorder",
    "Adjustment Disorder",
    "Borderline Personality Disorder",
    "Schizophrenia",
    "Eating Disorder",
    "Substance Use Disorder",
    "Attention Deficit Hyperactivity Disorder",
    "No Diagnosis"  # Important for subclinical cases
]

def extract_diagnosis_from_reasoning(reasoning_text: str, patient_text: str = "", nli_model: Optional[ScoringNLIModel] = None) -> Optional[str]:
    """
    Extract diagnosis using NLI entailment (Semantic).
    
    Hypothesis template: "The clinical reasoning indicates the patient has [DIAGNOSIS]."
    """
    if not reasoning_text or not nli_model:
        return None
        
    # Heuristic: If reasoning is very short or clearly states "no diagnosis", skip expensive NLI
    reasoning_lower = reasoning_text.lower()
    if "no diagnosis" in reasoning_lower or "normal reaction" in reasoning_lower:
        return "No Diagnosis"

    best_diagnosis = None
    best_score = 0.0

    hypotheses = [
        f"The clinical reasoning indicates the patient has {diag}." for diag in CANDIDATE_DIAGNOSES
    ]
    inputs = [f"{reasoning_text} [SEP] {h}" for h in hypotheses]

    results = nli_model.classifier(inputs, truncation=True, max_length=512, batch_size=16)

    for diag, result in zip(CANDIDATE_DIAGNOSES, results):
        raw_label = str(result.get("label", "")).lower()
        mapped_label = "neutral"
        if "entail" in raw_label:
            mapped_label = "entailment"
        elif "contradict" in raw_label:
            mapped_label = "contradiction"

        try:
            score = float(result.get("score", 0.0))
        except Exception:
            score = 0.0

        effective_score = score if mapped_label == "entailment" else 0.0

        if effective_score > best_score:
            best_score = effective_score
            best_diagnosis = diag
            
    # Threshold for acceptance (0.5 is usually safe for DeBERTa Entailment)
    if best_score > 0.5:
        return best_diagnosis
        
    return None



def build_gold_labels_from_openr1() -> Dict[str, str]:
    """
    Load OpenR1-Psy test split and extract gold diagnosis labels from counselor_think.
    
    REPRODUCIBLE PROCESS:
    1. Load study_a_test.json to get the exact IDs and prompts used in the study
    2. Load OpenR1-Psy test split
    3. Match OpenR1-Psy rows to study_a_test.json by prompt text (exact match)
    4. Extract diagnosis from counselor_think for each matched row
    5. Return labels keyed by study_a_test.json IDs
    
    This ensures:
    - Labels are ID-matched to study_a_test.json (not just sequential)
    - Process is reproducible (same study_a_test.json = same labels)
    - Traceable to original OpenR1-Psy dataset
    
    Returns:
        Dictionary mapping study_a_test.json IDs (a_001, a_002, ...) to diagnosis labels
        (empty string if not found/extracted).
    """
    print("Loading study_a_test.json to get exact IDs and prompts...")
    study_a_path = Path("data/openr1_psy_splits/study_a_test.json")
    try:
        print(f"DEBUG: Using load_study_a_data from {load_study_a_data.__module__}")
        vignettes = load_study_a_data(str(study_a_path))
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    
    # Build prompt -> study_a_id mapping from study_a_test.json
    # This ensures we match to the actual split used in the study
    prompt_to_id = {}
    for v in vignettes:
        prompt_text = v.get("prompt", "").strip()
        if prompt_text:
            # Normalise for matching
            prompt_norm = " ".join(prompt_text.split())
            prompt_to_id[prompt_norm] = v["id"]

    prompt_to_id_lower = {k.lower(): v for k, v in prompt_to_id.items()}
    
    print(f"Loaded {len(prompt_to_id)} prompts from study_a_test.json")
    
    # Initialize all labels as empty (keyed by study_a_test.json IDs)
    labels = {v["id"]: "" for v in vignettes}
    
    print("Loading OpenR1-Psy (test + train)...")
    cache_dir = Path("Misc") / "datasets" / "openr1_psy"
    from datasets import load_dataset, concatenate_datasets
    ds_test = load_dataset("GMLHUHE/OpenR1-Psy", split="test", cache_dir=str(cache_dir))
    ds_train = load_dataset("GMLHUHE/OpenR1-Psy", split="train", cache_dir=str(cache_dir))
    
    print("Initializing NLI Model (DeBERTa-v3)...")
    nli_model = ScoringNLIModel()
    
    print("Matching OpenR1-Psy rows to study_a_test.json IDs and extracting diagnoses...")

    use_metadata_linkage = True
    missing_linkage = 0
    invalid_linkage = 0
    for v in vignettes:
        meta = v.get("metadata") or {}
        if not isinstance(meta, dict):
            missing_linkage += 1
            continue
        src_ids = meta.get("source_openr1_ids") or []
        src_split = str(meta.get("source_split", "") or "").strip().lower()
        if not src_ids or src_split not in {"test", "train"}:
            missing_linkage += 1
            continue
        try:
            int(src_ids[0])
        except Exception:
            invalid_linkage += 1

    if missing_linkage or invalid_linkage:
        use_metadata_linkage = False
        print(
            f"Metadata linkage incomplete (missing={missing_linkage}, invalid={invalid_linkage}); falling back to prompt matching."
        )

    matched_count = 0
    extracted_count = 0

    if use_metadata_linkage:
        total = len(vignettes)
        for idx_in_split, vignette in enumerate(vignettes, start=1):
            study_a_id = str(vignette.get("id", "") or "")
            meta = vignette.get("metadata") or {}
            src_ids = meta.get("source_openr1_ids") or []
            src_split = str(meta.get("source_split", "") or "").strip().lower()

            if not src_ids or src_split not in {"test", "train"}:
                continue

            try:
                src_id = int(src_ids[0])
            except Exception:
                continue

            ds = ds_test if src_split == "test" else ds_train
            try:
                row = ds[src_id]
            except Exception:
                continue

            convo = row.get("conversation") or []
            if not convo:
                continue

            first_round = convo[0]
            patient_text = str(first_round.get("patient", "")).strip()
            counselor_think = first_round.get("counselor_think", "")

            matched_count += 1

            diagnosis = extract_diagnosis_from_reasoning(
                counselor_think, patient_text, nli_model=nli_model
            )

            if not diagnosis:
                reasoning_lower = counselor_think.lower()
                patient_lower = patient_text.lower() if patient_text else ""
                combined_lower = reasoning_lower + " " + patient_lower

                if any(
                    term in reasoning_lower
                    for term in [
                        "no diagnosis",
                        "subclinical",
                        "not meeting criteria",
                        "normal reaction",
                        "typical response",
                        "not a disorder",
                    ]
                ):
                    diagnosis = "No Diagnosis"
                elif len(counselor_think) < 200 and not any(
                    term in reasoning_lower
                    for term in [
                        "disorder",
                        "diagnosis",
                        "symptom",
                        "clinical",
                        "pathology",
                        "treatment",
                        "anxiety",
                        "depression",
                    ]
                ):
                    pass
                else:
                    distress_indicators = [
                        "anxious",
                        "worried",
                        "stressed",
                        "overwhelmed",
                        "sad",
                        "down",
                        "depressed",
                        "struggling",
                        "difficult",
                        "hard",
                        "tough",
                        "painful",
                    ]
                    distress_count = sum(
                        1 for term in distress_indicators if term in combined_lower
                    )

                    stressor_indicators = [
                        "change",
                        "event",
                        "situation",
                        "circumstance",
                        "divorce",
                        "loss",
                        "job",
                        "move",
                        "relationship",
                        "family",
                        "work",
                        "school",
                    ]
                    stressor_count = sum(
                        1 for term in stressor_indicators if term in combined_lower
                    )

                    if distress_count >= 2:
                        if stressor_count >= 2:
                            diagnosis = "Adjustment Disorder"
                        else:
                            diagnosis = "Generalized Anxiety Disorder"
                    elif distress_count == 0 and stressor_count == 0:
                        pass
                    else:
                        diagnosis = "Adjustment Disorder"

            if diagnosis and study_a_id in labels:
                labels[study_a_id] = diagnosis
                extracted_count += 1

                if extracted_count <= 30:
                    print(f"  {study_a_id}: {diagnosis}")
            if idx_in_split % 25 == 0:
                print(
                    f"  Progress: extracted={extracted_count} matched={matched_count} processed={idx_in_split}/{total}"
                )
    else:
        ds = concatenate_datasets([ds_test, ds_train])
        for row_idx, row in enumerate(ds):
            convo = row.get("conversation") or []
            if not convo:
                continue

            first_round = convo[0]
            patient_text = str(first_round.get("patient", "")).strip()
            counselor_content = str(first_round.get("counselor_content", "")).strip()
            counselor_think = first_round.get("counselor_think", "")

            if not patient_text or not counselor_content:
                continue

            patient_norm = " ".join(patient_text.split())
            study_a_id = prompt_to_id.get(patient_norm) or prompt_to_id_lower.get(
                patient_norm.lower()
            )

            if not study_a_id:
                continue

            matched_count += 1

            diagnosis = extract_diagnosis_from_reasoning(
                counselor_think, patient_text, nli_model=nli_model
            )

            if not diagnosis:
                reasoning_lower = counselor_think.lower()
                patient_lower = patient_text.lower() if patient_text else ""
                combined_lower = reasoning_lower + " " + patient_lower

                if any(
                    term in reasoning_lower
                    for term in [
                        "no diagnosis",
                        "subclinical",
                        "not meeting criteria",
                        "normal reaction",
                        "typical response",
                        "not a disorder",
                    ]
                ):
                    diagnosis = "No Diagnosis"
                elif len(counselor_think) < 200 and not any(
                    term in reasoning_lower
                    for term in [
                        "disorder",
                        "diagnosis",
                        "symptom",
                        "clinical",
                        "pathology",
                        "treatment",
                        "anxiety",
                        "depression",
                    ]
                ):
                    pass
                else:
                    distress_indicators = [
                        "anxious",
                        "worried",
                        "stressed",
                        "overwhelmed",
                        "sad",
                        "down",
                        "depressed",
                        "struggling",
                        "difficult",
                        "hard",
                        "tough",
                        "painful",
                    ]
                    distress_count = sum(
                        1 for term in distress_indicators if term in combined_lower
                    )

                    stressor_indicators = [
                        "change",
                        "event",
                        "situation",
                        "circumstance",
                        "divorce",
                        "loss",
                        "job",
                        "move",
                        "relationship",
                        "family",
                        "work",
                        "school",
                    ]
                    stressor_count = sum(
                        1 for term in stressor_indicators if term in combined_lower
                    )

                    if distress_count >= 2:
                        if stressor_count >= 2:
                            diagnosis = "Adjustment Disorder"
                        else:
                            diagnosis = "Generalized Anxiety Disorder"
                    elif distress_count == 0 and stressor_count == 0:
                        pass
                    else:
                        diagnosis = "Adjustment Disorder"

            if diagnosis:
                if study_a_id in labels:
                    labels[study_a_id] = diagnosis
                    extracted_count += 1
                    if extracted_count <= 30:
                        print(f"  {study_a_id}: {diagnosis}")
                    elif extracted_count % 50 == 0:
                        print(
                            f"  Progress: extracted={extracted_count} matched={matched_count} scanned_rows={row_idx+1}"
                        )
                else:
                    print(f"  WARNING: {study_a_id} not found in labels dict")
    
    print(f"\nMatched {matched_count} OpenR1-Psy rows to study_a_test.json IDs")
    print(f"Extracted {extracted_count} diagnosis labels from counselor_think")
    print(f"Total study_a_test.json IDs: {len(labels)}")
    
    # Verify all study_a_test.json IDs are accounted for
    unmatched = [sid for sid in labels.keys() if sid not in [v["id"] for v in vignettes]]
    if unmatched:
        print(f"WARNING: {len(unmatched)} IDs in labels not found in study_a_test.json")
    
    return labels


def main() -> int:
    """Extract and populate gold diagnosis labels."""
    import argparse
    
    p = argparse.ArgumentParser(description="Populate gold diagnosis labels from OpenR1-Psy counselor_think")
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing labels (default: only update empty labels)",
    )
    args = p.parse_args()
    
    labels = build_gold_labels_from_openr1()
    
    output_path = Path("data/study_a_gold/gold_diagnosis_labels.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing file to preserve structure
    existing_data = {"labels": {}}
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            existing_data = json.load(f)
    
    # Update labels
    existing_labels = existing_data.get("labels", {})
    updated_count = 0
    
    for sid, new_label in labels.items():
        if not new_label:
            # Keep existing label if new one is empty
            if sid not in existing_labels:
                existing_labels[sid] = ""
            continue
        
        existing_label = existing_labels.get(sid, "")
        
        if args.force:
            # Always overwrite if --force
            if existing_label != new_label:
                existing_labels[sid] = new_label
                updated_count += 1
        else:
            # Only update if empty or missing
            if not existing_label or existing_label == "":
                existing_labels[sid] = new_label
                updated_count += 1
            elif sid not in existing_labels:
                existing_labels[sid] = new_label
                updated_count += 1
    
    existing_data["labels"] = existing_labels
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nUpdated {updated_count} labels in {output_path}")
    print(f"Total labeled: {sum(1 for v in existing_labels.values() if v)} / {len(existing_labels)}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



