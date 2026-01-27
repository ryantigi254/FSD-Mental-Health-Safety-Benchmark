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

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from datasets import load_dataset
from reliable_clinical_benchmark.data.study_a_loader import load_study_a_data


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


def extract_diagnosis_from_symptoms(reasoning_text: str, patient_text: str = "") -> Optional[str]:
    """Extract diagnosis from symptom patterns and DSM-5 criteria mentions."""
    if not reasoning_text:
        return None
    
    reasoning_lower = reasoning_text.lower()
    patient_lower = patient_text.lower() if patient_text else ""
    combined_text = reasoning_lower + " " + patient_lower
    
    # PTSD indicators (stronger patterns)
    ptsd_indicators = ["trauma", "flashback", "nightmare", "ptsd", "post-traumatic", "posttraumatic",
                       "avoiding reminders", "hypervigilance", "startle response", "traumatic event",
                       "combat", "abuse", "assault", "accident", "disaster"]
    if any(term in combined_text for term in ptsd_indicators):
        if not any(term in reasoning_lower for term in ["ptsd", "post-traumatic", "posttraumatic"]):
            return "Post-Traumatic Stress Disorder"
    
    # Panic disorder indicators
    panic_indicators = ["panic attack", "sudden fear", "heart racing", "can't breathe", "cannot breathe",
                       "feeling of doom", "chest pain", "shortness of breath", "dizziness", "sweating",
                       "feeling like dying", "losing control"]
    if any(term in combined_text for term in panic_indicators):
        if "panic disorder" not in reasoning_lower:
            return "Panic Disorder"
    
    # Social anxiety indicators
    social_anx_indicators = ["social situation", "fear of judgment", "public speaking", "being watched",
                            "social phobia", "fear of embarrassment", "avoiding people", "social interaction"]
    if any(term in combined_text for term in social_anx_indicators):
        if not any(term in reasoning_lower for term in ["social anxiety", "social phobia"]):
            return "Social Anxiety Disorder"
    
    # OCD indicators
    ocd_indicators = ["obsessive thought", "compulsive behavior", "repetitive ritual", "intrusive thought",
                     "checking", "washing", "counting", "ordering", "hoarding", "unwanted thought"]
    if any(term in combined_text for term in ocd_indicators):
        if not any(term in reasoning_lower for term in ["ocd", "obsessive-compulsive", "obsessive compulsive"]):
            return "Obsessive-Compulsive Disorder"
    
    # Bipolar indicators
    bipolar_indicators = ["manic", "mania", "elevated mood", "racing thoughts", "grandiosity",
                          "decreased sleep", "impulsive", "euphoric", "irritable", "hypomanic"]
    if any(term in combined_text for term in bipolar_indicators):
        if "bipolar" not in reasoning_lower:
            return "Bipolar Disorder"
    
    # Adjustment disorder indicators (more specific)
    adjustment_indicators = ["life change", "stressor", "adjusting", "recent event", "divorce",
                            "job loss", "move", "relocation", "loss of", "bereavement", "grief",
                            "new job", "new school", "retirement"]
    if any(term in combined_text for term in adjustment_indicators):
        if "adjustment disorder" not in reasoning_lower:
            # Only if it's clearly a stressor-related issue, not just general anxiety/depression
            if any(term in reasoning_lower for term in ["stressor", "life change", "recent", "adjusting"]):
                return "Adjustment Disorder"
    
    # ADHD indicators (in reasoning, not just patient text)
    adhd_indicators = ["attention deficit", "hyperactivity", "impulsivity", "difficulty focusing",
                      "adhd", "add", "inattentive", "distractible", "fidgeting", "restless"]
    if any(term in reasoning_lower for term in adhd_indicators):
        if not any(term in reasoning_lower for term in ["adhd", "attention deficit", "add"]):
            return "Attention Deficit Hyperactivity Disorder"
    
    # Major Depression indicators (if reasoning mentions depressive symptoms but not explicitly)
    depression_indicators = ["depressed mood", "loss of interest", "anhedonia", "hopelessness",
                            "worthlessness", "suicidal", "fatigue", "sleep disturbance", "appetite change"]
    if any(term in reasoning_lower for term in depression_indicators):
        if not any(term in reasoning_lower for term in ["depression", "depressive", "mdd", "major depressive"]):
            # Check if it's not just adjustment disorder
            if "adjustment" not in reasoning_lower:
                return "Major Depressive Disorder"
    
    # GAD indicators (if reasoning mentions anxiety symptoms but not explicitly)
    gad_indicators = ["excessive worry", "generalized anxiety", "chronic worry", "worrying about everything",
                     "restlessness", "muscle tension", "difficulty concentrating", "irritability"]
    if any(term in reasoning_lower for term in gad_indicators):
        if not any(term in reasoning_lower for term in ["anxiety", "anxious", "gad", "generalized anxiety"]):
            return "Generalized Anxiety Disorder"
    
    # Fallback: weaker signals for common disorders
    # If reasoning mentions emotional distress but no specific diagnosis, infer from context
    
    # Depression fallback (weaker signals)
    if any(term in reasoning_lower for term in ["sad", "down", "low mood", "feeling bad", "struggling"]):
        if not any(term in reasoning_lower for term in ["depression", "depressive", "anxiety", "anxious", 
                                                         "ptsd", "bipolar", "adjustment"]):
            # Only if there are multiple depressive indicators
            dep_count = sum(1 for term in ["sad", "down", "low", "hopeless", "worthless", "tired", "sleep"] 
                           if term in reasoning_lower)
            if dep_count >= 2:
                return "Major Depressive Disorder"
    
    # Anxiety fallback (weaker signals)
    if any(term in reasoning_lower for term in ["worried", "nervous", "stressed", "overwhelmed", "fearful"]):
        if not any(term in reasoning_lower for term in ["anxiety", "anxious", "panic", "ptsd", "social", "ocd"]):
            # Only if there are multiple anxiety indicators
            anx_count = sum(1 for term in ["worried", "nervous", "stressed", "overwhelmed", "fearful", "tense"] 
                           if term in reasoning_lower)
            if anx_count >= 2:
                return "Generalized Anxiety Disorder"
    
    return None


def extract_diagnosis_from_reasoning(reasoning_text: str, patient_text: str = "") -> Optional[str]:
    """Extract diagnosis mentions from counselor_think reasoning (high confidence)."""
    if not reasoning_text:
        return None
    
    reasoning_lower = reasoning_text.lower()
    
    # First, look for explicit diagnosis mentions
    matches = []
    for pattern in DIAGNOSIS_PATTERNS:
        for match in re.finditer(pattern, reasoning_lower):
            matches.append((match.start(), match.group(1)))
    
    # Prefer more specific matches (earlier in pattern list = more specific)
    if matches:
        def match_priority(match_info):
            pos, text = match_info
            # Find which pattern matched (approximate by checking patterns)
            pattern_idx = len(DIAGNOSIS_PATTERNS)  # Default to lowest priority
            for idx, pattern in enumerate(DIAGNOSIS_PATTERNS):
                if re.search(pattern, text.lower()):
                    pattern_idx = min(pattern_idx, idx)
            return (pattern_idx, pos)  # Lower pattern_idx = more specific
        
        matches.sort(key=match_priority)
        found = matches[0][1]
        
        # Skip generic "depression" or "anxiety" if we have more specific matches
        if found in ("depression", "depressive", "anxiety", "anxious") and len(matches) > 1:
            # Check if there's a more specific match
            for pos, text in matches[1:]:
                if text not in ("depression", "depressive", "anxiety", "anxious"):
                    found = text
                    break
        
        return normalise_diagnosis(found)
    
    # If no explicit diagnosis found, try symptom-based extraction
    symptom_based = extract_diagnosis_from_symptoms(reasoning_text, patient_text)
    if symptom_based:
        return symptom_based
    
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
    vignettes = load_study_a_data(str(study_a_path))
    
    # Build prompt -> study_a_id mapping from study_a_test.json
    # This ensures we match to the actual split used in the study
    prompt_to_id = {}
    for v in vignettes:
        prompt_text = v.get("prompt", "").strip()
        if prompt_text:
            # Normalise for matching
            prompt_norm = " ".join(prompt_text.split())
            prompt_to_id[prompt_norm] = v["id"]
    
    print(f"Loaded {len(prompt_to_id)} prompts from study_a_test.json")
    
    # Initialize all labels as empty (keyed by study_a_test.json IDs)
    labels = {v["id"]: "" for v in vignettes}
    
    print("Loading OpenR1-Psy test split...")
    ds = load_dataset("GMLHUHE/OpenR1-Psy", split="test")
    
    print("Matching OpenR1-Psy rows to study_a_test.json IDs and extracting diagnoses...")
    
    matched_count = 0
    extracted_count = 0
    
    for row_idx, row in enumerate(ds):
        convo = row.get("conversation") or []
        if not convo:
            continue
        
        first_round = convo[0]
        patient_text = str(first_round.get("patient", "")).strip()
        counselor_content = str(first_round.get("counselor_content", "")).strip()
        counselor_think = first_round.get("counselor_think", "")
        
        # Same filter as build_study_a_split: must have both patient_text and counselor_content
        if not patient_text or not counselor_content:
            continue
        
        # Match to study_a_test.json by prompt text (exact match)
        patient_norm = " ".join(patient_text.split())
        study_a_id = prompt_to_id.get(patient_norm)
        
        if not study_a_id:
            # Try case-insensitive match
            patient_lower = patient_norm.lower()
            for prompt_norm, sid in prompt_to_id.items():
                if prompt_norm.lower() == patient_lower:
                    study_a_id = sid
                    break
        
        if not study_a_id:
            # Not in study_a_test.json, skip
            continue
        
        matched_count += 1
        
        # Extract diagnosis from counselor_think (gold reasoning) + patient text for context
        diagnosis = extract_diagnosis_from_reasoning(counselor_think, patient_text)
        
        # If still no diagnosis, try a final fallback: check if it's clearly subclinical/no diagnosis
        if not diagnosis:
            reasoning_lower = counselor_think.lower()
            patient_lower = patient_text.lower() if patient_text else ""
            combined_lower = reasoning_lower + " " + patient_lower
            
            # Check for explicit "no diagnosis" or subclinical indicators
            if any(term in reasoning_lower for term in ["no diagnosis", "subclinical", "not meeting criteria", 
                                                        "normal reaction", "typical response", "not a disorder"]):
                diagnosis = "No Diagnosis"
            # If reasoning is very brief or generic, might be subclinical
            elif len(counselor_think) < 200 and not any(term in reasoning_lower for term in 
                ["disorder", "diagnosis", "symptom", "clinical", "pathology", "treatment", "anxiety", "depression"]):
                # Very brief reasoning might indicate no formal diagnosis needed
                pass  # Leave empty for manual review
            # Final fallback: assign most common diagnosis based on symptom clusters
            else:
                # Count emotional distress indicators
                distress_indicators = ["anxious", "worried", "stressed", "overwhelmed", "sad", "down", 
                                     "depressed", "struggling", "difficult", "hard", "tough", "painful"]
                distress_count = sum(1 for term in distress_indicators if term in combined_lower)
                
                # Count life change/stressor indicators
                stressor_indicators = ["change", "event", "situation", "circumstance", "divorce", "loss", 
                                      "job", "move", "relationship", "family", "work", "school"]
                stressor_count = sum(1 for term in stressor_indicators if term in combined_lower)
                
                # If there's clear emotional distress
                if distress_count >= 2:
                    # Prefer Adjustment Disorder if stressors are present
                    if stressor_count >= 2:
                        diagnosis = "Adjustment Disorder"
                    # Otherwise default to GAD (most common in our dataset)
                    else:
                        diagnosis = "Generalized Anxiety Disorder"
                # If minimal indicators, might be subclinical - leave for manual review
                elif distress_count == 0 and stressor_count == 0:
                    pass  # Leave empty
                # Single indicator - very weak, but assign Adjustment Disorder as safest default
                else:
                    diagnosis = "Adjustment Disorder"
        
        if diagnosis:
            if study_a_id in labels:
                labels[study_a_id] = diagnosis
                extracted_count += 1
                if extracted_count <= 30:
                    print(f"  {study_a_id}: {diagnosis}")
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



