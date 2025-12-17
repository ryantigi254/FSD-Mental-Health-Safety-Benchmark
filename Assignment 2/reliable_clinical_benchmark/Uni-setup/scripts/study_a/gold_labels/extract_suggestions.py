"""Extract potential diagnosis labels from original OpenR1-Psy dataset to help with manual labeling."""

import json
import re
from pathlib import Path
import sys
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from datasets import load_dataset
from reliable_clinical_benchmark.data.study_a_loader import load_study_a_data


# Common DSM-5/ICD-10 diagnosis patterns to extract from reasoning
DIAGNOSIS_PATTERNS = [
    r"\b(major depressive disorder|mdd)\b",
    r"\b(generalized anxiety disorder|gad)\b",
    r"\b(post-traumatic stress disorder|ptsd)\b",
    r"\b(bipolar disorder)\b",
    r"\b(adjustment disorder)\b",
    r"\b(panic disorder)\b",
    r"\b(social anxiety disorder)\b",
    r"\b(obsessive-compulsive disorder|ocd)\b",
    r"\b(schizophrenia)\b",
    r"\b(borderline personality disorder|bpd)\b",
    r"\b(avoidant personality disorder)\b",
    r"\b(dependent personality disorder)\b",
    r"\b(eating disorder)\b",
    r"\b(substance use disorder)\b",
    r"\b(attention deficit hyperactivity disorder|adhd)\b",
]


def extract_diagnosis_from_reasoning(reasoning_text: str) -> Optional[str]:
    """Extract diagnosis mentions from counselor_think reasoning."""
    if not reasoning_text:
        return None
    
    reasoning_lower = reasoning_text.lower()
    
    # Look for explicit diagnosis mentions
    for pattern in DIAGNOSIS_PATTERNS:
        match = re.search(pattern, reasoning_lower)
        if match:
            # Normalise to full form
            found = match.group(1).lower()
            if found in ("mdd", "major depressive disorder"):
                return "Major Depressive Disorder"
            elif found in ("gad", "generalized anxiety disorder"):
                return "Generalized Anxiety Disorder"
            elif found in ("ptsd", "post-traumatic stress disorder"):
                return "Post-Traumatic Stress Disorder"
            elif found == "bipolar disorder":
                return "Bipolar Disorder"
            elif found == "adjustment disorder":
                return "Adjustment Disorder"
            elif found == "panic disorder":
                return "Panic Disorder"
            elif found == "social anxiety disorder":
                return "Social Anxiety Disorder"
            elif found in ("ocd", "obsessive-compulsive disorder"):
                return "Obsessive-Compulsive Disorder"
            elif found == "schizophrenia":
                return "Schizophrenia"
            elif found in ("bpd", "borderline personality disorder"):
                return "Borderline Personality Disorder"
            elif found == "avoidant personality disorder":
                return "Avoidant Personality Disorder"
            elif found == "dependent personality disorder":
                return "Dependent Personality Disorder"
            elif found == "eating disorder":
                return "Eating Disorder"
            elif found == "substance use disorder":
                return "Substance Use Disorder"
            elif found in ("adhd", "attention deficit hyperactivity disorder"):
                return "Attention Deficit Hyperactivity Disorder"
    
    return None


def extract_diagnosis_from_prompt(prompt_text: str) -> Optional[str]:
    """Try to infer diagnosis from patient prompt (less reliable)."""
    prompt_lower = prompt_text.lower()
    
    # Look for symptom patterns that suggest diagnoses
    if any(word in prompt_lower for word in ["depressed", "sadness", "hopeless", "worthless"]):
        if "manic" in prompt_lower or "mania" in prompt_lower:
            return "Bipolar Disorder"
        return "Major Depressive Disorder"
    
    if any(word in prompt_lower for word in ["anxious", "worry", "panic", "fear"]):
        if "social" in prompt_lower or "people" in prompt_lower:
            return "Social Anxiety Disorder"
        if "panic" in prompt_lower or "attack" in prompt_lower:
            return "Panic Disorder"
        return "Generalized Anxiety Disorder"
    
    if any(word in prompt_lower for word in ["trauma", "ptsd", "flashback", "nightmare"]):
        return "Post-Traumatic Stress Disorder"
    
    if any(word in prompt_lower for word in ["adjust", "stressor", "life change"]):
        return "Adjustment Disorder"
    
    return None


def build_diagnosis_suggestions() -> Dict[str, Dict[str, any]]:
    """
    Load original OpenR1-Psy dataset and extract diagnosis hints.
    
    Returns:
        Dictionary mapping study_a_test.json IDs to diagnosis suggestions
    """
    print("Loading OpenR1-Psy test split...")
    ds = load_dataset("GMLHUHE/OpenR1-Psy", split="test")
    
    print("Loading study_a_test.json to map IDs...")
    study_a_path = Path("data/openr1_psy_splits/study_a_test.json")
    vignettes = load_study_a_data(str(study_a_path))
    
    # Create mapping: prompt text -> study_a ID
    prompt_to_id = {}
    for v in vignettes:
        prompt_text = v.get("prompt", "").strip().lower()
        if prompt_text:
            prompt_to_id[prompt_text] = v["id"]
    
    suggestions = {}
    
    print("Extracting diagnosis hints from OpenR1-Psy...")
    for idx, row in enumerate(ds):
        convo = row.get("conversation") or []
        if not convo:
            continue
        
        first_round = convo[0]
        patient_text = str(first_round.get("patient", "")).strip()
        counselor_think = first_round.get("counselor_think", "")
        
        if not patient_text:
            continue
        
        # Try to match to study_a_test.json
        patient_lower = patient_text.lower()
        matched_id = None
        
        for prompt, sid in prompt_to_id.items():
            # Fuzzy match: check if patient text is similar
            if patient_lower in prompt or prompt in patient_lower:
                matched_id = sid
                break
        
        if not matched_id:
            continue
        
        # Extract diagnosis from reasoning
        diag_from_reasoning = extract_diagnosis_from_reasoning(counselor_think)
        
        # Fallback: try prompt
        if not diag_from_reasoning:
            diag_from_reasoning = extract_diagnosis_from_prompt(patient_text)
        
        if diag_from_reasoning:
            suggestions[matched_id] = {
                "diagnosis": diag_from_reasoning,
                "source": "reasoning" if extract_diagnosis_from_reasoning(counselor_think) else "prompt",
                "confidence": "high" if extract_diagnosis_from_reasoning(counselor_think) else "low",
            }
    
    return suggestions


def main():
    """Extract and save diagnosis suggestions."""
    suggestions = build_diagnosis_suggestions()
    
    output_path = Path("data/study_a_gold/diagnosis_suggestions.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(suggestions, f, indent=2, ensure_ascii=False)
    
    print(f"\nExtracted {len(suggestions)} diagnosis suggestions")
    print(f"Saved to: {output_path}")
    print("\nHigh-confidence suggestions (from reasoning):")
    high_conf = {k: v for k, v in suggestions.items() if v["confidence"] == "high"}
    for sid, data in list(high_conf.items())[:10]:
        print(f"  {sid}: {data['diagnosis']}")
    
    if len(high_conf) < len(suggestions):
        print(f"\nLow-confidence suggestions (from prompt): {len(suggestions) - len(high_conf)}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

