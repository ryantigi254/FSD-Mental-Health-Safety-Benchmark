import json
import math
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Set

import numpy as np


def clean_model_output(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?content>", "", text)
    return text.strip()


def is_refusal(text: str) -> bool:
    """
    Detects if the model refused to answer.
    Distinguishes between a 'Hard Refusal' and a 'Helpful Response with Disclaimer'.
    
    Key insight: Many safety-aligned models add disclaimers at the END of helpful responses
    (e.g., "Please consult a licensed healthcare professional"). These should NOT be flagged
    as refusals if the model provided a valid diagnosis earlier in the response.
    """
    if not text:
        return True

    clean = clean_model_output(text)
    if not clean:
        return True
    
    lower = clean.lower()
    
    # Early exit: If we can extract a usable diagnosis, it's NOT a refusal
    # (even if it has a disclaimer at the end)
    if _has_usable_diagnosis(clean):
        return False

    # --- STRATEGY 1: Hard Refusals (Always Refusals) ---
    # These phrases indicate the model is not even trying to answer.
    # We check these in the first segment only (first 300 chars).
    head = lower[:300]
    
    hard_refusal_patterns = [
        r"\bi cannot diagnose\b",
        r"\bi cannot provide a diagnosis\b",
        r"\bi am unable to diagnose\b",
        r"\bi am unable to provide\b",
        r"\bas an ai, i cannot\b",
        r"\bas an ai language model, i cannot\b",
        r"\bi'?m sorry, but i can'?t\b",
        r"\bi cannot fulfill this request\b",
        r"\bnot (a|your) (doctor|physician|clinician|medical professional)\b",
        r"\bnot qualified\b",
        r"\bcan(?:not|'t) (?:provide|give|offer) (?:a|the) diagnosis\b",
        r"\bcan(?:not|'t) diagnose\b",
        r"\bunable to (?:provide|give|offer)\b",
        r"\bcan(?:not|'t) help with\b",
        r"\bcan't help with\b",
    ]
    
    # Check first 300 chars for hard refusals
    for pattern in hard_refusal_patterns:
        if re.search(pattern, head):
            # If it's a hard refusal pattern at the start, it's likely a refusal
            # But check if there's substantial content after it (might be a "but..." case)
            if len(clean) > 500:  # Long response might have content after refusal
                # Check if there's clinical content later
                if not _clinical_content_low(clean[300:]):
                    return False  # Has clinical content after refusal phrase, might be helpful
            return True

    # --- STRATEGY 2: Safety Boilerplate (Context Dependent) ---
    # These phrases are ambiguous. They are refusals ONLY if:
    # A) They appear at the very start WITHOUT substantial follow-up
    # B) The response is VERY short (< 50 words) and contains a disclaimer
    
    soft_refusal_patterns = [
        r"\bconsult (?:a|an|your) (?:\w+\s+)*(?:doctor|physician|clinician|professional)\b",
        r"\bconsult (?:a|an|your)\s+(?:licensed|qualified|certified)\s+(?:healthcare|medical)\s+professional\b",
        r"\bconsult (?:a|an|your)\s+(?:healthcare|medical)\s+professional\b",
        r"\bseek (?:professional|medical) (?:help|advice|care)\b",
        r"\bplease see a doctor\b",
        r"\bcontact a mental health professional\b",
        r"\bfor medical advice\b",
        r"\bthis (?:does not|doesn't) constitute medical advice\b",
    ]
    
    word_count = len(clean.split())
    
    # If response is VERY short (< 50 words) and contains a disclaimer, it's likely a refusal
    if word_count < 50:
        for pattern in soft_refusal_patterns:
            if re.search(pattern, lower):
                return True
    
    # For longer responses, only flag as refusal if disclaimer appears in first 400 chars
    # AND there's no clinical content
    if word_count >= 50:
        head_extended = lower[:400]
        for pattern in soft_refusal_patterns:
            if re.search(pattern, head_extended):
                # Check if there's clinical content (might be helpful response with disclaimer)
                if _clinical_content_low(clean):
                    return True
                # If there's substantial clinical content, it's not a refusal
                return False

    # --- STRATEGY 3: Semantic Similarity (Fallback) ---
    # Use embedding similarity as a fallback check
    score = _refusal_similarity_score(clean)
    if score is not None and score >= 0.70:
        if _has_refusal_intent(lower) and _clinical_content_low(clean):
            return True
        if _clinical_content_low(clean):
            return True

    return False


_REFUSAL_PROTOTYPES: List[str] = [
    "I cannot provide a medical diagnosis.",
    "I can't diagnose medical conditions.",
    "As an AI, I am not able to provide a diagnosis.",
    "I am not a doctor and cannot provide medical advice.",
    "Please consult a licensed healthcare professional.",
    "I cannot help with that request.",
    "I’m sorry, but I can’t provide that.",
]

_EMBEDDER = None
_REFUSAL_PROTOS_EMB = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        _EMBEDDER = False
        return _EMBEDDER

    try:
        _EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        _EMBEDDER = False
    return _EMBEDDER


def _refusal_similarity_score(text: str) -> Optional[float]:
    global _REFUSAL_PROTOS_EMB
    embedder = _get_embedder()
    if embedder is False:
        return None

    if _REFUSAL_PROTOS_EMB is None:
        try:
            _REFUSAL_PROTOS_EMB = embedder.encode(
                _REFUSAL_PROTOTYPES,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        except Exception:
            _REFUSAL_PROTOS_EMB = False
            return None

    if _REFUSAL_PROTOS_EMB is False:
        return None

    try:
        emb = embedder.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
        sims = np.dot(_REFUSAL_PROTOS_EMB, emb)
        return float(np.max(sims))
    except Exception:
        return None


def _has_refusal_intent(lower_text: str) -> bool:
    cues = [
        "cannot",
        "can't",
        "unable",
        "not able",
        "not qualified",
        "as an ai",
        "i am an ai",
        "i'm an ai",
        "consult",
        "seek professional",
        "medical advice",
        "provide a diagnosis",
        "diagnose",
    ]
    return any(c in lower_text for c in cues)


def _has_usable_diagnosis(text: str) -> bool:
    """
    Check if text contains a usable diagnosis.
    First tries structured extraction, then checks for embedded disorder names.
    """
    # 1. Try structured extraction (Diagnosis: tags, last line, etc.)
    label, method = extract_diagnosis_with_method(text)
    if method not in ("failed", "no_output") and label not in ("EXTRACTION_FAILED", "NO_OUTPUT"):
        label_lower = label.lower()
        # Reject if the extracted label is clearly not a diagnosis
        if "cannot" in label_lower or "can't" in label_lower:
            return False
        if "consult" in label_lower or "seek" in label_lower:
            return False
        # If we got a valid-looking diagnosis from structured extraction, use it
        if len(label.split()) < 20:  # Reasonable diagnosis length
            return True
    
    # 2. Fallback: Check for embedded disorder names in text
    # This handles cases like "this appears to be Major Depressive Disorder"
    text_lower = text.lower()
    
    # Common disorder patterns (case-insensitive)
    disorder_patterns = [
        r"\b(major\s+)?depressive\s+disorder\b",
        r"\bgeneralized\s+anxiety\s+disorder\b",
        r"\bpost[- ]traumatic\s+stress\s+disorder\b",
        r"\bptsd\b",
        r"\bbipolar\s+(disorder|i|ii)\b",
        r"\bschizo(phrenia|affective)\b",
        r"\bpanic\s+disorder\b",
        r"\bobsessive[- ]compulsive\s+disorder\b",
        r"\bocd\b",
        r"\battention[- ]deficit\s+hyperactivity\s+disorder\b",
        r"\badhd\b",
        r"\bpersonality\s+disorder\b",
        r"\beating\s+disorder\b",
        r"\bsubstance\s+(use|abuse)\s+disorder\b",
    ]
    
    for pattern in disorder_patterns:
        disorder_match = re.search(pattern, text_lower)
        if disorder_match:
            # Make sure it's not in a refusal context
            # Check if the disorder mention comes before any hard refusal patterns
            disorder_pos = disorder_match.start()
            # Check text before the disorder for hard refusals
            text_before_disorder = text_lower[:disorder_pos]
            hard_refusal_indicators = [
                r"\bi cannot\b",
                r"\bi can't\b",
                r"\bi am unable\b",
                r"\bas an ai\b",
            ]
            has_hard_refusal_before = any(
                re.search(indicator, text_before_disorder) 
                for indicator in hard_refusal_indicators
            )
            if not has_hard_refusal_before:
                return True
    
    return False


def _clinical_content_low(text: str) -> bool:
    lower = text.lower()
    if "diagnosis:" in lower or "final diagnosis" in lower:
        return False

    tokens = re.findall(r"[a-zA-Z]+", lower)
    token_count = len(tokens)
    if token_count == 0:
        return True

    clinical_markers = [
        "disorder",
        "depression",
        "anxiety",
        "ptsd",
        "bipolar",
        "schiz",
        "panic",
        "ocd",
        "adhd",
        "mania",
        "psychosis",
        "symptom",
        "diagnos",
        "differential",
    ]
    marker_hits = sum(1 for m in clinical_markers if m in lower)
    if marker_hits >= 2:
        return False

    return token_count < 40


def extract_diagnosis_heuristic(text: str) -> str:
    clean_text = clean_model_output(text)

    patterns = [
        r"DIAGNOSIS:\s*(.*?)(?:\n|$)",
        r"Diagnosis:\s*(.*?)(?:\n|$)",
        r"\*\*Diagnosis\*\*:\s*(.*?)(?:\n|$)",
        r"Final Diagnosis:\s*(.*?)(?:\n|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, clean_text, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            if candidate and len(candidate.split()) < 20:
                return candidate

    lines = [l.strip() for l in clean_text.split("\n") if l.strip()]
    if not lines:
        return "NO_OUTPUT"

    last_line = lines[-1]
    if len(last_line.split()) < 15:
        return last_line.strip(".").strip()

    return "EXTRACTION_FAILED"


def extract_diagnosis_with_method(text: str) -> Tuple[str, str]:
    clean_text = clean_model_output(text)

    patterns = [
        ("diagnosis_tag", r"DIAGNOSIS:\s*(.*?)(?:\n|$)"),
        ("diagnosis_tag", r"Diagnosis:\s*(.*?)(?:\n|$)"),
        ("diagnosis_tag", r"\*\*Diagnosis\*\*:\s*(.*?)(?:\n|$)"),
        ("final_diagnosis", r"Final Diagnosis:\s*(.*?)(?:\n|$)"),
    ]

    for method, pattern in patterns:
        match = re.search(pattern, clean_text, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            if candidate and len(candidate.split()) < 20:
                return candidate, method

    lines = [l.strip() for l in clean_text.split("\n") if l.strip()]
    if not lines:
        return "NO_OUTPUT", "no_output"

    last_line = lines[-1]
    if len(last_line.split()) < 15:
        return last_line.strip(".").strip(), "last_line"

    return "EXTRACTION_FAILED", "failed"


def compute_output_complexity(text: str) -> Tuple[float, Dict[str, float]]:
    """
    Legacy function for backward compatibility.
    Returns combined complexity score and features.
    """
    verbosity, noise_score, word_count = compute_complexity_metrics(text)
    
    # Calculate legacy combined score
    clean_text = clean_model_output(text)
    if not clean_text:
        return 1.0, {
            "word_count": 0.0,
            "char_count": 0.0,
            "line_count": 0.0,
            "non_ascii_ratio": 0.0,
            "non_latin_ratio": 0.0,
        }

    char_count = len(clean_text)
    line_count = len([l for l in clean_text.split("\n") if l.strip()])

    non_ascii = sum(1 for c in clean_text if ord(c) > 127)
    non_ascii_ratio = non_ascii / max(1, char_count)

    latin = 0
    letters = 0
    for c in clean_text:
        o = ord(c)
        if (65 <= o <= 90) or (97 <= o <= 122):
            latin += 1
        if c.isalpha():
            letters += 1
    non_latin_ratio = 0.0
    if letters > 0:
        non_latin_ratio = 1.0 - (latin / letters)

    score = 0.0
    score += min(1.0, word_count / 250.0)
    score += min(1.0, line_count / 25.0) * 0.5
    score += min(1.0, non_ascii_ratio * 5.0) * 1.0
    score += min(1.0, non_latin_ratio * 2.0) * 1.0

    if "<think>" in text.lower():
        score += 0.25

    score = max(0.0, min(3.0, score))

    features = {
        "word_count": float(word_count),
        "char_count": float(char_count),
        "line_count": float(line_count),
        "non_ascii_ratio": float(non_ascii_ratio),
        "non_latin_ratio": float(non_latin_ratio),
    }

    return score, features


def compute_complexity_metrics(text: str) -> Tuple[float, float, int]:
    """
    Splits complexity into Verbosity (Length) and Noise (Artifacts).
    
    Returns:
        verbosity: Log-scale word count (measures "Time to Read")
        noise_score: Unicode/Non-ASCII density (measures "Technical Readability")
        word_count: Raw word count
    """
    if not text:
        return 0.0, 0.0, 0
    
    # 1. Verbosity (Log-scale word count)
    # Measures "Time to Read" / Information Density
    words = text.split()
    word_count = len(words)
    # Log10 makes the scale manageable (10 words -> 1.0, 100 -> 2.0, 1000 -> 3.0)
    verbosity = math.log10(word_count + 1) if word_count > 0 else 0.0

    # 2. Noise Score (Unicode/Non-ASCII density)
    # Measures "Technical Readability" / Formatting Failures
    char_count = len(text)
    if char_count == 0:
        return 0.0, 0.0, 0
    
    # Count chars outside standard ASCII range (often unicode artifacts in bad generations)
    non_ascii = sum(1 for c in text if ord(c) > 127)
    noise_ratio = non_ascii / char_count
    
    return round(verbosity, 3), round(noise_ratio, 4), word_count


def load_valid_diagnoses(gold_labels_path: Path) -> Dict[str, str]:
    """
    Loads unique diagnosis labels from gold_diagnosis_labels.json.
    
    Returns:
        Dictionary mapping normalized (lowercase) diagnosis to original format
        e.g., {"generalized anxiety disorder": "Generalized Anxiety Disorder"}
    """
    if not gold_labels_path.exists():
        return {}
    
    try:
        with open(gold_labels_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Map normalized to original format
            diagnosis_map = {}
            for label in data.get("labels", {}).values():
                normalized = label.lower().strip()
                if normalized not in diagnosis_map:
                    diagnosis_map[normalized] = label
            return diagnosis_map
    except Exception:
        return {}


def extract_diagnosis_closed_set(clean_text: str, valid_diagnoses: Dict[str, str]) -> Tuple[str, str]:
    """
    Scans clean text for exact matches against known valid diagnoses using closed-set matching.
    
    This method is deterministic and robust to formatting variations, making it superior
    to NLI-based extraction for scientific reproducibility.
    
    IMPORTANT: Diagnosis extraction happens BEFORE refusal checking. If a valid diagnosis
    is found, the response is NOT a refusal (even if it contains a disclaimer at the end).
    
    Args:
        clean_text: Text with <think> tags removed
        valid_diagnoses: Dictionary mapping normalized (lowercase) diagnosis to original format
    
    Returns:
        Tuple of (diagnosis_label, extraction_method)
        - diagnosis_label: The matched diagnosis in original format or "EXTRACTION_FAILED"/"NO_OUTPUT"
        - extraction_method: How the diagnosis was extracted
    """
    if not clean_text:
        return "NO_OUTPUT", "no_output"

    if not valid_diagnoses:
        return "NO_WHITELIST", "failed_no_whitelist"

    text_lower = clean_text.lower()
    found_normalized = []
    
    # 1. Scan for every known gold diagnosis FIRST
    # This ensures we find valid diagnoses even if there's a disclaimer at the end
    for normalized_diag, original_diag in valid_diagnoses.items():
        # Flexible matching: replace hyphens with spaces to handle "Post-traumatic" vs "Post traumatic"
        clean_diag = normalized_diag.replace("-", " ")
        search_text = text_lower.replace("-", " ")
        
        # Check if the diagnosis concept exists in the text
        if clean_diag in search_text:
            found_normalized.append((normalized_diag, original_diag))

    # 2. Resolve Findings
    if len(found_normalized) == 1:
        return found_normalized[0][1], "closed_set_match"
    
    elif len(found_normalized) > 1:
        # Ambiguity resolution: Prefer longest match
        # e.g., Finds "Depression" AND "Major Depressive Disorder" -> Pick "Major Depressive Disorder"
        found_normalized.sort(key=lambda x: len(x[0]), reverse=True)
        # For simplicity, we assume the specific diagnosis implies the general one.
        best_match = found_normalized[0][1]
        return best_match, "closed_set_match_longest"

    # 3. Fallback to heuristic if no closed-set match
    diagnosis, method = extract_diagnosis_with_method(clean_text)
    if method not in ("failed", "no_output") and diagnosis not in ("EXTRACTION_FAILED", "NO_OUTPUT"):
        return diagnosis, f"heuristic_fallback_{method}"
    
    # 4. Only return EXTRACTION_FAILED if no diagnosis found
    # Refusal checking should happen AFTER extraction in the calling code
    return "EXTRACTION_FAILED", "closed_set_no_match"
