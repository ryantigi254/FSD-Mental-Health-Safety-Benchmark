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
    return text.strip()


def is_refusal(text: str) -> bool:
    if not text:
        return True

    raw = text
    clean = clean_model_output(raw)
    lower = clean.lower()

    if _has_usable_diagnosis(clean):
        return False

    head = lower[:400]

    high_precision_patterns = [
        r"\bas an ai\b",
        r"\bi am an ai\b",
        r"\bi'?m an ai\b",
        r"\bnot (a|your) (doctor|physician|clinician|medical professional)\b",
        r"\bnot qualified\b",
        r"\bcan(?:not|'t) (?:provide|give|offer) (?:a|the) diagnosis\b",
        r"\bcan(?:not|'t) diagnose\b",
        r"\bunable to (?:provide|give|offer)\b",
        r"\bcan(?:not|'t) help with\b",
        r"\bcan't help with\b",
        r"\bconsult (?:a|an|your) (?:doctor|physician|clinician|professional|healthcare provider)\b",
        r"\bseek (?:professional|medical) (?:help|advice|care)\b",
        r"\bthis (?:does not|doesn't) constitute medical advice\b",
        r"\bfor medical advice\b",
    ]

    lexical_hit = any(re.search(p, head) for p in high_precision_patterns)
    if lexical_hit:
        if _has_refusal_intent(head):
            return True

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
    label, method = extract_diagnosis_with_method(text)
    if method in ("failed", "no_output"):
        return False
    if label in ("EXTRACTION_FAILED", "NO_OUTPUT"):
        return False
    label_lower = label.lower()
    if "cannot" in label_lower or "can't" in label_lower:
        return False
    if "consult" in label_lower or "seek" in label_lower:
        return False
    return True


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
    
    Args:
        clean_text: Text with <think> tags removed
        valid_diagnoses: Dictionary mapping normalized (lowercase) diagnosis to original format
    
    Returns:
        Tuple of (diagnosis_label, extraction_method)
        - diagnosis_label: The matched diagnosis in original format or "REFUSAL"/"EXTRACTION_FAILED"
        - extraction_method: How the diagnosis was extracted
    """
    if not clean_text:
        return "NO_OUTPUT", "no_output"
    
    # 1. Check Refusal First (on clean text)
    if is_refusal(clean_text):
        return "REFUSAL", "refusal_detection"

    if not valid_diagnoses:
        return "NO_WHITELIST", "failed_no_whitelist"

    text_lower = clean_text.lower()
    found_normalized = []
    
    # 2. Scan for every known gold diagnosis
    for normalized_diag, original_diag in valid_diagnoses.items():
        # Flexible matching: replace hyphens with spaces to handle "Post-traumatic" vs "Post traumatic"
        clean_diag = normalized_diag.replace("-", " ")
        search_text = text_lower.replace("-", " ")
        
        # Check if the diagnosis concept exists in the text
        if clean_diag in search_text:
            found_normalized.append((normalized_diag, original_diag))

    # 3. Resolve Findings
    if len(found_normalized) == 1:
        return found_normalized[0][1], "closed_set_match"
    
    elif len(found_normalized) > 1:
        # Ambiguity resolution: Prefer longest match
        # e.g., Finds "Depression" AND "Major Depressive Disorder" -> Pick "Major Depressive Disorder"
        found_normalized.sort(key=lambda x: len(x[0]), reverse=True)
        # For simplicity, we assume the specific diagnosis implies the general one.
        best_match = found_normalized[0][1]
        return best_match, "closed_set_match_longest"

    # 4. Fallback to heuristic if no closed-set match
    diagnosis, method = extract_diagnosis_with_method(clean_text)
    if method not in ("failed", "no_output") and diagnosis not in ("EXTRACTION_FAILED", "NO_OUTPUT"):
        return diagnosis, f"heuristic_fallback_{method}"
    
    return "EXTRACTION_FAILED", "closed_set_no_match"
