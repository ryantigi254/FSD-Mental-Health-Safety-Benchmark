import re
from typing import Dict, Tuple, Optional, List

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
    words = clean_text.split()
    word_count = len(words)
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
