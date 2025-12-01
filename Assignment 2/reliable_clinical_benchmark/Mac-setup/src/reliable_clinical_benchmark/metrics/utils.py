"""Shared utilities for metric computation."""

import re
from typing import Set
import unicodedata


def normalize_text(text: str) -> str:
    """Normalise text for comparison."""
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> Set[str]:
    """Simple whitespace tokenisation."""
    return set(normalize_text(text).split())


def compute_token_overlap(text1: str, text2: str) -> float:
    """Compute Dice coefficient between two texts."""
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1 & tokens2
    dice = 2 * len(intersection) / (len(tokens1) + len(tokens2))

    return dice


def extract_diagnosis(text: str) -> str:
    """Extract diagnosis from free-form text."""
    text_lower = text.lower()

    match = re.search(r"diagnosis:\s*([^\n.]+)", text_lower)
    if match:
        return match.group(1).strip()

    match = re.search(r"the patient has\s+([^\n.]+)", text_lower)
    if match:
        return match.group(1).strip()

    sentences = re.split(r"[.!?]\s+", text)
    if sentences:
        return sentences[-1].strip()

    return text.strip()

