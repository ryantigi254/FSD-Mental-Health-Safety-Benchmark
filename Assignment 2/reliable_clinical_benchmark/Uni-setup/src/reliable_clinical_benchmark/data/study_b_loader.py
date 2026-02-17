"""Data loading for Study B (Sycophancy)."""

import json
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SycophancyItem:
    """Single sycophancy evaluation item."""

    id: str
    prompt: str
    gold_answer: str
    incorrect_opinion: str
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def load_study_b_data(data_path: str) -> List[SycophancyItem]:
    """
    Load Study B frozen test split.

    Args:
        data_path: Path to study_b_test.json

    Returns:
        List of SycophancyItem objects
    """
    path = Path(data_path)
    if not path.exists():
        logger.warning(f"Study B data file not found: {data_path}")
        return []

    with open(path, "r") as f:
        data = json.load(f)

    # Handle both new flat-list format and legacy dict format
    if isinstance(data, list):
        samples = data
    else:
        samples = data.get("samples", [])
        
    items = []
    for sample in samples:
        items.append(
            SycophancyItem(
                id=sample.get("id", ""),
                prompt=sample.get("prompt", ""),
                gold_answer=sample.get("gold_answer", ""),
                incorrect_opinion=sample.get("incorrect_opinion", ""),
                metadata=sample.get("metadata", {}),
            )
        )

    logger.info(f"Loaded {len(items)} Study B samples from {data_path}")
    return items


def load_multi_turn_cases(data_path: str) -> List[Dict]:
    """
    Load multi-turn sycophancy cases for Turn of Flip evaluation.

    Args:
        data_path: Path to study_b_multi_turn_test.json (or legacy study_b_test.json)

    Returns:
        List of multi-turn case dictionaries
    """
    path = Path(data_path)
    if not path.exists():
        return []

    with open(path, "r") as f:
        data = json.load(f)

    # Handle both new flat-list format and legacy dict format
    if isinstance(data, list):
        multi_turn_cases = data
    else:
        multi_turn_cases = data.get("multi_turn_cases", [])
        
    logger.info(f"Loaded {len(multi_turn_cases)} multi-turn cases from {data_path}")
    return multi_turn_cases

