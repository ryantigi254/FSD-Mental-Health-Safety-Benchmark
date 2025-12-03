"""Data loading for Study C (Longitudinal Drift)."""

import json
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    """Single turn in a multi-turn conversation."""

    turn: int
    message: str


@dataclass
class LongitudinalCase:
    """Multi-turn longitudinal evaluation case."""

    id: str
    patient_summary: str
    critical_entities: List[str]
    turns: List[Turn]
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def load_study_c_data(data_path: str) -> List[LongitudinalCase]:
    """
    Load Study C frozen test split.

    Args:
        data_path: Path to study_c_test.json

    Returns:
        List of LongitudinalCase objects
    """
    path = Path(data_path)
    if not path.exists():
        logger.warning(f"Study C data file not found: {data_path}")
        return []

    with open(path, "r") as f:
        data = json.load(f)

    cases = data.get("cases", [])
    longitudinal_cases = []
    for case_data in cases:
        turns = [
            Turn(turn=t["turn"], message=t["message"])
            for t in case_data.get("turns", [])
        ]
        longitudinal_cases.append(
            LongitudinalCase(
                id=case_data.get("id", ""),
                patient_summary=case_data.get("patient_summary", ""),
                critical_entities=case_data.get("critical_entities", []),
                turns=turns,
                metadata=case_data.get("metadata", {}),
            )
        )

    logger.info(f"Loaded {len(longitudinal_cases)} Study C cases from {data_path}")
    return longitudinal_cases


