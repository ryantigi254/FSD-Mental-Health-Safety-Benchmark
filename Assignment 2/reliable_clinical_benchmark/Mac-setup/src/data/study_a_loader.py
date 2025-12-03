"""Data loading for Study A (Faithfulness)."""

import json
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def load_study_a_data(data_path: str) -> List[Dict]:
    """
    Load Study A frozen test split.

    Args:
        data_path: Path to study_a_test.json

    Returns:
        List of vignette dictionaries with 'prompt', 'gold_answer', 'gold_reasoning'
    """
    path = Path(data_path)
    if not path.exists():
        logger.warning(f"Study A data file not found: {data_path}")
        return []

    with open(path, "r") as f:
        data = json.load(f)

    samples = data.get("samples", [])
    logger.info(f"Loaded {len(samples)} Study A samples from {data_path}")
    return samples

