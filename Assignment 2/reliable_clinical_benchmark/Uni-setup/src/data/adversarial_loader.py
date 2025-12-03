"""Data loading for adversarial bias cases."""

import json
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def load_adversarial_bias_cases(data_path: str) -> List[Dict]:
    """
    Load adversarial bias test cases.

    Args:
        data_path: Path to biased_vignettes.json

    Returns:
        List of adversarial case dictionaries with 'prompt', 'bias_feature', 'bias_label'
    """
    path = Path(data_path)
    if not path.exists():
        logger.warning(f"Adversarial bias data file not found: {data_path}")
        return []

    with open(path, "r") as f:
        data = json.load(f)

    cases = data.get("cases", [])
    logger.info(f"Loaded {len(cases)} adversarial bias cases from {data_path}")
    return cases


