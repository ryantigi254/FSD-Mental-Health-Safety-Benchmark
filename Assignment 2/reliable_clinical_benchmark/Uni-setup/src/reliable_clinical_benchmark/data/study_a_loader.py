"""Data loading for Study A (Faithfulness)."""

import json
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def load_study_a_data(
    data_path: str,
    gold_diagnosis_labels_path: Optional[str] = None,
) -> List[Dict]:
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

    # Use UTF-8 to handle the curated split (contains smart quotes)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("samples", [])

    labels_path: Optional[Path] = None
    if gold_diagnosis_labels_path:
        labels_path = Path(gold_diagnosis_labels_path)
    else:
        # Try new location first (data/study_a_gold/)
        candidate_new = path.parent.parent / "study_a_gold" / "gold_diagnosis_labels.json"
        if candidate_new.exists():
            labels_path = candidate_new
        else:
            # Fallback to old location for backwards compatibility
            candidate_old = path.parent / "study_a_gold_diagnosis_labels.json"
            if candidate_old.exists():
                labels_path = candidate_old

    if labels_path and labels_path.exists():
        try:
            with labels_path.open("r", encoding="utf-8") as f:
                labels_data = json.load(f)
            id_to_label = labels_data.get("labels", {})
            if isinstance(id_to_label, dict):
                for s in samples:
                    sid = s.get("id")
                    if sid in id_to_label and id_to_label[sid] is not None:
                        s["gold_diagnosis_label"] = str(id_to_label[sid])
                logger.info(
                    f"Merged gold_diagnosis_label for {len(id_to_label)} sample(s) from {labels_path}"
                )
        except Exception as e:
            logger.warning(f"Failed to load gold diagnosis labels from {labels_path}: {e}")
    logger.info(f"Loaded {len(samples)} Study A samples from {data_path}")
    return samples

