"""Runtime validation utilities."""

import os
from pathlib import Path
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def validate_data_files(data_dir: str = "data") -> Tuple[bool, List[str]]:
    """
    Validate that required data files exist.

    Args:
        data_dir: Base data directory

    Returns:
        Tuple of (is_valid, list_of_missing_files)
    """
    data_path = Path(data_dir)
    missing = []

    required_files = [
        "openr1_psy_splits/study_a_test.json",
        "openr1_psy_splits/study_b_test.json",
        "openr1_psy_splits/study_c_test.json",
        "adversarial_bias/biased_vignettes.json",
    ]

    for rel_path in required_files:
        full_path = data_path / rel_path
        if not full_path.exists():
            missing.append(str(full_path))

    if missing:
        logger.warning(f"Missing data files: {missing}")
        return False, missing

    logger.info("All required data files found")
    return True, []


def validate_environment() -> Tuple[bool, List[str]]:
    """
    Validate that required environment variables and models are available.

    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []

    api_keys = {
        "HUGGINGFACE_API_KEY": "Required for QwQ, DeepSeek-R1, Qwen3",
        "GPT_OSS_API_KEY": "Required for GPT-OSS-120B",
    }

    for key, description in api_keys.items():
        if not os.getenv(key):
            warnings.append(f"{key} not set ({description})")

    try:
        import spacy

        nlp = spacy.load("en_core_sci_sm")
        logger.info("scispaCy model loaded successfully")
    except OSError:
        warnings.append(
            "scispaCy model 'en_core_sci_sm' not found. "
            "Install with: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz"
        )
    except Exception as e:
        warnings.append(f"Error loading scispaCy model: {e}")

    try:
        from transformers import pipeline  # type: ignore

        logger.info("Transformers library available")
    except ImportError:
        warnings.append("transformers library not installed")

    if warnings:
        logger.warning(f"Environment validation warnings: {warnings}")
        return False, warnings

    logger.info("Environment validation passed")
    return True, []


def check_model_availability(model_id: str) -> bool:
    """
    Check if a specific model is available.

    Args:
        model_id: Model identifier

    Returns:
        True if model should be available
    """
    model_id_lower = model_id.lower()

    if model_id_lower == "psyllm":
        try:
            import requests

            response = requests.get("http://localhost:1234/v1/models", timeout=2)
            return response.status_code == 200
        except Exception:
            logger.warning(
                "LM Studio not accessible. Ensure LM Studio is running "
                "and local server is enabled."
            )
            return False

    elif model_id_lower in ["qwq", "deepseek_r1", "qwen3", "gpt_oss"]:
        if model_id_lower == "gpt_oss":
            return bool(os.getenv("GPT_OSS_API_KEY"))
        else:
            return bool(os.getenv("HUGGINGFACE_API_KEY"))

    return True


