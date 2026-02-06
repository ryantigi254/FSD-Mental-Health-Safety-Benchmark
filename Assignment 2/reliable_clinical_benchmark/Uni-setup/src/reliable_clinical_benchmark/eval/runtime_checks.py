"""Runtime validation utilities."""

import os
from pathlib import Path
from typing import List, Tuple
import logging
import json

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

    # Check for API keys (optional for local models)
    api_keys = {
        "HUGGINGFACE_API_KEY": "Required for QwQ, DeepSeek-R1, Qwen3",
        "GPT_OSS_API_KEY": "Required for GPT-OSS-120B",
    }

    for key, description in api_keys.items():
        if not os.getenv(key):
            warnings.append(f"{key} not set ({description})")

    # Try to load scispaCy model
    try:
        import spacy

        nlp = spacy.load("en_core_sci_sm")
        logger.info("scispaCy model loaded successfully")
    except OSError:
        warnings.append(
            "scispaCy model 'en_core_sci_sm' not found. "
            "Install with: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz"
        )
    except Exception as e:
        warnings.append(f"Error loading scispaCy model: {e}")

    # Try to load NLI model (optional, but recommended)
    try:
        from transformers import pipeline

        # Just check if transformers is available
        logger.info("Transformers library available")
    except ImportError:
        warnings.append("transformers library not installed")

    if warnings:
        logger.warning(f"Environment validation warnings: {warnings}")
        return False, warnings

    logger.info("Environment validation passed")
    return True, []


def validate_study_b_schema(data_dir: str = "data") -> Tuple[bool, List[str]]:
    """
    Validate Study B split has persona IDs + well-formed IDs before running generations.

    Args:
        data_dir: Base data directory (should contain openr1_psy_splits/)

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    data_path = Path(data_dir)
    study_b_path = data_path / "openr1_psy_splits" / "study_b_test.json"
    if not study_b_path.exists():
        return False, [f"Study B split not found: {study_b_path}"]

    try:
        payload = json.loads(study_b_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, [f"Study B split is not valid JSON: {study_b_path} ({e})"]

    errors: List[str] = []

    samples = payload.get("samples")
    multi_turn_cases = payload.get("multi_turn_cases")

    if not isinstance(samples, list):
        errors.append("Study B: 'samples' must be a list")
        samples = []
    if not isinstance(multi_turn_cases, list):
        errors.append("Study B: 'multi_turn_cases' must be a list")
        multi_turn_cases = []

    seen_ids = set()
    for i, item in enumerate(samples):
        if not isinstance(item, dict):
            errors.append(f"Study B sample[{i}]: must be an object")
            continue

        sid = item.get("id")
        if not isinstance(sid, str) or not sid.strip():
            errors.append(f"Study B sample[{i}]: missing/invalid 'id'")
        else:
            if sid in seen_ids:
                errors.append(f"Study B: duplicate sample id '{sid}'")
            seen_ids.add(sid)

        for key in ("prompt", "gold_answer", "incorrect_opinion"):
            v = item.get(key)
            if not isinstance(v, str) or not v.strip():
                errors.append(f"Study B sample[{sid or i}]: missing/invalid '{key}'")

        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            errors.append(f"Study B sample[{sid or i}]: missing/invalid 'metadata'")
        else:
            persona_id = metadata.get("persona_id")
            if not isinstance(persona_id, str) or not persona_id.strip():
                errors.append(f"Study B sample[{sid or i}]: missing/invalid metadata.persona_id")

    for j, case in enumerate(multi_turn_cases):
        if not isinstance(case, dict):
            errors.append(f"Study B multi_turn_cases[{j}]: must be an object")
            continue

        cid = case.get("id")
        if not isinstance(cid, str) or not cid.strip():
            errors.append(f"Study B multi_turn_cases[{j}]: missing/invalid 'id'")

        gold = case.get("gold_answer")
        if not isinstance(gold, str) or not gold.strip():
            errors.append(f"Study B multi_turn_cases[{cid or j}]: missing/invalid 'gold_answer'")

        turns = case.get("turns")
        if not isinstance(turns, list) or not turns:
            errors.append(f"Study B multi_turn_cases[{cid or j}]: missing/invalid 'turns'")
        else:
            for k, t in enumerate(turns):
                if not isinstance(t, dict):
                    errors.append(f"Study B multi_turn_cases[{cid or j}] turn[{k}]: must be an object")
                    continue
                msg = t.get("message")
                if not isinstance(msg, str) or not msg.strip():
                    errors.append(f"Study B multi_turn_cases[{cid or j}] turn[{k}]: missing/invalid 'message'")

        metadata = case.get("metadata")
        if not isinstance(metadata, dict):
            errors.append(f"Study B multi_turn_cases[{cid or j}]: missing/invalid 'metadata'")
        else:
            persona_id = metadata.get("persona_id")
            if not isinstance(persona_id, str) or not persona_id.strip():
                errors.append(f"Study B multi_turn_cases[{cid or j}]: missing/invalid metadata.persona_id")

    if errors:
        return False, errors
    return True, []


def validate_study_c_schema(data_dir: str = "data") -> Tuple[bool, List[str]]:
    """
    Validate Study C split has persona IDs + well-formed turn structure before running generations.

    Args:
        data_dir: Base data directory (should contain openr1_psy_splits/)

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    data_path = Path(data_dir)
    study_c_path = data_path / "openr1_psy_splits" / "study_c_test.json"
    if not study_c_path.exists():
        return False, [f"Study C split not found: {study_c_path}"]

    try:
        payload = json.loads(study_c_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, [f"Study C split is not valid JSON: {study_c_path} ({e})"]

    errors: List[str] = []
    cases = payload.get("cases")
    if not isinstance(cases, list):
        return False, ["Study C: 'cases' must be a list"]

    seen_ids = set()
    for i, case in enumerate(cases):
        if not isinstance(case, dict):
            errors.append(f"Study C cases[{i}]: must be an object")
            continue

        cid = case.get("id")
        if not isinstance(cid, str) or not cid.strip():
            errors.append(f"Study C cases[{i}]: missing/invalid 'id'")
            cid = f"idx:{i}"
        else:
            if cid in seen_ids:
                errors.append(f"Study C: duplicate case id '{cid}'")
            seen_ids.add(cid)

        ps = case.get("patient_summary")
        if not isinstance(ps, str) or not ps.strip():
            errors.append(f"Study C cases[{cid}]: missing/invalid 'patient_summary'")

        critical_entities = case.get("critical_entities")
        if not isinstance(critical_entities, list):
            errors.append(f"Study C cases[{cid}]: missing/invalid 'critical_entities'")
        else:
            for ent in critical_entities:
                if not isinstance(ent, str) or not ent.strip():
                    errors.append(f"Study C cases[{cid}]: critical_entities contains invalid entry")
                    break

        turns = case.get("turns")
        if not isinstance(turns, list) or not turns:
            errors.append(f"Study C cases[{cid}]: missing/invalid 'turns'")
        else:
            for j, t in enumerate(turns):
                if not isinstance(t, dict):
                    errors.append(f"Study C cases[{cid}] turn[{j}]: must be an object")
                    continue
                turn_no = t.get("turn")
                if not isinstance(turn_no, int):
                    errors.append(f"Study C cases[{cid}] turn[{j}]: missing/invalid 'turn'")
                msg = t.get("message")
                if not isinstance(msg, str) or not msg.strip():
                    errors.append(f"Study C cases[{cid}] turn[{j}]: missing/invalid 'message'")

        metadata = case.get("metadata")
        if not isinstance(metadata, dict):
            errors.append(f"Study C cases[{cid}]: missing/invalid 'metadata'")
        else:
            persona_id = metadata.get("persona_id")
            if not isinstance(persona_id, str) or not persona_id.strip():
                errors.append(f"Study C cases[{cid}]: missing/invalid metadata.persona_id")
            source_openr1_ids = metadata.get("source_openr1_ids")
            if not isinstance(source_openr1_ids, list):
                errors.append(f"Study C cases[{cid}]: missing/invalid metadata.source_openr1_ids")
            else:
                source_split = metadata.get("source_split")
                if source_split is not None:
                    if not isinstance(source_split, str) or not source_split.strip():
                        errors.append(f"Study C cases[{cid}]: missing/invalid metadata.source_split")
                    elif source_split not in {"test", "train", "generated"}:
                        errors.append(
                            f"Study C cases[{cid}]: metadata.source_split must be one of {{test, train, generated}}"
                        )

                # Enforce linkage unless explicitly synthetic.
                # Backwards compatibility: if source_split is missing, treat empty lists as synthetic.
                if not source_openr1_ids and source_split not in {None, "generated"}:
                    errors.append(
                        f"Study C cases[{cid}]: empty metadata.source_openr1_ids is only allowed when metadata.source_split == 'generated'"
                    )

    if errors:
        return False, errors
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
        # Check if LM Studio endpoint is accessible
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
        # Check for API key
        if model_id_lower == "gpt_oss":
            return bool(os.getenv("GPT_OSS_API_KEY"))
        else:
            return bool(os.getenv("HUGGINGFACE_API_KEY"))

    return True
