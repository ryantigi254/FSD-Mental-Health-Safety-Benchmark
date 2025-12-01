"""Integration tests for split builders (build_splits.py).

These tests do NOT touch the committed JSONs. Instead they call the builder
functions into a temporary directory and verify that repeated runs with the
same configuration are deterministic and schema-consistent.
"""

from pathlib import Path
import json
import importlib

import pytest


@pytest.mark.integration
@pytest.mark.slow
def test_build_study_a_split_is_deterministic(tmp_path):
    """OpenR1-Psy conversion: repeated runs with same seed give identical JSON."""
    try:
        build_splits = importlib.import_module("scripts.build_splits")
    except Exception as e:
        pytest.skip(f"build_splits not importable (datasets not installed?): {e}")

    out1 = tmp_path / "study_a_run1.json"
    out2 = tmp_path / "study_a_run2.json"

    # Use a small target_n for speed but a fixed seed for determinism
    build_splits.build_study_a_split(output_path=out1, target_n=20, seed=123)
    build_splits.build_study_a_split(output_path=out2, target_n=20, seed=123)

    assert out1.exists() and out2.exists()
    assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")

    data = json.loads(out1.read_text(encoding="utf-8"))
    assert "samples" in data
    assert len(data["samples"]) == 20


@pytest.mark.integration
def test_build_study_b_split_is_deterministic(tmp_path):
    """Study B synthetic sycophancy items: deterministic persona-grounded split."""
    try:
        build_splits = importlib.import_module("scripts.build_splits")
    except Exception as e:
        pytest.skip(f"build_splits not importable: {e}")

    out1 = tmp_path / "study_b_run1.json"
    out2 = tmp_path / "study_b_run2.json"

    build_splits.build_study_b_split(output_path=out1)
    build_splits.build_study_b_split(output_path=out2)

    assert out1.exists() and out2.exists()
    assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")

    data = json.loads(out1.read_text(encoding="utf-8"))
    assert "samples" in data and "multi_turn_cases" in data
    assert isinstance(data["samples"], list)
    assert isinstance(data["multi_turn_cases"], list)


@pytest.mark.integration
def test_build_study_c_split_is_deterministic(tmp_path):
    """Study C longitudinal drift cases: deterministic persona-grounded split."""
    try:
        build_splits = importlib.import_module("scripts.build_splits")
    except Exception as e:
        pytest.skip(f"build_splits not importable: {e}")

    out1 = tmp_path / "study_c_run1.json"
    out2 = tmp_path / "study_c_run2.json"

    build_splits.build_study_c_split(output_path=out1)
    build_splits.build_study_c_split(output_path=out2)

    assert out1.exists() and out2.exists()
    assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")

    data = json.loads(out1.read_text(encoding="utf-8"))
    assert "cases" in data
    assert isinstance(data["cases"], list)


@pytest.mark.integration
def test_build_adversarial_bias_cases_is_deterministic(tmp_path):
    """Adversarial bias set: deterministic biased_vignettes.json."""
    try:
        build_splits = importlib.import_module("scripts.build_splits")
    except Exception as e:
        pytest.skip(f"build_splits not importable: {e}")

    out1 = tmp_path / "biased_vignettes_run1.json"
    out2 = tmp_path / "biased_vignettes_run2.json"

    build_splits.build_adversarial_bias_cases(output_path=out1)
    build_splits.build_adversarial_bias_cases(output_path=out2)

    assert out1.exists() and out2.exists()
    assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")

    data = json.loads(out1.read_text(encoding="utf-8"))
    assert "cases" in data
    assert isinstance(data["cases"], list)


