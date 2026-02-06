"""Unit tests for frozen evaluation data splits (Studies A, B, C and adversarial bias).

These tests assert simple structural invariants over the committed JSONs, so that:
  - the splits remain present and well-formed, and
  - accidental edits to the frozen benchmark data are caught early.
"""

from pathlib import Path
import json

import pytest


BASE_DIR = Path(__file__).resolve().parents[3]


def _load_json(relative_path: str):
    path = BASE_DIR / relative_path
    assert path.exists(), f"Expected data file not found: {path}"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.unit
def test_study_a_split_invariants():
    """Study A: basic schema and size checks for frozen split."""
    data = _load_json("data/openr1_psy_splits/study_a_test.json")

    assert "samples" in data
    samples = data["samples"]
    assert isinstance(samples, list)
    # Split size (minimum)
    assert len(samples) >= 2000

    for sample in samples:
        for key in ("id", "prompt", "gold_answer", "gold_reasoning", "metadata"):
            assert key in sample
        assert isinstance(sample["id"], str) and sample["id"]
        assert isinstance(sample["prompt"], str) and sample["prompt"].strip()
        assert isinstance(sample["gold_answer"], str) and sample["gold_answer"].strip()
        assert isinstance(sample["gold_reasoning"], list)
        assert isinstance(sample["metadata"], dict)
        # Reasoning steps should be non-empty strings if present
        for step in sample["gold_reasoning"]:
            assert isinstance(step, str)
            assert step.strip()


@pytest.mark.unit
def test_study_b_split_invariants():
    """Study B: persona-grounded sycophancy items and ToF cases."""
    samples_payload = _load_json("data/openr1_psy_splits/study_b_test.json")
    multi_turn_cases = _load_json("data/openr1_psy_splits/study_b_multi_turn_test.json")

    # Study B single-turn items are committed as a flat list.
    assert isinstance(samples_payload, list)
    samples = samples_payload

    assert isinstance(multi_turn_cases, list)

    # Split size (minimum)
    assert len(samples) >= 2000

    for item in samples:
        for key in ("id", "prompt", "gold_answer", "incorrect_opinion", "metadata"):
            assert key in item
        assert isinstance(item["id"], str) and item["id"]
        assert isinstance(item["prompt"], str) and item["prompt"].strip()
        assert isinstance(item["gold_answer"], str) and item["gold_answer"].strip()
        assert isinstance(item["incorrect_opinion"], str) and item["incorrect_opinion"].strip()

        metadata = item["metadata"]
        assert isinstance(metadata, dict)
        assert "persona_id" in metadata
        assert isinstance(metadata["persona_id"], str) and metadata["persona_id"]
        assert "age" in metadata
        assert isinstance(metadata["age"], int)

    # Multi-turn "truth-under-pressure" cases (minimum)
    assert len(multi_turn_cases) >= 40

    for case in multi_turn_cases:
        for key in ("id", "gold_answer", "turns", "metadata"):
            assert key in case
        assert isinstance(case["id"], str) and case["id"]
        assert isinstance(case["gold_answer"], str) and case["gold_answer"].strip()

        turns = case["turns"]
        assert isinstance(turns, list)
        # Each ToF conversation should have at least 5 turns of escalating pressure
        assert len(turns) >= 5
        for t in turns:
            assert isinstance(t.get("turn"), int)
            assert isinstance(t.get("message"), str) and t["message"].strip()

        metadata = case["metadata"]
        assert isinstance(metadata, dict)
        assert "persona_id" in metadata
        assert isinstance(metadata["persona_id"], str) and metadata["persona_id"]
        assert "age" in metadata
        assert isinstance(metadata["age"], int)


@pytest.mark.unit
def test_study_c_split_invariants():
    """Study C: longitudinal drift cases are persona-grounded and 10-turn each."""
    data = _load_json("data/openr1_psy_splits/study_c_test.json")

    assert "cases" in data
    cases = data["cases"]
    assert isinstance(cases, list)

    # Split size (minimum)
    assert len(cases) >= 100

    for case in cases:
        for key in ("id", "patient_summary", "critical_entities", "turns", "metadata", "num_turns"):
            assert key in case

        assert isinstance(case["id"], str) and case["id"]
        assert isinstance(case["patient_summary"], str) and case["patient_summary"].strip()

        critical_entities = case["critical_entities"]
        assert isinstance(critical_entities, list)
        assert critical_entities  # At least one critical entity
        for ent in critical_entities:
            assert isinstance(ent, str) and ent.strip()

        turns = case["turns"]
        assert isinstance(turns, list)
        # Longitudinal histories are fixed at 20 turns
        assert len(turns) == 20
        assert case.get("num_turns") == 20
        for t in turns:
            assert isinstance(t.get("turn"), int)
            assert isinstance(t.get("message"), str) and t["message"].strip()

        metadata = case["metadata"]
        assert isinstance(metadata, dict)
        assert "persona_id" in metadata
        assert isinstance(metadata["persona_id"], str) and metadata["persona_id"]
        # New: provenance back to OpenR1-Psy train trajectories
        assert "source_openr1_ids" in metadata
        assert isinstance(metadata["source_openr1_ids"], list)


@pytest.mark.unit
def test_adversarial_bias_invariants():
    """Adversarial bias set: basic schema and coverage checks."""
    data = _load_json("data/adversarial_bias/biased_vignettes.json")

    assert "cases" in data
    cases = data["cases"]
    assert isinstance(cases, list)
    # Adversarial vignettes (minimum)
    assert len(cases) >= 12

    dimensions = set()

    for case in cases:
        for key in ("id", "prompt", "bias_feature", "bias_label", "metadata"):
            assert key in case

        assert isinstance(case["id"], str) and case["id"]
        assert isinstance(case["prompt"], str) and case["prompt"].strip()
        assert isinstance(case["bias_feature"], str) and case["bias_feature"].strip()
        assert isinstance(case["bias_label"], str) and case["bias_label"].strip()

        metadata = case["metadata"]
        assert isinstance(metadata, dict)
        assert "dimension" in metadata
        dimensions.add(metadata["dimension"])

    # Ensure we still cover the intended bias dimensions (plus new variants)
    for dim in ("race", "gender", "substance_use", "framing_bias", "intersection_triple", "socioeconomic", "synthetic_fill"):
        assert dim in dimensions
