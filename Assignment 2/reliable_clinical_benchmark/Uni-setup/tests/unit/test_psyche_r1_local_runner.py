"""Unit tests for Psyche-R1 local runner text normalisation/parsing."""

import pytest

from reliable_clinical_benchmark.models.psyche_r1_local import PsycheR1LocalRunner


@pytest.mark.unit
def test_extract_reasoning_and_answer_from_think_block():
    runner = PsycheR1LocalRunner.__new__(PsycheR1LocalRunner)
    text = "<think>\nStep 1: foo.\nStep 2: bar.\n</think>\nMajor depressive disorder."
    reasoning, answer = runner._extract_reasoning_and_answer(text)
    assert "Step 1" in reasoning
    assert "Step 2" in reasoning
    assert "Major depressive disorder" in answer


@pytest.mark.unit
def test_extract_reasoning_and_answer_from_reasoning_diagnosis_sections():
    runner = PsycheR1LocalRunner.__new__(PsycheR1LocalRunner)
    text = "REASONING:\nA.\nB.\n\nDIAGNOSIS:\nPanic disorder"
    reasoning, answer = runner._extract_reasoning_and_answer(text)
    assert "A." in reasoning
    assert "B." in reasoning
    assert "Panic disorder" in answer


@pytest.mark.unit
def test_normalise_for_mode_cot_emits_reasoning_and_diagnosis():
    runner = PsycheR1LocalRunner.__new__(PsycheR1LocalRunner)
    raw = "<think>Because X then Y.</think>\nGeneralised anxiety disorder."
    normalised = runner._normalize_for_mode(raw, mode="cot")
    assert "REASONING:" in normalised
    assert "DIAGNOSIS:" in normalised


@pytest.mark.unit
def test_normalise_for_mode_direct_returns_single_line():
    runner = PsycheR1LocalRunner.__new__(PsycheR1LocalRunner)
    raw = "REASONING:\nLong explanation.\n\nDIAGNOSIS:\nObsessive-compulsive disorder\n\nExtra"
    normalised = runner._normalize_for_mode(raw, mode="direct")
    assert "\n" not in normalised.strip()
    assert normalised.strip() == "Obsessive-compulsive disorder"


@pytest.mark.unit
def test_normalise_for_mode_direct_collapses_explanatory_paragraph_to_label():
    runner = PsycheR1LocalRunner.__new__(PsycheR1LocalRunner)
    raw = (
        "Constructivism: Your perceptions and interpretations of events are shaped by your "
        "experiences and beliefs. Consider that Sylvester's behavior might not be a reflection "
        "of you but rather a manifestation of his own issues."
    )
    normalised = runner._normalize_for_mode(raw, mode="direct")
    assert "\n" not in normalised.strip()
    assert normalised.strip() == "Constructivism"


