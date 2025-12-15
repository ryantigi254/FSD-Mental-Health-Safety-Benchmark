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
    reasoning, answer = runner._extract_reasoning_and_answer(raw)
    assert "Because X" in reasoning
    assert "Generalised anxiety disorder" in answer


@pytest.mark.unit
def test_normalise_for_mode_direct_returns_single_line():
    runner = PsycheR1LocalRunner.__new__(PsycheR1LocalRunner)
    raw = "REASONING:\nLong explanation.\n\nDIAGNOSIS:\nObsessive-compulsive disorder\n\nExtra"
    reasoning, answer = runner._extract_reasoning_and_answer(raw)
    assert "Long explanation" in reasoning
    assert "Obsessive-compulsive disorder" in answer


