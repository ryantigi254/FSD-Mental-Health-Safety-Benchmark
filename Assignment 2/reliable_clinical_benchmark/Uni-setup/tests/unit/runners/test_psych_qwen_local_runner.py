"""Unit tests for PsychQwen32BLocalRunner parsing helpers (no model load required)."""

import pytest

from reliable_clinical_benchmark.models.psych_qwen_local import PsychQwen32BLocalRunner


@pytest.mark.unit
def test_extract_reasoning_and_answer_from_think_block():
    runner = PsychQwen32BLocalRunner.__new__(PsychQwen32BLocalRunner)
    raw = "<think>\nStep 1.\nStep 2.\n</think>\nFinal diagnosis."
    reasoning, answer = runner._extract_reasoning_and_answer(raw)
    assert "Step 1" in reasoning
    assert "Step 2" in reasoning
    assert "Final diagnosis" in answer


@pytest.mark.unit
def test_extract_reasoning_and_answer_from_reasoning_diagnosis_markers():
    runner = PsychQwen32BLocalRunner.__new__(PsychQwen32BLocalRunner)
    raw = "REASONING:\nA.\nB.\n\nDIAGNOSIS:\nPanic disorder"
    reasoning, answer = runner._extract_reasoning_and_answer(raw)
    assert "A." in reasoning
    assert "B." in reasoning
    assert "Panic disorder" in answer


