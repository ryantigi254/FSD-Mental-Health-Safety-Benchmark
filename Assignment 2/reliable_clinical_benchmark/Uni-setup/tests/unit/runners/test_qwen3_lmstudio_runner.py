"""Unit tests for Qwen3 LM Studio runner parsing helpers (no LM Studio required)."""

import pytest

from reliable_clinical_benchmark.models.lmstudio_qwen3 import extract_answer_and_reasoning


@pytest.mark.unit
def test_extract_answer_and_reasoning_from_think_block():
    raw = "<think>Reasoning here.</think>\nFinal diagnosis."
    answer, reasoning = extract_answer_and_reasoning(raw)
    assert reasoning == "Reasoning here."
    assert "Final diagnosis" in answer


@pytest.mark.unit
def test_extract_answer_and_reasoning_from_sections():
    raw = "Some reasoning...\nDiagnosis: Panic disorder"
    answer, reasoning = extract_answer_and_reasoning(raw)
    assert "Panic disorder" in answer
    assert "Some reasoning" in reasoning


