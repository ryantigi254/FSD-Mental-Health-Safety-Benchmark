"""Unit tests for QwQLMStudioRunner parsing helpers (no LM Studio required)."""

import pytest

from reliable_clinical_benchmark.models.lmstudio_qwq import extract_answer_and_reasoning


@pytest.mark.unit
def test_extract_answer_and_reasoning_from_think_block():
    raw = "<think>\nReasoning here.\n</think>\nFinal answer."
    answer, reasoning = extract_answer_and_reasoning(raw)
    assert "Reasoning here" in reasoning
    assert "Final answer" in answer


@pytest.mark.unit
def test_extract_answer_and_reasoning_from_sections():
    raw = "Some reasoning...\nDiagnosis: Panic disorder"
    answer, reasoning = extract_answer_and_reasoning(raw)
    assert "Panic disorder" in answer
    assert "Some reasoning" in reasoning


@pytest.mark.unit
def test_extract_answer_and_reasoning_no_tags():
    raw = "Just a plain response without reasoning tags."
    answer, reasoning = extract_answer_and_reasoning(raw)
    assert answer == raw
    assert reasoning == raw


@pytest.mark.unit
def test_extract_answer_and_reasoning_empty_response():
    raw = ""
    answer, reasoning = extract_answer_and_reasoning(raw)
    assert answer == ""
    assert reasoning == ""


@pytest.mark.unit
def test_extract_answer_and_reasoning_think_tag_variant():
    raw = "<think>\nInternal reasoning.\n</think>\nFinal output."
    answer, reasoning = extract_answer_and_reasoning(raw)
    assert "Internal reasoning" in reasoning
    assert "Final output" in answer

