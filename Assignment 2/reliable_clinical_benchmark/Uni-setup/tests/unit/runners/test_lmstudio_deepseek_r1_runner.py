"""Unit tests for DeepSeekR1LMStudioRunner parsing helpers (no LM Studio required)."""

import pytest

from reliable_clinical_benchmark.models.lmstudio_deepseek_r1 import extract_answer_and_reasoning


@pytest.mark.unit
def test_extract_answer_and_reasoning_from_think_block():
    raw = "<think>\nReasoning here.\n</think>\nFinal answer."
    answer, reasoning = extract_answer_and_reasoning(raw)
    assert "Reasoning here" in reasoning
    assert "Final answer" in answer


