"""Unit tests for PsyLLMGMLLocalRunner parsing helpers (no model load required)."""

import pytest

from reliable_clinical_benchmark.models.psyllm_gml_local import PsyLLMGMLLocalRunner


@pytest.mark.unit
def test_extract_reasoning_and_answer_from_think_block():
    runner = PsyLLMGMLLocalRunner.__new__(PsyLLMGMLLocalRunner)
    raw = "<think>\nStep 1.\nStep 2.\n</think>\nFinal output."
    reasoning, answer = runner._extract_reasoning_and_answer(raw)
    assert "Step 1" in reasoning
    assert "Step 2" in reasoning
    assert "Final output" in answer


