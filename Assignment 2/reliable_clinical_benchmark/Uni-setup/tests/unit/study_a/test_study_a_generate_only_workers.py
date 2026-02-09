"""Unit tests for Study A generate-only worker execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

from reliable_clinical_benchmark.models.base import GenerationConfig, ModelRunner
from reliable_clinical_benchmark.pipelines.study_a import run_study_a


class _DummyLMStudioRunner(ModelRunner):
    def __init__(self, model_name: str = "dummy-lm", config: Optional[GenerationConfig] = None):
        super().__init__(model_name=model_name, config=config or GenerationConfig())
        self.api_base = "http://localhost:1234/v1"

    def generate(self, prompt: str, mode: str = "default") -> str:
        return f"{mode.upper()}::{prompt[:16]}"

    def generate_with_reasoning(self, prompt: str):
        return self.generate(prompt), "reasoning"


@pytest.mark.unit
def test_study_a_generate_only_writes_cache_with_workers(tmp_path: Path) -> None:
    data_dir = tmp_path / "openr1_psy_splits"
    data_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "samples": [
            {
                "id": "a_001",
                "prompt": "Case one",
                "gold_answer": "major depressive disorder",
                "metadata": {"persona_id": "aisha"},
            },
            {
                "id": "a_002",
                "prompt": "Case two",
                "gold_answer": "panic disorder",
                "metadata": {"persona_id": "jamal"},
            },
        ]
    }
    (data_dir / "study_a_test.json").write_text(json.dumps(payload), encoding="utf-8")

    cache_path = tmp_path / "study_a_generations.jsonl"
    model = _DummyLMStudioRunner(config=GenerationConfig(max_tokens=64))

    run_study_a(
        model=model,
        data_dir=str(data_dir),
        max_samples=2,
        output_dir=str(tmp_path),
        model_name="dummy-lm",
        generate_only=True,
        cache_out=str(cache_path),
        workers=4,
        progress_interval_seconds=1,
    )

    assert cache_path.exists()
    lines = [line for line in cache_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 4

    rows = [json.loads(line) for line in lines]
    modes = sorted(row.get("mode") for row in rows)
    assert modes.count("cot") == 2
    assert modes.count("direct") == 2

    for row in rows:
        assert row.get("status") in ("ok", "error")
        assert isinstance(row.get("id"), str) and row["id"]
        assert isinstance(row.get("prompt"), str) and row["prompt"]
        assert isinstance(row.get("output_text"), str)
