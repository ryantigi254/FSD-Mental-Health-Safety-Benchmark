"""Unit tests for worker runtime helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

from reliable_clinical_benchmark.models.base import GenerationConfig, ModelRunner
from reliable_clinical_benchmark.utils.worker_runtime import (
    append_jsonl_with_retry,
    is_lmstudio_runner,
    iter_threaded_results,
    resolve_worker_count,
)


class _DummyRunner(ModelRunner):
    def __init__(self, model_name: str = "dummy", config: Optional[GenerationConfig] = None):
        super().__init__(model_name=model_name, config=config or GenerationConfig())

    def generate(self, prompt: str, mode: str = "default") -> str:
        return f"{mode}:{prompt}"

    def generate_with_reasoning(self, prompt: str):
        return self.generate(prompt), "reasoning"


class _DummyLMStudioRunner(_DummyRunner):
    def __init__(self, model_name: str = "dummy-lm", config: Optional[GenerationConfig] = None):
        super().__init__(model_name=model_name, config=config)
        self.api_base = "http://localhost:1234/v1"


@pytest.mark.unit
def test_worker_count_fallback_for_non_lm_runner() -> None:
    runner = _DummyRunner()
    assert not is_lmstudio_runner(runner)
    assert resolve_worker_count(4, runner) == 1


@pytest.mark.unit
def test_worker_count_allows_parallel_for_lm_runner() -> None:
    runner = _DummyLMStudioRunner()
    assert is_lmstudio_runner(runner)
    assert resolve_worker_count(4, runner) == 4
    assert resolve_worker_count(None, runner) == 4


@pytest.mark.unit
def test_append_jsonl_with_retry_writes_entry(tmp_path: Path) -> None:
    out_path = tmp_path / "cache" / "rows.jsonl"
    payload = {"id": "row_1", "status": "ok"}
    assert append_jsonl_with_retry(out_path, payload)
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows == [payload]


@pytest.mark.unit
def test_iter_threaded_results_returns_all_jobs() -> None:
    jobs = [1, 2, 3, 4]
    seen = []

    def _worker(job: int) -> int:
        return job * 10

    for job, result in iter_threaded_results(jobs, worker_count=2, worker_fn=_worker):
        seen.append((job, result))

    assert sorted(seen) == [(1, 10), (2, 20), (3, 30), (4, 40)]
