"""Unit tests for Study A direct-mode reasoning leakage detection."""

import json
from pathlib import Path

import pytest

from reliable_clinical_benchmark.models.base import GenerationConfig, ModelRunner
from reliable_clinical_benchmark.pipelines.study_a import run_study_a


class _MockModel(ModelRunner):
    def __init__(self):
        super().__init__("mock", GenerationConfig())

    def generate(self, prompt: str, mode: str = "default") -> str:
        return ""

    def generate_with_reasoning(self, prompt: str):
        return "", ""


@pytest.mark.unit
def test_direct_reasoning_leak_rate_from_cache(tmp_path: Path):
    data_dir = tmp_path / "data" / "openr1_psy_splits"
    data_dir.mkdir(parents=True, exist_ok=True)

    study_a_path = data_dir / "study_a_test.json"
    study_a_payload = {
        "samples": [
            {
                "id": "s1",
                "prompt": "Case",
                "gold_answer": "Major Depressive Disorder",
                "gold_reasoning": [],
            }
        ]
    }
    study_a_path.write_text(json.dumps(study_a_payload), encoding="utf-8")

    cache_path = tmp_path / "study_a_generations.jsonl"
    cache_entries = [
        {
            "id": "s1",
            "mode": "cot",
            "prompt": "Case",
            "output_text": "DIAGNOSIS: Major Depressive Disorder",
            "status": "ok",
            "timestamp": "2024-01-01T00:00:00Z",
        },
        {
            "id": "s1",
            "mode": "direct",
            "prompt": "Case",
            "output_text": "<think>hidden</think>\nDIAGNOSIS: Major Depressive Disorder",
            "status": "ok",
            "timestamp": "2024-01-01T00:00:00Z",
        },
    ]
    cache_path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in cache_entries) + "\n",
        encoding="utf-8",
    )

    model = _MockModel()
    output_dir = tmp_path / "results"

    run_study_a(
        model=model,
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        model_name="mock_model",
        from_cache=str(cache_path),
    )

    results_path = output_dir / "mock_model" / "study_a_results.json"
    payload = json.loads(results_path.read_text(encoding="utf-8"))

    assert payload["direct_reasoning_leak_rate"] == pytest.approx(1.0, abs=0.01)
    assert payload["direct_reasoning_leak_count"] == 1
    assert payload["direct_reasoning_total_checked"] == 1
