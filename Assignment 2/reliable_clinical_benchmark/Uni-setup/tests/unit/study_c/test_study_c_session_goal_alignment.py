"""Unit tests for Study C session goal alignment integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

from reliable_clinical_benchmark.models.base import ModelRunner, GenerationConfig
from reliable_clinical_benchmark.pipelines.study_c import run_study_c
import reliable_clinical_benchmark.pipelines.study_c as study_c_pipeline


class _DummyRunner(ModelRunner):
    def __init__(self, model_name: str = "dummy", config: Optional[GenerationConfig] = None):
        super().__init__(model_name=model_name, config=config or GenerationConfig())

    def generate(self, prompt: str, mode: str = "default") -> str:
        return "Patient has mdd and takes fluoxetine."

    def generate_with_reasoning(self, prompt: str):
        return "answer", "reasoning"

    def chat(self, messages, mode: str = "default") -> str:
        return "I recommend CBT and medication management."


class _StubNER:
    def extract_clinical_entities(self, text: str):
        return set(str(text).lower().split())


@pytest.mark.unit
def test_study_c_includes_alignment_when_gold_plan_exists(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "openr1_psy_splits"
    data_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "cases": [
            {
                "id": "c_001",
                "patient_summary": "Patient has MDD and takes fluoxetine.",
                "critical_entities": ["mdd", "fluoxetine"],
                "turns": [
                    {"turn": 1, "message": "Turn one"},
                    {"turn": 2, "message": "Turn two"},
                ],
                "metadata": {"persona_id": "aisha", "source_openr1_ids": [16]},
            }
        ]
    }
    (data_dir / "study_c_test.json").write_text(json.dumps(payload), encoding="utf-8")

    gold_dir = tmp_path / "study_c_gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    (gold_dir / "target_plans.json").write_text(
        json.dumps({"plans": {"c_001": {"plan": "Medication + CBT"}}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(study_c_pipeline, "MedicalNER", lambda: _StubNER())
    monkeypatch.setattr(study_c_pipeline, "calculate_alignment_score", lambda actions, plan: 0.75)

    out_dir = tmp_path / "results"
    model = _DummyRunner(config=GenerationConfig(max_tokens=64))

    run_study_c(
        model=model,
        data_dir=str(data_dir),
        max_cases=1,
        output_dir=str(out_dir),
        model_name="dummy",
        use_nli=False,
    )

    result_path = out_dir / "dummy" / "study_c_results.json"
    assert result_path.exists()
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert "session_goal_alignment" in result
    assert result["session_goal_alignment"] == pytest.approx(0.75, abs=1e-9)


@pytest.mark.unit
def test_study_c_omits_alignment_when_no_gold_plan(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "openr1_psy_splits"
    data_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "cases": [
            {
                "id": "c_001",
                "patient_summary": "Patient has MDD and takes fluoxetine.",
                "critical_entities": ["mdd", "fluoxetine"],
                "turns": [{"turn": 1, "message": "Turn one"}],
                "metadata": {"persona_id": "aisha", "source_openr1_ids": [16]},
            }
        ]
    }
    (data_dir / "study_c_test.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(study_c_pipeline, "MedicalNER", lambda: _StubNER())
    monkeypatch.setattr(study_c_pipeline, "calculate_alignment_score", lambda actions, plan: 0.75)

    out_dir = tmp_path / "results"
    model = _DummyRunner(config=GenerationConfig(max_tokens=64))

    run_study_c(
        model=model,
        data_dir=str(data_dir),
        max_cases=1,
        output_dir=str(out_dir),
        model_name="dummy",
        use_nli=False,
    )

    result_path = out_dir / "dummy" / "study_c_results.json"
    assert result_path.exists()
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert "session_goal_alignment" not in result
