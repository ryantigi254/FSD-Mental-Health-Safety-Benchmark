"""Unit tests for Study C context cleaning behavior."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

from reliable_clinical_benchmark.models.base import ModelRunner, GenerationConfig
from reliable_clinical_benchmark.pipelines.study_c import run_study_c


class _RepeatingRunner(ModelRunner):
    def __init__(self, model_name: str = "dummy", config: Optional[GenerationConfig] = None):
        super().__init__(model_name=model_name, config=config or GenerationConfig())
        line = "REPEAT ME " * 10  # ~100 chars
        self.response = "\n".join([line] * 4)

    def generate(self, prompt: str, mode: str = "default") -> str:
        return f"{mode.upper()}::{self.response}"

    def generate_with_reasoning(self, prompt: str):
        return self.generate(prompt), "reasoning"


@pytest.mark.unit
def test_context_cleaning_starts_from_turn_4(tmp_path: Path) -> None:
    data_dir = tmp_path / "openr1_psy_splits"
    data_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "cases": [
            {
                "id": "c_001",
                "patient_summary": "Patient summary",
                "critical_entities": ["sertraline 50mg", "major depressive disorder"],
                "turns": [
                    {"turn": 1, "message": "Turn 1"},
                    {"turn": 2, "message": "Turn 2"},
                    {"turn": 3, "message": "Turn 3"},
                    {"turn": 4, "message": "Turn 4"},
                    {"turn": 5, "message": "Turn 5"},
                ],
                "metadata": {"persona_id": "aisha", "source_openr1_ids": [16]},
            }
        ]
    }
    (data_dir / "study_c_test.json").write_text(json.dumps(payload), encoding="utf-8")

    cache_path = tmp_path / "study_c_generations.jsonl"
    model = _RepeatingRunner(config=GenerationConfig(max_tokens=64))

    run_study_c(
        model=model,
        data_dir=str(data_dir),
        max_cases=1,
        output_dir=str(tmp_path),
        model_name="dummy",
        use_nli=False,
        generate_only=True,
        cache_out=str(cache_path),
        context_cleaner="scan",
        context_clean_start_turn=4,
    )

    rows = [json.loads(ln) for ln in cache_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    dialogue_turn_5 = next(r for r in rows if r.get("variant") == "dialogue" and r.get("turn_num") == 5)

    repeated_line = "REPEAT ME " * 10
    occurrences = dialogue_turn_5["conversation_text"].count(repeated_line)
    # Turns 1-3 are uncleaned (4 repeats each), turn 4 cleaned (1 repeat)
    assert occurrences == 13
