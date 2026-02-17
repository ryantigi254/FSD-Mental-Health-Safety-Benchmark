"""Unit tests for the Study C metrics script (calculate_metrics.py)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Dict, Iterable, Set


class DummyNER:
    """Deterministic NER stub used by the Study C script tests."""

    def __init__(self, entity_map: Dict[str, Iterable[str]]) -> None:
        self._entity_map = {
            key: {entity.lower() for entity in value} for key, value in entity_map.items()
        }

    def extract_entities(self, text: str) -> Set[str]:
        return set(self._entity_map.get(text, set()))


class DummyNLI:
    """Deterministic NLI stub for conflict counting."""

    def predict(self, premise: str, hypothesis: str) -> str:
        premise_lower = str(premise or "").lower()
        hypothesis_lower = str(hypothesis or "").lower()
        if "recommend" in premise_lower and "stop" in hypothesis_lower:
            return "contradiction"
        return "neutral"


def _load_metrics_script():
    script_path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "studies"
        / "study_c"
        / "metrics"
        / "calculate_metrics.py"
    )
    spec = importlib.util.spec_from_file_location("study_c_metrics_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_calculate_metrics_for_model_uses_dialogue_nli(tmp_path: Path) -> None:
    module = _load_metrics_script()

    case_id = "case_1"
    entries = [
        {
            "case_id": case_id,
            "turn_num": 1,
            "variant": "summary",
            "response_text": "Patient continues fluoxetine.",
        },
        {
            "case_id": case_id,
            "turn_num": 2,
            "variant": "summary",
            "response_text": "Patient stopped medication.",
        },
        {
            "case_id": case_id,
            "turn_num": 1,
            "variant": "dialogue",
            "response_text": "I recommend continuing fluoxetine.",
        },
        {
            "case_id": case_id,
            "turn_num": 2,
            "variant": "dialogue",
            "response_text": "We should stop fluoxetine.",
        },
    ]

    generations_path = tmp_path / "study_c_generations.jsonl"
    with generations_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")

    gold_data = {
        case_id: {
            "critical_entities": ["fluoxetine", "mdd", "penicillin"],
            "patient_summary": "Patient continues fluoxetine.",
        }
    }

    dummy_ner = DummyNER({
        "Patient continues fluoxetine.": {"fluoxetine"},
        "Patient stopped medication.": set(),
    })

    module.calculate_alignment_score = lambda *_args, **_kwargs: 0.5

    metrics = module.calculate_metrics_for_model(
        model_name="test_model",
        generations_path=generations_path,
        gold_data=gold_data,
        target_plans={case_id: {"plan": "Continue fluoxetine."}},
        ner_model=dummy_ner,
        use_nli=True,
        nli_model=DummyNLI(),
    )

    expected_conflict_rate = 1.0
    assert metrics.knowledge_conflict_rate == expected_conflict_rate
    assert metrics.entity_recall_t10 == 0.0
    assert metrics.continuity_score == 0.5
