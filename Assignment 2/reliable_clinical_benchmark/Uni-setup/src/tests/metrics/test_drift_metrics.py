import sys
import types
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from reliable_clinical_benchmark.metrics import drift as drift_metrics


def _install_dummy_sentence_transformers() -> None:
    class DummySentenceTransformer:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def encode(self, text: str) -> np.ndarray:
            return np.array([1.0, 0.0], dtype=float)

    sys.modules["sentence_transformers"] = types.SimpleNamespace(
        SentenceTransformer=DummySentenceTransformer
    )


def test_extract_clinical_actions_strips_think_tags() -> None:
    text = (
        "<think>ignore this</think> You should try journalling each night. "
        "Let's meet next session to review progress."
    )
    actions = drift_metrics._extract_clinical_actions(text)
    assert "You should try journalling" in actions
    assert "next session" in actions.lower()


def test_filter_entities_removes_generic_and_short() -> None:
    entities = {"patient", "MDD", "fluoxetine 20mg", "a"}
    filtered = drift_metrics._filter_entities(entities)
    assert "patient" not in filtered
    assert "a" not in filtered
    assert "mdd" in filtered
    assert "fluoxetine 20mg" in filtered


def test_is_negated_detects_window() -> None:
    assert drift_metrics._is_negated(
        "penicillin allergy", "No penicillin allergy reported."
    )
    assert not drift_metrics._is_negated(
        "penicillin allergy", "Penicillin allergy reported."
    )


def test_compute_entity_set_metrics_negation_penalises() -> None:
    recall, precision, f1_score, hallucinated = drift_metrics._compute_entity_set_metrics(
        gold_entities={"penicillin allergy"},
        predicted_entities={"penicillin allergy"},
        summary_text="No penicillin allergy reported.",
        nli_model=None,
        apply_negation=True,
    )
    assert recall == 0.0
    assert precision == 0.0
    assert f1_score == 0.0
    assert hallucinated == 1.0


def test_alignment_score_none_when_no_actions() -> None:
    _install_dummy_sentence_transformers()
    score = drift_metrics.calculate_alignment_score(
        ["Hello there."], "plan text", mode="actions"
    )
    assert score is None


def test_alignment_curve_actions_none_then_score() -> None:
    _install_dummy_sentence_transformers()
    curve = drift_metrics.calculate_alignment_curve_actions(
        ["Hello there.", "You should breathe slowly."], "plan"
    )
    assert curve[0] is None
    assert curve[1] is not None


def test_knowledge_conflict_rate_stride_sampling() -> None:
    class DummyNLI:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, premise: str, hypothesis: str) -> str:
            self.calls += 1
            return "contradiction"

    responses_by_case = {
        "case-1": [
            "You should sleep more.",
            "You should drink water.",
            "You should walk daily.",
            "You should journal nightly.",
        ]
    }

    dummy_nli = DummyNLI()
    score = drift_metrics.calculate_knowledge_conflict_rate_from_responses(
        responses_by_case, dummy_nli, nli_stride=2
    )

    assert dummy_nli.calls == 2
    assert score == 1.0
