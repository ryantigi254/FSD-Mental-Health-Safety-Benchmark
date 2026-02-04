"""Unit tests for drift metrics."""

import pytest
from reliable_clinical_benchmark.metrics.drift import (
    compute_entity_recall_curve,
    compute_entity_recall_metrics,
    calculate_knowledge_conflict_rate,
    compute_drift_slope,
    _extract_advice,
    _compute_entity_set_metrics,
)
from reliable_clinical_benchmark.data.study_c_loader import LongitudinalCase, Turn


class MockModel:
    """Mock model for testing."""

    def __init__(self, summaries):
        self.summaries = summaries
        self.call_count = 0

    def generate(self, prompt, mode="default"):
        summary = self.summaries[self.call_count % len(self.summaries)]
        self.call_count += 1
        return summary


class DummyNER:
    """Deterministic NER stub for unit tests."""

    def __init__(self, entity_map):
        self.entity_map = {
            key: {entity.lower() for entity in value}
            for key, value in entity_map.items()
        }

    def extract_clinical_entities(self, text):
        return set(self.entity_map.get(text, set()))


@pytest.fixture
def sample_case():
    return LongitudinalCase(
        id="test_case_1",
        patient_summary="32-year-old female with major depressive disorder, currently on fluoxetine 20mg daily. Known allergy to penicillin.",
        critical_entities=["major depressive disorder", "fluoxetine", "penicillin"],
        turns=[
            Turn(turn=1, message="The fluoxetine doesn't seem to be helping."),
            Turn(turn=2, message="I've developed a sore throat."),
            Turn(turn=3, message="Can you summarize my treatment plan?"),
        ],
    )


@pytest.mark.unit
def test_compute_entity_recall_curve(sample_case):
    """Test entity recall curve computation."""
    summaries = [
        "Patient has major depressive disorder, on fluoxetine, allergic to penicillin.",
        "Patient has major depressive disorder, on fluoxetine.",
        "Patient has major depressive disorder.",
    ]

    model = MockModel(summaries)
    dummy_ner = DummyNER(
        {
            sample_case.patient_summary: {
                "major depressive disorder",
                "fluoxetine",
                "penicillin",
            },
            summaries[0]: {"major depressive disorder", "fluoxetine", "penicillin"},
            summaries[1]: {"major depressive disorder", "fluoxetine"},
            summaries[2]: {"major depressive disorder"},
        }
    )
    recall_curve = compute_entity_recall_curve(model, sample_case, dummy_ner)

    assert len(recall_curve) == len(sample_case.turns)
    assert all(0.0 <= recall <= 1.0 for recall in recall_curve)
    # Recall should generally decrease over turns
    assert recall_curve[0] >= recall_curve[-1]


@pytest.mark.unit
def test_entity_recall_metrics_critical_only():
    patient_summary = "Baseline summary text"
    summary_text = "Patient continues fluoxetine."
    sample_case = LongitudinalCase(
        id="critical_only",
        patient_summary=patient_summary,
        critical_entities=["fluoxetine"],
        turns=[Turn(turn=1, message="Checking in")],
    )

    dummy_ner = DummyNER(
        {
            patient_summary: {"fluoxetine", "morning"},
            summary_text: {"fluoxetine"},
        }
    )
    model = MockModel([summary_text])

    metrics = compute_entity_recall_metrics(model, sample_case, dummy_ner)

    assert metrics.recall_curve_critical == [1.0]
    assert metrics.precision_curve_critical == [1.0]
    assert metrics.f1_curve_critical == [1.0]
    assert metrics.hallucinated_rate_curve_critical == [0.0]
    assert metrics.recall_curve_extended == [pytest.approx(0.5)]
    assert metrics.f1_curve_extended == [pytest.approx(2.0 / 3.0)]


@pytest.mark.unit
def test_entity_metrics_negation_excludes_recall():
    gold_entities = {"penicillin allergy"}
    predicted_entities = {"penicillin allergy"}
    summary_text = "No penicillin allergy reported."

    recall, precision, f1_score, hallucinated_rate = _compute_entity_set_metrics(
        gold_entities=gold_entities,
        predicted_entities=predicted_entities,
        summary_text=summary_text,
        nli_model=None,
        apply_negation=True,
    )

    assert recall == 0.0
    assert precision == 0.0
    assert f1_score == 0.0
    assert hallucinated_rate == 1.0


@pytest.mark.unit
def test_entity_metrics_precision_f1_hallucinated():
    gold_entities = {"fluoxetine", "penicillin"}
    predicted_entities = {"fluoxetine", "sertraline"}
    summary_text = "Patient continues fluoxetine and mentions sertraline."

    recall, precision, f1_score, hallucinated_rate = _compute_entity_set_metrics(
        gold_entities=gold_entities,
        predicted_entities=predicted_entities,
        summary_text=summary_text,
        nli_model=None,
        apply_negation=False,
    )

    assert recall == pytest.approx(0.5)
    assert precision == pytest.approx(0.5)
    assert f1_score == pytest.approx(0.5)
    assert hallucinated_rate == pytest.approx(0.5)


@pytest.mark.unit
def test_compute_drift_slope():
    """Test drift slope computation."""
    # Decreasing recall curve
    decreasing_curve = [1.0, 0.9, 0.8, 0.7, 0.6]
    slope = compute_drift_slope(decreasing_curve)
    assert slope < 0.0

    # Increasing recall (shouldn't happen, but test the function)
    increasing_curve = [0.6, 0.7, 0.8, 0.9, 1.0]
    slope = compute_drift_slope(increasing_curve)
    assert slope > 0.0

    # Constant recall
    constant_curve = [0.8, 0.8, 0.8, 0.8, 0.8]
    slope = compute_drift_slope(constant_curve)
    assert abs(slope) < 0.01


@pytest.mark.unit
def test_extract_advice():
    """Test advice extraction."""
    text = "I recommend continuing fluoxetine for 6-8 weeks. The patient should also consider therapy."
    advice = _extract_advice(text)

    assert "fluoxetine" in advice.lower()
    assert "recommend" in advice.lower() or "should" in advice.lower()


@pytest.mark.unit
def test_extract_advice_no_keywords():
    """Test advice extraction when no keywords present."""
    text = "The patient is doing well. Thank you for the update."
    advice = _extract_advice(text)

    assert advice == ""


@pytest.mark.unit
def test_calculate_knowledge_conflict_rate():
    """Test knowledge conflict rate calculation."""
    class _MockNLIModel:
        def predict(self, premise: str, hypothesis: str) -> str:
            return "contradiction"

    cases = [
        LongitudinalCase(
            id="test_1",
            patient_summary="Patient with MDD",
            critical_entities=[],
            turns=[
                Turn(turn=1, message="What should I do?"),
                Turn(turn=2, message="Any other options?"),
            ],
        )
    ]

    # Mock responses that contradict
    responses = [
        "I recommend continuing fluoxetine.",
        "We should stop fluoxetine immediately.",
    ]

    model = MockModel(responses)

    nli_model = _MockNLIModel()
    k_conflict = calculate_knowledge_conflict_rate(model, cases, nli_model)
    assert 0.0 <= k_conflict <= 1.0


@pytest.mark.unit
def test_entity_recall_empty_case():
    """Test entity recall with empty case."""
    empty_case = LongitudinalCase(
        id="empty",
        patient_summary="",
        critical_entities=[],
        turns=[],
    )

    model = MockModel([])
    dummy_ner = DummyNER({"": set()})
    recall_curve = compute_entity_recall_curve(model, empty_case, dummy_ner)
    assert recall_curve == []

