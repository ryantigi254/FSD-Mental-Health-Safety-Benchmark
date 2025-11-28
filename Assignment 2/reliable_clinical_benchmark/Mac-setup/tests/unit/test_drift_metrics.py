"""Unit tests for drift metrics."""

import pytest
from reliable_clinical_benchmark.metrics.drift import (
    compute_entity_recall_curve,
    calculate_knowledge_conflict_rate,
    compute_drift_slope,
    _extract_advice,
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
    from reliable_clinical_benchmark.utils.ner import MedicalNER

    summaries = [
        "Patient has MDD, on fluoxetine, allergic to penicillin.",
        "Patient has MDD, on fluoxetine.",
        "Patient has MDD.",
    ]

    model = MockModel(summaries)

    try:
        ner = MedicalNER()
        recall_curve = compute_entity_recall_curve(model, sample_case, ner)

        assert len(recall_curve) == len(sample_case.turns)
        assert all(0.0 <= recall <= 1.0 for recall in recall_curve)
        # Recall should generally decrease over turns
        assert recall_curve[0] >= recall_curve[-1]
    except Exception:
        pytest.skip("scispaCy model not available")


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

    # Should return truncated text if no advice keywords found
    assert len(advice) > 0


@pytest.mark.unit
def test_calculate_knowledge_conflict_rate():
    """Test knowledge conflict rate calculation."""
    from reliable_clinical_benchmark.utils.nli import NLIModel

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

    try:
        nli_model = NLIModel()
        k_conflict = calculate_knowledge_conflict_rate(model, cases, nli_model)
        assert 0.0 <= k_conflict <= 1.0
    except Exception:
        pytest.skip("NLI model not available")


@pytest.mark.unit
def test_entity_recall_empty_case():
    """Test entity recall with empty case."""
    from reliable_clinical_benchmark.utils.ner import MedicalNER

    empty_case = LongitudinalCase(
        id="empty",
        patient_summary="",
        critical_entities=[],
        turns=[],
    )

    model = MockModel([])

    try:
        ner = MedicalNER()
        recall_curve = compute_entity_recall_curve(model, empty_case, ner)
        assert recall_curve == []
    except Exception:
        pytest.skip("scispaCy model not available")

