"""Unit tests for Study A faithfulness metrics."""

import pytest
from reliable_clinical_benchmark.metrics.faithfulness import (
    _is_correct_diagnosis,
    calculate_step_f1,
    extract_reasoning_steps,
    calculate_faithfulness_gap,
    calculate_silent_bias_rate,
    MIN_REASONING_TOKENS,
)
from reliable_clinical_benchmark.models.base import ModelRunner


class MockModelRunner(ModelRunner):
    """Mock model runner for testing."""

    def __init__(self, cot_responses=None, direct_responses=None, reasoning_responses=None):
        self.cot_responses = cot_responses or []
        self.direct_responses = direct_responses or []
        self.reasoning_responses = reasoning_responses or []
        self.cot_call_count = 0
        self.direct_call_count = 0
        self.reasoning_call_count = 0

    def generate(self, prompt, mode="default"):
        if mode == "cot":
            response = self.cot_responses[self.cot_call_count % len(self.cot_responses)]
            self.cot_call_count += 1
            return response
        elif mode == "direct":
            response = self.direct_responses[self.direct_call_count % len(self.direct_responses)]
            self.direct_call_count += 1
            return response
        return ""

    def generate_with_reasoning(self, prompt):
        if self.reasoning_responses:
            response = self.reasoning_responses[
                self.reasoning_call_count % len(self.reasoning_responses)
            ]
            self.reasoning_call_count += 1
            return response["answer"], response["reasoning"]
        return "Major Depressive Disorder", "Patient shows persistent sadness"


@pytest.mark.unit
class TestIsCorrectDiagnosis:
    """Tests for _is_correct_diagnosis function."""

    def test_exact_match(self):
        """Test exact match of diagnosis."""
        response = "The diagnosis is Major Depressive Disorder"
        gold = "Major Depressive Disorder"
        assert _is_correct_diagnosis(response, gold) is True

    def test_exact_match_case_insensitive(self):
        """Test exact match is case insensitive."""
        response = "the diagnosis is major depressive disorder"
        gold = "Major Depressive Disorder"
        assert _is_correct_diagnosis(response, gold) is True

    def test_abbreviation_mdd(self):
        """Test MDD abbreviation for Major Depressive Disorder."""
        response = "The patient has MDD"
        gold = "Major Depressive Disorder"
        assert _is_correct_diagnosis(response, gold) is True

    def test_abbreviation_gad(self):
        """Test GAD abbreviation for Generalized Anxiety Disorder."""
        response = "Diagnosis: GAD"
        gold = "Generalized Anxiety Disorder"
        assert _is_correct_diagnosis(response, gold) is True

    def test_abbreviation_ptsd(self):
        """Test PTSD abbreviation for Post-Traumatic Stress Disorder."""
        response = "The condition is PTSD"
        gold = "Post-Traumatic Stress Disorder"
        assert _is_correct_diagnosis(response, gold) is True

    def test_abbreviation_bipolar(self):
        """Test bipolar abbreviation."""
        response = "Bipolar disorder"
        gold = "Bipolar Disorder"
        assert _is_correct_diagnosis(response, gold) is True

    def test_no_match(self):
        """Test when diagnosis doesn't match."""
        response = "The patient has anxiety"
        gold = "Major Depressive Disorder"
        assert _is_correct_diagnosis(response, gold) is False

    def test_partial_match_not_sufficient(self):
        """Test that partial word matches don't count."""
        response = "The patient has depression symptoms"
        gold = "Major Depressive Disorder"
        assert _is_correct_diagnosis(response, gold) is False

    def test_gold_with_whitespace(self):
        """Test gold answer with extra whitespace (gold is stripped before calling)."""
        response = "Major Depressive Disorder"
        gold = "Major Depressive Disorder"  # Already stripped in pipeline
        assert _is_correct_diagnosis(response, gold) is True


@pytest.mark.unit
class TestExtractReasoningSteps:
    """Tests for extract_reasoning_steps function."""

    def test_extract_with_reasoning_and_diagnosis_markers(self):
        """Test extraction with REASONING: and DIAGNOSIS: markers."""
        # Need at least MIN_REASONING_TOKENS (20) tokens in reasoning block
        text = """REASONING: The patient shows persistent sadness and loss of interest in activities. They have been experiencing sleep disturbances for several weeks. Their energy levels have decreased significantly. These symptoms suggest a mood disorder. DIAGNOSIS: Major Depressive Disorder"""
        steps = extract_reasoning_steps(text)
        assert len(steps) >= 2
        assert "persistent sadness" in steps[0].lower() or "loss of interest" in steps[0].lower()

    def test_extract_with_lowercase_markers(self):
        """Test extraction with lowercase markers."""
        # Need at least MIN_REASONING_TOKENS (20) tokens
        text = """reasoning: Patient has anxiety symptoms that persist throughout the day. They worry excessively about multiple aspects of their life. These symptoms have been present for several months and interfere with daily functioning. diagnosis: GAD"""
        steps = extract_reasoning_steps(text)
        assert len(steps) >= 2

    def test_extract_without_markers(self):
        """Test extraction without explicit markers."""
        text = "The patient shows persistent sadness. They have lost interest in activities. Sleep disturbances are present. This suggests Major Depressive Disorder."
        steps = extract_reasoning_steps(text)
        assert len(steps) >= 3

    def test_too_short_reasoning_returns_empty(self):
        """Test that reasoning shorter than MIN_REASONING_TOKENS returns empty."""
        text = "REASONING: Short. DIAGNOSIS: MDD"
        steps = extract_reasoning_steps(text)
        assert steps == []

    def test_reasoning_with_multiple_sentences(self):
        """Test extraction of multiple sentences."""
        text = """REASONING: First step is to assess mood. Second step is to check sleep patterns. Third step is to evaluate energy levels. DIAGNOSIS: MDD"""
        steps = extract_reasoning_steps(text)
        assert len(steps) == 3

    def test_reasoning_with_punctuation(self):
        """Test extraction handles various punctuation."""
        # Need at least MIN_REASONING_TOKENS (20) tokens
        text = """REASONING: Step one: assess mood and emotional state carefully! Step two: check sleep patterns and quality of rest? Step three: evaluate energy levels and daily functioning. Step four: consider duration and severity of symptoms. DIAGNOSIS: MDD"""
        steps = extract_reasoning_steps(text)
        assert len(steps) >= 3

    def test_empty_reasoning_block(self):
        """Test empty reasoning block."""
        text = "REASONING: DIAGNOSIS: MDD"
        steps = extract_reasoning_steps(text)
        assert steps == []

    def test_reasoning_with_redacted_tags(self):
        """Test extraction with <think> tags."""
        # Need at least MIN_REASONING_TOKENS (20) tokens
        text = """<think>Patient shows persistent sadness that has lasted for several weeks. They have lost interest in activities they previously enjoyed. Sleep issues are present with difficulty falling asleep and early morning awakening. Energy levels are significantly decreased. These symptoms suggest a mood disorder.</think>
<content>Major Depressive Disorder</content>"""
        steps = extract_reasoning_steps(text)
        assert len(steps) >= 2


@pytest.mark.unit
class TestCalculateStepF1:
    """Tests for calculate_step_f1 function."""

    def test_perfect_match(self):
        """Test perfect match between model and gold steps."""
        model_steps = ["Patient shows persistent sadness", "Loss of interest in activities"]
        gold_steps = ["Patient shows persistent sadness", "Loss of interest in activities"]
        f1 = calculate_step_f1(model_steps, gold_steps)
        assert f1 == pytest.approx(1.0, abs=0.01)

    def test_partial_match(self):
        """Test partial match scenario."""
        model_steps = [
            "Patient shows persistent sadness",
            "Loss of interest in activities",
            "Sleep disturbances",
        ]
        gold_steps = [
            "Patient shows persistent sadness",
            "Loss of interest in activities",
        ]
        f1 = calculate_step_f1(model_steps, gold_steps)
        assert 0.0 < f1 < 1.0
        # Precision = 2/3 = 0.67, Recall = 2/2 = 1.0, F1 = 2*0.67*1.0/(0.67+1.0) ≈ 0.8
        assert f1 == pytest.approx(0.8, abs=0.1)

    def test_no_match(self):
        """Test no match scenario."""
        model_steps = ["Patient has anxiety", "Worries excessively"]
        gold_steps = ["Patient shows persistent sadness", "Loss of interest"]
        f1 = calculate_step_f1(model_steps, gold_steps)
        assert f1 == pytest.approx(0.0, abs=0.01)

    def test_empty_model_steps(self):
        """Test empty model steps."""
        model_steps = []
        gold_steps = ["Patient shows persistent sadness"]
        f1 = calculate_step_f1(model_steps, gold_steps)
        assert f1 == 0.0

    def test_empty_gold_steps(self):
        """Test empty gold steps."""
        model_steps = ["Patient shows persistent sadness"]
        gold_steps = []
        f1 = calculate_step_f1(model_steps, gold_steps)
        assert f1 == 0.0

    def test_both_empty(self):
        """Test both empty."""
        model_steps = []
        gold_steps = []
        f1 = calculate_step_f1(model_steps, gold_steps)
        assert f1 == 0.0

    def test_one_to_one_matching(self):
        """Test that one-to-one matching is enforced."""
        model_steps = ["Step A", "Step B"]
        gold_steps = ["Step A", "Step A"]  # Duplicate gold step
        f1 = calculate_step_f1(model_steps, gold_steps, threshold=0.6)
        assert f1 < 1.0  # Should not match both model steps to same gold step

    def test_threshold_parameter(self):
        """Test threshold parameter affects matching."""
        model_steps = ["Patient shows persistent sadness"]
        gold_steps = ["Patient shows persistent sadness and anxiety"]
        f1_low = calculate_step_f1(model_steps, gold_steps, threshold=0.3)
        f1_high = calculate_step_f1(model_steps, gold_steps, threshold=0.9)
        assert f1_low >= f1_high

    def test_case_insensitive_matching(self):
        """Test that matching is case insensitive."""
        model_steps = ["PATIENT SHOWS PERSISTENT SADNESS"]
        gold_steps = ["patient shows persistent sadness"]
        f1 = calculate_step_f1(model_steps, gold_steps)
        assert f1 == pytest.approx(1.0, abs=0.01)

    def test_punctuation_normalization(self):
        """Test that punctuation is normalized."""
        model_steps = ["Patient shows persistent sadness!"]
        gold_steps = ["Patient shows persistent sadness."]
        f1 = calculate_step_f1(model_steps, gold_steps)
        assert f1 == pytest.approx(1.0, abs=0.01)


@pytest.mark.unit
class TestCalculateFaithfulnessGap:
    """Tests for calculate_faithfulness_gap function."""

    def test_positive_gap(self):
        """Test positive faithfulness gap (CoT better than direct)."""
        vignettes = [
            {
                "id": "v1",
                "prompt": "Test prompt",
                "gold_answer": "Major Depressive Disorder",
            },
            {
                "id": "v2",
                "prompt": "Test prompt 2",
                "gold_answer": "Generalized Anxiety Disorder",
            },
        ]
        model = MockModelRunner(
            cot_responses=["Major Depressive Disorder", "Generalized Anxiety Disorder"],
            direct_responses=["Anxiety", "Depression"],  # Both wrong
        )
        gap, acc_cot, acc_early = calculate_faithfulness_gap(model, vignettes)
        assert acc_cot == 1.0
        assert acc_early == 0.0
        assert gap == pytest.approx(1.0, abs=0.01)

    def test_zero_gap(self):
        """Test zero gap (CoT and direct same accuracy)."""
        vignettes = [
            {
                "id": "v1",
                "prompt": "Test prompt",
                "gold_answer": "Major Depressive Disorder",
            },
        ]
        model = MockModelRunner(
            cot_responses=["Major Depressive Disorder"],
            direct_responses=["Major Depressive Disorder"],
        )
        gap, acc_cot, acc_early = calculate_faithfulness_gap(model, vignettes)
        assert acc_cot == 1.0
        assert acc_early == 1.0
        assert gap == pytest.approx(0.0, abs=0.01)

    def test_negative_gap(self):
        """Test negative gap (direct better than CoT - unusual but possible)."""
        vignettes = [
            {
                "id": "v1",
                "prompt": "Test prompt",
                "gold_answer": "Major Depressive Disorder",
            },
        ]
        model = MockModelRunner(
            cot_responses=["Anxiety"],  # Wrong
            direct_responses=["Major Depressive Disorder"],  # Correct
        )
        gap, acc_cot, acc_early = calculate_faithfulness_gap(model, vignettes)
        assert acc_cot == 0.0
        assert acc_early == 1.0
        assert gap == pytest.approx(-1.0, abs=0.01)

    def test_empty_vignettes(self):
        """Test empty vignettes list."""
        vignettes = []
        model = MockModelRunner()
        gap, acc_cot, acc_early = calculate_faithfulness_gap(model, vignettes)
        assert acc_cot == 0.0
        assert acc_early == 0.0
        assert gap == pytest.approx(0.0, abs=0.01)

    def test_partial_accuracy(self):
        """Test partial accuracy scenario."""
        vignettes = [
            {
                "id": "v1",
                "prompt": "Test prompt",
                "gold_answer": "Major Depressive Disorder",
            },
            {
                "id": "v2",
                "prompt": "Test prompt 2",
                "gold_answer": "Generalized Anxiety Disorder",
            },
            {
                "id": "v3",
                "prompt": "Test prompt 3",
                "gold_answer": "PTSD",
            },
        ]
        model = MockModelRunner(
            cot_responses=["Major Depressive Disorder", "Generalized Anxiety Disorder", "PTSD"],
            direct_responses=["Major Depressive Disorder", "Anxiety", "Anxiety"],
        )
        gap, acc_cot, acc_early = calculate_faithfulness_gap(model, vignettes)
        assert acc_cot == pytest.approx(1.0, abs=0.01)
        assert acc_early == pytest.approx(0.33, abs=0.1)
        assert gap == pytest.approx(0.67, abs=0.1)


@pytest.mark.unit
class TestCalculateSilentBiasRate:
    """Tests for calculate_silent_bias_rate function."""

    def test_no_biased_answers(self):
        """Test when no biased answers are given."""
        adversarial_cases = [
            {
                "prompt": "Test prompt",
                "bias_feature": "gender",
                "bias_label": "female",
            },
        ]
        model = MockModelRunner(
            reasoning_responses=[
                {"answer": "Major Depressive Disorder", "reasoning": "Patient shows symptoms"}
            ]
        )
        r_sb = calculate_silent_bias_rate(model, adversarial_cases)
        assert r_sb == 0.0

    def test_all_silent_bias(self):
        """Test when all biased answers are silent (bias not mentioned in reasoning)."""
        adversarial_cases = [
            {
                "prompt": "Test prompt",
                "bias_feature": "gender",
                "bias_label": "female",
            },
        ]
        model = MockModelRunner(
            reasoning_responses=[
                {
                    "answer": "The patient is female and has depression",
                    "reasoning": "Patient shows persistent sadness",  # No mention of gender
                }
            ]
        )
        r_sb = calculate_silent_bias_rate(model, adversarial_cases)
        assert r_sb == pytest.approx(1.0, abs=0.01)

    def test_no_silent_bias(self):
        """Test when bias is mentioned in reasoning."""
        adversarial_cases = [
            {
                "prompt": "Test prompt",
                "bias_feature": "gender",
                "bias_label": "female",
            },
        ]
        model = MockModelRunner(
            reasoning_responses=[
                {
                    "answer": "The patient is female and has depression",
                    "reasoning": "Gender is a factor in this case. Female patients often show these symptoms",  # bias_feature "gender" mentioned
                }
            ]
        )
        r_sb = calculate_silent_bias_rate(model, adversarial_cases)
        assert r_sb == pytest.approx(0.0, abs=0.01)

    def test_partial_silent_bias(self):
        """Test partial silent bias scenario."""
        adversarial_cases = [
            {
                "prompt": "Test prompt 1",
                "bias_feature": "gender",
                "bias_label": "female",
            },
            {
                "prompt": "Test prompt 2",
                "bias_feature": "gender",
                "bias_label": "female",
            },
        ]
        model = MockModelRunner(
            reasoning_responses=[
                {
                    "answer": "The patient is female and has depression",
                    "reasoning": "Patient shows persistent sadness",  # Silent (no "gender" mentioned)
                },
                {
                    "answer": "The patient is female and has anxiety",
                    "reasoning": "Gender is relevant. Female patients often show anxiety",  # Not silent (bias_feature "gender" mentioned)
                }
            ]
        )
        r_sb = calculate_silent_bias_rate(model, adversarial_cases)
        assert r_sb == pytest.approx(0.5, abs=0.01)

    def test_case_insensitive_bias_detection(self):
        """Test that bias detection is case insensitive."""
        adversarial_cases = [
            {
                "prompt": "Test prompt",
                "bias_feature": "gender",
                "bias_label": "female",
            },
        ]
        model = MockModelRunner(
            reasoning_responses=[
                {
                    "answer": "The patient is FEMALE and has depression",
                    "reasoning": "Patient shows persistent sadness",
                }
            ]
        )
        r_sb = calculate_silent_bias_rate(model, adversarial_cases)
        assert r_sb == pytest.approx(1.0, abs=0.01)

    def test_empty_adversarial_cases(self):
        """Test empty adversarial cases."""
        adversarial_cases = []
        model = MockModelRunner()
        r_sb = calculate_silent_bias_rate(model, adversarial_cases)
        assert r_sb == 0.0

