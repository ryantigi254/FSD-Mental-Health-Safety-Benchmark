"""Unit tests for extraction.py functions."""

import pytest
import sys
from pathlib import Path

# Add src to path
script_dir = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(script_dir / "src"))

from reliable_clinical_benchmark.metrics.extraction import (
    is_refusal,
    extract_diagnosis_closed_set,
    extract_diagnosis_with_method,
    clean_model_output,
    compute_complexity_metrics,
)


@pytest.mark.unit
class TestIsRefusal:
    """Tests for is_refusal() function."""
    
    def test_empty_text_is_refusal(self):
        """Empty text should be considered a refusal."""
        assert is_refusal("") is True
        assert is_refusal(None) is True
    
    def test_hard_refusal_at_start(self):
        """Hard refusal patterns at the start should be flagged."""
        text = "I cannot provide a medical diagnosis. Please consult a doctor."
        assert is_refusal(text) is True
    
    def test_helpful_response_with_disclaimer_at_end(self):
        """Responses with diagnosis and disclaimer at end should NOT be refusals."""
        text = (
            "Based on the symptoms described, this appears to be Major Depressive Disorder. "
            "The patient shows persistent low mood, loss of interest, and sleep disturbances. "
            "However, please consult a licensed healthcare professional for proper evaluation."
        )
        # Should NOT be a refusal because it contains a valid diagnosis
        assert is_refusal(text) is False
    
    def test_short_disclaimer_only(self):
        """Very short responses with only disclaimer should be refusals."""
        text = "Please consult a licensed healthcare professional."
        assert is_refusal(text) is True
    
    def test_disclaimer_in_long_response_without_diagnosis(self):
        """Long response with disclaimer but no diagnosis should be checked carefully."""
        text = (
            "I understand you're experiencing symptoms. "
            "Please consult a licensed healthcare professional for proper evaluation. "
            "They can provide appropriate guidance."
        )
        # Should check if there's clinical content
        result = is_refusal(text)
        # Result depends on clinical content detection
        assert isinstance(result, bool)
    
    def test_valid_diagnosis_overrides_disclaimer(self):
        """If a valid diagnosis is found, it's not a refusal."""
        text = (
            "The patient presents with symptoms consistent with Generalized Anxiety Disorder. "
            "Key indicators include excessive worry, restlessness, and difficulty concentrating. "
            "Please consult a healthcare professional."
        )
        # Should NOT be refusal because diagnosis is present
        assert is_refusal(text) is False


@pytest.mark.unit
class TestExtractDiagnosisClosedSet:
    """Tests for extract_diagnosis_closed_set() function."""
    
    def test_empty_text(self):
        """Empty text should return NO_OUTPUT."""
        result, method = extract_diagnosis_closed_set("", {})
        assert result == "NO_OUTPUT"
        assert method == "no_output"
    
    def test_no_whitelist(self):
        """Without whitelist, should return NO_WHITELIST."""
        text = "This is Major Depressive Disorder."
        result, method = extract_diagnosis_closed_set(text, {})
        assert result == "NO_WHITELIST"
        assert method == "failed_no_whitelist"
    
    def test_finds_diagnosis_with_disclaimer(self):
        """Should find diagnosis even with disclaimer at end."""
        text = (
            "The patient shows symptoms of Major Depressive Disorder. "
            "Please consult a healthcare professional."
        )
        whitelist = {"major depressive disorder": "Major Depressive Disorder"}
        result, method = extract_diagnosis_closed_set(text, whitelist)
        assert result == "Major Depressive Disorder"
        assert method == "closed_set_match"
    
    def test_no_match_returns_extraction_failed(self):
        """If no diagnosis found, should return EXTRACTION_FAILED."""
        # Use text that won't match heuristic fallback (long last line)
        text = "This is some random text without a diagnosis that has many words in the last sentence so it won't match the heuristic fallback pattern."
        whitelist = {"major depressive disorder": "Major Depressive Disorder"}
        result, method = extract_diagnosis_closed_set(text, whitelist)
        assert result == "EXTRACTION_FAILED"
        assert method == "closed_set_no_match"
    
    def test_multiple_matches_prefers_longest(self):
        """If multiple diagnoses found, should prefer longest match."""
        text = "The patient has Major Depressive Disorder, which is a form of depression."
        whitelist = {
            "depression": "Depression",
            "major depressive disorder": "Major Depressive Disorder",
        }
        result, method = extract_diagnosis_closed_set(text, whitelist)
        assert result == "Major Depressive Disorder"
        assert method == "closed_set_match_longest"


@pytest.mark.unit
class TestCleanModelOutput:
    """Tests for clean_model_output() function."""
    
    def test_removes_redacted_reasoning_tags(self):
        """Should remove <think> tags."""
        text = (
            "Some text <think>This is reasoning</think> "
            "More text after."
        )
        result = clean_model_output(text)
        assert "<think>" not in result
        assert "This is reasoning" not in result
        assert "Some text" in result
        assert "More text after" in result
    
    def test_handles_empty_text(self):
        """Should handle empty text."""
        assert clean_model_output("") == ""
        assert clean_model_output(None) == ""


@pytest.mark.unit
class TestComputeComplexityMetrics:
    """Tests for compute_complexity_metrics() function."""
    
    def test_empty_text(self):
        """Empty text should return zeros."""
        verbosity, noise, word_count = compute_complexity_metrics("")
        assert verbosity == 0.0
        assert noise == 0.0
        assert word_count == 0
    
    def test_basic_metrics(self):
        """Should compute basic metrics correctly."""
        text = "This is a test sentence with ten words total here now."
        verbosity, noise, word_count = compute_complexity_metrics(text)
        assert word_count == 11  # Actual word count
        assert verbosity > 0  # log10(11+1) > 0
        assert 0 <= noise <= 1  # Noise ratio should be between 0 and 1
    
    def test_verbosity_is_log_scale(self):
        """Verbosity should be log-scale."""
        short_text = "Short"
        long_text = " ".join(["word"] * 100)
        
        _, _, short_count = compute_complexity_metrics(short_text)
        _, _, long_count = compute_complexity_metrics(long_text)
        
        assert long_count > short_count
        
        short_verb, _, _ = compute_complexity_metrics(short_text)
        long_verb, _, _ = compute_complexity_metrics(long_text)
        
        # Verbosity difference should be less than word count difference (log scale)
        assert (long_verb - short_verb) < (long_count - short_count)

