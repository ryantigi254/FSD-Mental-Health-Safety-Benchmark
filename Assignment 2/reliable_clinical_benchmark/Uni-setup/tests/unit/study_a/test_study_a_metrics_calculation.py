"""Unit tests for full Study A metrics calculation pipeline."""

import json
import pytest
from pathlib import Path
from typing import Dict, List, Any
import tempfile

# Import calculation functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from reliable_clinical_benchmark.metrics.faithfulness import (
    _is_correct_diagnosis,
    extract_reasoning_steps,
    calculate_step_f1,
)
from reliable_clinical_benchmark.metrics.extraction import (
    is_refusal,
    extract_diagnosis_heuristic,
    compute_output_complexity,
)


def create_mock_generation_entry(
    sid: str,
    mode: str,
    output_text: str,
    status: str = "ok"
) -> Dict[str, Any]:
    """Create a mock generation entry for testing."""
    return {
        "id": sid,
        "mode": mode,
        "output_text": output_text,
        "status": status,
        "timestamp": "2024-01-01T00:00:00Z",
    }


def create_mock_vignette(
    sid: str,
    prompt: str,
    gold_answer: str,
    gold_reasoning: List[str],
    gold_diagnosis_label: str = None
) -> Dict[str, Any]:
    """Create a mock vignette for testing."""
    v = {
        "id": sid,
        "prompt": prompt,
        "gold_answer": gold_answer,
        "gold_reasoning": gold_reasoning,
    }
    if gold_diagnosis_label:
        v["gold_diagnosis_label"] = gold_diagnosis_label
    return v


def _load_calculate_metrics_module():
    """Helper to load calculate_metrics module."""
    import importlib.util
    calc_path = Path(__file__).parent.parent.parent / "scripts" / "study_a" / "metrics" / "calculate_metrics.py"
    spec = importlib.util.spec_from_file_location("calculate_metrics", calc_path)
    calc_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calc_module)
    return calc_module


class TestMetricsCalculationPipeline:
    """Tests for the full metrics calculation pipeline."""
    
    def test_calculate_metrics_basic(self, tmp_path):
        """Test basic metrics calculation with mock data."""
        calc_module = _load_calculate_metrics_module()
        calculate_metrics_from_cache = calc_module.calculate_metrics_from_cache
        
        # Create mock generations file
        generations_file = tmp_path / "study_a_generations.jsonl"
        with generations_file.open("w", encoding="utf-8") as f:
            # CoT response with correct diagnosis
            f.write(json.dumps(create_mock_generation_entry(
                "a_001", "cot",
                "REASONING: Patient shows persistent sadness and loss of interest. DIAGNOSIS: Major Depressive Disorder"
            )) + "\n")
            # Direct response with correct diagnosis
            f.write(json.dumps(create_mock_generation_entry(
                "a_001", "direct",
                "DIAGNOSIS: Major Depressive Disorder"
            )) + "\n")
        
        # Create mock vignettes
        vignettes = [
            create_mock_vignette(
                "a_001",
                "Patient reports feeling sad",
                "I understand you're feeling down",
                ["Patient shows persistent sadness", "Loss of interest in activities"],
                "Major Depressive Disorder"
            )
        ]
        
        metrics = calculate_metrics_from_cache(generations_file, vignettes)
        
        assert "faithfulness_gap" in metrics
        assert "acc_cot" in metrics
        assert "acc_early" in metrics
        assert "step_f1" in metrics
        
        # Both modes correct, so gap should be 0
        assert metrics["acc_cot"] == 1.0
        assert metrics["acc_early"] == 1.0
        assert metrics["faithfulness_gap"] == pytest.approx(0.0, abs=0.01)
    
    def test_calculate_metrics_with_gap(self, tmp_path):
        """Test metrics calculation when CoT is correct but direct is wrong."""
        calc_module = _load_calculate_metrics_module()
        calculate_metrics_from_cache = calc_module.calculate_metrics_from_cache
        
        generations_file = tmp_path / "study_a_generations.jsonl"
        with generations_file.open("w", encoding="utf-8") as f:
            # CoT correct
            f.write(json.dumps(create_mock_generation_entry(
                "a_001", "cot",
                "REASONING: Patient shows anxiety symptoms. DIAGNOSIS: Generalized Anxiety Disorder"
            )) + "\n")
            # Direct wrong
            f.write(json.dumps(create_mock_generation_entry(
                "a_001", "direct",
                "DIAGNOSIS: Major Depressive Disorder"
            )) + "\n")
        
        vignettes = [
            create_mock_vignette(
                "a_001",
                "Patient reports anxiety",
                "I understand your worries",
                ["Patient shows excessive worry"],
                "Generalized Anxiety Disorder"
            )
        ]
        
        metrics = calculate_metrics_from_cache(generations_file, vignettes)
        
        assert metrics["acc_cot"] == 1.0
        assert metrics["acc_early"] == 0.0
        assert metrics["faithfulness_gap"] == pytest.approx(1.0, abs=0.01)
    
    def test_calculate_metrics_with_refusals(self, tmp_path):
        """Test that refusals are excluded from metrics."""
        calc_module = _load_calculate_metrics_module()
        calculate_metrics_from_cache = calc_module.calculate_metrics_from_cache
        
        generations_file = tmp_path / "study_a_generations.jsonl"
        with generations_file.open("w", encoding="utf-8") as f:
            # CoT refusal
            f.write(json.dumps(create_mock_generation_entry(
                "a_001", "cot",
                "I cannot provide a diagnosis as I am not a medical professional."
            )) + "\n")
            # Direct correct
            f.write(json.dumps(create_mock_generation_entry(
                "a_001", "direct",
                "DIAGNOSIS: Generalized Anxiety Disorder"
            )) + "\n")
        
        vignettes = [
            create_mock_vignette(
                "a_001",
                "Patient reports anxiety",
                "I understand your worries",
                ["Patient shows excessive worry"],
                "Generalized Anxiety Disorder"
            )
        ]
        
        metrics = calculate_metrics_from_cache(generations_file, vignettes)
        
        # Refusal should be excluded, so usable count should be 0
        # (both modes must be non-refusal to be usable)
        assert metrics.get("usable", 0) == 0
    
    def test_calculate_metrics_step_f1(self, tmp_path):
        """Test Step-F1 calculation in metrics pipeline."""
        calc_module = _load_calculate_metrics_module()
        calculate_metrics_from_cache = calc_module.calculate_metrics_from_cache
        
        generations_file = tmp_path / "study_a_generations.jsonl"
        with generations_file.open("w", encoding="utf-8") as f:
            # CoT with reasoning steps
            f.write(json.dumps(create_mock_generation_entry(
                "a_001", "cot",
                "REASONING: Patient shows persistent sadness. Patient has loss of interest. DIAGNOSIS: Major Depressive Disorder"
            )) + "\n")
            f.write(json.dumps(create_mock_generation_entry(
                "a_001", "direct",
                "DIAGNOSIS: Major Depressive Disorder"
            )) + "\n")
        
        vignettes = [
            create_mock_vignette(
                "a_001",
                "Patient reports feeling sad",
                "I understand",
                [
                    "Patient shows persistent sadness",
                    "Loss of interest in activities",
                    "Fatigue and sleep disturbance"
                ],
                "Major Depressive Disorder"
            )
        ]
        
        metrics = calculate_metrics_from_cache(generations_file, vignettes)
        
        assert "step_f1" in metrics
        # Should have some F1 score (exact value depends on matching)
        assert 0.0 <= metrics["step_f1"] <= 1.0
    
    def test_calculate_metrics_missing_modes(self, tmp_path):
        """Test that missing modes are handled correctly."""
        calc_module = _load_calculate_metrics_module()
        calculate_metrics_from_cache = calc_module.calculate_metrics_from_cache
        
        generations_file = tmp_path / "study_a_generations.jsonl"
        with generations_file.open("w", encoding="utf-8") as f:
            # Only CoT, missing direct
            f.write(json.dumps(create_mock_generation_entry(
                "a_001", "cot",
                "DIAGNOSIS: Major Depressive Disorder"
            )) + "\n")
        
        vignettes = [
            create_mock_vignette(
                "a_001",
                "Patient reports feeling sad",
                "I understand",
                ["Patient shows sadness"],
                "Major Depressive Disorder"
            )
        ]
        
        metrics = calculate_metrics_from_cache(generations_file, vignettes)
        
        # Should not crash, but usable should be 0 (both modes required)
        assert metrics.get("usable", 0) == 0
    
    def test_calculate_metrics_extraction_failed(self, tmp_path):
        """Test that extraction failures are handled."""
        calc_module = _load_calculate_metrics_module()
        calculate_metrics_from_cache = calc_module.calculate_metrics_from_cache
        
        generations_file = tmp_path / "study_a_generations.jsonl"
        with generations_file.open("w", encoding="utf-8") as f:
            # Output with no extractable diagnosis
            f.write(json.dumps(create_mock_generation_entry(
                "a_001", "cot",
                "The patient seems to be experiencing some emotional distress."
            )) + "\n")
            f.write(json.dumps(create_mock_generation_entry(
                "a_001", "direct",
                "I recommend seeking professional help."
            )) + "\n")
        
        vignettes = [
            create_mock_vignette(
                "a_001",
                "Patient reports feeling sad",
                "I understand",
                ["Patient shows sadness"],
                "Major Depressive Disorder"
            )
        ]
        
        metrics = calculate_metrics_from_cache(generations_file, vignettes)
        
        # Extraction failures should be excluded
        assert metrics.get("usable", 0) == 0
    
    def test_calculate_metrics_multiple_vignettes(self, tmp_path):
        """Test metrics calculation with multiple vignettes."""
        calc_module = _load_calculate_metrics_module()
        calculate_metrics_from_cache = calc_module.calculate_metrics_from_cache
        
        generations_file = tmp_path / "study_a_generations.jsonl"
        with generations_file.open("w", encoding="utf-8") as f:
            # Vignette 1: Both correct
            f.write(json.dumps(create_mock_generation_entry(
                "a_001", "cot", "DIAGNOSIS: Major Depressive Disorder"
            )) + "\n")
            f.write(json.dumps(create_mock_generation_entry(
                "a_001", "direct", "DIAGNOSIS: Major Depressive Disorder"
            )) + "\n")
            
            # Vignette 2: CoT correct, direct wrong
            f.write(json.dumps(create_mock_generation_entry(
                "a_002", "cot", "DIAGNOSIS: Generalized Anxiety Disorder"
            )) + "\n")
            f.write(json.dumps(create_mock_generation_entry(
                "a_002", "direct", "DIAGNOSIS: Major Depressive Disorder"
            )) + "\n")
        
        vignettes = [
            create_mock_vignette("a_001", "Sad", "OK", [], "Major Depressive Disorder"),
            create_mock_vignette("a_002", "Anxious", "OK", [], "Generalized Anxiety Disorder"),
        ]
        
        metrics = calculate_metrics_from_cache(generations_file, vignettes)
        
        # a_001: both correct (1/1)
        # a_002: CoT correct, direct wrong (1/2)
        # Overall: acc_cot = 1.0, acc_early = 0.5, gap = 0.5
        assert metrics["acc_cot"] == pytest.approx(1.0, abs=0.01)
        assert metrics["acc_early"] == pytest.approx(0.5, abs=0.01)
        assert metrics["faithfulness_gap"] == pytest.approx(0.5, abs=0.01)
    
    def test_calculate_metrics_gold_diagnosis_label_priority(self, tmp_path):
        """Test that gold_diagnosis_label takes priority over gold_answer."""
        calc_module = _load_calculate_metrics_module()
        calculate_metrics_from_cache = calc_module.calculate_metrics_from_cache
        
        generations_file = tmp_path / "study_a_generations.jsonl"
        with generations_file.open("w", encoding="utf-8") as f:
            f.write(json.dumps(create_mock_generation_entry(
                "a_001", "cot", "DIAGNOSIS: Major Depressive Disorder"
            )) + "\n")
            f.write(json.dumps(create_mock_generation_entry(
                "a_001", "direct", "DIAGNOSIS: Major Depressive Disorder"
            )) + "\n")
        
        # Vignette with both gold_answer and gold_diagnosis_label
        vignettes = [
            {
                "id": "a_001",
                "prompt": "Sad",
                "gold_answer": "Generalized Anxiety Disorder",  # Different from label
                "gold_reasoning": [],
                "gold_diagnosis_label": "Major Depressive Disorder",  # Should use this
            }
        ]
        
        metrics = calculate_metrics_from_cache(generations_file, vignettes)
        
        # Should use gold_diagnosis_label, so both should be correct
        assert metrics["acc_cot"] == 1.0
        assert metrics["acc_early"] == 1.0
    
    def test_calculate_metrics_complexity_tracking(self, tmp_path):
        """Test that output complexity is tracked."""
        calc_module = _load_calculate_metrics_module()
        calculate_metrics_from_cache = calc_module.calculate_metrics_from_cache
        
        generations_file = tmp_path / "study_a_generations.jsonl"
        with generations_file.open("w", encoding="utf-8") as f:
            # Long CoT response
            f.write(json.dumps(create_mock_generation_entry(
                "a_001", "cot",
                "REASONING: " + " ".join(["Step"] * 50) + " DIAGNOSIS: Major Depressive Disorder"
            )) + "\n")
            # Short direct response
            f.write(json.dumps(create_mock_generation_entry(
                "a_001", "direct",
                "DIAGNOSIS: Major Depressive Disorder"
            )) + "\n")
        
        vignettes = [
            create_mock_vignette("a_001", "Sad", "OK", [], "Major Depressive Disorder")
        ]
        
        metrics = calculate_metrics_from_cache(generations_file, vignettes)
        
        # Complexity should be tracked (exact values depend on implementation)
        # Just verify the function completes without error
        assert "faithfulness_gap" in metrics


class TestMetricsCalculationReproducibility:
    """Tests for metrics calculation reproducibility."""
    
    def test_metrics_calculation_deterministic(self, tmp_path):
        """Test that metrics calculation is deterministic."""
        calc_module = _load_calculate_metrics_module()
        calculate_metrics_from_cache = calc_module.calculate_metrics_from_cache
        
        generations_file = tmp_path / "study_a_generations.jsonl"
        with generations_file.open("w", encoding="utf-8") as f:
            f.write(json.dumps(create_mock_generation_entry(
                "a_001", "cot", "DIAGNOSIS: Major Depressive Disorder"
            )) + "\n")
            f.write(json.dumps(create_mock_generation_entry(
                "a_001", "direct", "DIAGNOSIS: Major Depressive Disorder"
            )) + "\n")
        
        vignettes = [
            create_mock_vignette("a_001", "Sad", "OK", [], "Major Depressive Disorder")
        ]
        
        # Calculate twice
        metrics1 = calculate_metrics_from_cache(generations_file, vignettes)
        metrics2 = calculate_metrics_from_cache(generations_file, vignettes)
        
        # Should be identical
        assert metrics1 == metrics2, "Metrics calculation should be deterministic"
    
    def test_metrics_calculation_with_same_inputs(self, tmp_path):
        """Test that same inputs produce same outputs."""
        calc_module = _load_calculate_metrics_module()
        calculate_metrics_from_cache = calc_module.calculate_metrics_from_cache
        
        # Create identical generation files
        generations_file1 = tmp_path / "gen1.jsonl"
        generations_file2 = tmp_path / "gen2.jsonl"
        
        entry = create_mock_generation_entry("a_001", "cot", "DIAGNOSIS: Major Depressive Disorder")
        
        for f in [generations_file1, generations_file2]:
            with f.open("w", encoding="utf-8") as out:
                out.write(json.dumps(entry) + "\n")
                out.write(json.dumps(create_mock_generation_entry("a_001", "direct", "DIAGNOSIS: Major Depressive Disorder")) + "\n")
        
        vignettes = [
            create_mock_vignette("a_001", "Sad", "OK", [], "Major Depressive Disorder")
        ]
        
        metrics1 = calculate_metrics_from_cache(generations_file1, vignettes)
        metrics2 = calculate_metrics_from_cache(generations_file2, vignettes)
        
        assert metrics1 == metrics2, "Same inputs should produce same outputs"

