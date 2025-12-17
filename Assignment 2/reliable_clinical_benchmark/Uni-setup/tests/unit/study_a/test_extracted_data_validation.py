"""Unit tests for extracted data validation."""

import pytest
from pathlib import Path
import sys
import json
from collections import defaultdict

# Add scripts to path to import validation function
script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir / "scripts" / "study_a" / "metrics"))

from validate_extracted_data import validate_extracted_file


@pytest.mark.unit
class TestExtractedDataValidation:
    """Tests for extracted data validation."""
    
    @pytest.fixture
    def processed_dir(self):
        """Get processed directory path."""
        return script_dir / "processed" / "study_a_extracted"
    
    def test_processed_directory_exists(self, processed_dir):
        """Test that processed directory exists."""
        assert processed_dir.exists(), f"Processed directory not found: {processed_dir}"
    
    def test_all_models_have_extracted_files(self, processed_dir):
        """Test that all model directories have extracted files."""
        if not processed_dir.exists():
            pytest.skip("Processed directory does not exist - run extraction first")
        
        model_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
        assert len(model_dirs) > 0, "No model directories found in processed directory"
        
        for model_dir in model_dirs:
            extracted_file = model_dir / "study_a_extracted.jsonl"
            assert extracted_file.exists(), f"Missing extracted file for {model_dir.name}: {extracted_file}"
    
    def test_all_extracted_files_are_valid(self, processed_dir):
        """Test that all extracted files pass validation."""
        if not processed_dir.exists():
            pytest.skip("Processed directory does not exist - run extraction first")
        
        model_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
        assert len(model_dirs) > 0, "No model directories found"
        
        validation_failures = []
        
        for model_dir in sorted(model_dirs):
            extracted_file = model_dir / "study_a_extracted.jsonl"
            if not extracted_file.exists():
                validation_failures.append(f"{model_dir.name}: File not found")
                continue
            
            result = validate_extracted_file(extracted_file)
            
            if not result["is_valid"]:
                issues_summary = f"{len(result['issues'])} issues"
                if result["issues"]:
                    issues_summary += f": {result['issues'][:3]}"  # Show first 3 issues
                validation_failures.append(f"{model_dir.name}: {issues_summary}")
            
            # Check specific requirements
            stats = result["stats"]
            
            # Must have new format
            assert stats["has_new_format"], f"{model_dir.name}: Missing new format fields"
            
            # Must not have old format
            assert not stats["has_old_format"], f"{model_dir.name}: Contains old format fields"
            
            # Must have entries
            assert stats["total_entries"] > 0, f"{model_dir.name}: No entries found"
            
            # Must have no missing required fields
            assert len(stats["missing_fields"]) == 0, \
                f"{model_dir.name}: Missing fields: {dict(stats['missing_fields'])}"
        
        if validation_failures:
            pytest.fail(f"Validation failures:\n" + "\n".join(f"  - {f}" for f in validation_failures))
    
    def test_extracted_files_have_required_fields(self, processed_dir):
        """Test that all entries have required fields."""
        if not processed_dir.exists():
            pytest.skip("Processed directory does not exist - run extraction first")
        
        required_fields = [
            "id", "mode", "model_name", "status", "is_refusal",
            "extracted_diagnosis", "extraction_method",
            "response_verbosity", "format_noise_score", "word_count"
        ]
        
        model_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
        
        for model_dir in model_dirs:
            extracted_file = model_dir / "study_a_extracted.jsonl"
            if not extracted_file.exists():
                continue
            
            with open(extracted_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    entry = json.loads(line)
                    
                    for field in required_fields:
                        assert field in entry, \
                            f"{model_dir.name} line {line_num}: Missing required field '{field}'"
    
    def test_extracted_files_have_new_format_only(self, processed_dir):
        """Test that extracted files use new format and not old format."""
        if not processed_dir.exists():
            pytest.skip("Processed directory does not exist - run extraction first")
        
        old_format_fields = ["output_complexity", "complexity_features"]
        new_format_fields = ["response_verbosity", "format_noise_score", "word_count"]
        
        model_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
        
        for model_dir in model_dirs:
            extracted_file = model_dir / "study_a_extracted.jsonl"
            if not extracted_file.exists():
                continue
            
            with open(extracted_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    entry = json.loads(line)
                    
                    # Must have new format fields
                    for field in new_format_fields:
                        assert field in entry, \
                            f"{model_dir.name} line {line_num}: Missing new format field '{field}'"
                    
                    # Must not have old format fields
                    for field in old_format_fields:
                        assert field not in entry, \
                            f"{model_dir.name} line {line_num}: Contains old format field '{field}'"
    
    def test_extracted_files_have_valid_numeric_types(self, processed_dir):
        """Test that numeric fields have correct types."""
        if not processed_dir.exists():
            pytest.skip("Processed directory does not exist - run extraction first")
        
        numeric_fields = ["response_verbosity", "format_noise_score", "word_count"]
        
        model_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
        
        for model_dir in model_dirs:
            extracted_file = model_dir / "study_a_extracted.jsonl"
            if not extracted_file.exists():
                continue
            
            with open(extracted_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    entry = json.loads(line)
                    
                    for field in numeric_fields:
                        if field in entry:
                            value = entry[field]
                            assert isinstance(value, (int, float)), \
                                f"{model_dir.name} line {line_num}: Field '{field}' has invalid type {type(value)}, expected int or float"
    
    def test_extracted_files_have_valid_extraction_methods(self, processed_dir):
        """Test that extraction methods are valid."""
        if not processed_dir.exists():
            pytest.skip("Processed directory does not exist - run extraction first")
        
        valid_methods = [
            "closed_set_match",
            "closed_set_match_longest",
            "closed_set_no_match",
            "heuristic_fallback_diagnosis_tag",
            "heuristic_fallback_last_line",
            "heuristic_fallback_final_diagnosis",
            "refusal_detection",
            "no_output",
            "failed_no_whitelist",
        ]
        
        model_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
        
        for model_dir in model_dirs:
            extracted_file = model_dir / "study_a_extracted.jsonl"
            if not extracted_file.exists():
                continue
            
            with open(extracted_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    entry = json.loads(line)
                    
                    if "extraction_method" in entry:
                        method = entry["extraction_method"]
                        # Allow heuristic_fallback_* methods
                        is_valid = (
                            method in valid_methods or
                            method.startswith("heuristic_fallback_")
                        )
                        assert is_valid, \
                            f"{model_dir.name} line {line_num}: Invalid extraction_method '{method}'"
    
    def test_extraction_statistics_are_reasonable(self, processed_dir):
        """Test that extraction statistics are within reasonable ranges."""
        if not processed_dir.exists():
            pytest.skip("Processed directory does not exist - run extraction first")
        
        model_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
        
        for model_dir in model_dirs:
            extracted_file = model_dir / "study_a_extracted.jsonl"
            if not extracted_file.exists():
                continue
            
            result = validate_extracted_file(extracted_file)
            stats = result["stats"]
            
            # Check verbosity range (log10 scale, so 0-4 is reasonable for 1-10000 words)
            if stats["verbosity_range"][0] != float("inf"):
                assert 0 <= stats["verbosity_range"][0] <= 4, \
                    f"{model_dir.name}: Verbosity min out of range: {stats['verbosity_range'][0]}"
                assert 0 <= stats["verbosity_range"][1] <= 4, \
                    f"{model_dir.name}: Verbosity max out of range: {stats['verbosity_range'][1]}"
            
            # Check noise range (0-1 ratio)
            if stats["noise_range"][0] != float("inf"):
                assert 0 <= stats["noise_range"][0] <= 1, \
                    f"{model_dir.name}: Noise min out of range: {stats['noise_range'][0]}"
                assert 0 <= stats["noise_range"][1] <= 1, \
                    f"{model_dir.name}: Noise max out of range: {stats['noise_range'][1]}"
            
            # Check word count range (reasonable for clinical responses)
            if stats["word_count_range"][0] != float("inf"):
                assert stats["word_count_range"][0] >= 0, \
                    f"{model_dir.name}: Word count min negative: {stats['word_count_range'][0]}"
                assert stats["word_count_range"][1] <= 10000, \
                    f"{model_dir.name}: Word count max too high: {stats['word_count_range'][1]}"

