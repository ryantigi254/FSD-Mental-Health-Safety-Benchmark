"""Validate extracted data for completeness and correctness."""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))


def validate_extracted_file(file_path: Path) -> Dict[str, Any]:
    """Validate a single extracted JSONL file."""
    issues = []
    stats = {
        "total_entries": 0,
        "missing_fields": defaultdict(int),
        "invalid_values": [],
        "extraction_methods": defaultdict(int),
        "refusal_count": 0,
        "extraction_failed_count": 0,
        "has_old_format": False,
        "has_new_format": False,
        "verbosity_range": [float("inf"), float("-inf")],
        "noise_range": [float("inf"), float("-inf")],
        "word_count_range": [float("inf"), float("-inf")],
    }
    
    required_fields = [
        "id",
        "mode",
        "model_name",
        "status",
        "is_refusal",
        "extracted_diagnosis",
        "extraction_method",
        "response_verbosity",
        "format_noise_score",
        "word_count",
    ]
    
    old_format_fields = ["output_complexity", "complexity_features"]
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                    stats["total_entries"] += 1
                    
                    # Check for old format
                    has_old = any(field in entry for field in old_format_fields)
                    has_new = all(field in entry for field in ["response_verbosity", "format_noise_score", "word_count"])
                    
                    if has_old:
                        stats["has_old_format"] = True
                    if has_new:
                        stats["has_new_format"] = True
                    
                    # Check required fields
                    for field in required_fields:
                        if field not in entry:
                            stats["missing_fields"][field] += 1
                            issues.append(f"Line {line_num}: Missing field '{field}'")
                    
                    # Validate field values
                    if "extraction_method" in entry:
                        stats["extraction_methods"][entry["extraction_method"]] += 1
                    
                    if entry.get("is_refusal"):
                        stats["refusal_count"] += 1
                    
                    if entry.get("extracted_diagnosis") in ("EXTRACTION_FAILED", "NO_OUTPUT", "REFUSAL"):
                        stats["extraction_failed_count"] += 1
                    
                    # Check numeric fields
                    if "response_verbosity" in entry:
                        verb = entry["response_verbosity"]
                        if isinstance(verb, (int, float)):
                            stats["verbosity_range"][0] = min(stats["verbosity_range"][0], verb)
                            stats["verbosity_range"][1] = max(stats["verbosity_range"][1], verb)
                        else:
                            issues.append(f"Line {line_num}: Invalid response_verbosity type: {type(verb)}")
                    
                    if "format_noise_score" in entry:
                        noise = entry["format_noise_score"]
                        if isinstance(noise, (int, float)):
                            stats["noise_range"][0] = min(stats["noise_range"][0], noise)
                            stats["noise_range"][1] = max(stats["noise_range"][1], noise)
                        else:
                            issues.append(f"Line {line_num}: Invalid format_noise_score type: {type(noise)}")
                    
                    if "word_count" in entry:
                        wc = entry["word_count"]
                        if isinstance(wc, (int, float)):
                            stats["word_count_range"][0] = min(stats["word_count_range"][0], wc)
                            stats["word_count_range"][1] = max(stats["word_count_range"][1], wc)
                        else:
                            issues.append(f"Line {line_num}: Invalid word_count type: {type(wc)}")
                    
                    # Check for invalid extraction methods
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
                    if "extraction_method" in entry and entry["extraction_method"] not in valid_methods:
                        if not entry["extraction_method"].startswith("heuristic_fallback_"):
                            issues.append(f"Line {line_num}: Unknown extraction_method: {entry['extraction_method']}")
                
                except json.JSONDecodeError as e:
                    issues.append(f"Line {line_num}: JSON decode error: {e}")
    
    except FileNotFoundError:
        issues.append(f"File not found: {file_path}")
    except Exception as e:
        issues.append(f"Error reading file: {e}")
    
    # Fix infinity values
    if stats["verbosity_range"][0] == float("inf"):
        stats["verbosity_range"] = [0, 0]
    if stats["noise_range"][0] == float("inf"):
        stats["noise_range"] = [0, 0]
    if stats["word_count_range"][0] == float("inf"):
        stats["word_count_range"] = [0, 0]
    
    return {
        "issues": issues,
        "stats": stats,
        "is_valid": len(issues) == 0 and stats["has_new_format"] and not stats["has_old_format"],
    }


def main():
    """Validate all extracted files."""
    script_dir = Path(__file__).parent
    uni_setup_root = script_dir.parent.parent.parent
    processed_dir = uni_setup_root / "processed" / "study_a_extracted"
    
    if not processed_dir.exists():
        print(f"ERROR: Processed directory not found: {processed_dir}")
        return 1
    
    model_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
    
    if not model_dirs:
        print("ERROR: No model directories found")
        return 1
    
    print(f"Validating {len(model_dirs)} models...\n")
    print("=" * 80)
    
    all_valid = True
    summary = {}
    
    for model_dir in sorted(model_dirs):
        model_name = model_dir.name
        extracted_file = model_dir / "study_a_extracted.jsonl"
        
        print(f"\nModel: {model_name}")
        print("-" * 80)
        
        if not extracted_file.exists():
            print(f"  [FAIL] File not found: {extracted_file}")
            all_valid = False
            continue
        
        result = validate_extracted_file(extracted_file)
        summary[model_name] = result
        
        stats = result["stats"]
        issues = result["issues"]
        
        # Print statistics
        print(f"  Total entries: {stats['total_entries']}")
        print(f"  Format: {'[OK] New format' if stats['has_new_format'] else '[FAIL] Missing new format'}")
        if stats['has_old_format']:
            print(f"  [WARN] Contains old format fields")
        
        if stats['missing_fields']:
            print(f"  [FAIL] Missing fields: {dict(stats['missing_fields'])}")
        else:
            print(f"  [OK] All required fields present")
        
        print(f"\n  Extraction Methods:")
        for method, count in sorted(stats['extraction_methods'].items()):
            pct = (count / stats['total_entries'] * 100) if stats['total_entries'] > 0 else 0
            print(f"    {method}: {count} ({pct:.1f}%)")
        
        print(f"\n  Metrics:")
        print(f"    Refusals: {stats['refusal_count']} ({stats['refusal_count']/stats['total_entries']*100:.1f}%)")
        print(f"    Extraction failed: {stats['extraction_failed_count']} ({stats['extraction_failed_count']/stats['total_entries']*100:.1f}%)")
        print(f"    Response verbosity: {stats['verbosity_range'][0]:.3f} - {stats['verbosity_range'][1]:.3f}")
        print(f"    Format noise score: {stats['noise_range'][0]:.4f} - {stats['noise_range'][1]:.4f}")
        print(f"    Word count: {int(stats['word_count_range'][0])} - {int(stats['word_count_range'][1])}")
        
        if issues:
            print(f"\n  [FAIL] Issues found ({len(issues)}):")
            for issue in issues[:10]:  # Show first 10 issues
                print(f"    {issue}")
            if len(issues) > 10:
                print(f"    ... and {len(issues) - 10} more issues")
            all_valid = False
        else:
            print(f"\n  [OK] No issues found")
        
        if not result["is_valid"]:
            all_valid = False
    
    print("\n" + "=" * 80)
    print("\nSUMMARY")
    print("=" * 80)
    
    if all_valid:
        print("[OK] All models validated successfully!")
        print("\nAll files have:")
        print("  [OK] New format fields (response_verbosity, format_noise_score, word_count)")
        print("  [OK] No old format fields")
        print("  [OK] All required fields present")
        print("  [OK] Valid extraction methods")
        print("  [OK] Proper data types")
    else:
        print("[FAIL] Validation issues found. Please review the output above.")
        print("\nModels with issues:")
        for model_name, result in summary.items():
            if not result["is_valid"]:
                print(f"  [FAIL] {model_name}: {len(result['issues'])} issues")
    
    return 0 if all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

