# Metric Calculation Quick Reference

## File Locations

### Input Files (Raw Generations)
- **Location**: `Uni-setup/results/{model_name}/study_a_generations.jsonl`
- **Format**: JSONL (one JSON object per line)
- **Fields**: `id`, `mode` ("cot" or "direct"), `output_text`, `status`, etc.
- **Never Modified**: These files are read-only for metric calculation

### Gold Standard Data
- **Location**: `Uni-setup/data/openr1_psy_splits/study_a_test.json`
- **Format**: JSON with `{"samples": [...]}`
- **Fields per sample**: `id`, `prompt`, `gold_answer`, `gold_reasoning`
- **Frozen**: Never modified (ensures reproducibility)

### Output Files (Calculated Metrics)
- **Location**: `Uni-setup/metric-results/`
- **Individual**: `{model_name}_metrics.json`
- **Combined**: `all_models_metrics.json`
- **Format**: JSON with metric values

## Calculation Script

**Script**: `Uni-setup/scripts/calculate_study_a_metrics.py`

**Usage**:
```bash
cd Uni-setup
python scripts/calculate_study_a_metrics.py
```

**What it does**:
1. Reads raw generations from `results/`
2. Loads gold data from `data/openr1_psy_splits/study_a_test.json`
3. Calculates metrics using objective functions
4. Saves to `metric-results/`

## Metric Functions (Implementation)

### Core Functions
- **`_is_correct_diagnosis(response, gold)`**: Checks if response matches gold answer
  - Location: `src/reliable_clinical_benchmark/metrics/faithfulness.py:91`
  - Rules: Exact match + abbreviation mapping

- **`extract_reasoning_steps(text)`**: Extracts reasoning steps from CoT output
  - Location: `src/reliable_clinical_benchmark/metrics/faithfulness.py:242`
  - Rules: REASONING:...DIAGNOSIS: markers, min 20 tokens, sentence splitting

- **`calculate_step_f1(model_steps, gold_steps)`**: Computes Step-F1 score
  - Location: `src/reliable_clinical_benchmark/metrics/faithfulness.py:115`
  - Formula: F1 = 2PR/(P+R) where P = matched/predicted, R = matched/gold

- **`normalize_text(text)`**: Normalises text for comparison
  - Location: `src/reliable_clinical_benchmark/metrics/utils.py:8`
  - Rules: Lowercase, remove punctuation, normalise whitespace

- **`compute_token_overlap(text1, text2)`**: Dice coefficient
  - Location: `src/reliable_clinical_benchmark/metrics/utils.py:22`
  - Formula: Dice = 2 × |A ∩ B| / (|A| + |B|)

## LaTeX Specification Alignment

| Metric | LaTeX Section | Equation | Implementation File |
|--------|---------------|----------|-------------------|
| Faithfulness Gap | 6.1 | Equation 1 | `metrics/faithfulness.py:31` |
| Step-F1 | 6.2 | Equation 2 | `metrics/faithfulness.py:115` |
| Silent Bias Rate | 6.3 | Equation 3 | `metrics/faithfulness.py:187` |

## Verification

**Unit Tests**: `tests/unit/test_faithfulness_metrics.py`
- 38 tests covering all metric functions
- Verifies formula correctness
- Tests edge cases

**Run Tests**:
```bash
cd Uni-setup
python -m pytest tests/unit/test_faithfulness_metrics.py -v
```

## Example Metric Output

```json
{
  "faithfulness_gap": 0.0,
  "acc_cot": 0.0,
  "acc_early": 0.0,
  "step_f1": 0.09749378298765646,
  "n_samples": 300,
  "n_step_f1_samples": 300,
  "correct_cot": 0,
  "correct_early": 0
}
```

**Interpretation**:
- `faithfulness_gap = 0.0`: No difference between CoT and Early accuracy
- `step_f1 = 0.097`: Low reasoning quality (9.7% overlap with gold)
- `n_samples = 300`: All 300 samples processed successfully

