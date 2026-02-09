# Walkthrough: Metric Pipeline Updates

I have addressed the critical risks identified in the supervisory review by updating the metric calculation scripts and utilities.

## Changes Overview

### 1. Statistical Rigor (Bootstrap CIs)
**Status**: ✅ Implemented

- **New File**: `src/reliable_clinical_benchmark/metrics/stats.py`
  - Implements `compute_bootstrap_ci` using 1000 resamples (default) and percentiles.
- **Updated**: `scripts/studies/study_a/metrics/calculate_metrics.py`
  - Now calculates 95% Confidence Intervals for:
    - Faithfulness Gap
    - CoT Accuracy
    - Early Answer Accuracy
    - Step-F1
  - Adds fields `_ci_low` and `_ci_high` to the output JSONs in `metric-results/`.

### 2. Study C Continuity Score
**Status**: ✅ Enabled

- **Updated**: `scripts/studies/study_c/metrics/calculate_metrics.py`
  - Now loads `data/study_c_gold/target_plans.json` (verified as present).
  - Imports `calculate_continuity_score` from `drift.py`.
  - Calculates the score for each case and averages it.
  - Adds `continuity_score` to output JSONs.
- **Note**: Requires `sentence-transformers`. If missing, the code gracefully handles failure (returns `None`), but for the final run, ensure it is installed.

### 3. Study B Sycophancy Agreement Logic
**Status**: ✅ Improved

- **Updated**: `src/reliable_clinical_benchmark/metrics/sycophancy.py`
  - Replaced simple "agree" string match with robust logic.
  - Excluding "disagree", "not agree", and nuanced phrases like "You are correct that [X] but...".
  - Explicitly checking for negation.
- **Environment**: Fixed NLI model loading by resolving `pydantic`/`transformers` conflicts in `mh-llm-benchmark-env`.

## How to Verify

### 1. Environment Setup
The environment `mh-llm-benchmark-env` has been recreated and patched.
```powershell
conda activate mh-llm-benchmark-env
```

### 2. Run Pipeline
Re-run the metric calculations to regenerate the JSON files:

```powershell
# Study A (Faithfulness)
python scripts/studies/study_a/metrics/calculate_metrics.py

# Study B (Sycophancy)
python scripts/studies/study_b/metrics/calculate_metrics.py

# Study C (Drift)
python scripts/studies/study_c/metrics/calculate_metrics.py
```

### 3. Check Outputs
- `metric-results/study_a/*_metrics.json`: Check for `faithfulness_gap_ci_low/high`.
- `metric-results/study_b/sycophancy_metrics.json`: Check that `evidence_hallucination` is verified (likely 0.0 if agreement is low).
- `metric-results/study_c/drift_metrics.json`: Check for `continuity_score`.

## Supervisor Checklist Coverage
- [x] **Audit of Folder Structure**: Scripts correctly read from `results/` or `processed/` and write to `metric-results/`.
- [x] **Confidence Intervals**: implemented and serialized.
- [x] **Study C Gap**: Addressed by enabling the Continuity Score with existing data.
- [x] **Sycophancy Logic**: Hardened against false positives; NLI model loading verified.
