# Documentation Index

This directory contains comprehensive documentation for the mental health LLM benchmark.

## Structure

### Core Documentation

- **`data/`** - Data documentation
  - `STUDY_A_GOLD_LABELS_MAPPING.md` - Study A gold diagnosis labels mapping and extraction process

- **`metrics/`** - Metrics documentation
  - `METRIC_CALCULATION_PIPELINE.md` - Detailed metric calculation pipeline
  - `QUICK_REFERENCE.md` - Quick reference for all metrics

- **`studies/`** - Study-specific documentation
  - `study_a_faithfulness.md` - Study A (Faithfulness) overview
  - `study_b_sycophancy.md` - Study B (Sycophancy) overview
  - `study_c_drift.md` - Study C (Longitudinal Drift) overview

- **`models/`** - Model documentation
  - `MODEL_RUNNERS.md` - Model runner implementations and usage

- **`testing/`** - Testing documentation
  - `TESTING.md` - Testing strategy and guidelines

- **`environment/`** - Environment setup
  - `ENVIRONMENT.md` - Environment configuration and setup

- **`evaluation/`** - Evaluation protocols
  - `EVALUATION_PROTOCOL.md` - Evaluation procedures and protocols

## Quick Links

### Study A (Faithfulness)
- [Study Overview](studies/study_a_faithfulness.md)
- [Gold Labels Mapping](data/STUDY_A_GOLD_LABELS_MAPPING.md)
- [Metrics Pipeline](metrics/METRIC_CALCULATION_PIPELINE.md)
- [Metrics Reference](metrics/QUICK_REFERENCE.md)

### Study B (Sycophancy)
- [Study Overview](studies/study_b_sycophancy.md)

### Study C (Longitudinal Drift)
- [Study Overview](studies/study_c_drift.md)

### General
- [Model Runners](models/MODEL_RUNNERS.md)
- [Testing Strategy](testing/TESTING.md)
- [Environment Setup](environment/ENVIRONMENT.md)
- [Evaluation Protocol](evaluation/EVALUATION_PROTOCOL.md)

## File Locations

### Data Files
- **Test Splits**: `data/openr1_psy_splits/`
  - `study_a_test.json`, `study_b_test.json`, `study_c_test.json`
- **Gold Labels**: `data/study_a_gold/`
  - `gold_diagnosis_labels.json`, `gold_labels_mapping.json`

### Scripts
- **Study A Gold Labels**: `scripts/study_a/gold_labels/`
- **Study A Metrics**: `scripts/study_a/metrics/`
- **Main Evaluation**: `scripts/run_evaluation.py`

### Results
- **Model Generations**: `results/<model-name>/`
- **Calculated Metrics**: `metric-results/`

## Getting Started

1. **Setup Environment**: See [Environment Setup](environment/ENVIRONMENT.md)
2. **Understand Studies**: Read study-specific docs in `studies/`
3. **Run Evaluations**: See [Evaluation Protocol](evaluation/EVALUATION_PROTOCOL.md)
4. **Calculate Metrics**: See [Metrics Pipeline](metrics/METRIC_CALCULATION_PIPELINE.md)

