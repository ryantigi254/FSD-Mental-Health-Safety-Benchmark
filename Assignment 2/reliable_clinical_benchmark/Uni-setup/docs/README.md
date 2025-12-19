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
  - `studies_summary.md` - **Benchmark Overview** - High-level summary of all studies, models, and workflow
  - See `testing/TESTING_GUIDE.md` - Comprehensive testing guide for all studies
  - `study_a/` - Study A documentation
    - `study_a_faithfulness.md` - Study A (Faithfulness) overview
    - `study_a_bias.md` - Silent Bias Rate (R_SB) evaluation: setup, commands, and workflow
  - `study_b/` - Study B documentation
    - `study_b_sycophancy.md` - Study B (Sycophancy) implementation guide
    - `study_b_commands.md` - Model-specific generation commands
  - `study_c/` - Study C documentation
    - `study_c_drift.md` - Study C (Longitudinal Drift) overview
    - `study_c_commands.md` - Model-specific generation commands

- **`models/`** - Model documentation
  - `MODEL_RUNNERS.md` - Model runner implementations and usage

- **`testing/`** - Testing documentation
  - `TESTING.md` - Testing strategy and guidelines

- **`environment/`** - Environment setup
  - `ENVIRONMENT.md` - Two-environment setup (general benchmark + local HF inference)
    - `mh-llm-benchmark-env`: For LM Studio runners, evaluation, tests
    - `mh-llm-local-env`: For local PyTorch models (Piaget, Psyche-R1, etc.) requiring modern transformers

- **`evaluation/`** - Evaluation protocols
  - `EVALUATION_PROTOCOL.md` - Evaluation procedures and protocols

## Quick Links

### Benchmark Overview
- **[Benchmark Overview](studies/studies_summary.md)** - High-level summary of all studies, models, and workflow

### Study A (Faithfulness)
- [Study Overview](studies/study_a/study_a_faithfulness.md)
- [Bias Evaluation](studies/study_a/study_a_bias.md)
- [Gold Labels Mapping](data/STUDY_A_GOLD_LABELS_MAPPING.md)
- [Metrics Pipeline](metrics/METRIC_CALCULATION_PIPELINE.md)
- [Metrics Reference](metrics/QUICK_REFERENCE.md)

### Study B (Sycophancy)
- [Study Overview](studies/study_b/study_b_sycophancy.md)
- [Generation Commands](studies/study_b/study_b_commands.md)

### Study C (Longitudinal Drift)
- [Study Overview](studies/study_c/study_c_drift.md)
- [Generation Commands](studies/study_c/study_c_commands.md)

### General
- [Model Runners](models/MODEL_RUNNERS.md) - ModelRunner interface (`generate`, `chat`, `generate_with_reasoning`)
- [Testing Guide](testing/TESTING_GUIDE.md) - Smoke tests and unit tests for all studies
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

**Generation Scripts** (standalone, per-model):
- **Study A Bias**: `hf-local-scripts/run_study_a_bias_generate_only.py`
  - Generates CoT responses for adversarial bias cases
  - Output: `processed/study_a_bias/{model-id}/study_a_bias_generations.jsonl`
- **Study B**: `hf-local-scripts/run_study_b_generate_only.py`
  - Generates single-turn (control + injected) and multi-turn responses
  - Output: `results/{model-id}/study_b_generations.jsonl`
- **Study C**: `hf-local-scripts/run_study_c_generate_only.py`
  - Generates summary + dialogue variants for multi-turn cases
  - Output: `results/{model-id}/study_c_generations.jsonl`

**Metric Calculation Scripts**:
- **Study A Gold Labels**: `scripts/study_a/gold_labels/`
- **Study A Metrics**: `scripts/study_a/metrics/`
  - `calculate_metrics.py` - Main metrics (faithfulness, accuracy, step F1)
  - `calculate_bias.py` - Silent Bias Rate (R_SB)
- **Main Evaluation**: `scripts/run_evaluation.py`
  - Supports `from_cache` mode for calculating metrics from cached generations

### Results

**Generation Cache Files**:
- **Study A Main**: `results/{model-id}/study_a_generations.jsonl`
  - Contains CoT and Direct mode responses per sample
- **Study A Bias**: `processed/study_a_bias/{model-id}/study_a_bias_generations.jsonl`
  - Contains CoT responses for adversarial bias cases
  - Saved in `processed/study_a_bias/` directory (separate from main Study A generations)
- **Study B**: `results/{model-id}/study_b_generations.jsonl`
  - Contains single-turn (control + injected) and multi-turn responses
- **Study C**: `results/{model-id}/study_c_generations.jsonl`
  - Contains summary + dialogue variants per turn

**Calculated Metrics**:
- **Study A**: `metric-results/all_models_metrics.json` (includes `study_a_bias_metrics.json`)
- **Study B**: `results/{model-id}/study_b_results.json`
- **Study C**: `results/{model-id}/study_c_results.json`

## Getting Started

1. **Read Overview**: Start with [Benchmark Overview](studies/studies_summary.md) for a high-level understanding
2. **Setup Environment**: See [Environment Setup](environment/ENVIRONMENT.md)
3. **Understand Architecture**: See `src/README.md` for package structure and study architectures
4. **Understand Studies**: Read study-specific docs in `studies/`
5. **Run Generations**: Use generation scripts in `hf-local-scripts/` (see study command docs)
6. **Calculate Metrics**: Use `from_cache` mode or metric calculation scripts
7. **Run Tests**: See [Testing Guide](testing/TESTING_GUIDE.md) for smoke tests and unit tests

## Architecture Overview

For detailed architecture information, see:
- **Package Structure**: `src/README.md` - Complete package layout and ModelRunner interface
- **Test Structure**: `tests/README.md` - Pytest unit tests and integration tests
- **Study Architectures**: `src/README.md` - Single-turn/multi-turn separation, generation modes, variants

