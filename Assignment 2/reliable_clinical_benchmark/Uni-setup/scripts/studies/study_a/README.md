# Study A Scripts

Scripts for Study A (Faithfulness) evaluation, organised by function.

## Directory Structure

```
scripts/study_a/
├── gold_labels/     # Gold diagnosis label management
└── metrics/        # Metric calculation and analysis
```

## Gold Labels (`gold_labels/`)

Scripts for managing gold diagnosis labels extracted from OpenR1-Psy:

- **`populate_from_openr1.py`** - Extract labels from OpenR1-Psy `counselor_think` reasoning
- **`verify_mapping.py`** - Verify mapping to OpenR1-Psy dataset
- **`verify_id_matching.py`** - Verify ID matching to `study_a_test.json`
- **`manual_label.py`** - Interactive manual labelling CLI
- **`init_labels.py`** - Initialise empty labels file
- **`populate_remaining.py`** - Populate remaining unlabelled cases with defaults
- **`extract_suggestions.py`** - Extract diagnosis suggestions from OpenR1-Psy
- **`label_with_openai.py`** - Alternative: Use OpenAI API to extract diagnoses from prompts (backup method)

See `data/study_a_gold/README.md` for details on gold label files.

## Metrics (`metrics/`)

Scripts for calculating and analysing Study A metrics:

- **`calculate_metrics.py`** - Calculate faithfulness metrics from cached generations
- **`extract_predictions.py`** - Extract model predictions from generations
- **`analyze_extraction.py`** - Analyse model outputs for extractability, refusal rate, and complexity

## Quick Start

### Extract gold labels from OpenR1-Psy

```bash
python scripts/study_a/gold_labels/populate_from_openr1.py --force
python scripts/study_a/gold_labels/verify_id_matching.py
```

### Calculate metrics

```bash
python scripts/study_a/metrics/calculate_metrics.py
```

### Manual labelling

```bash
python scripts/study_a/gold_labels/manual_label.py --only-unlabeled --show-consensus
```

### Analyse model outputs

```bash
python scripts/study_a/metrics/analyze_extraction.py --results-dir results
```

### Alternative labelling (OpenAI API)

```bash
python scripts/study_a/gold_labels/label_with_openai.py --model gpt-4o-mini
```

