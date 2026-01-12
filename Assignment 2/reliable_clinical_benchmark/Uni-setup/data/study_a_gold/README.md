# Study A Gold Diagnosis Labels

This directory contains all gold diagnosis labels and related files for Study A (Faithfulness).

## Files

### Core Files

- **`gold_diagnosis_labels.json`** - Main gold diagnosis labels file
  - Keys: Study A IDs (`a_001`, `a_002`, ..., `a_300`)
  - Values: DSM-5/ICD-10 diagnosis labels
  - Format: `{"labels": {"a_001": "Generalized Anxiety Disorder", ...}}`

- **`gold_labels_mapping.json`** - Full mapping to OpenR1-Psy dataset
  - Maps each Study A ID to its corresponding OpenR1-Psy row
  - Includes prompt previews, `counselor_think` previews, and verification status
  - Used for traceability and verification

- **`diagnosis_suggestions.json`** - Diagnosis hints extracted from OpenR1-Psy
  - Extracted from `counselor_think` reasoning and patient prompts
  - Used as suggestions during manual labelling
  - Format: `{"a_001": {"diagnosis": "...", "confidence": "high/low", "source": "reasoning/prompt"}}`

## Scripts

All scripts for managing gold labels are in `scripts/study_a/gold_labels/`:

- **`populate_from_openr1.py`** - Extract labels from OpenR1-Psy `counselor_think`
- **`verify_mapping.py`** - Verify mapping to OpenR1-Psy dataset
- **`verify_id_matching.py`** - Verify ID matching to `study_a_test.json`
- **`manual_label.py`** - Interactive manual labelling CLI
- **`init_labels.py`** - Initialise empty labels file
- **`populate_remaining.py`** - Populate remaining unlabelled cases
- **`extract_suggestions.py`** - Extract diagnosis suggestions from OpenR1-Psy
- **`label_with_openai.py`** - Alternative: Use OpenAI API to extract diagnoses (backup method)

## Usage

### Extract labels from OpenR1-Psy

```bash
python scripts/study_a/gold_labels/populate_from_openr1.py --force
```

### Verify ID matching

```bash
python scripts/study_a/gold_labels/verify_id_matching.py
```

### Manual labelling

```bash
python scripts/study_a/gold_labels/manual_label.py --only-unlabeled --show-consensus --show-openr1
```

## Integration

The gold labels are automatically loaded by `study_a_loader.py` when loading Study A data:

```python
from reliable_clinical_benchmark.data.study_a_loader import load_study_a_data

# Automatically loads gold_diagnosis_labels.json if present
vignettes = load_study_a_data("data/openr1_psy_splits/study_a_test.json")
# Each vignette will have 'gold_diagnosis_label' if available
```

## Reproducibility

The extraction process is fully reproducible:
- Labels are ID-matched to `study_a_test.json` (not just sequential)
- Matching by prompt text ensures consistency
- Same `study_a_test.json` → same labels

See `docs/data/STUDY_A_GOLD_LABELS_MAPPING.md` for detailed documentation.

