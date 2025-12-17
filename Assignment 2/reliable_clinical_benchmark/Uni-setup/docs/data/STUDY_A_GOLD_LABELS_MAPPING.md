# Study A Gold Diagnosis Labels - Mapping to OpenR1-Psy

## Overview

All 300 Study A vignettes have been assigned gold diagnosis labels extracted from the [OpenR1-Psy dataset](https://huggingface.co/datasets/GMLHUHE/OpenR1-Psy) `counselor_think` reasoning field (gold standard therapeutic reasoning).

## Mapping Methodology

**REPRODUCIBLE PROCESS** - Labels are ID-matched to `study_a_test.json` (not just sequential):

1. **Load study_a_test.json**: Get exact IDs and prompts used in the study
2. **Load OpenR1-Psy test split**: Access original dataset with `counselor_think` reasoning
3. **Match by Prompt Text**: Match OpenR1-Psy rows to `study_a_test.json` IDs by exact prompt text matching
4. **Extract Diagnoses**: Labels extracted from `counselor_think` using:
   - Explicit diagnosis mentions (DSM-5/ICD-10 terms)
   - Symptom pattern matching (DSM-5 criteria indicators)
   - Fallback logic for cases with minimal diagnostic information
5. **ID-Matched Output**: Labels are keyed by `study_a_test.json` IDs, ensuring perfect alignment with your study split

### Reproducibility

The process is **fully reproducible**:
- Same `study_a_test.json` → same labels (matching by prompt text)
- Independent of OpenR1-Psy row order
- Traceable to original dataset via `study_a_gold_labels_mapping.json`

## Verification

- **Mapping Accuracy**: 100% (300/300 cases matched to OpenR1-Psy)
- **Label Coverage**: 100% (300/300 cases labeled)
- **Source**: All labels derived from OpenR1-Psy `counselor_think` (gold reasoning), not model predictions

## Label Distribution

| Diagnosis | Count | Percentage |
|-----------|-------|------------|
| Adjustment Disorder | 154 | 51.3% |
| Generalized Anxiety Disorder | 100 | 33.3% |
| Major Depressive Disorder | 28 | 9.3% |
| Post-Traumatic Stress Disorder | 6 | 2.0% |
| Obsessive-Compulsive Disorder | 4 | 1.3% |
| Social Anxiety Disorder | 3 | 1.0% |
| Bipolar Disorder | 2 | 0.7% |
| Attention Deficit Hyperactivity Disorder | 2 | 0.7% |
| Substance Use Disorder | 1 | 0.3% |

## Files

All gold label files are in `data/study_a_gold/`:

- **Gold Labels**: `data/study_a_gold/gold_diagnosis_labels.json`
  - Keys: `study_a_test.json` IDs (`a_001`, `a_002`, ..., `a_300`)
  - Values: Diagnosis labels (DSM-5/ICD-10 standard names)
- **Full Mapping**: `data/study_a_gold/gold_labels_mapping.json`
  - Complete mapping with OpenR1-Psy row indices, post_ids, and prompt previews
- **Suggestions**: `data/study_a_gold/diagnosis_suggestions.json`
  - Diagnosis hints extracted from OpenR1-Psy for manual labeling

## Scripts

All scripts are in `scripts/study_a/gold_labels/`:

- **Extraction**: `scripts/study_a/gold_labels/populate_from_openr1.py`
  - ID-matched extraction (matches to `study_a_test.json` by prompt text)
- **Verification Scripts**:
  - `scripts/study_a/gold_labels/verify_mapping.py` - Verifies mapping to OpenR1-Psy
  - `scripts/study_a/gold_labels/verify_id_matching.py` - Verifies ID matching to `study_a_test.json`
- **Manual Labeling**: `scripts/study_a/gold_labels/manual_label.py`
  - Interactive CLI for manual diagnosis labeling

See `data/study_a_gold/README.md` and `scripts/study_a/README.md` for details.

## Extraction Process

### High-Confidence Extraction (133 cases)
- Explicit diagnosis mentions in `counselor_think`
- Strong symptom patterns matching DSM-5 criteria

### Symptom-Based Extraction (153 cases)
- Symptom clusters indicating specific disorders
- PTSD: trauma, flashbacks, nightmares
- Panic Disorder: panic attacks, sudden fear
- Social Anxiety: fear of judgment, social situations
- OCD: obsessive thoughts, compulsive behaviors
- Bipolar: manic/mania indicators
- Adjustment Disorder: life changes, stressors
- ADHD: attention/hyperactivity indicators

### Fallback Assignment (14 cases)
- Cases with minimal diagnostic information
- Assigned "Adjustment Disorder" as safest default
- Based on presence of emotional distress indicators

## Reference

- **OpenR1-Psy Dataset**: [https://huggingface.co/datasets/GMLHUHE/OpenR1-Psy](https://huggingface.co/datasets/GMLHUHE/OpenR1-Psy)
- **Paper**: [arXiv:2505.15715](https://arxiv.org/abs/2505.15715) - "Beyond Empathy: Integrating Diagnostic and Therapeutic Reasoning with Large Language Models for Mental Health Counseling"

## Reproducibility

To regenerate labels (e.g., after updating `study_a_test.json`):

```bash
python scripts/study_a/gold_labels/populate_from_openr1.py --force
python scripts/study_a/gold_labels/verify_id_matching.py  # Verify ID matching
```

The process is **deterministic and reproducible**:
- Same `study_a_test.json` → same labels (matching by prompt text)
- Labels are ID-matched to your study split, not just sequential
- Independent of OpenR1-Psy dataset iteration order

## Notes

- All labels are extracted from the **gold standard** `counselor_think` reasoning, ensuring objectivity
- Labels are **ID-matched to `study_a_test.json`** (not just sequential order)
- Labels follow DSM-5/ICD-10 diagnostic standards
- Some cases may benefit from clinical review for refinement
- The mapping is traceable: each label links to its OpenR1-Psy source via `study_a_gold_labels_mapping.json`

