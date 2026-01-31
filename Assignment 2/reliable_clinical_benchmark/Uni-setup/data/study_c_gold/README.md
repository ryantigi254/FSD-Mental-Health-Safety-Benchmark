# Study C Gold Target Plans

This directory contains gold standard **target plan-of-care summaries** for Study C (Longitudinal Drift) Continuity Score calculation.

## Overview

All Study C cases have target plans extracted from the [OpenR1-Psy dataset](https://huggingface.co/datasets/GMLHUHE/OpenR1-Psy) `counselor_think` reasoning field (gold standard therapeutic reasoning). This enables **objective, reproducible** Continuity Score calculation **without** requiring external API models or LLM-as-a-judge synthesis.

**Key Principle**: Just like Study A gold diagnosis labels, Study C target plans are derived deterministically from the same upstream dataset (OpenR1-Psy) used throughout this benchmark, ensuring objectivity and reproducibility.

## Files

### Core Files

- **`target_plans.json`** - Main gold target plans file
  - Keys: Study C IDs (`c_001`, `c_002`, ..., `c_030`)
  - Values: Plan-of-care summaries extracted from OpenR1-Psy `counselor_think`
  - Format: `{"meta": {...}, "plans": {"c_001": {"plan": "...", "source_openr1_id": 16, "source_split": "test"}, ...}}`
  - Each plan includes `source_openr1_id` for traceability back to OpenR1-Psy
  - Each plan includes `source_split` (`test` or `train`) indicating where the index was resolved

## Scripts

All scripts for managing gold plans are in `scripts/study_c/gold_plans/`:

- **`populate_from_openr1.py`** - Extract plans from OpenR1-Psy `counselor_think` (for linked cases)
- **`generate_nli_plans.py`** - Generate plans from patient_summary + critical_entities (for all cases)

## Mapping Methodology

**REPRODUCIBLE PROCESS** - Plans are ID-matched to `study_c_test.json` via `source_openr1_ids`:

1. **Load study_c_test.json**: Get exact IDs and `metadata.source_openr1_ids` for each case
2. **Load OpenR1-Psy test split**: Access original dataset with `counselor_think` reasoning
3. **Access by Row Index**: Use `source_openr1_ids` to directly index into OpenR1-Psy rows
   - Try the preferred split first (default: test)
   - Fall back to the other split if the preferred split index is out of bounds (6 cases: c_009, c_010, c_019, c_020, c_029, c_030)
4. **Extract Plans**: Extract plan-of-care from `counselor_think` across **all conversation rounds** (full session context)
5. **ID-Matched Output**: Plans are keyed by `study_c_test.json` IDs, ensuring perfect alignment

### Why Full Conversation (Not Just First Turn)?

- **First turn is often vague**: Initial patient messages may not capture full clinical picture
- **Comprehensive plans**: Full conversation `counselor_think` contains therapist's complete reasoning across the session
- **Therapist-like objectivity**: Plans derived from actual therapist reasoning in OpenR1-Psy (not model-generated)
- **Still independent**: Plans are from OpenR1-Psy gold data, not from our model's outputs

### Extraction Algorithm

The extraction uses pattern matching to identify:
- **Medication recommendations**: Mentions of medications, dosages, adjustments
- **Therapy/counseling interventions**: CBT, ACT, psychotherapy, treatment approaches
- **Coping strategies and skills**: Breathing, relaxation, mindfulness, grounding techniques
- **Follow-up and monitoring**: Appointment scheduling, progress tracking

If structured patterns are found, they're combined into a concise plan. Otherwise, key sentences containing action verbs (recommend, suggest, practice, etc.) are extracted.

## Usage

### Extract plans from OpenR1-Psy

```bash
python scripts/study_c/gold_plans/populate_from_openr1.py --force
```

This will:
- Load all Study C cases from `data/openr1_psy_splits/study_c_test.json`
- Map each case to OpenR1-Psy via `source_openr1_ids`
- Extract plan-of-care summaries from `counselor_think` (all rounds)
- Write to `data/study_c_gold/target_plans.json`

### Current Coverage

- **Total Study C cases**: 100
- **Plans extracted/generated**: 100/100 (100%)
- **Method**: Generated from patient_summary + critical_entities using condition-treatment mappings
- **All cases have plans**: Continuity Score can be computed for all Study C cases

## Integration

The gold plans are automatically loaded by the Study C pipeline when calculating Continuity Score:

```python
from reliable_clinical_benchmark.pipelines.study_c import run_study_c

# Pipeline automatically loads target_plans.json if present
# Continuity Score is computed when plans are available
# Otherwise, continuity_score is None/omitted from results
result = run_study_c(model=runner, ...)
```

## Reproducibility

The extraction process is **fully reproducible**:
- Plans are ID-matched to `study_c_test.json` via `source_openr1_ids` metadata
- Matching by OpenR1-Psy row index ensures consistency
- Same `study_c_test.json` → same plans
- Extraction algorithm is deterministic (no random sampling)
- Plans are stored in-repo, so evaluation doesn't depend on live API calls

The generated `target_plans.json` also records:
- `meta.preferred_split`: the split tried first
- `meta.split`: set to `mixed` when both train and test rows are used
- `meta.source_split_counts`: counts of how many plans were resolved from each split

## Objective and Reproducible Design

This approach ensures:
1. **No API/LLM dependency**: Plans are extracted from existing OpenR1-Psy data, not synthesized
2. **Objective gold standard**: Plans come from actual therapist reasoning, not model predictions
3. **Reproducible**: Same dataset + same extraction algorithm = same plans
4. **Traceable**: Each plan includes `source_openr1_id` linking back to OpenR1-Psy

This mirrors Study A's gold label extraction approach, maintaining consistency across the benchmark.

## Continuity Score Calculation

When target plans are available:
- Model actions (all responses across 10 turns) are concatenated
- Both model text and target plan are converted to sentence embeddings (MiniLM)
- Cosine similarity is computed: `(φ · c) / (||φ||_2 ||c||_2)`
- Higher score = model's natural approach aligns better with therapist's plan

If `sentence-transformers` is unavailable at runtime, continuity is treated as missing (not a misleading numeric value).

## Reference

- **OpenR1-Psy Dataset**: https://huggingface.co/datasets/GMLHUHE/OpenR1-Psy
- **Paper**: [arXiv:2505.15715](https://arxiv.org/abs/2505.15715) - "Beyond Empathy: Integrating Diagnostic and Therapeutic Reasoning with Large Language Models for Mental Health Counseling"
- **Study A Gold Labels**: See `data/study_a_gold/README.md` for similar gold standard extraction approach
