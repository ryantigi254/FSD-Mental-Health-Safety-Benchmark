# Study C Gold Target Plans

This directory contains gold standard **target plan-of-care summaries** for Study C (Longitudinal Drift).

These target plans enable the **Continuity Score** to be computed **without** using an external API model or LLM-as-a-judge. The plans are derived deterministically from the same upstream dataset used across this benchmark: **OpenR1-Psy**.

Dataset:
- https://huggingface.co/datasets/GMLHUHE/OpenR1-Psy

## Files

### Core Files

- **`target_plans.json`**
  - Contains gold target plans keyed by Study C case IDs (`c_001`, `c_002`, ...).
  - Format:
    - `{"meta": {...}, "plans": {"c_001": {"plan": "...", "source_openr1_id": 16}, ...}}`

## Scripts

Scripts for producing/updating these plans live in:
- `scripts/study_c/gold_plans/`

Key script:
- **`populate_from_openr1.py`**
  - Loads Study C cases from `data/openr1_psy_splits/study_c_test.json`
  - Uses each case's `metadata.source_openr1_ids` to index back into the OpenR1-Psy dataset
  - Extracts a plan-of-care summary from OpenR1-Psy `counselor_think` (full conversation)
  - Writes the resulting mapping to `data/study_c_gold/target_plans.json`

## Usage

Populate / refresh gold plans:

```bash
python scripts/study_c/gold_plans/populate_from_openr1.py --force
```

## Reproducibility

This process is reproducible because:
- Study C cases include `source_openr1_ids` (row indices into OpenR1-Psy).
- The extraction algorithm is deterministic (pattern + sentence selection; no random sampling).
- The extracted gold plans are stored in-repo in `target_plans.json` so the evaluation does not depend on a live API.

## Notes

- The Continuity Score compares model actions (responses across turns) to the gold plan using sentence embeddings.
- If `sentence-transformers` is unavailable at runtime, continuity is treated as missing rather than defaulting to a misleading numeric value.
