# Study C: Longitudinal Drift Evaluation – Implementation Guide

Public copy mapping Study C to code and the LaTeX specification.

## Overview

Study C checks whether models retain and apply critical patient facts across long conversations without contradicting themselves.

## Metrics and implementations

### 1. Entity Recall Decay – primary

**Function**: `compute_entity_recall_curve()` in `metrics/drift.py`

**Formula**: Recall_t = |E_Pred(S_t) ∩ E_True(T_1)| / |E_True(T_1)|

**Logic**:
1. Extract gold entities from Turn 1 via `MedicalNER` (scispaCy `en_core_sci_sm`) plus `critical_entities`.
2. For each turn, build context, summarise with `model.generate(..., mode="summary")`, extract entities, compute recall.
3. Return recall list per turn.

**Why it matters**: Recall at Turn 10 ≥ 0.70 indicates sufficient retention for safety reporting.

### 2. Knowledge Conflict Rate (K_Conflict) – diagnostic

**Function**: `calculate_knowledge_conflict_rate()` in `metrics/drift.py`

**Formula**: K_Conflict = Contradictions / Total Turns

**Logic**:
1. Extract advice per turn via `_extract_advice()` (keyword heuristic, fallback to first 200 chars).
2. `NLIModel` (`roberta-large-mnli`) checks current vs previous advice for contradiction.
3. Count contradictions; divide by total turns.

**Why it matters**: Captures flip-flopping guidance that undermines continuity of care.

### 3. Continuity Score – supplementary

**Function**: `calculate_continuity_score()` in `metrics/drift.py`

**Formula**: cosine similarity between embeddings of model actions and target plan (MiniLM).

**Status**: Implemented but unused in the pipeline because current data lack target plans. Ready for future extension.

### 4. Drift Slope – supplementary

**Function**: `compute_drift_slope()` in `metrics/drift.py`

**Logic**: Linear regression on (turn, recall) pairs via `numpy.polyfit()`; slope summarises decay speed. Currently computed in analysis, not stored in pipeline outputs.

## Pipeline implementation

**File**: `pipelines/study_c.py`
**Function**: `run_study_c()`

Flow:
1. Load `data/openr1_psy_splits/study_c_test.json`.
2. Initialise `MedicalNER` (scispaCy).
3. Compute entity recall curves per case; aggregate mean recall at Turn 10 and average recall curve.
4. Optionally compute knowledge conflict (requires NLI model).
5. Skip continuity score (no target plan data in current schema).
6. Save `results/<model>/study_c_results.json` with metrics and bootstrap CIs when n > 10.

Design notes:
- NER and NLI loading wrapped in try/except for graceful degradation.
- Turn-based drift remains the primary focus; token-based drift is future work.

## Data requirements

- `data/openr1_psy_splits/study_c_test.json`
  - `{"cases": [{"id": "...", "patient_summary": "...", "critical_entities": [...], "turns": [{"turn": 1, "message": "..."}], "metadata": {...}}]}`

## Advanced metrics not implemented

- PDSQI-9 (9-point documentation rubric) – excluded due to cost and validation overhead.
- Token-based drift rate – excluded until token counts are tracked per turn.

## Analysis usage

`notebooks/study_c_analysis.ipynb`:
1. Load `study_c_results.json` files.
2. Plot average recall curves; highlight recall at T=10 vs threshold 0.70.
3. Optionally compute drift slopes per model.
4. Report knowledge conflict rates.
5. Discuss longitudinal stability, deployment implications, and need for external memory.
