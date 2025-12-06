# Study C: Longitudinal Drift – Implementation Guide (concise copy)

Public summary of Study C implementation.

## Metrics

- **Entity Recall Decay** (`compute_entity_recall_curve`): scispaCy entities at Turn 1 vs each turn; target recall at T10 > 0.70.
- **Knowledge Conflict Rate** (`calculate_knowledge_conflict_rate`): NLI contradictions between consecutive advice turns (`roberta-large-mnli`).
- **Continuity Score** (`calculate_continuity_score`): MiniLM cosine similarity vs target plan; implemented, unused (no plan data yet).
- **Drift Slope** (`compute_drift_slope`): linear slope of recall over turns; used in analysis, not stored.

## Pipeline

- File: `pipelines/study_c.py`
- Flow: load `data/openr1_psy_splits/study_c_test.json`; compute recall curves; aggregate mean recall at T10 and average curve; optionally compute knowledge conflict; skip continuity score; save `results/<model>/study_c_results.json` with bootstrap CIs.
- NER/NLI loading wrapped in try/except for robustness.

## Data

- `cases` with `patient_summary`, `critical_entities`, `turns`, and optional metadata.

## Analysis

- Plots and tables in `notebooks/study_c_analysis.ipynb` (recall curves, conflict rates, optional drift slopes).
