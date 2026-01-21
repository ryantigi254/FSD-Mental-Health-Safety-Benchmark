# Metric Updates (Reproducibility + Correctness)

This note summarizes targeted metric fixes made to ensure results are accurate, reproducible, and not misleading.

## Study C (Longitudinal Drift): Continuity Score

- The Continuity Score implementation (`calculate_continuity_score()` in `metrics/drift.py`) requires `target_plan` data.
- Study C gold target plans can be populated deterministically from OpenR1-Psy therapist reasoning (`counselor_think`) into `data/study_c_gold/target_plans.json`.
- The pipeline computes `continuity_score` when at least one gold plan is available.
- The pipeline does not report a misleading `0.0` value:
  - `continuity_score` is treated as missing when not computed (e.g., no gold plan available).
  - The saved results JSON omits `continuity_score` entirely when it is missing.

## Study B (Sycophancy): Evidence Hallucination ($H_{Ev}$)

- `$H_{Ev}$` is computed using an NLI model and can fail for individual samples.
- Sampling is deterministic (fixed index order starting from 0) to ensure reproducibility.
- A small failure buffer is used to increase the chance of collecting 50 successful scores.
- The results JSON includes diagnostic counters:
  - `evidence_hallucination_n_attempted`
  - `evidence_hallucination_n_scored`

## Study A (Faithfulness): Step-F1 strict one-to-one matching

- Step-F1 enforces one-to-one matching between predicted steps and gold steps.
- Repeated model steps cannot receive multiple credit for a single gold step.
- A unit test was added to prevent score inflation from repeated steps.
