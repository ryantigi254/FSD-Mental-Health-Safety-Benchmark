### Overview

Frozen Mac evaluation splits with v2 personas. Persona definitions live at `../Uni-setup/docs/personas/persona_registry_v2.json`.

---

## Study A: Faithfulness (context)

- File: `Mac-setup/data/openr1_psy_splits/study_a_test.json`
- 300 samples; current JSON lacks persona metadata (counts confirmed on 2025-12-06).
- Bias stressors: `data/adversarial_bias/biased_vignettes.json` (58 items).

---

## Study B: Sycophancy (single-turn + ToF)

- File: `Mac-setup/data/openr1_psy_splits/study_b_test.json`
- Shape:
  - `samples` (single-turn): id, prompt, gold_answer, incorrect_opinion, metadata.persona_id, metadata.age
  - `multi_turn_cases` (ToF): id, gold_answer, turns[*].message, metadata.persona_id, metadata.age
- Coverage (validated 2025-12-06):
  - Single-turn: 40 total, 4 each across 10 personas (aisha, jamal, eleni, maya, sam, leo, priya, noor, tomas, kai)
  - Multi-turn: 10 total, one per persona (same set)

---

## Study C: Longitudinal drift

- File: `Mac-setup/data/openr1_psy_splits/study_c_test.json`
- Shape: `cases` with `id`, `patient_summary`, `critical_entities`, `turns`, optional `metadata.persona_id`.
- Coverage (validated 2025-12-06):
  - 30 cases total, 3 per persona across the same 10 personas used in Study B.

---

## Validation & provenance

- Deterministic generation; canonical files: `study_b_test.json`, `study_c_test.json`.
- Archives: `data/openr1_psy_splits/archive/study_b_test_v1.json` and `.../study_c_test_v1.json`; reference v2 copies remain at `study_b_test_v2.json` and `study_c_test_v2.json`.
- Rebuild/validate:

```bash
cd Assignment\ 2/reliable_clinical_benchmark/Mac-setup
source .mh-llm-benchmark-env/bin/activate
python scripts/build_splits.py
```

---

## Notes for evaluation

- Study B personas align with the v2 registry; keep LM Studio/runner persona handling consistent.
- Study C longitudinal cases rely on persona-linked `critical_entities`; metric checks depend on scispaCy + NLI readiness.

