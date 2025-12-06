### Overview

Public report on the frozen evaluation datasets for Studies B (sycophancy) and C (longitudinal drift), with persona grounding and quality controls.

---

## Study B: Sycophancy (single-turn + ToF)

### Dataset shape and schema

- File: `Mac-setup/data/openr1_psy_splits/study_b_test.json`
- Schema:
  - `samples` (single-turn):
    - `id`: `b_001`…
    - `prompt`: persona-grounded vignette
    - `gold_answer`: correct diagnosis/label
    - `incorrect_opinion`: plausible but wrong user belief
    - `metadata`: `{ "persona_id": "<id>", "age": <int> }`
  - `multi_turn_cases` (ToF):
    - `id`: `b_mt_001`…
    - `gold_answer`
    - `turns[*].message`: user message per turn
    - `metadata`: `{ "persona_id": "<id>", "age": <int> }`

Counts and coverage (from `validate_persona_splits()`):

- Total single-turn samples: 40
- Single-turn per persona_id (10 personas): 4 each
- Total multi-turn cases: 10 (one per persona)

### Why persona-grounded?

Persona grounding preserves stylistic pressure (tone, insistence, vulnerability) and role asymmetry that affect sycophancy. Each persona has consistent voice and failure modes, enabling per-persona analysis of agreement and flip timing.

### Single-turn construction

Per persona:
- 1–2 vignette patterns matched to context (e.g., depression, panic, OCD, PTSD, ADHD, social anxiety, grief, anger dysregulation, autistic overload).
- Two plausible but incorrect opinions per vignette; each yields a single-turn sample.
- Metadata retains persona_id and age for stratified analysis.

Design choices favour a smaller, high-quality, hand-crafted set over large synthetic permutations to expose nuanced failures.

### Multi-turn “truth-under-pressure” design

- 10 cases, one per persona, 5 turns each.
- Each case centres on a single misframing (e.g., “heart condition, not panic”; “fussy phase, not autism”).
- Pressure escalates across turns; ToF records the first incorrect turn.

This supports measuring when models capitulate and whether timing varies by persona or condition.

---

## Study C: Longitudinal drift

### Dataset shape and schema

- File: `Mac-setup/data/openr1_psy_splits/study_c_test.json`
- Schema:
  - `cases`: histories with `id`, `patient_summary`, `critical_entities`, `turns`, and optional `metadata` (persona_id, provenance identifiers).

Counts:
- Total Study C cases: 30
- Cases per persona_id: 3 each (10 personas)

### Persona-based longitudinal design

Each case is persona-aligned and includes:
- `patient_summary` linking demographics, diagnosis, meds, allergy/risk, and psychosocial context.
- `critical_entities` capturing diagnosis, key meds/doses, allergy/risk, and contextual anchors.
- 10 temporally coherent turns stressing entities (e.g., repeated antibiotic discussion with an allergy; panic management with asthma; sensory overload scenarios).

Persona linkage maintains stylistic cues and varied risk surfaces (meds, allergies, psychosocial factors) so forgetting is clinically meaningful.

---

## Validation & quality control

`validate_persona_splits()` reports:
- Study B: total counts, per-persona distributions for single-turn and multi-turn.
- Study C: total cases, per-persona distribution, optional provenance identifiers.

Example output:
```
=== Validation: Study B persona coverage ===
Total single-turn samples: 40
Single-turn samples per persona_id: {...}
Total multi-turn cases: 10
Multi-turn cases per persona_id: {...}

=== Validation: Study C persona coverage ===
Total Study C cases: 30
Cases per persona_id: {...}
Distinct provenance ids per persona_id: {...}
```

This ensures coverage balance and auditable provenance.

### Frozen splits and reproducibility

- Deterministic generation; no random sampling in B/C.
- Canonical files now point to v2 personas: `data/openr1_psy_splits/study_b_test.json` and `data/openr1_psy_splits/study_c_test.json`.
- Archived v1 splits: `data/openr1_psy_splits/archive/study_b_test_v1.json` and `data/openr1_psy_splits/archive/study_c_test_v1.json`. Reference copies of v2 remain at `data/openr1_psy_splits/study_b_test_v2.json` and `data/openr1_psy_splits/study_c_test_v2.json`.
- Single command builds and validates:

```bash
cd Assignment\ 2/reliable_clinical_benchmark/Mac-setup
source .mh-llm-benchmark-env/bin/activate
python scripts/build_splits.py
```

- JSON outputs are intended as frozen evaluation artefacts; reported results should reference the v2 persona files.
- Study A two-phase option (traceable/resumable):
  - Generate only: `python scripts/run_evaluation.py --study A --model <model> --generate-only --cache-out results/<model>/study_a_generations.jsonl`
  - Metrics from cache: `python scripts/run_evaluation.py --study A --model <model> --from-cache results/<model>/study_a_generations.jsonl`
  - Default remains single-pass (generate + metrics) if no cache flags are provided.

---

## Best-practice alignment (sources)

- Faithfulness, sycophancy, and drift metric design from the benchmark specification.
- Literature grounding: faithfulness (Lanham et al., 2023), sycophancy (Wei et al., 2023; Turpin et al., 2023), drift and dialogue NLI practices.
- Synthetic clinical data guidance: frozen splits, human-designed prompts, documented label semantics and failure conditions.
