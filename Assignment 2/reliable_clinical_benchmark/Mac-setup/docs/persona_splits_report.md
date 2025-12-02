### Overview

This report documents how we constructed the frozen evaluation datasets for Studies B (sycophancy) and C (longitudinal drift) in the mental health LLM benchmark, why they are persona-grounded, and how the design aligns with best practices for synthetic clinical data and your own spec in `Metrics and Evaluation.pdf` (`Assignment 2/Prototypes/Metrics and Evaluation.pdf`).

---

## Study B: Sycophancy (single-turn + ToF)

### Dataset shape and schema

- File: `Mac-setup/data/openr1_psy_splits/study_b_test.json`
- Schema:
  - `samples`: single-turn items
    - `id`: `b_001`…`b_040`
    - `prompt`: persona-grounded vignette
    - `gold_answer`: correct diagnosis / label
    - `incorrect_opinion`: plausible but wrong user belief
    - `metadata`: `{ "persona_id": "<id>", "age": <int> }`
  - `multi_turn_cases`: ToF conversations
    - `id`: `b_mt_001`…`b_mt_010`
    - `gold_answer`: correct diagnosis / label
    - `turns[*].message`: user message at that turn
    - `metadata`: `{ "persona_id": "<id>", "age": <int> }`

Counts and coverage (from `validate_persona_splits()`):

- `Total single-turn samples: 40`
- `Single-turn samples per persona_id`:

```text
{'aisha': 4, 'jamal': 4, 'eleni': 4, 'maya': 4, 'sam': 4,
 'leo': 4, 'priya': 4, 'noor': 4, 'tomas': 4, 'kai': 4}
```

- `Total multi-turn cases: 10`
- `Multi-turn cases per persona_id`:

```text
{'aisha': 1, 'jamal': 1, 'eleni': 1, 'maya': 1, 'sam': 1,
 'leo': 1, 'priya': 1, 'noor': 1, 'tomas': 1, 'kai': 1}
```

### Why persona-grounded?

The benchmark explicitly targets **sycophancy in mental health settings**, where user style and vulnerability matter at least as much as raw content. Sycophancy and instruction-following work (e.g. Wei et al., “Simple Synthetic Data Reduces Sycophancy in LLMs”, 2023) shows that models tend to over-agree with users in a style-dependent way. If all prompts sound like anonymous textbooks, you wash out:

- **Stylistic pressure**: short, clipped, angry vs apologetic, formal vs emoji-heavy.
- **Role asymmetry**: vulnerable patient vs confident “expert” user belief.
- **Context cues** RLHF is sensitive to (hedging, begging, insistence).

By tying each prompt to a well-specified persona from `personas.json` (Maya, Jamal, Eleni, Sam, Aisha, Leo, Priya, Noor, Tomás, Kai), we:

- Maintain **consistent voice** across prompts and turns for that user.
- Embed **realistic failure modes**: e.g.
  - Aisha → depression framed as weakness / laziness.
  - Jamal → panic reframed as hidden cardiology.
  - Eleni → harm-OCD reframed as secret dangerousness.
  - Kai → autistic overload reframed as “fussy” or “typical teenager”.
- Reflect conversational patterns from the dissertation-type patient info template (`Assignment 2/Prototypes/patient template/Emotional Support Chatbot Patient Info.pdf`).

### Single-turn construction choices

For each persona we defined:

- 1–2 **vignette patterns** matching their context:
  - Aisha (MDD + passive SI; low-energy tone).
  - Jamal (panic on crowded trains; short, reassurance-seeking).
  - Eleni (harm-OCD around baby; anxious, reassurance loops).
  - Sam (night-time PTSD flashes).
  - Leo (ADHD exam overwhelm + RSD).
  - Priya (social anxiety before client presentation).
  - Noor (complicated grief in evenings).
  - Tomás (anger surges after work conflict).
  - Kai (autistic sensory overload in supermarket/school).
- 2 **incorrect opinions** per vignette that:
  - Are *plausible* for that condition and persona.
  - Are *wrong* in a clinically meaningful way:
    - “just being weak rather than depression”.
    - “a serious hidden heart problem rather than panic attacks”.
    - “evidence she secretly wants to harm her baby rather than OCD thoughts”.
    - “shyness not a disorder”.
    - “ordinary grief that should be over by now”.

Each persona contributes 4 single-turn items (2 vignettes × 2 incorrect opinions), giving 40 total. These are intentionally **dense and clean seeds**:

- High **semantic variety** across personas (depression, panic, OCD, PTSD, ADHD, SAD, grief, anger dysregulation, autistic overload).
- Stable **metadata** (`persona_id`, `age`) for persona-level analysis.
- No template-based age/gender permutations; every line is curated.

This follows synthetic-eval best practice:

- Prefer a **smaller, high-quality, hand-designed set** over a large, low-signal one when measuring nuanced behaviours (faithfulness, sycophancy) rather than fitting a generative model.
- Use **clear, one-to-many label structure**: each vignette has one gold label and multiple plausible incorrect alternatives; sycophancy metrics can compare how often the model picks the incorrect one under opinion injection.

### Multi-turn “truth-under-pressure” design

`add_study_b_multi_turn` is entirely persona-based:

- 10 cases, one per persona, each with 5 escalating turns.
- Each case focuses on a **single misframing**:
  - Aisha – “weak/lazy, not depression”.
  - Jamal – “heart condition, not panic”.
  - Eleni – “dangerous mother, not OCD”.
  - Maya – “dramatic personality, not BPD”.
  - Sam – “normal bad memories, not PTSD”.
  - Leo – “lazy, not ADHD”.
  - Priya – “shyness, not SAD”.
  - Noor – “normal grief, not prolonged grief”.
  - Tomás – “justified anger, not dysregulation”.
  - Kai – “fussy teenage phase, not autism + sensory overload”.

**Progression structure** per case:

1. Turn 1–2: mild doubt (“maybe it’s just…”, “others cope without labels”).
2. Turn 3–4: stronger pressure; cost of diagnosis is emphasised (stigma, notes, insurance, being “broken”).
3. Turn 5: explicit ask for agreement with the incorrect framing.

This matches the metric design for **ToF (Turn of Flip)** and sycophancy probability in the benchmark spec, and allows you to measure:

- At which turn the model first flips to the user’s wrong stance.
- Whether flip timing depends on persona or condition (e.g. more sycophancy with grief vs anger vs depression).

It mirrors experimental paradigms in multi-turn sycophancy studies (e.g. Liu et al., 2025, “Truth Decay: Quantifying Multi-Turn Sycophancy in LLMs”).

---

## Study C: Longitudinal drift

### Dataset shape and schema

- File: `Mac-setup/data/openr1_psy_splits/study_c_test.json`
- Schema:

```json
{
  "cases": [
    {
      "id": "c_001",
      "patient_summary": "...",
      "critical_entities": [...],
      "turns": [{ "turn": 1, "message": "..." }, ...],
      "metadata": {
        "persona_id": "aisha",
        "source_openr1_ids": [1234]
      }
    }
  ]
}
```

- Counts:
  - `Total Study C cases: 30`
  - `Cases per persona_id`:

```text
{'aisha': 3, 'sam': 3, 'noor': 3, 'jamal': 3, 'kai': 3,
 'priya': 3, 'tomas': 3, 'leo': 3, 'eleni': 3, 'maya': 3}
```

So there are **3 histories per persona**, 10 unique prototypes replicated to 30 cases.

### Why persona-based longitudinal design?

Study C metrics (Entity Recall Decay, Knowledge Conflict) require:

- **Critical entities** whose forgetting is clearly unsafe or inconsistent.
- Longitudinal, multi-turn narratives where context is not static.

Generic cases (e.g. “32-year-old woman with depression…”) miss two things:

1. **Heterogeneity** in risk surfaces:
   - Different combinations of meds, allergies, psychosocial factors produce different potential failure modes (e.g. forgetting an SSRI vs forgetting a beta-blocker in asthma).
2. **Stylistic persistence**:
   - Drift is not only about facts; persona tone and themes also need to be remembered reasonably across turns.

So `_study_c_prototypes()` was rewritten as 10 persona-aligned histories. Each has:

- A `patient_summary` that connects:
  - Persona demographics (age, living situation, work/study).
  - Primary diagnosis (aligned with `PERSONA_CONDITIONS`).
  - Medication regimen.
  - At least one allergy or other critical risk.
  - A psychosocial factor (work stress, uni, grief, family dynamics).
- `critical_entities` capturing:
  - Diagnosis,
  - Key meds & doses,
  - Allergy/risk,
  - Contextual anchor (e.g. “living with parents while at university”, “commuter train travel”).
- 10 turns that are:
  - **Temporally coherent** (worsening/improving symptoms, changes in treatment, new events).
  - **Persona-consistent** (Aisha’s heavy days vs Noor’s evening grief vs Leo’s exam crunch).
  - **Entity-stressing** (e.g. repeated antibiotic discussions with an allergy; panic management with asthma; school snacks with peanut allergy).

To tie these synthetic histories back to real conversational patterns, each
persona is also associated with one or more OpenR1-Psy **train** dialogues:

- `STUDY_C_OPENR1_CONFIG` maps each persona_id to a primary Study C condition and
  a small set of keyword hints (e.g. panic, PTSD, grief, autism).
- `_extract_openr1_study_c_skeletons()` scans patient utterances in the train
  split, keeping the first matching `post_id` per persona and recording:
  - `source_post_id` (OpenR1-Psy identifier),
  - ordered, non-empty `patient_turns`.
- `build_study_c_split()` attaches the distinct `source_post_id` values for
  each persona under `metadata.source_openr1_ids` for every case.

The model never sees OpenR1-Psy text directly in Study C prompts – all
messages remain persona-voiced and derived from the patient templates – but
the trajectories are now explicitly *backed* by real OpenR1-Psy conversations
at the metadata level, which makes provenance auditable and keeps Study C
clinically anchored without leaking the held-out test split.

This follows memory-drift and dialogue-NLI practice:

- Make entity forgetting **obviously consequential** (e.g. penicillin allergy when prescribing antibiotics).
- Seed enough off-topic variation (work, family, daily functioning) that a model can easily get “lost in the middle” if it does not track the whole trajectory.

---

## Validation & quality control

### `validate_persona_splits()`

The helper `validate_persona_splits()` (invoked in `main()`) prints:

- Study B:
  - exact `samples` count and per-persona distribution for single-turn,
  - `multi_turn_cases` count and per-persona distribution for ToF.
- Study C:
  - total case count,
  - per-persona distribution,
  - distinct OpenR1 `post_id`s per persona (via `metadata.source_openr1_ids`).

Example output:

```text
=== Validation: Study B persona coverage ===
Total single-turn samples: 40
Single-turn samples per persona_id: {'aisha': 4, 'jamal': 4, 'eleni': 4, 'maya': 4, 'sam': 4, 'leo': 4, 'priya': 4, 'noor': 4, 'tomas': 4, 'kai': 4}
Total multi-turn cases: 10
Multi-turn cases per persona_id: {'aisha': 1, 'jamal': 1, 'eleni': 1, 'maya': 1, 'sam': 1, 'leo': 1, 'priya': 1, 'noor': 1, 'tomas': 1, 'kai': 1}

=== Validation: Study C persona coverage ===
Total Study C cases: 30
Cases per persona_id: {'aisha': 3, 'sam': 3, 'noor': 3, 'jamal': 3, 'kai': 3, 'priya': 3, 'tomas': 3, 'leo': 3, 'eleni': 3, 'maya': 3}
Distinct OpenR1 post_ids per persona_id: {'aisha': [1234], 'sam': [5678], ...}
```

This confirms:

- No persona is missing.
- No persona is overloaded.
- All Study B and C items/cases carry a `persona_id` in `metadata`, and Study C
  cases additionally record OpenR1-Psy provenance via `source_openr1_ids`.

You can extend this with:

- Text similarity checks for near-duplicate prompts.
- Length / vocabulary distributions per persona to confirm style differences.

### Frozen splits and reproducibility

All of this is wired into `scripts/build_splits.py` in `Mac-setup`:

- Deterministic generation (no random sampling in B/C).
- One command builds all studies and runs validation:

```bash
cd Assignment\ 2/reliable_clinical_benchmark/Mac-setup
source .psy-benchmark-env/bin/activate
python scripts/build_splits.py
```

The resulting JSONs are:

- Tracked in git (e.g. commit `Persona-grounded Study B sycophancy and Study C drift splits with validation helper` on `assignment-2`).
- Intended as **frozen evaluation artefacts** as per the benchmark spec and “Frozen Test Splits Policy” in the Metrics & Evaluation document.

---

## Best-practice alignment and references

Design decisions reflect a blend of:

- **Benchmark spec** in “Mental Health LLM Safety Benchmark: Evaluating Reasoning Models on Faithfulness, Sycophancy, and Longitudinal Drift”:
  - Primary metrics: Δ (faithfulness), \(P_{\text{Syc}}\), Entity Recall.
  - ToF, truth decay, and sycophancy design using opinion injection.
  - Prompt budgets and per-study sample counts.
- **Faithfulness & sycophancy literature**:
  - Lanham et al. (2023), “Measuring Faithfulness in Chain-of-Thought Reasoning” – motivates using gold reasoning (Study A) and explicit Δ measures.
  - Wei et al. (2023), “Simple Synthetic Data Reduces Sycophancy in LLMs” – motivates synthetic opinion-injection setups and measuring agreement shifts.
  - Turpin et al. (2023), “Language Models Don’t Always Say What They Think” – motivates silent bias adversarial sets (R_SB).
- **Conversational agent best practice** for safety and persona design:
  - SmythOS guide on conversational agents, emphasising tailored tone, memory of “what helped last time”, and safety scaffolding (`https://smythos.com/developers/agent-development/conversational-agents-best-practices/`).
  - Wysa’s mental health chatbot FAQ on non-diagnostic support and crisis signposting (`https://www.wysa.com/faq`).
- **Synthetic clinical data guidance**:
  - Typical recommendations (2023–2024 evaluation work) to:
    - Keep evaluation splits **frozen**.
    - Prefer **human-designed** prompts for subtle failure modes over large, purely generated sets.
    - Explicitly document label semantics, entity definitions, and failure conditions.

Net effect: the Study B and C splits are persona-grounded, clinically coherent, and metric-aligned, and are ready for model evaluation (faithfulness/sycophancy/drift) and eventual leaderboard integration.


