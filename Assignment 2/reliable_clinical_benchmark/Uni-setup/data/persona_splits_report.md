### Overview

This report documents how we constructed the frozen evaluation datasets for Studies B (sycophancy) and C (longitudinal drift) in the mental health LLM benchmark, why they are persona-grounded, and how the design aligns with best practices for synthetic clinical data and the benchmark spec in `Assignment 2/Prototypes/Metrics and Evaluation.pdf`.

The content mirrors the report in `Mac-setup/data/persona_splits_report.md` so that Mac-setup and Uni-setup share the same prompt design rationale.

---

## Study B: Sycophancy (single-turn + ToF)

### Dataset shape and schema

- File: `Uni-setup/data/openr1_psy_splits/study_b_test.json`
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

Counts and coverage (from `validate_persona_splits()` in Mac-setup, which mirrors Uni-setup):

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

`add_study_b_multi_turn` in Mac-setup (mirrored in Uni-setup data) is entirely persona-based:

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
- Whether flip timing depends on persona or condition.

---

## Study C: Longitudinal drift

### Dataset shape and schema

- File: `Uni-setup/data/openr1_psy_splits/study_c_test.json`
- Schema:

```json
{
  "cases": [
    {
      "id": "c_001",
      "patient_summary": "...",
      "critical_entities": [...],
      "turns": [{ "turn": 1, "message": "..." }, ...],
      "metadata": { "persona_id": "aisha" }
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

Generic cases miss:

- Variation in risk surfaces (different combos of meds, allergies, psychosocial factors).
- Persistent persona tone and themes.

The 10 persona-aligned histories each include:

- `patient_summary` linking diagnosis, meds, allergy/critical risk, and psychosocial context.
- `critical_entities` listing those explicitly.
- 10 persona-consistent turns that stress those entities over time.

This makes entity forgetting and drift both **plausible** and **consequential**, in line with your benchmark design.

---

## Validation and freezing

- Persona coverage and distributions are validated via `validate_persona_splits()` in Mac-setup.
- Splits are generated deterministically from `scripts/build_splits.py`.
- JSON files in `Uni-setup/data/openr1_psy_splits/` mirror Mac-setup and are treated as **frozen evaluation artefacts** for Studies B and C.


