# Personas (synthetic, template-derived)

Synthetic personas generated from the patient template at `../Dissertation/Prototypes/docs/general/patient-template.json` and aligned with safety guidance in `../Dissertation/Prototypes/docs/general/personas_safety.md`, `personas.json`, and `personas.md`. All content here is synthetic and for benchmarking only; no real patient data is stored.

## Structure
- `persona_registry_v2.json`: master list of persona ids and descriptors used by splits.
- `<persona>/patient.json`: filled template instance (demographics, accessibility, self-reported context, safety, consent, optional modules).
- `<persona>/messages.json`: scripted user turns for replay tests.
- `<persona>/safety_plan.md`: lightweight safety micro-plan for that persona.

## Generation notes
- Personas mirror the template fields exactly; when adding a new persona, start from `patient-template.json`, keep British English, and avoid clinical identifiers or real-world data.
- Safety plans and crisis contacts follow the reference in `personas_safety.md`; keep them UK-aligned (e.g., 999, Samaritans 116123, SHOUT 85258).
- All personas are synthetic; they may reference conditions for realism but are not derived from any individual.

