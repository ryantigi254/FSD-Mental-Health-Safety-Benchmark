## OpenR1-Psy frozen splits (Study A/B/C)

This directory on the Uni setup mirrors the Mac setup and holds the **frozen
JSON test splits** used by the benchmark pipelines:

- `study_a_test.json` – faithfulness vignettes with gold reasoning
- `study_b_test.json` – sycophancy prompts (control + injected) and optional multi-turn cases
- `study_c_test.json` – longitudinal multi-turn cases for drift

The code expects these files exactly here (paths are relative to the Uni-setup root):

- Study A loader: `data/openr1_psy_splits/study_a_test.json`
- Study B loader: `data/openr1_psy_splits/study_b_test.json`
- Study C loader: `data/openr1_psy_splits/study_c_test.json`

See the Mac-setup `data/openr1_psy_splits/README.md` for the JSON schemas and
an example script for deriving `study_a_test.json` from the OpenR1‑Psy dataset.

Once generated, these splits should be treated as **frozen test sets** for
published experiments.

## Related Files

**Gold diagnosis labels** for Study A are now in a separate directory:
- `data/study_a_gold/` - Contains all Study A gold label files
  - See `data/study_a_gold/README.md` for details

This separation keeps the frozen test splits clean and organized.


