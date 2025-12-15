## `src/` layout (Uni-setup)

The canonical Python package is:

- `src/reliable_clinical_benchmark/`

Everything under that package uses absolute imports like `reliable_clinical_benchmark.models...`.

Notes:

- `src/tests/` contains **manual smoke tests** (not pytest) for LM Studio / local HF runners.
- `__pycache__/` directories are runtime artefacts created by Python and should be ignored.


