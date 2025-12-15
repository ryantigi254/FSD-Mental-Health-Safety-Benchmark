## `src/` layout (Mac-setup)

The canonical Python package is:

- `src/reliable_clinical_benchmark/`

Everything under that package uses imports like `reliable_clinical_benchmark.models...`.

Notes:

- `src/tests/` contains **manual smoke scripts** (not pytest).
- `__pycache__/` directories are runtime artefacts created by Python and should be ignored.


