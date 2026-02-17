## Tests (`tests/`) vs smoke scripts (`src/tests/`)

This repo has **two** test locations on purpose:

- **`tests/`**: pytest-driven tests.
  - **`tests/unit/`**: fast, deterministic tests that do **not** require GPU / model weights / LM Studio.
  - **`tests/integration/`**: slower tests that may require optional dependencies and/or local data.

- **`src/tests/`**: ad-hoc smoke scripts and capture helpers (manual runs).
  - These are not part of pytest because they can require LM Studio running and/or model downloads.


