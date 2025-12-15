## Tests (`tests/`) vs smoke scripts (`src/tests/`)

This repo has **two** test locations on purpose:

- **`tests/`**: pytest-driven tests.
  - **`tests/unit/`**: fast, deterministic tests that do **not** require GPU / model weights / LM Studio.
  - **`tests/integration/`**: slower tests that may require optional dependencies and/or local data.

- **`src/tests/`**: ad-hoc **smoke tests** and capture scripts.
  - These are meant for manual runs (often to confirm LM Studio connectivity or local HF inference).
  - Some of these scripts can use the GPU and/or load models.


