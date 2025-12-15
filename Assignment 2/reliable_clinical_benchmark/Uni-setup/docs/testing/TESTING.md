# Testing (Uni-setup)

This repo intentionally separates **pytest tests** from **manual smoke scripts**.

## Pytest tests (`tests/`)

Pytest tests are for correctness and regression prevention:

- **Unit tests**: `tests/unit/`
  - Must be fast and deterministic.
  - Must not require GPU, local weights, or LM Studio.
  - Examples: reasoning extraction helpers, metric calculations, data split invariants.

- **Integration tests**: `tests/integration/`
  - Exercise pipeline wiring and higher-level components.
  - May skip if optional dependencies are missing.

Run:

```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
$Env:PYTHONPATH="src"
pytest tests/unit -v
pytest tests/integration -v
```

## Smoke tests (`src/tests/`)

Smoke tests are simple scripts that prove a runner can:

- load a model (local HF), or connect to LM Studio,
- generate output,
- optionally detect `<think>...</think>` blocks.

They are intentionally **not** part of the pytest suite because they can:

- take minutes,
- require GPU/VRAM,
- require model weights present locally,
- require LM Studio running.

Examples:

- `src/tests/test_piaget_local.py`
- `src/tests/test_psyche_r1_local.py`
- `src/tests/test_psych_qwen_local.py`
- `src/tests/test_psyllm_gml_local.py`
- `src/tests/test_qwen3_lmstudio.py`
- `src/tests/test_lmstudio_capture_qwq.py`
- `src/tests/test_lmstudio_capture_deepseek_r1_14b.py`


