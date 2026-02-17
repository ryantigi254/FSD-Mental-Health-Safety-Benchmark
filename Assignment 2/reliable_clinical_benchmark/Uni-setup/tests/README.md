## Tests (`tests/`) vs smoke scripts (`src/tests/`)

This repo has **two** test locations on purpose:

- **`tests/`**: pytest-driven tests.
  - **`tests/unit/`**: fast, deterministic tests that do **not** require GPU / model weights / LM Studio.
  - **`tests/integration/`**: slower tests that may require optional dependencies and/or local data.

- **`src/tests/studies/`**: ad-hoc **smoke tests** and capture scripts.
  - These are meant for manual runs (often to confirm LM Studio connectivity or local HF inference).
  - Some of these scripts can use the GPU and/or load models.
  - Organised by study: `study_a/`, `study_b/`, `study_c/`

### Test Structure

```
tests/
├── unit/                          # Fast, deterministic unit tests
│   ├── data/                      # Data loading and validation tests
│   │   ├── test_data_splits_invariants.py
│   │   └── test_gold_label_generation.py
│   ├── metrics/                   # Metric calculation tests
│   │   ├── test_faithfulness_metrics.py
│   │   ├── test_sycophancy_metrics.py
│   │   └── test_drift_metrics.py
│   ├── runners/                   # Model runner interface tests
│   │   ├── test_lmstudio_deepseek_r1_runner.py
│   │   ├── test_lmstudio_qwq_runner.py
│   │   ├── test_lmstudio_qwen3_runner.py
│   │   ├── test_psyllm_gml_local_runner.py
│   │   ├── test_psyche_r1_local_runner.py
│   │   └── test_psych_qwen_local_runner.py
│   ├── study_a/                    # Study A specific tests
│   │   ├── test_study_a_bias_generate_only_cache.py
│   │   ├── test_study_a_metrics_calculation.py
│   │   └── test_extracted_data_validation.py
│   ├── study_b/                    # Study B specific tests
│   │   └── test_study_b_generate_only_cache.py
│   └── study_c/                    # Study C specific tests
│       └── test_study_c_generate_only_cache.py
└── integration/                   # Integration tests (may require GPU/data)
    ├── test_build_splits_deterministic.py
    ├── test_pipelines.py
    └── test_psyche_r1_local_inference.py
```

### Running Tests

**All unit tests** (fast, no GPU required):
```powershell
cd Uni-setup
pytest tests/unit/ -v
```

**Specific study tests**:
```powershell
pytest tests/unit/study_a/ -v
pytest tests/unit/study_b/ -v
pytest tests/unit/study_c/ -v
```

**Integration tests** (may require GPU/model weights):
```powershell
pytest tests/integration/ -v
```

**Specific test categories**:
```powershell
pytest tests/unit/metrics/ -v          # Metric calculation tests
pytest tests/unit/runners/ -v          # Model runner tests
pytest tests/unit/data/ -v             # Data validation tests
```

### Smoke Tests (Manual)

Smoke tests are located in `src/tests/studies/` and are **not** pytest-driven. They are organised by study:

- **Study A**: `src/tests/studies/study_a/`
  - `lmstudio/bias/` - 4 tests (LM Studio models for bias)
  - `lmstudio/generations/` - 4 tests (LM Studio models for main generation)
  - `models/bias/` - 4 tests (local HF models for bias)
  - `models/generations/` - 4 tests (local HF models for main generation)

- **Study B**: `src/tests/studies/study_b/`
  - `lmstudio/` - 4 tests (LM Studio models)
  - `models/` - 4 tests (local HF models)

- **Study C**: `src/tests/studies/study_c/`
  - `lmstudio/` - 4 tests (LM Studio models)
  - `models/` - 4 tests (local HF models)

See `docs/studies/TESTING_GUIDE.md` for detailed smoke test commands.


