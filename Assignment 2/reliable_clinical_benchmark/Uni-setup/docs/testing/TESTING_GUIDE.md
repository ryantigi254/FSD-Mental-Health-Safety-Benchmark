# Testing Guide for All Studies

## Overview

This guide covers smoke tests and unit tests for Study A (Faithfulness + Bias), Study B (Sycophancy), and Study C (Longitudinal Drift).

## Smoke Tests

Smoke tests verify that:
- Model connections work
- Data loading works
- Generation pipeline works
- Cache files are written correctly

### Study A Bias Smoke Tests (Per Model)

#### QwQ-32B (LM Studio)
```powershell
cd Uni-setup
python src/tests/studies/study_a/lmstudio/bias/test_study_a_bias_qwq.py
```

#### DeepSeek-R1 (LM Studio distill)
```powershell
python src/tests/studies/study_a/lmstudio/bias/test_study_a_bias_deepseek_r1_lmstudio.py
```

#### GPT-OSS-20B (LM Studio)
```powershell
python src/tests/studies/study_a/lmstudio/bias/test_study_a_bias_gpt_oss.py
```

#### Qwen3-8B (LM Studio)
```powershell
python src/tests/studies/study_a/lmstudio/bias/test_study_a_bias_qwen3_lmstudio.py
```

#### PsyLLM (HF local, GMLHUHE/PsyLLM)
```powershell
python src/tests/studies/study_a/models/bias/test_study_a_bias_psyllm_gml_local.py
```

#### Piaget-8B (HF local)
```powershell
python src/tests/studies/study_a/models/bias/test_study_a_bias_piaget_local.py
```

#### Psyche-R1 (HF local)
```powershell
python src/tests/studies/study_a/models/bias/test_study_a_bias_psyche_r1_local.py
```

#### Psych-Qwen-32B (HF local, 4-bit)
```powershell
python src/tests/studies/study_a/models/bias/test_study_a_bias_psych_qwen_local.py
```

#### Generic (any model)
```powershell
cd Uni-setup
python src/tests/studies/study_a/test_study_a_bias_generate_only.py --model-id qwen3_lmstudio --max-cases 3
```

### Study A Generations Smoke Tests (Per Model)

#### QwQ-32B (LM Studio)
```powershell
cd Uni-setup
python src/tests/studies/study_a/lmstudio/generations/test_study_a_generation_qwq.py
```

#### DeepSeek-R1 (LM Studio distill)
```powershell
python src/tests/studies/study_a/lmstudio/generations/test_study_a_generation_deepseek_r1_lmstudio.py
```

#### GPT-OSS-20B (LM Studio)
```powershell
python src/tests/studies/study_a/lmstudio/generations/test_study_a_generation_gpt_oss.py
```

#### Qwen3-8B (LM Studio)
```powershell
python src/tests/studies/study_a/lmstudio/generations/test_study_a_generation_qwen3_lmstudio.py
```

#### PsyLLM (HF local, GMLHUHE/PsyLLM)
```powershell
python src/tests/studies/study_a/models/generations/test_study_a_generation_psyllm_gml_local.py
```

#### Piaget-8B (HF local)
```powershell
python src/tests/studies/study_a/models/generations/test_study_a_generation_piaget_local.py
```

#### Psyche-R1 (HF local)
```powershell
python src/tests/studies/study_a/models/generations/test_study_a_generation_psyche_r1_local.py
```

#### Psych-Qwen-32B (HF local, 4-bit)
```powershell
python src/tests/studies/study_a/models/generations/test_study_a_generation_psych_qwen_local.py
```

### Study B Smoke Tests (Per Model)

#### QwQ-32B (LM Studio)
```powershell
cd Uni-setup
python src/tests/studies/study_b/lmstudio/test_study_b_qwq.py
```

#### DeepSeek-R1 (LM Studio distill)
```powershell
python src/tests/studies/study_b/lmstudio/test_study_b_deepseek_r1_lmstudio.py
```

#### GPT-OSS-20B (LM Studio)
```powershell
python src/tests/studies/study_b/lmstudio/test_study_b_gpt_oss.py
```

#### Qwen3-8B (LM Studio)
```powershell
python src/tests/studies/study_b/lmstudio/test_study_b_qwen3_lmstudio.py
```

#### PsyLLM (HF local, GMLHUHE/PsyLLM)
```powershell
python src/tests/studies/study_b/models/test_study_b_psyllm_gml_local.py
```

#### Piaget-8B (HF local)
```powershell
python src/tests/studies/study_b/models/test_study_b_piaget_local.py
```

#### Psyche-R1 (HF local)
```powershell
python src/tests/studies/study_b/models/test_study_b_psyche_r1_local.py
```

#### Psych-Qwen-32B (HF local, 4-bit)
```powershell
python src/tests/studies/study_b/models/test_study_b_psych_qwen_local.py
```

### Study C Smoke Tests (Per Model)

#### PsyLLM (LM Studio)
```powershell
cd Uni-setup
python src/tests/studies/study_c/lmstudio/test_study_c_psyllm.py
```

#### PsyLLM (HF local, GMLHUHE/PsyLLM)
```powershell
python src/tests/studies/study_c/models/test_study_c_psyllm_gml_local.py
```

#### QwQ-32B (LM Studio)
```powershell
python src/tests/studies/study_c/lmstudio/test_study_c_qwq.py
```

#### DeepSeek-R1 (LM Studio distill)
```powershell
python src/tests/studies/study_c/lmstudio/test_study_c_deepseek_r1_lmstudio.py
```

#### GPT-OSS-20B (LM Studio)
```powershell
python src/tests/studies/study_c/lmstudio/test_study_c_gpt_oss.py
```

#### Qwen3-8B (LM Studio)
```powershell
python src/tests/studies/study_c/lmstudio/test_study_c_qwen3_lmstudio.py
```

#### Piaget-8B (HF local)
```powershell
python src/tests/studies/study_c/models/test_study_c_piaget_local.py
```

#### Psyche-R1 (HF local)
```powershell
python src/tests/studies/study_c/models/test_study_c_psyche_r1_local.py
```

#### Psych-Qwen-32B (HF local, 4-bit)
```powershell
python src/tests/studies/study_c/models/test_study_c_psych_qwen_local.py
```

### Generic Smoke Tests (Any Model)

For Study B and C, you can also use the generic tests with `--model-id`:
```powershell
python src/tests/studies/study_b/test_study_b_generate_only.py --model-id qwen3_lmstudio --max-samples 1
python src/tests/studies/study_c/test_study_c_generate_only.py --model-id qwen3_lmstudio --max-cases 1
```

**Output**: Creates smoke test cache files in `results/{model-id}/` with `.smoke-{timestamp}.jsonl` suffix.

## Unit Tests

Unit tests verify cache file structure and schema correctness using dummy models (no actual API calls).

### Study A Bias
```powershell
cd Uni-setup
pytest tests/unit/study_a/test_study_a_bias_generate_only_cache.py -v
```

### Study B
```powershell
cd Uni-setup
pytest tests/unit/study_b/test_study_b_generate_only_cache.py -v
```

### Study C
```powershell
cd Uni-setup
pytest tests/unit/study_c/test_study_c_generate_only_cache.py -v
```

### Run All Unit Tests
```powershell
cd Uni-setup
pytest tests/unit/study_*/test_*_generate_only_cache.py -v
```

## Test Files Structure

```
Uni-setup/
├── src/tests/
│   └── studies/
│       ├── study_a/
│       │   ├── lmstudio/                              [LM Studio model tests]
│       │   │   ├── bias/                              [4 bias tests]
│       │   │   │   ├── test_study_a_bias_qwq.py
│       │   │   │   ├── test_study_a_bias_deepseek_r1_lmstudio.py
│       │   │   │   ├── test_study_a_bias_gpt_oss.py
│       │   │   │   └── test_study_a_bias_qwen3_lmstudio.py
│       │   │   └── generations/                       [4 generation tests]
│       │   │       ├── test_study_a_generation_qwq.py
│       │   │       ├── test_study_a_generation_deepseek_r1_lmstudio.py
│       │   │       ├── test_study_a_generation_gpt_oss.py
│       │   │       └── test_study_a_generation_qwen3_lmstudio.py
│       │   ├── models/                                [Local/HF model tests]
│       │   │   ├── bias/                               [4 bias tests]
│       │   │   │   ├── test_study_a_bias_psyllm_gml_local.py
│       │   │   │   ├── test_study_a_bias_piaget_local.py
│       │   │   │   ├── test_study_a_bias_psyche_r1_local.py
│       │   │   │   └── test_study_a_bias_psych_qwen_local.py
│       │   │   └── generations/                        [4 generation tests]
│       │   │       ├── test_study_a_generation_psyllm_gml_local.py
│       │   │       ├── test_study_a_generation_piaget_local.py
│       │   │       ├── test_study_a_generation_psyche_r1_local.py
│       │   │       └── test_study_a_generation_psych_qwen_local.py
│       │   ├── test_study_a_bias_generate_only.py    [Generic]
│       │   └── __init__.py
│       ├── study_b/
│       │   ├── lmstudio/                              [4 LM Studio model tests]
│       │   │   ├── test_study_b_qwq.py
│       │   │   ├── test_study_b_deepseek_r1_lmstudio.py
│       │   │   ├── test_study_b_gpt_oss.py
│       │   │   └── test_study_b_qwen3_lmstudio.py
│       │   ├── models/                                [4 Local/HF model tests]
│       │   │   ├── test_study_b_psyllm_gml_local.py
│       │   │   ├── test_study_b_piaget_local.py
│       │   │   ├── test_study_b_psyche_r1_local.py
│       │   │   └── test_study_b_psych_qwen_local.py
│       │   ├── test_study_b_generate_only.py        [Generic]
│       │   └── __init__.py
│       └── study_c/
│           ├── lmstudio/                              [LM Studio model tests]
│           │   ├── test_study_c_psyllm.py
│           │   ├── test_study_c_qwq.py
│           │   ├── test_study_c_deepseek_r1_lmstudio.py
│           │   ├── test_study_c_gpt_oss.py
│           │   └── test_study_c_qwen3_lmstudio.py
│           ├── models/                                [Local/HF model tests]
│           │   ├── test_study_c_psyllm_gml_local.py
│           │   ├── test_study_c_piaget_local.py
│           │   ├── test_study_c_psyche_r1_local.py
│           │   └── test_study_c_psych_qwen_local.py
│           ├── test_study_c_generate_only.py        [Generic]
│           └── __init__.py
└── tests/unit/
    ├── study_a/
    │   └── test_study_a_bias_generate_only_cache.py
    ├── study_b/
    │   └── test_study_b_generate_only_cache.py
    └── study_c/
        └── test_study_c_generate_only_cache.py
```

## What Each Test Validates

### Smoke Tests
- ✅ Model runner can be instantiated
- ✅ Data files can be loaded
- ✅ Model.generate() works (actual API call)
- ✅ Cache files are written with correct structure
- ✅ Basic error handling works

### Unit Tests
- ✅ Cache file schema is correct
- ✅ Required fields are present
- ✅ Field types are correct
- ✅ Variant values are valid
- ✅ Timestamps and run_ids are present
- ✅ No actual API calls (uses dummy model)

## Running Tests Before Full Generation

**Recommended workflow**:
1. Run smoke test for your specific model to verify connection:
   ```powershell
   # Example for Study B with QwQ
   python src/tests/studies/study_b/lmstudio/test_study_b_qwq.py
   
   # Example for Study C with Qwen3
   python src/tests/studies/study_c/lmstudio/test_study_c_qwen3_lmstudio.py
   ```
2. If smoke tests pass, proceed with full generation runs
3. Run unit tests to verify cache structure:
   ```powershell
   pytest tests/unit/study_*/test_*_generate_only_cache.py -v
   ```

## Troubleshooting

### Smoke Test Fails
- Check model connection (LM Studio running? HF model path correct?)
- Verify data files exist in expected locations
- Check model ID is correct
- Review error messages for specific issues

### Unit Test Fails
- Check cache file structure matches expected schema
- Verify all required fields are present
- Check field types match expectations
- Review test output for specific validation failures

## Notes

- Smoke tests use minimal samples (1-3) to keep runtime short
- Unit tests use dummy models (no API calls) for fast execution
- Both test types validate cache file structure
- Smoke tests verify end-to-end functionality
- Unit tests verify schema correctness

