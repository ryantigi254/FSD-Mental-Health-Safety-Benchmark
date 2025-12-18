# Study A Bias Generation Commands

## Overview

Study A bias evaluation measures **Silent Bias Rate (R_SB)** - detecting when models make biased decisions but don't mention the demographic feature in their reasoning.

**File**: `hf-local-scripts/run_study_a_bias_generate_only.py`

**What it does**:
- Validates bias data structure before generating
- Generates CoT responses for adversarial bias cases
- Writes to: `processed/study_a_bias/<model-id>/study_a_bias_generations.jsonl`

**Important**:
- **Each model must be run separately** - one command per model with a unique `--model-id`
- **Each model gets its own output file**: `processed/study_a_bias/{model-id}/study_a_bias_generations.jsonl`
- Output is saved in `processed/study_a_bias/` (separate from main Study A generations in `results/`)
- You cannot run multiple models in a single command

**Note**: This is a separate run from main Study A (faithfulness). Bias cases only need CoT mode, not Direct mode.

## Environment Requirements

Bias generations use the **same environments** as main Study A generations:

- **`mh-llm-benchmark-env`**: For LM Studio models (`qwq`, `deepseek_r1_lmstudio`, `gpt_oss`, `qwen3_lmstudio`)
- **`mh-llm-local-env`**: For local HF models (`psyllm`, `psyllm_gml_local`, `piaget_local`, `psyche_r1_local`, `psych_qwen_local`)

See `docs/environment/ENVIRONMENT.md` for setup instructions.

**Note**: Adjust the Anaconda path (`D:\Anaconda3\Scripts\activate`) in the commands below if your Anaconda installation is in a different location.

## Commands per Model

For each model, run these steps in order:
1. **Activate environment** and set up paths
2. **Run unit tests** (extraction logic - run once, shared across all models)
3. **Run smoke test** (model-specific inference test)
4. **Run full generation** (all bias cases)

---

### PsyLLM (Local HF)

**Environment**: `mh-llm-local-env`

#### 1. Activate Environment
```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-local-env
$Env:PYTHONNOUSERSITE="1"
$Env:PYTHONPATH="src"
```

#### 2. Unit Tests (Run Once - Shared Across All Models)
```powershell
pytest tests/unit/metrics/test_extraction.py -v
```

#### 3. Smoke Test
```powershell
python src/tests/studies/study_a/models/bias/test_study_a_bias_psyllm_gml_local.py
```

#### 4. Full Generation
```powershell
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id psyllm
```

---

### QwQ-32B (LM Studio)

**Environment**: `mh-llm-benchmark-env`

#### 1. Activate Environment
```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-benchmark-env
$Env:PYTHONPATH="src"
```

#### 2. Unit Tests (Run Once - Shared Across All Models)
```powershell
pytest tests/unit/metrics/test_extraction.py -v
```

#### 3. Smoke Test
```powershell
python src/tests/studies/study_a/lmstudio/bias/test_study_a_bias_qwq.py
```

#### 4. Full Generation
```powershell
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id qwq
```

---

### DeepSeek-R1 (LM Studio distill)

**Environment**: `mh-llm-benchmark-env`

#### 1. Activate Environment
```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-benchmark-env
$Env:PYTHONPATH="src"
```

#### 2. Unit Tests (Run Once - Shared Across All Models)
```powershell
pytest tests/unit/metrics/test_extraction.py -v
```

#### 3. Smoke Test
```powershell
python src/tests/studies/study_a/lmstudio/bias/test_study_a_bias_deepseek_r1_lmstudio.py
```

#### 4. Full Generation
```powershell
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id deepseek_r1_lmstudio
```

---

### GPT-OSS-20B (LM Studio)

**Environment**: `mh-llm-benchmark-env`

#### 1. Activate Environment
```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-benchmark-env
$Env:PYTHONPATH="src"
```

#### 2. Unit Tests (Run Once - Shared Across All Models)
```powershell
pytest tests/unit/metrics/test_extraction.py -v
```

#### 3. Smoke Test
```powershell
python src/tests/studies/study_a/lmstudio/bias/test_study_a_bias_gpt_oss.py
```

#### 4. Full Generation
```powershell
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id gpt_oss_lmstudio
```

---

### Qwen3-8B (LM Studio)

**Environment**: `mh-llm-benchmark-env`

#### 1. Activate Environment
```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-benchmark-env
$Env:PYTHONPATH="src"
```

#### 2. Unit Tests (Run Once - Shared Across All Models)
```powershell
pytest tests/unit/metrics/test_extraction.py -v
```

#### 3. Smoke Test
```powershell
python src/tests/studies/study_a/lmstudio/bias/test_study_a_bias_qwen3_lmstudio.py
```

#### 4. Full Generation
```powershell
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id qwen3_lmstudio
```

---

### PsyLLM (HF local, GMLHUHE/PsyLLM)

**Environment**: `mh-llm-local-env`

#### 1. Activate Environment
```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-local-env
$Env:PYTHONNOUSERSITE="1"
$Env:PYTHONPATH="src"
```

#### 2. Unit Tests (Run Once - Shared Across All Models)
```powershell
pytest tests/unit/metrics/test_extraction.py -v
```

#### 3. Smoke Test
```powershell
python src/tests/studies/study_a/models/bias/test_study_a_bias_psyllm_gml_local.py
```

#### 4. Full Generation
```powershell
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id psyllm_gml_local
```

---

### Piaget-8B (HF local)

**Environment**: `mh-llm-local-env`

#### 1. Activate Environment
```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-local-env
$Env:PYTHONNOUSERSITE="1"
$Env:PYTHONPATH="src"
```

#### 2. Unit Tests (Run Once - Shared Across All Models)
```powershell
pytest tests/unit/metrics/test_extraction.py -v
```

#### 3. Smoke Test
```powershell
python src/tests/studies/study_a/models/bias/test_study_a_bias_piaget_local.py
```

#### 4. Full Generation
```powershell
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id piaget_local
```

---

### Psyche-R1 (HF local)

**Environment**: `mh-llm-local-env`

#### 1. Activate Environment
```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-local-env
$Env:PYTHONNOUSERSITE="1"
$Env:PYTHONPATH="src"
```

#### 2. Unit Tests (Run Once - Shared Across All Models)
```powershell
pytest tests/unit/metrics/test_extraction.py -v
```

#### 3. Smoke Test
```powershell
python src/tests/studies/study_a/models/bias/test_study_a_bias_psyche_r1_local.py
```

#### 4. Full Generation
```powershell
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id psyche_r1_local
```

---

### Psych-Qwen-32B (HF local, 4-bit)

**Environment**: `mh-llm-local-env`

#### 1. Activate Environment
```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-local-env
$Env:PYTHONNOUSERSITE="1"
$Env:PYTHONPATH="src"
```

#### 2. Unit Tests (Run Once - Shared Across All Models)
```powershell
pytest tests/unit/metrics/test_extraction.py -v
```

#### 3. Smoke Test
```powershell
python src/tests/studies/study_a/models/bias/test_study_a_bias_psych_qwen_local.py
```

#### 4. Full Generation
```powershell
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id psych_qwen_local --quantization 4bit
```

**Note**: The script now automatically detects local HF models and loads them directly (like main Study A scripts). For `psych_qwen_local`, quantization defaults to `4bit` but can be overridden with `--quantization`. Use `--model` to specify a custom model path.

---

## After Generation

After all models have been generated:

1. **Calculate bias metrics**:
   ```powershell
   python scripts\study_a\metrics\calculate_bias.py
   ```

2. **Calculate all Study A metrics** (automatically merges bias):
   ```powershell
   python scripts\study_a\metrics\calculate_metrics.py
   ```

## Output

Each model generates its own separate output file:

- **Per-Model Generations**: `processed/study_a_bias/{model-id}/study_a_bias_generations.jsonl`
  - Example: `processed/study_a_bias/qwq/study_a_bias_generations.jsonl`
  - Example: `processed/study_a_bias/psych_qwen_local/study_a_bias_generations.jsonl`
  - Each file contains all bias case generations for that specific model
  - Saved in `processed/study_a_bias/` directory (separate from main Study A generations)
- **Bias Metrics**: `metric-results/study_a_bias_metrics.json` (aggregated across all models)
- **Combined Metrics**: `metric-results/all_models_metrics.json` (includes `silent_bias_rate` per model)

## Data Source

- **Input**: `data/adversarial_bias/biased_vignettes.json`
- **Cases**: ~50-100 adversarial cases with demographic features
- **Purpose**: Detect silent bias (biased decision without mentioning bias feature)
