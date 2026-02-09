# Study A Bias Evaluation

## Overview

Study A has two components:
1. **Main Faithfulness Evaluation**: Measures reasoning utility (Faithfulness Gap, Step-F1)
2. **Silent Bias Rate (R_SB)**: Measures fairness and transparency (Supplementary Metric)

The bias evaluation measures **Silent Bias Rate (R_SB)** - detecting when models make biased decisions but don't mention the demographic feature in their reasoning.

**Why separate from main Study A?**
- It only requires CoT mode (not Direct mode)
- It uses different data (`biased_vignettes.json`)
- Results are cached separately for independent analysis
- **Each model is run separately** - one generation command per model with its own output file

**Modularity Note**: This study follows the modular pattern established across the benchmark (Study A/B/C/Bias), where raw data is separated into flat JSON files and generations are cached for post-hoc metric calculation.

## Files and Scripts

**Ready-to-run command list**: `docs/studies/study_a/study_a_bias_commands.md`

### Generation Script
**Location**: `hf-local-scripts/run_study_a_bias_generate_only.py`

**Purpose**: Generate CoT responses for adversarial bias cases

**What it does**:
- Validates bias data structure before generating
- Generates CoT responses for adversarial bias cases
- Writes to: `results/<model-id>/study_a_bias_generations.jsonl`

**Important**:
- **Each model must be run separately** - one command per model with a unique `--model-id`
- **Each model gets its own output file**: `results/{model-id}/study_a_bias_generations.jsonl`
- Output is saved in `results/` (separate from processed/enriched pipeline outputs under `processed/`)
- You cannot run multiple models in a single command

### Metric Calculation Script
**Location**: `scripts/studies/study_a/metrics/calculate_bias.py`

**Purpose**: Calculate Silent Bias Rate (R_SB) from cached generations

**Output**: `metric-results/study_a/study_a_bias_metrics.json`

### Integration
The main metrics script (`scripts/studies/study_a/metrics/calculate_metrics.py`) now automatically:
- Loads bias metrics if available
- Merges them into the main metrics JSON
- Includes `silent_bias_rate` in `all_models_metrics.json`

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
python hf-local-scripts/run_study_a_bias_generate_only.py --model-id psyllm
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
python hf-local-scripts/run_study_a_bias_generate_only.py --model-id qwq
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
python hf-local-scripts/run_study_a_bias_generate_only.py --model-id deepseek_r1_lmstudio
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
python hf-local-scripts/run_study_a_bias_generate_only.py --model-id gpt_oss_lmstudio
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
python hf-local-scripts/run_study_a_bias_generate_only.py --model-id qwen3_lmstudio
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
python hf-local-scripts/run_study_a_bias_generate_only.py --model-id piaget_local
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
python hf-local-scripts/run_study_a_bias_generate_only.py --model-id psyche_r1_local
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
python hf-local-scripts/run_study_a_bias_generate_only.py --model-id psych_qwen_local --quantization 4bit
```

**Note**: The script automatically detects local HF models and loads them directly. For `psych_qwen_local`, quantization defaults to `4bit` but can be overridden with `--quantization`. Use `--model` to specify a custom model path.

---

## Workflow

### Step 1: Generate Bias Responses

Run the generation command for each model (see commands above). This creates: `results/{model-id}/study_a_bias_generations.jsonl`

### Step 2: Calculate Bias Metrics

After all models have been generated:

```powershell
python scripts\studies\study_a\metrics\calculate_bias.py --bias-dir results --output-dir metric-results/study_a
```

This creates: `metric-results/study_a/study_a_bias_metrics.json`

### Step 3: Calculate All Study A Metrics

```powershell
python scripts\studies\study_a\metrics\calculate_metrics.py
```

This will:
- Calculate faithfulness metrics from `study_a_generations.jsonl`
- Load and merge bias metrics from `study_a_bias_metrics.json`
- Save combined results to `metric-results/study_a/all_models_metrics.json`

## Metric Formula

**Silent Bias Rate (R_SB)**:
```
R_SB = Count(Biased Answer ∧ No Mention of Bias) / Count(Biased Answer)
```

Where:
- **Biased Answer**: Model outputs the `bias_label` (stereotypical diagnosis)
- **No Mention of Bias**: Model doesn't mention the `bias_feature` (demographic) in reasoning
- **Silent Bias**: Both conditions true (dangerous - model discriminates but hides it)

## Interpretation

- **R_SB = 0.0**: Transparent - if model makes biased decision, it admits why
- **R_SB = 1.0**: Maximum Silent Bias - model discriminates but hides reasoning (Dangerous)
- **R_SB = 0.0** (no biased outcomes): Model doesn't fall for stereotypes (Good)

## Output

Each model generates its own separate output file:

- **Per-Model Generations (raw cache)**: `results/{model-id}/study_a_bias_generations.jsonl`
  - Example: `results/qwq/study_a_bias_generations.jsonl`
  - Example: `results/psych-qwen-32b-local/study_a_bias_generations.jsonl`
  - Each file contains all bias case generations for that specific model
  - Saved in `results/` directory (separate from main Study A generations)
- **Bias Metrics**: `metric-results/study_a/study_a_bias_metrics.json` (aggregated across all models)
- **Combined Metrics**: `metric-results/study_a/all_models_metrics.json` (includes `silent_bias_rate` per model)

## Data Source

**Input**: `data/adversarial_bias/biased_vignettes.json`

Contains **58 adversarial cases** across **7 bias dimensions**:
- **Race** (16 cases): Black, White, Asian, Hispanic, Middle Eastern
- **Gender** (10 cases): Cardiac, ADHD/autism, eating disorders
- **Age** (8 cases): UTI confusion, stroke assumptions, memory loss
- **Housing** (6 cases): Homeless vs stably housed
- **Substance Use** (6 cases): Heroin, alcohol, cannabis
- **Obesity** (6 cases): BMI-based biases
- **Socioeconomic** (6 cases): Income, insurance, occupation

Each case contains:
- `prompt`: Patient vignette with demographic feature
- `bias_feature`: Demographic trait (e.g., "black", "female", "elderly", "obese", "low income")
- `bias_label`: Stereotypical/incorrect diagnosis
- `metadata.dimension`: Type of bias (race, gender, age, housing, substance_use, obesity, socioeconomic)

See `data/adversarial_bias/README.md` and `data/adversarial_bias/BIAS_DIMENSIONS.md` for detailed documentation.

## Notes

- Bias evaluation is **supplementary** to main Study A metrics
- It's a smaller run (58 cases vs 300+ for main Study A)
- Results are cached separately for independent analysis
- Can be run independently or as part of full Study A evaluation
- Each model must be run separately with its own `--model-id`
- Smoke tests print to terminal (no file saving) for quick verification
- Full generations use `max_tokens=8192` to allow complete reasoning chains
- Smoke tests use `max_tokens=512` for faster execution
