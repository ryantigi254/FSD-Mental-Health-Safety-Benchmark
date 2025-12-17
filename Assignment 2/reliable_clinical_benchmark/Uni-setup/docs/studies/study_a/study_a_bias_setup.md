# Study A Bias Evaluation Setup

## Overview

Study A has two components:
1. **Main Faithfulness Evaluation**: Measures reasoning utility (Faithfulness Gap, Step-F1)
2. **Silent Bias Rate (R_SB)**: Measures fairness and transparency (Supplementary Metric)

The bias evaluation is run **separately** because:
- It only requires CoT mode (not Direct mode)
- It uses different data (`biased_vignettes.json`)
- Results are cached separately for independent analysis

## Files Created

### 1. Generation Script
**Location**: `hf-local-scripts/run_study_a_bias_generate_only.py`

**Purpose**: Generate CoT responses for adversarial bias cases

**Usage**:
```bash
python hf-local-scripts/run_study_a_bias_generate_only.py --model-id qwq-lmstudio
```

**Output**: `results/{model-id}/study_a_bias_generations.jsonl`

### 2. Metric Calculation Script
**Location**: `scripts/study_a/metrics/calculate_bias.py`

**Purpose**: Calculate Silent Bias Rate (R_SB) from cached generations

**Usage**:
```bash
python scripts/study_a/metrics/calculate_bias.py
```

**Output**: `metric-results/study_a_bias_metrics.json`

### 3. Integration
The main metrics script (`scripts/study_a/metrics/calculate_metrics.py`) now automatically:
- Loads bias metrics if available
- Merges them into the main metrics JSON
- Includes `silent_bias_rate` in `all_models_metrics.json`

## Environment Requirements

Bias generations use the **same environments** as main Study A generations:

- **`mh-llm-benchmark-env`**: For LM Studio models (`psyllm`, `qwq`, `deepseek_r1_lmstudio`, `gpt_oss`, `qwen3_lmstudio`)
- **`mh-llm-local-env`**: For local HF models (`psyllm_gml_local`, `piaget_local`, `psyche_r1_local`, `psych_qwen_local`)

See `docs/environment/ENVIRONMENT.md` for setup instructions.

## Workflow

### Step 1: Generate Bias Responses

**For LM Studio models** (use `mh-llm-benchmark-env`):
```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-benchmark-env
$Env:PYTHONPATH="src"
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id qwq
```

**For local HF models** (use `mh-llm-local-env`):
```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-local-env
$Env:PYTHONNOUSERSITE="1"
$Env:PYTHONPATH="src"
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id psych_qwen_local
```

This creates: `results/{model-id}/study_a_bias_generations.jsonl`

### Step 2: Calculate Bias Metrics
```bash
python scripts/study_a/metrics/calculate_bias.py
```

This creates: `metric-results/study_a_bias_metrics.json`

### Step 3: Calculate All Study A Metrics
```bash
python scripts/study_a/metrics/calculate_metrics.py
```

This will:
- Calculate faithfulness metrics from `study_a_generations.jsonl`
- Load and merge bias metrics from `study_a_bias_metrics.json`
- Save combined results to `metric-results/all_models_metrics.json`

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

## Data Source

**Input**: `data/adversarial_bias/biased_vignettes.json`

Contains adversarial cases with:
- `prompt`: Patient vignette with demographic feature
- `bias_feature`: Demographic trait (e.g., "black", "female", "elderly")
- `bias_label`: Stereotypical/incorrect diagnosis
- `metadata.dimension`: Type of bias (race, gender, age)

## Notes

- Bias evaluation is **supplementary** to main Study A metrics
- It's a smaller run (~50-100 cases vs 300+ for main Study A)
- Results are cached separately for independent analysis
- Can be run independently or as part of full Study A evaluation

