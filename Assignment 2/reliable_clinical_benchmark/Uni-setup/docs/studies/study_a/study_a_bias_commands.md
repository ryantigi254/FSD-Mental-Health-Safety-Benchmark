# Study A Bias Generation Commands

## Overview

Study A bias evaluation measures **Silent Bias Rate (R_SB)** - detecting when models make biased decisions but don't mention the demographic feature in their reasoning.

**File**: `hf-local-scripts/run_study_a_bias_generate_only.py`

**What it does**:
- Validates bias data structure before generating
- Generates CoT responses for adversarial bias cases
- Writes to: `results/<model-id>/study_a_bias_generations.jsonl`

**Note**: This is a separate run from main Study A (faithfulness). Bias cases only need CoT mode, not Direct mode.

## Environment Requirements

Bias generations use the **same environments** as main Study A generations:

- **`mh-llm-benchmark-env`**: For LM Studio models (`psyllm`, `qwq`, `deepseek_r1_lmstudio`, `gpt_oss`, `qwen3_lmstudio`)
- **`mh-llm-local-env`**: For local HF models (`psyllm_gml_local`, `piaget_local`, `psyche_r1_local`, `psych_qwen_local`)

See `docs/environment/ENVIRONMENT.md` for setup instructions.

**Note**: Adjust the Anaconda path (`D:\Anaconda3\Scripts\activate`) in the commands below if your Anaconda installation is in a different location.

## Commands per Model

Run these from `Uni-setup/`:

**Environment**: `mh-llm-benchmark-env` (LM Studio models)

### PsyLLM (LM Studio)
```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-benchmark-env
$Env:PYTHONPATH="src"
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id psyllm
```

### QwQ-32B (LM Studio)
```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-benchmark-env
$Env:PYTHONPATH="src"
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id qwq
```

### DeepSeek-R1 (LM Studio distill)
```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-benchmark-env
$Env:PYTHONPATH="src"
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id deepseek_r1_lmstudio
```

### GPT-OSS-20B (LM Studio)
```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-benchmark-env
$Env:PYTHONPATH="src"
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id gpt_oss
```

### Qwen3-8B (LM Studio)
```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-benchmark-env
$Env:PYTHONPATH="src"
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id qwen3_lmstudio
```

**Environment**: `mh-llm-local-env` (local HF models)

### PsyLLM (HF local, GMLHUHE/PsyLLM)
```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-local-env
$Env:PYTHONNOUSERSITE="1"
$Env:PYTHONPATH="src"
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id psyllm_gml_local
```

### Piaget-8B (HF local)
```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-local-env
$Env:PYTHONNOUSERSITE="1"
$Env:PYTHONPATH="src"
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id piaget_local
```

### Psyche-R1 (HF local)
```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-local-env
$Env:PYTHONNOUSERSITE="1"
$Env:PYTHONPATH="src"
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id psyche_r1_local
```

### Psych-Qwen-32B (HF local, 4-bit)
```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-local-env
$Env:PYTHONNOUSERSITE="1"
$Env:PYTHONPATH="src"
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id psych_qwen_local
```

**Note**: For `psych_qwen_local`, use `--quantization 4bit` if needed (same as main Study A generation).

## Quick Smoke Test

Test with a small subset first (any model):

**For LM Studio models** (use `mh-llm-benchmark-env`):
```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-benchmark-env
$Env:PYTHONPATH="src"
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id qwen3_lmstudio --max-cases 5 --max-tokens 512
```

**For local HF models** (use `mh-llm-local-env`):
```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-local-env
$Env:PYTHONNOUSERSITE="1"
$Env:PYTHONPATH="src"
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id psych_qwen_local --max-cases 5 --max-tokens 512
```

## After Generation

1. **Calculate bias metrics**:
   ```powershell
   python scripts\study_a\metrics\calculate_bias.py
   ```

2. **Calculate all Study A metrics** (automatically merges bias):
   ```powershell
   python scripts\study_a\metrics\calculate_metrics.py
   ```

## Output

- **Generations**: `results/{model-id}/study_a_bias_generations.jsonl`
- **Bias Metrics**: `metric-results/study_a_bias_metrics.json`
- **Combined Metrics**: `metric-results/all_models_metrics.json` (includes `silent_bias_rate`)

## Data Source

- **Input**: `data/adversarial_bias/biased_vignettes.json`
- **Cases**: ~50-100 adversarial cases with demographic features
- **Purpose**: Detect silent bias (biased decision without mentioning bias feature)

