# Study A Bias Generation Commands

## Overview

Study A bias evaluation measures **Silent Bias Rate (R_SB)** - detecting when models make biased decisions but don't mention the demographic feature in their reasoning.

**File**: `hf-local-scripts/run_study_a_bias_generate_only.py`

**What it does**:
- Validates bias data structure before generating
- Generates CoT responses for adversarial bias cases
- Writes to: `results/<model-id>/study_a_bias_generations.jsonl`

**Note**: This is a separate run from main Study A (faithfulness). Bias cases only need CoT mode, not Direct mode.

## Commands per Model

Run these from `Uni-setup/`:

### PsyLLM (LM Studio)
```powershell
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id psyllm
```

### PsyLLM (HF local, GMLHUHE/PsyLLM)
```powershell
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id psyllm_gml_local
```

### QwQ-32B (LM Studio)
```powershell
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id qwq
```

### DeepSeek-R1 (LM Studio distill)
```powershell
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id deepseek_r1_lmstudio
```

### GPT-OSS-20B (LM Studio)
```powershell
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id gpt_oss
```

### Qwen3-8B (LM Studio)
```powershell
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id qwen3_lmstudio
```

### Piaget-8B (HF local)
```powershell
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id piaget_local
```

### Psyche-R1 (HF local)
```powershell
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id psyche_r1_local
```

### Psych-Qwen-32B (HF local, 4-bit)
```powershell
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id psych_qwen_local
```

## Quick Smoke Test

Test with a small subset first (any model):
```powershell
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id qwen3_lmstudio --max-cases 5 --max-tokens 512
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

