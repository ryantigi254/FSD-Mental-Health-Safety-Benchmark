# Study A Generation Commands

## Overview

Study A evaluates **Faithfulness** with paired outputs per sample:
- `cot` response (reasoning)
- `direct` response (no reasoning)

Generation cache output:
- `results/<model-folder>/study_a_generations.jsonl`

Study A now uses the same style as Study B/C/Bias:
- `hf-local-scripts/run_study_a_generate_only.py --model-id ...`

## Primary Script

- `hf-local-scripts/run_study_a_generate_only.py`

Accepted model IDs:
- `qwen3_lmstudio`
- `qwq`
- `deepseek_r1_lmstudio`
- `gpt_oss`
- `psyllm_gml_local`
- `piaget_local`
- `psyche_r1_local`
- `psych_qwen_local`

## Commands per Model

Run from `Uni-setup/`.

### Qwen3-8B (LM Studio)
```powershell
python hf-local-scripts\run_study_a_generate_only.py --model-id qwen3_lmstudio
```

### QwQ-32B (LM Studio)
```powershell
python hf-local-scripts\run_study_a_generate_only.py --model-id qwq
```

### DeepSeek-R1 (LM Studio distill)
```powershell
python hf-local-scripts\run_study_a_generate_only.py --model-id deepseek_r1_lmstudio
```

### GPT-OSS-20B (LM Studio)
```powershell
python hf-local-scripts\run_study_a_generate_only.py --model-id gpt_oss
```

### PsyLLM (HF local, GMLHUHE/PsyLLM)
```powershell
python hf-local-scripts\run_study_a_generate_only.py --model-id psyllm_gml_local
```

### Piaget-8B (HF local)
```powershell
python hf-local-scripts\run_study_a_generate_only.py --model-id piaget_local
```

### Psyche-R1 (HF local)
```powershell
python hf-local-scripts\run_study_a_generate_only.py --model-id psyche_r1_local
```

### Psych-Qwen-32B (HF local, 4-bit)
```powershell
python hf-local-scripts\run_study_a_generate_only.py --model-id psych_qwen_local
```

## Smoke Test

```powershell
python hf-local-scripts\run_study_a_generate_only.py --model-id gpt_oss --max-samples 5
```

## Custom Options

### Explicit cache path
```powershell
python hf-local-scripts\run_study_a_generate_only.py --model-id gpt_oss --cache-out results\gpt-oss-20b\study_a_generations.jsonl
```

### Custom data directory
```powershell
python hf-local-scripts\run_study_a_generate_only.py --model-id gpt_oss --data-dir data\openr1_psy_splits
```

### Custom max tokens
```powershell
python hf-local-scripts\run_study_a_generate_only.py --model-id gpt_oss --max-tokens 16384
```

## After Generation (Metrics)

### Faithfulness metrics
```powershell
python scripts\studies\study_a\metrics\calculate_metrics.py --use-cleaned --output-dir metric-results\study_a
```

### Bias metrics merge
```powershell
python scripts\studies\study_a\metrics\calculate_bias.py --use-cleaned --output-dir metric-results\study_a
python scripts\studies\study_a\metrics\calculate_metrics.py --use-cleaned --output-dir metric-results\study_a
```

## Output

- Generation cache: `results/{model}/study_a_generations.jsonl`
- Per-model metrics: `metric-results/study_a/<model>_metrics.json`
- Combined metrics: `metric-results/study_a/all_models_metrics.json`

---

## Automatic Cross-Platform Runner (Preferred)

Use this one command style on both Mac and PC:

```bash
python scripts/dev/run_generation_auto.py --study study_a --model-id gpt_oss --env mh-llm-benchmark-env
```

Study variants:

```bash
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id gpt_oss_lmstudio --env mh-llm-benchmark-env --workers 8
python scripts/dev/run_generation_auto.py --study study_b --model-id gpt_oss --env mh-llm-benchmark-env
python scripts/dev/run_generation_auto.py --study study_b_multi_turn --model-id gpt_oss --env mh-llm-benchmark-env
python scripts/dev/run_generation_auto.py --study study_c --model-id gpt_oss --env mh-llm-benchmark-env
```

## Dual Path Commands (PC + Mac, Legacy)

### PC Path (Uni setup)

```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-benchmark-env
$Env:PYTHONPATH="src"
python hf-local-scripts\run_study_a_generate_only.py --model-id gpt_oss
python scripts\studies\study_a\metrics\calculate_metrics.py --use-cleaned --output-dir metric-results\study_a
```

### Mac Path (Uni setup)

```bash
cd "/Users/ryangichuru/Documents/SSD-K/Uni/3rd year/NLP/Assignment 2/reliable_clinical_benchmark/Uni-setup"
export PYTHONPATH=src
python hf-local-scripts/run_study_a_generate_only.py --model-id gpt_oss
python scripts/studies/study_a/metrics/calculate_metrics.py --use-cleaned --output-dir metric-results/study_a
```

## Worker Commands (Absolute Bottom)

Study A generation runner does not expose `--workers`.
Use LM Studio `Max Concurrent Predictions` for Study A generation throughput.

If you need explicit Python workers, use the bias runner:

```powershell
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id gpt_oss_lmstudio --workers 8
```

```bash
python hf-local-scripts/run_study_a_bias_generate_only.py --model-id gpt_oss_lmstudio --workers 8
```
