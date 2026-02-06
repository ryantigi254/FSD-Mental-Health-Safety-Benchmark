# Study A Bias Commands (Generation + Metrics)

## Run Location

Run every command from:

```bash
cd "/Users/ryangichuru/Documents/SSD-K/Uni/3rd year/NLP/Assignment 2/reliable_clinical_benchmark/Uni-setup"
```

## Shared Variables

```bash
export PYTHONPATH=src
RUN_TAG="$(date +%Y%m%d_%H%M)"
OUT_ROOT="metric-results/misc/${RUN_TAG}"
mkdir -p "${OUT_ROOT}/study_a"
```

## Study A Bias Generation Commands (Per Model)

### LM Studio models (`mh-llm-benchmark-env`)

```bash
conda run -n mh-llm-benchmark-env env PYTHONPATH=src PYTHONUNBUFFERED=1 \
  python hf-local-scripts/run_study_a_bias_generate_only.py --model-id qwen3_lmstudio --workers 4

conda run -n mh-llm-benchmark-env env PYTHONPATH=src PYTHONUNBUFFERED=1 \
  python hf-local-scripts/run_study_a_bias_generate_only.py --model-id qwq --workers 4

conda run -n mh-llm-benchmark-env env PYTHONPATH=src PYTHONUNBUFFERED=1 \
  python hf-local-scripts/run_study_a_bias_generate_only.py --model-id deepseek_r1_lmstudio --workers 4

conda run -n mh-llm-benchmark-env env PYTHONPATH=src PYTHONUNBUFFERED=1 \
  python hf-local-scripts/run_study_a_bias_generate_only.py --model-id gpt_oss_lmstudio --workers 4
```

### Local HF models (`mh-llm-local-env`)

```bash
conda run -n mh-llm-local-env env PYTHONNOUSERSITE=1 PYTHONPATH=src PYTHONUNBUFFERED=1 \
  python hf-local-scripts/run_study_a_bias_generate_only.py --model-id psyllm_gml_local

conda run -n mh-llm-local-env env PYTHONNOUSERSITE=1 PYTHONPATH=src PYTHONUNBUFFERED=1 \
  python hf-local-scripts/run_study_a_bias_generate_only.py --model-id piaget_local

conda run -n mh-llm-local-env env PYTHONNOUSERSITE=1 PYTHONPATH=src PYTHONUNBUFFERED=1 \
  python hf-local-scripts/run_study_a_bias_generate_only.py --model-id psyche_r1_local

conda run -n mh-llm-local-env env PYTHONNOUSERSITE=1 PYTHONPATH=src PYTHONUNBUFFERED=1 \
  python hf-local-scripts/run_study_a_bias_generate_only.py --model-id psych_qwen_local --quantization 4bit
```

## Study A Bias Metric Commands

### Silent Bias Rate (all models)

```bash
conda run -n mh-llm-benchmark-env env PYTHONPATH=src \
  python scripts/studies/study_a/metrics/calculate_bias.py \
  --use-cleaned \
  --output-dir "${OUT_ROOT}/study_a"
```

### Merge bias into Study A metrics bundle

```bash
conda run -n mh-llm-benchmark-env env PYTHONPATH=src \
  python scripts/studies/study_a/metrics/calculate_metrics.py \
  --use-cleaned \
  --output-dir "${OUT_ROOT}/study_a"
```

## Diagnostics Checks

```bash
jq '. | to_entries[] | {model: .key, silent_bias_rate: .value.silent_bias_rate, silent_bias_rate_ci_low: .value.silent_bias_rate_ci_low, silent_bias_rate_ci_high: .value.silent_bias_rate_ci_high, n_biased_outcomes: .value.n_biased_outcomes, n_silent: .value.n_silent}' \
  "${OUT_ROOT}/study_a/study_a_bias_metrics.json"
```

## Output Files

```text
results/<model>/study_a_bias_generations.jsonl
${OUT_ROOT}/study_a/study_a_bias_metrics.json
${OUT_ROOT}/study_a/all_models_metrics.json
```

---

## Dual Path Commands (PC + Mac)

### PC Path (Uni setup)

```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-benchmark-env
$Env:PYTHONPATH="src"
$Env:PYTHONUNBUFFERED="1"
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id gpt_oss_lmstudio --workers 8
python scripts\studies\study_a\metrics\calculate_bias.py --use-cleaned --output-dir metric-results\study_a
python scripts\studies\study_a\metrics\calculate_metrics.py --use-cleaned --output-dir metric-results\study_a
```

### Mac Path (Uni setup)

```bash
cd "/Users/ryangichuru/Documents/SSD-K/Uni/3rd year/NLP/Assignment 2/reliable_clinical_benchmark/Uni-setup"
export PYTHONPATH=src
PYTHONUNBUFFERED=1 python hf-local-scripts/run_study_a_bias_generate_only.py --model-id gpt_oss_lmstudio --workers 8
python scripts/studies/study_a/metrics/calculate_bias.py --use-cleaned --output-dir metric-results/study_a
python scripts/studies/study_a/metrics/calculate_metrics.py --use-cleaned --output-dir metric-results/study_a
```

## Worker Commands (Absolute Bottom)

```powershell
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id qwen3_lmstudio --workers 8
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id qwq --workers 8
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id deepseek_r1_lmstudio --workers 8
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id gpt_oss_lmstudio --workers 8
```

```bash
python hf-local-scripts/run_study_a_bias_generate_only.py --model-id qwen3_lmstudio --workers 8
python hf-local-scripts/run_study_a_bias_generate_only.py --model-id qwq --workers 8
python hf-local-scripts/run_study_a_bias_generate_only.py --model-id deepseek_r1_lmstudio --workers 8
python hf-local-scripts/run_study_a_bias_generate_only.py --model-id gpt_oss_lmstudio --workers 8
```
