# Study A Commands (Generation + Metrics)

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

## Study A Generation Commands (Per Model)

### LM Studio models (`mh-llm-benchmark-env`)

```bash
conda run -n mh-llm-benchmark-env env PYTHONPATH=src \
  python hf-local-scripts/run_qwen3_lmstudio.py study-a --generate-only \
  --api-identifier qwen3-8b --model-name qwen3-lmstudio

conda run -n mh-llm-benchmark-env env PYTHONPATH=src \
  python hf-local-scripts/run_qwen3_lmstudio.py study-a --generate-only \
  --api-identifier QwQ-32B-GGUF --model-name qwq

conda run -n mh-llm-benchmark-env env PYTHONPATH=src \
  python hf-local-scripts/run_qwen3_lmstudio.py study-a --generate-only \
  --api-identifier deepseek-r1-distill-qwen-14b --model-name deepseek-r1-lmstudio

conda run -n mh-llm-benchmark-env env PYTHONPATH=src \
  python hf-local-scripts/run_qwen3_lmstudio.py study-a --generate-only \
  --api-identifier gpt-oss-20b --model-name gpt-oss-20b
```

### Local HF models (`mh-llm-local-env`)

```bash
conda run -n mh-llm-local-env env PYTHONNOUSERSITE=1 PYTHONPATH=src \
  python hf-local-scripts/run_psyllm_gml.py study-a --generate-only --model-name psyllm-gml-local

conda run -n mh-llm-local-env env PYTHONNOUSERSITE=1 PYTHONPATH=src \
  python hf-local-scripts/run_psyche_r1.py study-a --generate-only --model-name psyche-r1-local

conda run -n mh-llm-local-env env PYTHONNOUSERSITE=1 PYTHONPATH=src \
  python hf-local-scripts/run_psych_qwen_32b.py study-a --generate-only --model-name psych-qwen-32b-local --quantization 4bit

conda run -n mh-llm-local-env env PYTHONNOUSERSITE=1 PYTHONPATH=src \
  python scripts/evaluation/run_evaluation.py --model piaget_local --study A --generate-only --output-dir results
```

## Study A Metrics Commands

### Faithfulness Gap + Step-F1 (all models from cleaned pipeline)

```bash
conda run -n mh-llm-benchmark-env env PYTHONPATH=src \
  python scripts/studies/study_a/metrics/calculate_metrics.py \
  --use-cleaned \
  --output-dir "${OUT_ROOT}/study_a"
```

### Faithfulness Gap + Step-F1 (per model)

```bash
for MODEL in qwen3-lmstudio qwq deepseek-r1-lmstudio gpt-oss-20b psyllm-gml-local piaget-8b-local psyche-r1-local psych-qwen-32b-local; do
  conda run -n mh-llm-benchmark-env env PYTHONPATH=src \
    python scripts/studies/study_a/metrics/calculate_metrics.py \
    --use-cleaned \
    --model "${MODEL}" \
    --output-dir "${OUT_ROOT}/study_a"
done
```

### Silent Bias Rate merge

```bash
conda run -n mh-llm-benchmark-env env PYTHONPATH=src \
  python scripts/studies/study_a/metrics/calculate_bias.py \
  --use-cleaned \
  --output-dir "${OUT_ROOT}/study_a"

conda run -n mh-llm-benchmark-env env PYTHONPATH=src \
  python scripts/studies/study_a/metrics/calculate_metrics.py \
  --use-cleaned \
  --output-dir "${OUT_ROOT}/study_a"
```

## Output Files

```text
${OUT_ROOT}/study_a/all_models_metrics.json
${OUT_ROOT}/study_a/study_a_bias_metrics.json
${OUT_ROOT}/study_a/<model>_metrics.json
```

Bias generation commands are in:
`docs/studies/study_a/study_a_bias_commands.md`
