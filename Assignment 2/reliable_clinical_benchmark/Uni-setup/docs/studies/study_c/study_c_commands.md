# Study C Commands (Generation + Metrics)

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
mkdir -p "${OUT_ROOT}/study_c"
```

## Study C Generation Commands (Per Model)

### LM Studio models (`mh-llm-benchmark-env`)

```bash
conda run -n mh-llm-benchmark-env env PYTHONPATH=src \
  python hf-local-scripts/run_study_c_generate_only.py --model-id qwen3_lmstudio

conda run -n mh-llm-benchmark-env env PYTHONPATH=src \
  python hf-local-scripts/run_study_c_generate_only.py --model-id qwq

conda run -n mh-llm-benchmark-env env PYTHONPATH=src \
  python hf-local-scripts/run_study_c_generate_only.py --model-id deepseek_r1_lmstudio

conda run -n mh-llm-benchmark-env env PYTHONPATH=src \
  python hf-local-scripts/run_study_c_generate_only.py --model-id gpt_oss
```

### Local HF models (`mh-llm-local-env`)

```bash
conda run -n mh-llm-local-env env PYTHONNOUSERSITE=1 PYTHONPATH=src \
  python hf-local-scripts/run_study_c_generate_only.py --model-id psyllm_gml_local

conda run -n mh-llm-local-env env PYTHONNOUSERSITE=1 PYTHONPATH=src \
  python hf-local-scripts/run_study_c_generate_only.py --model-id piaget_local

conda run -n mh-llm-local-env env PYTHONNOUSERSITE=1 PYTHONPATH=src \
  python hf-local-scripts/run_study_c_generate_only.py --model-id psyche_r1_local

conda run -n mh-llm-local-env env PYTHONNOUSERSITE=1 PYTHONPATH=src \
  python hf-local-scripts/run_study_c_generate_only.py --model-id psych_qwen_local
```

## Study C Metrics Commands

### Full Study C metrics (Entity Recall Decay, Knowledge Conflict Rate, Session Goal Alignment)

```bash
conda run -n mh-llm-benchmark-env env PYTHONPATH=src PYTHONUNBUFFERED=1 \
  python scripts/studies/study_c/metrics/calculate_metrics.py \
  --use-cleaned \
  --use-nli \
  --nli-stride 2 \
  --output-dir "${OUT_ROOT}/study_c"
```

### Per-model Study C metrics

```bash
for MODEL in qwen3-lmstudio qwq deepseek-r1-lmstudio gpt-oss-20b psyllm-gml-local piaget-8b-local psyche-r1-local psych-qwen-32b-local; do
  conda run -n mh-llm-benchmark-env env PYTHONPATH=src PYTHONUNBUFFERED=1 \
    python scripts/studies/study_c/metrics/calculate_metrics.py \
    --use-cleaned \
    --use-nli \
    --nli-stride 2 \
    --model "${MODEL}" \
    --output-dir "${OUT_ROOT}/study_c"
done
```

## Diagnostics Checks

```bash
jq '.[] | {model, entity_recall_t10, entity_recall_t10_ci_low, entity_recall_t10_ci_high, knowledge_conflict_rate, knowledge_conflict_rate_ci_low, knowledge_conflict_rate_ci_high, continuity_score, continuity_source}' \
  "${OUT_ROOT}/study_c/drift_metrics.json"
```

## Output Files

```text
${OUT_ROOT}/study_c/drift_metrics.json
```
