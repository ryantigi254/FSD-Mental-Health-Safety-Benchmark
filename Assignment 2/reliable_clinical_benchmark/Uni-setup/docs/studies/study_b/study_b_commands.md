# Study B Commands (Generation + Metrics)

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
mkdir -p "${OUT_ROOT}/study_b"
```

## Study B Generation Commands (Per Model)

### LM Studio models (`mh-llm-benchmark-env`)

```bash
conda run -n mh-llm-benchmark-env env PYTHONPATH=src \
  python hf-local-scripts/run_study_b_generate_only.py --model-id qwen3_lmstudio

conda run -n mh-llm-benchmark-env env PYTHONPATH=src \
  python hf-local-scripts/run_study_b_generate_only.py --model-id qwq

conda run -n mh-llm-benchmark-env env PYTHONPATH=src \
  python hf-local-scripts/run_study_b_generate_only.py --model-id deepseek_r1_lmstudio

conda run -n mh-llm-benchmark-env env PYTHONPATH=src \
  python hf-local-scripts/run_study_b_generate_only.py --model-id gpt_oss
```

### Local HF models (`mh-llm-local-env`)

```bash
conda run -n mh-llm-local-env env PYTHONNOUSERSITE=1 PYTHONPATH=src \
  python hf-local-scripts/run_study_b_generate_only.py --model-id psyllm_gml_local

conda run -n mh-llm-local-env env PYTHONNOUSERSITE=1 PYTHONPATH=src \
  python hf-local-scripts/run_study_b_generate_only.py --model-id piaget_local

conda run -n mh-llm-local-env env PYTHONNOUSERSITE=1 PYTHONPATH=src \
  python hf-local-scripts/run_study_b_generate_only.py --model-id psyche_r1_local

conda run -n mh-llm-local-env env PYTHONNOUSERSITE=1 PYTHONPATH=src \
  python hf-local-scripts/run_study_b_generate_only.py --model-id psych_qwen_local
```

### Multi-turn generation (Turn-of-Flip input)

```bash
conda run -n mh-llm-benchmark-env env PYTHONPATH=src \
  python hf-local-scripts/run_study_b_multi_turn_generate_only.py --model-id qwen3_lmstudio

conda run -n mh-llm-benchmark-env env PYTHONPATH=src \
  python hf-local-scripts/run_study_b_multi_turn_generate_only.py --model-id qwq

conda run -n mh-llm-benchmark-env env PYTHONPATH=src \
  python hf-local-scripts/run_study_b_multi_turn_generate_only.py --model-id deepseek_r1_lmstudio

conda run -n mh-llm-benchmark-env env PYTHONPATH=src \
  python hf-local-scripts/run_study_b_multi_turn_generate_only.py --model-id gpt_oss

conda run -n mh-llm-local-env env PYTHONNOUSERSITE=1 PYTHONPATH=src \
  python hf-local-scripts/run_study_b_multi_turn_generate_only.py --model-id psyllm_gml_local

conda run -n mh-llm-local-env env PYTHONNOUSERSITE=1 PYTHONPATH=src \
  python hf-local-scripts/run_study_b_multi_turn_generate_only.py --model-id piaget_local

conda run -n mh-llm-local-env env PYTHONNOUSERSITE=1 PYTHONPATH=src \
  python hf-local-scripts/run_study_b_multi_turn_generate_only.py --model-id psyche_r1_local

conda run -n mh-llm-local-env env PYTHONNOUSERSITE=1 PYTHONPATH=src \
  python hf-local-scripts/run_study_b_multi_turn_generate_only.py --model-id psych_qwen_local
```

## Study B Metrics Commands

### Full Study B metrics (P_Syc, Flip, H_Ev agree-only, H_Ev all-injected, ToF)

```bash
conda run -n mh-llm-benchmark-env env PYTHONPATH=src PYTHONUNBUFFERED=1 \
  python scripts/studies/study_b/metrics/calculate_metrics.py \
  --use-cleaned \
  --output-dir "${OUT_ROOT}/study_b"
```

### Per-model Study B metrics

```bash
for MODEL in qwen3-lmstudio qwq deepseek-r1-lmstudio gpt-oss-20b psyllm-gml-local piaget-8b-local psyche-r1-local psych-qwen-32b-local; do
  conda run -n mh-llm-benchmark-env env PYTHONPATH=src PYTHONUNBUFFERED=1 \
    python scripts/studies/study_b/metrics/calculate_metrics.py \
    --use-cleaned \
    --model "${MODEL}" \
    --output-dir "${OUT_ROOT}/study_b"
done
```

## Diagnostics Checks

```bash
jq '.[] | {model, claims_extracted_total, injected_responses_scored, h_ev_agree, h_ev_all}' \
  "${OUT_ROOT}/study_b/hev_diagnostics.json"

jq '.[] | {model, sycophancy_probability, flip_rate: (.flip_rate // null), bad_flip_rate: (.bad_flip_rate // null), good_flip_rate: (.good_flip_rate // null), net_harm_rate: (.net_harm_rate // null), evidence_hallucination, evidence_hallucination_all, turn_of_flip}' \
  "${OUT_ROOT}/study_b/sycophancy_metrics.json"
```

## Output Files

```text
${OUT_ROOT}/study_b/sycophancy_metrics.json
${OUT_ROOT}/study_b/hev_diagnostics.json
```
