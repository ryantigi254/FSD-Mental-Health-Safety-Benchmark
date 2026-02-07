# Study A Bias Commands

## Scope
Study A bias generation writes to `results/<model-folder>/study_a_bias_generations.jsonl` and metrics write to `metric-results/.../study_a`.

## One-Time Setup

### Mac/Linux
```bash
cd "/Users/ryangichuru/Documents/SSD-K/Uni/3rd year/NLP/Assignment 2/reliable_clinical_benchmark/Uni-setup"
```

### Windows (PC)
```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
```

## Generation Commands (Automatic Cross-Platform Runner)

### Mac/Linux
```bash
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id qwen3_lmstudio --env mh-llm-benchmark-env --workers 8
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id qwq --env mh-llm-benchmark-env --workers 8
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id deepseek_r1_lmstudio --env mh-llm-benchmark-env --workers 8
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id gpt_oss_lmstudio --env mh-llm-benchmark-env --workers 8
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id psyllm_gml_local --env mh-llm-local-env
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id piaget_local --env mh-llm-local-env
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id psyche_r1_local --env mh-llm-local-env
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id psych_qwen_local --env mh-llm-local-env --quantization 4bit
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id qwen3_lmstudio --env mh-llm-benchmark-env --workers 8
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id qwq --env mh-llm-benchmark-env --workers 8
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id deepseek_r1_lmstudio --env mh-llm-benchmark-env --workers 8
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id gpt_oss_lmstudio --env mh-llm-benchmark-env --workers 8
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id psyllm_gml_local --env mh-llm-local-env
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id piaget_local --env mh-llm-local-env
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id psyche_r1_local --env mh-llm-local-env
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id psych_qwen_local --env mh-llm-local-env --quantization 4bit
```

## Metrics Commands

### Mac/Linux
```bash
RUN_TAG="$(date +%Y%m%d_%H%M)"; OUT_ROOT="metric-results/misc/${RUN_TAG}"; mkdir -p "${OUT_ROOT}/study_a"; conda run -n mh-llm-benchmark-env env PYTHONPATH=src python scripts/studies/study_a/metrics/calculate_bias.py --use-cleaned --output-dir "${OUT_ROOT}/study_a"
RUN_TAG="$(date +%Y%m%d_%H%M)"; OUT_ROOT="metric-results/misc/${RUN_TAG}"; mkdir -p "${OUT_ROOT}/study_a"; conda run -n mh-llm-benchmark-env env PYTHONPATH=src python scripts/studies/study_a/metrics/calculate_metrics.py --use-cleaned --output-dir "${OUT_ROOT}/study_a"
```

### Windows (PC)
```powershell
$RUN_TAG=Get-Date -Format "yyyyMMdd_HHmm"; $OUT_ROOT="metric-results/misc/$RUN_TAG"; New-Item -ItemType Directory -Force -Path "$OUT_ROOT/study_a" | Out-Null; conda run -n mh-llm-benchmark-env python scripts/studies/study_a/metrics/calculate_bias.py --use-cleaned --output-dir "$OUT_ROOT/study_a"
$RUN_TAG=Get-Date -Format "yyyyMMdd_HHmm"; $OUT_ROOT="metric-results/misc/$RUN_TAG"; New-Item -ItemType Directory -Force -Path "$OUT_ROOT/study_a" | Out-Null; conda run -n mh-llm-benchmark-env python scripts/studies/study_a/metrics/calculate_metrics.py --use-cleaned --output-dir "$OUT_ROOT/study_a"
```

## Workers
`study_a_bias` generation supports `--workers`. If not passed, default is `1`.

## Useful Checks
```bash
python scripts/dev/run_generation_auto.py --study study_a_bias --model-id gpt_oss_lmstudio --check-only
python hf-local-scripts/run_study_a_bias_generate_only.py --model-id gpt_oss_lmstudio --workers 8 --max-cases 5
```
