# Study C Commands

## Scope
Study C generation writes to `results/<model-folder>/study_c_generations.jsonl` and metrics write to `metric-results/.../study_c`.

## One-Time Setup

### Mac/Linux
```bash
cd "/Users/ryangichuru/Documents/SSD-K/Uni/3rd year/NLP/Assignment 2/reliable_clinical_benchmark/Uni-setup"
```

### Windows (PC)
```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
```

## Optional Gold Plans (for continuity score stability)

### Mac/Linux
```bash
conda run -n mh-llm-benchmark-env env PYTHONPATH=src python scripts/studies/study_c/gold_plans/generate_nli_plans.py --force
```

### Windows (PC)
```powershell
conda run -n mh-llm-benchmark-env python scripts/studies/study_c/gold_plans/generate_nli_plans.py --force
```

## Generation Commands

### Mac/Linux
```bash
python scripts/dev/run_generation_auto.py --study study_c --model-id qwen3_lmstudio --env mh-llm-benchmark-env
python scripts/dev/run_generation_auto.py --study study_c --model-id qwq --env mh-llm-benchmark-env
python scripts/dev/run_generation_auto.py --study study_c --model-id deepseek_r1_lmstudio --env mh-llm-benchmark-env
python scripts/dev/run_generation_auto.py --study study_c --model-id gpt_oss --env mh-llm-benchmark-env
python scripts/dev/run_generation_auto.py --study study_c --model-id psyllm --env mh-llm-benchmark-env
python scripts/dev/run_generation_auto.py --study study_c --model-id psyllm_gml_local --env mh-llm-local-env
python scripts/dev/run_generation_auto.py --study study_c --model-id piaget_local --env mh-llm-local-env
python scripts/dev/run_generation_auto.py --study study_c --model-id psyche_r1_local --env mh-llm-local-env
python scripts/dev/run_generation_auto.py --study study_c --model-id psych_qwen_local --env mh-llm-local-env
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study study_c --model-id qwen3_lmstudio --env mh-llm-benchmark-env
python scripts/dev/run_generation_auto.py --study study_c --model-id qwq --env mh-llm-benchmark-env
python scripts/dev/run_generation_auto.py --study study_c --model-id deepseek_r1_lmstudio --env mh-llm-benchmark-env
python scripts/dev/run_generation_auto.py --study study_c --model-id gpt_oss --env mh-llm-benchmark-env
python scripts/dev/run_generation_auto.py --study study_c --model-id psyllm --env mh-llm-benchmark-env
python scripts/dev/run_generation_auto.py --study study_c --model-id psyllm_gml_local --env mh-llm-local-env
python scripts/dev/run_generation_auto.py --study study_c --model-id piaget_local --env mh-llm-local-env
python scripts/dev/run_generation_auto.py --study study_c --model-id psyche_r1_local --env mh-llm-local-env
python scripts/dev/run_generation_auto.py --study study_c --model-id psych_qwen_local --env mh-llm-local-env
```

## Metrics Commands

### Mac/Linux
```bash
RUN_TAG="$(date +%Y%m%d_%H%M)"; OUT_ROOT="metric-results/misc/${RUN_TAG}"; mkdir -p "${OUT_ROOT}/study_c"; conda run -n mh-llm-benchmark-env env PYTHONPATH=src python scripts/studies/study_c/metrics/calculate_metrics.py --use-cleaned --use-nli --nli-stride 2 --output-dir "${OUT_ROOT}/study_c"
```

### Windows (PC)
```powershell
$RUN_TAG=Get-Date -Format "yyyyMMdd_HHmm"; $OUT_ROOT="metric-results/misc/$RUN_TAG"; New-Item -ItemType Directory -Force -Path "$OUT_ROOT/study_c" | Out-Null; conda run -n mh-llm-benchmark-env python scripts/studies/study_c/metrics/calculate_metrics.py --use-cleaned --use-nli --nli-stride 2 --output-dir "$OUT_ROOT/study_c"
```

## Workers
- `run_study_c_generate_only.py` exposes `--workers` and `--progress-interval-seconds`.
- Default worker policy is fail-closed:
  - LM Studio runners: auto `4` workers.
  - Non-LM runners: auto `1` worker.
- Study C parallelises by case only; each case preserves turn order and context accumulation.

## Useful Checks
```bash
python scripts/dev/run_generation_auto.py --study study_c --model-id gpt_oss --check-only
python hf-local-scripts/run_study_c_generate_only.py --model-id gpt_oss --max-cases 2 --workers 4 --progress-interval-seconds 10
```
