# Study A Generation Commands

## Scope
Study A generation writes to `results/<model-folder>/study_a_generations.jsonl`.

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
python scripts/dev/run_generation_auto.py --study study_a --model-id qwen3_lmstudio --env mh-llm-benchmark-env
python scripts/dev/run_generation_auto.py --study study_a --model-id qwq --env mh-llm-benchmark-env
python scripts/dev/run_generation_auto.py --study study_a --model-id deepseek_r1_lmstudio --env mh-llm-benchmark-env
python scripts/dev/run_generation_auto.py --study study_a --model-id gpt_oss --env mh-llm-benchmark-env
python scripts/dev/run_generation_auto.py --study study_a --model-id psyllm_gml_local --env mh-llm-local-env
python scripts/dev/run_generation_auto.py --study study_a --model-id piaget_local --env mh-llm-local-env
python scripts/dev/run_generation_auto.py --study study_a --model-id psyche_r1_local --env mh-llm-local-env
python scripts/dev/run_generation_auto.py --study study_a --model-id psych_qwen_local --env mh-llm-local-env
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study study_a --model-id qwen3_lmstudio --env mh-llm-benchmark-env
python scripts/dev/run_generation_auto.py --study study_a --model-id qwq --env mh-llm-benchmark-env
python scripts/dev/run_generation_auto.py --study study_a --model-id deepseek_r1_lmstudio --env mh-llm-benchmark-env
python scripts/dev/run_generation_auto.py --study study_a --model-id gpt_oss --env mh-llm-benchmark-env
python scripts/dev/run_generation_auto.py --study study_a --model-id psyllm_gml_local --env mh-llm-local-env
python scripts/dev/run_generation_auto.py --study study_a --model-id piaget_local --env mh-llm-local-env
python scripts/dev/run_generation_auto.py --study study_a --model-id psyche_r1_local --env mh-llm-local-env
python scripts/dev/run_generation_auto.py --study study_a --model-id psych_qwen_local --env mh-llm-local-env
```

## Metrics Commands

### Mac/Linux
```bash
conda run -n mh-llm-benchmark-env env PYTHONPATH=src python scripts/studies/study_a/metrics/calculate_metrics.py --use-cleaned --output-dir metric-results/study_a
conda run -n mh-llm-benchmark-env env PYTHONPATH=src python scripts/studies/study_a/metrics/calculate_bias.py --use-cleaned --output-dir metric-results/study_a
conda run -n mh-llm-benchmark-env env PYTHONPATH=src python scripts/studies/study_a/metrics/calculate_metrics.py --use-cleaned --output-dir metric-results/study_a
```

### Windows (PC)
```powershell
conda run -n mh-llm-benchmark-env python scripts/studies/study_a/metrics/calculate_metrics.py --use-cleaned --output-dir metric-results/study_a
conda run -n mh-llm-benchmark-env python scripts/studies/study_a/metrics/calculate_bias.py --use-cleaned --output-dir metric-results/study_a
conda run -n mh-llm-benchmark-env python scripts/studies/study_a/metrics/calculate_metrics.py --use-cleaned --output-dir metric-results/study_a
```

## Workers
`study_a` generation does not expose a `--workers` flag. Throughput for Study A is controlled by LM Studio model load settings (`Max Concurrent Predictions`).

## Useful Checks
```bash
python scripts/dev/run_generation_auto.py --study study_a --model-id gpt_oss --check-only
python hf-local-scripts/run_study_a_generate_only.py --model-id gpt_oss --max-samples 5
```
