# Study B Commands

## Scope
Study B has two generation runners:
- Single-turn: `results/<model-folder>/study_b_generations.jsonl`
- Multi-turn: `results/<model-folder>/study_b_multi_turn_generations.jsonl`

Metrics write to `metric-results/.../study_b`.

## One-Time Setup

### Mac/Linux
```bash
cd "/Users/ryangichuru/Documents/SSD-K/Uni/3rd year/NLP/Assignment 2/reliable_clinical_benchmark/Uni-setup"
```

### Windows (PC)
```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
```

## Single-Turn Generation Commands

### Mac/Linux
```bash
python scripts/dev/run_generation_auto.py --study study_b --model-id qwen3_lmstudio --env mh-llm-benchmark-env --workers 6
python scripts/dev/run_generation_auto.py --study study_b --model-id qwq --env mh-llm-benchmark-env --workers 6
python scripts/dev/run_generation_auto.py --study study_b --model-id deepseek_r1_lmstudio --env mh-llm-benchmark-env --workers 4
python scripts/dev/run_generation_auto.py --study study_b --model-id gpt_oss --env mh-llm-benchmark-env --workers 2
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study study_b --model-id qwen3_lmstudio --env mh-llm-benchmark-env --workers 6
python scripts/dev/run_generation_auto.py --study study_b --model-id qwq --env mh-llm-benchmark-env --workers 6
python scripts/dev/run_generation_auto.py --study study_b --model-id deepseek_r1_lmstudio --env mh-llm-benchmark-env --workers 4
python scripts/dev/run_generation_auto.py --study study_b --model-id gpt_oss --env mh-llm-benchmark-env --workers 2
```

## vLLM (Local HF Models Only)

This section is for the four HF-local models when served via vLLM's OpenAI-compatible server.

- Client `--workers` should remain `1`.
- Concurrency is controlled server-side via `--max-num-seqs` (recommended sweep: `2 → 4 → 8 → 12`).

### vLLM Server (per model)

#### PsyLLM-8B (`psyllm_gml_vllm`, default port 8101)

Mac/Linux:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model "GMLHUHE/PsyLLM-8B" \
  --download-dir "./models/vllm" \
  --host 0.0.0.0 \
  --port 8101 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 4 \
  --enforce-eager
```

Windows (PC) / WSL:
```bash
python -m vllm.entrypoints.openai.api_server --model "GMLHUHE/PsyLLM-8B" --download-dir "./models/vllm" --host 0.0.0.0 --port 8101 --gpu-memory-utilization 0.9 --max-num-seqs 4 --enforce-eager
```

#### Piaget-8B (`piaget_vllm`, default port 8102)

Mac/Linux:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model "gustavecortal/Piaget-8B" \
  --download-dir "./models/vllm" \
  --host 0.0.0.0 \
  --port 8102 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 4 \
  --enforce-eager
```

Windows (PC) / WSL:
```bash
python -m vllm.entrypoints.openai.api_server --model "gustavecortal/Piaget-8B" --download-dir "./models/vllm" --host 0.0.0.0 --port 8102 --gpu-memory-utilization 0.9 --max-num-seqs 4 --enforce-eager
```

#### Psyche-R1 (`psyche_r1_vllm`, default port 8103)

Mac/Linux:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model "MindIntLab/Psyche-R1" \
  --download-dir "./models/vllm" \
  --host 0.0.0.0 \
  --port 8103 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 4 \
  --enforce-eager
```

Windows (PC) / WSL:
```bash
python -m vllm.entrypoints.openai.api_server --model "MindIntLab/Psyche-R1" --download-dir "./models/vllm" --host 0.0.0.0 --port 8103 --gpu-memory-utilization 0.9 --max-num-seqs 4 --enforce-eager
```

#### Psych_Qwen_32B (`psych_qwen_vllm`, default port 8104)

Mac/Linux:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model "Compumacy/Psych_Qwen_32B" \
  --download-dir "./models/vllm" \
  --host 0.0.0.0 \
  --port 8104 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 4 \
  --enforce-eager \
  --quantization bitsandbytes \
  --load-format bitsandbytes
```

Windows (PC) / WSL:
```bash
python -m vllm.entrypoints.openai.api_server --model "Compumacy/Psych_Qwen_32B" --download-dir "./models/vllm" --host 0.0.0.0 --port 8104 --gpu-memory-utilization 0.9 --max-num-seqs 4 --enforce-eager --quantization bitsandbytes --load-format bitsandbytes
```

### vLLM Single-Turn Generation Commands

#### Mac/Linux

```bash
python scripts/dev/run_generation_auto.py --study study_b --model-id psyllm_gml_vllm --env mh-llm-benchmark-env
python scripts/dev/run_generation_auto.py --study study_b --model-id piaget_vllm --env mh-llm-benchmark-env
python scripts/dev/run_generation_auto.py --study study_b --model-id psyche_r1_vllm --env mh-llm-benchmark-env
python scripts/dev/run_generation_auto.py --study study_b --model-id psych_qwen_vllm --env mh-llm-benchmark-env
```

#### Windows (PC)

```powershell
python scripts/dev/run_generation_auto.py --study study_b --model-id psyllm_gml_vllm --env mh-llm-benchmark-env
python scripts/dev/run_generation_auto.py --study study_b --model-id piaget_vllm --env mh-llm-benchmark-env
python scripts/dev/run_generation_auto.py --study study_b --model-id psyche_r1_vllm --env mh-llm-benchmark-env
python scripts/dev/run_generation_auto.py --study study_b --model-id psych_qwen_vllm --env mh-llm-benchmark-env
```

## Multi-Turn Generation Commands

### Mac/Linux
```bash
python scripts/dev/run_generation_auto.py --study study_b_multi_turn --model-id qwen3_lmstudio --env mh-llm-benchmark-env --workers 6
python scripts/dev/run_generation_auto.py --study study_b_multi_turn --model-id qwq --env mh-llm-benchmark-env --workers 6
python scripts/dev/run_generation_auto.py --study study_b_multi_turn --model-id deepseek_r1_lmstudio --env mh-llm-benchmark-env --workers 4
python scripts/dev/run_generation_auto.py --study study_b_multi_turn --model-id gpt_oss --env mh-llm-benchmark-env --workers 2
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study study_b_multi_turn --model-id qwen3_lmstudio --env mh-llm-benchmark-env --workers 6
python scripts/dev/run_generation_auto.py --study study_b_multi_turn --model-id qwq --env mh-llm-benchmark-env --workers 6
python scripts/dev/run_generation_auto.py --study study_b_multi_turn --model-id deepseek_r1_lmstudio --env mh-llm-benchmark-env --workers 4
python scripts/dev/run_generation_auto.py --study study_b_multi_turn --model-id gpt_oss --env mh-llm-benchmark-env --workers 2
```

### vLLM Multi-Turn Generation Commands (Local HF Models Only)

#### Mac/Linux

```bash
python scripts/dev/run_generation_auto.py --study study_b_multi_turn --model-id psyllm_gml_vllm --env mh-llm-benchmark-env
python scripts/dev/run_generation_auto.py --study study_b_multi_turn --model-id piaget_vllm --env mh-llm-benchmark-env
python scripts/dev/run_generation_auto.py --study study_b_multi_turn --model-id psyche_r1_vllm --env mh-llm-benchmark-env
python scripts/dev/run_generation_auto.py --study study_b_multi_turn --model-id psych_qwen_vllm --env mh-llm-benchmark-env
```

#### Windows (PC)

```powershell
python scripts/dev/run_generation_auto.py --study study_b_multi_turn --model-id psyllm_gml_vllm --env mh-llm-benchmark-env
python scripts/dev/run_generation_auto.py --study study_b_multi_turn --model-id piaget_vllm --env mh-llm-benchmark-env
python scripts/dev/run_generation_auto.py --study study_b_multi_turn --model-id psyche_r1_vllm --env mh-llm-benchmark-env
python scripts/dev/run_generation_auto.py --study study_b_multi_turn --model-id psych_qwen_vllm --env mh-llm-benchmark-env
```

## Metrics Commands

### Mac/Linux
```bash
RUN_TAG="$(date +%Y%m%d_%H%M)"; OUT_ROOT="metric-results/misc/${RUN_TAG}"; mkdir -p "${OUT_ROOT}/study_b"; conda run -n mh-llm-benchmark-env env PYTHONPATH=src python scripts/studies/study_b/metrics/calculate_metrics.py --use-cleaned --output-dir "${OUT_ROOT}/study_b"
```

### Windows (PC)
```powershell
$RUN_TAG=Get-Date -Format "yyyyMMdd_HHmm"; $OUT_ROOT="metric-results/misc/$RUN_TAG"; New-Item -ItemType Directory -Force -Path "$OUT_ROOT/study_b" | Out-Null; conda run -n mh-llm-benchmark-env python scripts/studies/study_b/metrics/calculate_metrics.py --use-cleaned --output-dir "$OUT_ROOT/study_b"
```

## Workers
- `run_study_b_generate_only.py` and `run_study_b_multi_turn_generate_only.py` expose `--workers` and `--progress-interval-seconds`.
- Default worker policy is fail-closed:
  - LM Studio runners: auto `4` workers.
  - Non-LM runners: auto `1` worker.
- Single-turn generation parallelises per prompt variant.
- Multi-turn generation parallelises by case only; each case stays sequential by turn.

## Useful Checks
```bash
python scripts/dev/run_generation_auto.py --study study_b --model-id gpt_oss --check-only
python scripts/dev/run_generation_auto.py --study study_b_multi_turn --model-id gpt_oss --check-only
python hf-local-scripts/run_study_b_generate_only.py --model-id gpt_oss --max-samples 5 --workers 4 --progress-interval-seconds 10
python hf-local-scripts/run_study_b_multi_turn_generate_only.py --model-id gpt_oss --max-samples 2 --workers 4 --progress-interval-seconds 10
```
