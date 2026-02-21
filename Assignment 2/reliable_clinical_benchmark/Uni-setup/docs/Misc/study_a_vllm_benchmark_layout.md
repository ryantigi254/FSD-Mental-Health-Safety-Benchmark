# Study A vLLM Benchmark Layout (HF Local Models)

This document describes where Study A vLLM benchmark artefacts should live for the four HF-local models when using the vLLM-backed runners.

## Target models

- `psyllm_gml_vllm` → results folder: `results/psyllm-gml-local`
- `piaget_vllm` → results folder: `results/piaget-8b-local`
- `psyche_r1_vllm` → results folder: `results/psyche-r1-local`
- `psych_qwen_vllm` → results folder: `results/psych-qwen-32b-local`

## Benchmark control variable (vLLM concurrency)

vLLM throughput is controlled **server-side** (engine batching), not by the client `--workers` flag.

- **vLLM knob (server-side):** `--max-num-seqs <N>`
- **Directory naming convention:** `maxseq_<N>/`
- **Sweep order (recommended):** `2 → 4 → 8 → 12`

The generation client should remain at:

- `--workers 1` (vLLM handles batching internally)

## Required vLLM server config fields (per run)

Record these values in the per-run report (`reports/*.md` + `reports/*.json`) for reproducibility:

- `--model` (HF repo / local weights path)
- `--download-dir`
- `--host`, `--port`
- `--gpu-memory-utilization`
- `--max-num-seqs` (**this is the “worker count equivalent” for vLLM**)
- `--max-model-len`
- `--enforce-eager` (WSL-only; disables CUDAGraph but avoids nvcc issues)

Example server commands (per model):

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
  --enforce-eager \
  --max-model-len 4096
```

Windows (PC):
```powershell
python -m vllm.entrypoints.openai.api_server --model "GMLHUHE/PsyLLM-8B" --download-dir "./models/vllm" --host 0.0.0.0 --port 8101 --gpu-memory-utilization 0.9 --max-num-seqs 4 --enforce-eager --max-model-len 4096
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
  --enforce-eager \
  --max-model-len 4096
```

Windows (PC):
```powershell
python -m vllm.entrypoints.openai.api_server --model "gustavecortal/Piaget-8B" --download-dir "./models/vllm" --host 0.0.0.0 --port 8102 --gpu-memory-utilization 0.9 --max-num-seqs 4 --enforce-eager --max-model-len 4096
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
  --enforce-eager \
  --max-model-len 4096
```

Windows (PC):
```powershell
python -m vllm.entrypoints.openai.api_server --model "MindIntLab/Psyche-R1" --download-dir "./models/vllm" --host 0.0.0.0 --port 8103 --gpu-memory-utilization 0.9 --max-num-seqs 4 --enforce-eager --max-model-len 4096
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
  --max-model-len 4096 \
  --quantization bitsandbytes \
  --load-format bitsandbytes
```

Windows (PC):
```powershell
python -m vllm.entrypoints.openai.api_server --model "Compumacy/Psych_Qwen_32B" --download-dir "./models/vllm" --host 0.0.0.0 --port 8104 --gpu-memory-utilization 0.9 --max-num-seqs 4 --enforce-eager --max-model-len 4096 --quantization bitsandbytes --load-format bitsandbytes
```

Important constraint:

- The client `--max-tokens` must satisfy: `max_tokens <= max_model_len - max_expected_input_tokens`.

## Folder structure per model

Under each model’s `results/<model-folder>/misc` directory:

```text
results/
  <model-folder>/
    misc/
      vllm_benchmark/
        temp_generations/
          maxseq_2/
            study_a_generations_temp_maxseq2_s3_<date>.jsonl
          maxseq_4/
            study_a_generations_temp_maxseq4_s3_<date>.jsonl
          maxseq_8/
            study_a_generations_temp_maxseq8_s3_<date>.jsonl
          maxseq_12/
            study_a_generations_temp_maxseq12_s3_<date>.jsonl
        reports/
          study_a_vllm_benchmark_<date>.md
          study_a_vllm_benchmark_<date>.json
```

Notes:

- `maxseq_<N>` folders correspond to the vLLM `--max-num-seqs` setting used for that run.
- Filenames follow the pattern `study_a_generations_temp_maxseq<N>_s3_<date>.jsonl` for `max-samples=3`.
- Reports mirror the LM Studio worker-cutoff style but are scoped to vLLM concurrency settings (server-side `--max-num-seqs`).

## Example Study A command (per model-id)

With a vLLM server running for e.g. `psyllm_gml_vllm`:

```powershell
conda activate mh-llm-benchmark-env

python scripts/dev/run_generation_auto.py `
  --study study_a `
  --model-id psyllm_gml_vllm `
  --max-samples 3 `
  --workers 1 `
  --max-tokens 2048 `
  --cache-out results/psyllm-gml-local/misc/vllm_benchmark/temp_generations/maxseq_4/study_a_generations_temp_maxseq4_s3_20260212.jsonl `
  --progress-interval-seconds 10
```

The only change between runs is the `maxseq_<N>` directory and matching filename, corresponding to the `--max-num-seqs` value configured on the vLLM server.

