# Model Configuration and Precision Strategy (Mac)

Mac runs are limited to 8B-class models. Larger models (32B+) remain on the Uni setup for full-precision reporting.

## Mac-local canonical configs

- **Qwen3-8B (MLX or HF Inference)**
  - Weights: `Qwen/Qwen2.5-8B-Instruct`.
  - Preferred Mac path: load into LM Studio with MLX, start `http://localhost:1234/v1`, and set `model_name` to the LM Studio identifier (e.g., `qwen3-8b-mlx`) when constructing `PsyLLMRunner`.
  - Alternative: Hugging Face Inference API via `HUGGINGFACE_API_KEY` (precision handled server-side; avoids local VRAM limits).

- **PsyLLM-8B (LM Studio)**
  - Weights: `GMLHUHE/PsyLLM`.
  - Run through LM Studio Local Server at `http://localhost:1234/v1` via `PsyLLMRunner`.
  - Precision: full precision on MLX/MPS; fall back to CPU only for debugging.

## Excluded from Mac

- QwQ-32B, DeepSeek-R1-32B, GPT-OSS-120B and other ≥32B models are evaluated on the Uni setup only; do not schedule them on the Mac.

## Quick invocation examples (Mac)

```bash
cd "Assignment 2/reliable_clinical_benchmark/Mac-setup"
source .mh-llm-benchmark-env/bin/activate

# Qwen3-8B via LM Studio (keep runs small)
PYTHONPATH=src python scripts/run_evaluation.py --model qwen3-8b-mlx --study A --max-samples 5

# PsyLLM-8B via LM Studio
PYTHONPATH=src python scripts/run_evaluation.py --model PsyLLM-8B --study B --max-samples 5
```
