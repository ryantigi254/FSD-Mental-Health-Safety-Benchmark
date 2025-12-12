# Model Configuration and Precision Strategy

Public summary of canonical model configurations used across Studies A, B, and C. All models run at full precision (fp16/bf16 or fp32 as applicable) except GPT-OSS-120B, which follows its backend precision.

## 1. Canonical configurations

Use a fixed configuration per model for all reported results.

- **PsyLLM-8B**
  - Weights: `GMLHUHE/PsyLLM` (full checkpoint).
  - Precision: fp16 on GPU; fp32 on CPU/MPS if needed.
  - Device: primary GPU workstation (dual RTX 4090) or CPU/MPS fallback for development.

- **Qwen3-8B**
  - Weights: `Qwen/Qwen2.5-8B-Instruct`.
  - Precision: fp16 on GPU; fp32 on CPU/MPS if needed.
  - Device: primary GPU workstation.

- **QwQ-32B**
  - Weights: `Qwen/QwQ-32B-Preview`.
  - Precision: fp16 on GPU.
  - Device: primary GPU workstation with sufficient VRAM.

- **DeepSeek-R1-14B**
  - Weights: `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`.
  - Precision: fp16 on GPU.
  - Device: primary GPU workstation.

- **DeepSeek-R1-Distill-Llama-70B**
  - Weights: `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` (LM Studio GGUF Q6_K for local runs).
  - Precision: Q6_K quant in LM Studio (fixed for reported runs).
  - Device: high-VRAM LM Studio host.

- **GPT-OSS-20B**
  - Weights: LM Studio GGUF (e.g., `lmstudio-community/openai-gpt-oss-20b-gguf-*`).
  - Precision: LM Studio quant (recorded as deployed, e.g., MXFP4/FP16; keep fixed).
  - Device: LM Studio host with GPU.

- **DeepSeek-R1-Distill-Llama-70B**
  - Weights: `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` (LM Studio GGUF Q6_K for local runs).
  - Precision: Q6_K quant in LM Studio (fixed for reported runs).
  - Device: high-VRAM LM Studio host.

- **GPT-OSS-20B**
  - Weights: LM Studio GGUF (e.g., `lmstudio-community/openai-gpt-oss-20b-gguf-*`).
  - Precision: LM Studio quant (recorded as deployed, e.g., MXFP4/FP16; keep fixed).
  - Device: LM Studio host with GPU.

- **GPT-OSS-120B**
  - Weights: GPT-OSS 120B (API/remote endpoint).
  - Precision: backend-defined (API-managed reduced precision).
  - Device: high-VRAM server accessed via API.

Example methods text:

> Reported results use full precision for locally run models (fp16 on GPU, fp32 on CPU/MPS as needed). GPT-OSS-120B uses its provider’s backend precision.

## 2. Mac vs primary workstation

- **Primary workstation (dual RTX 4090)**
  - Runs canonical experiments for all locally executed models.
  - JSON results consumed by analysis notebooks and any published tables/figures.
  - Precision stays fixed per model as above.

- **Mac setup**
  - Used for development and smoke tests (prompts, small subsets, plotting).
  - May use fp16/fp32 as resources allow.
  - Not a source of final reported metrics.

## 3. Device selection in local helpers

`psy-llm-local/infer.py` policy:
- Respects `PSY_DEVICE` (`cuda | mps | cpu | mlx`).
- Otherwise auto-detects CUDA > MPS > CPU.
- Prints a notice if `PSY_DEVICE=mlx` because MLX requires a separate conversion path (see `Assignment 2/docs/psyllm_setup.md`).

This keeps local experiments aligned with documented device semantics while preserving the canonical GPU configurations for reported results.
