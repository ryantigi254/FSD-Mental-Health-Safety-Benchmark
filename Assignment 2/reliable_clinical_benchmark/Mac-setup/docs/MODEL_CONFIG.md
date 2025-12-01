# Model Configuration and Precision Strategy

This file documents the **canonical model configurations** used for the benchmark and how they relate to the Mac vs Uni setups. For publication, all reported results must come from **one fixed configuration per model**.

## 1. Canonical (publication) configurations

These are the configs to use for the actual Study A/B/C runs that go into tables, figures, and safety cards.

- **PsyLLM-8B**
  - **Weights**: full Hugging Face `GMLHUHE/PsyLLM` (no further quantisation).
  - **Precision**: `float16` (GPU) or `float32` (MPS/CPU if needed).
  - **Device (canonical)**: Uni GPU (preferred) or Uni CPU/MPS if necessary.

- **Qwen3-8B**
  - **Weights**: standard HF instruct checkpoint (`Qwen/Qwen2.5-8B-Instruct`).
  - **Precision**: `float16` on GPU (or `float32` on MPS/CPU).
  - **Device (canonical)**: Uni GPU.

- **QwQ-32B**
  - **Weights**: `Qwen/QwQ-32B-Preview`.
  - **Precision**: 8‑bit quantised (e.g. via bitsandbytes/HF 8‑bit loading).
  - **Device (canonical)**: Uni GPU with sufficient VRAM.

- **DeepSeek-R1-32B**
  - **Weights**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`.
  - **Precision**: 8‑bit quantised.
  - **Device (canonical)**: Uni GPU.

- **GPT-OSS-120B**
  - **Weights**: GPT‑OSS 120B checkpoint (API/remote where applicable).
  - **Precision**: 8‑bit or backend‑specific reduced precision, as required by the serving stack.
  - **Device (canonical)**: high‑VRAM server (A100/H100 class), accessed via API from Uni.

For the **methods section**, you should describe these as the “deployment‑grade” configurations, e.g.:

> “All reported benchmark results were obtained using fp16 for 8B models (PsyLLM, Qwen3‑8B) and 8‑bit quantised weights for the 32B and 120B models (QwQ‑32B, DeepSeek‑R1‑32B, GPT‑OSS‑120B).”

## 2. Mac vs Uni usage

- **Uni setup**
  - Runs the **canonical experiments** for all models.
  - All JSON result files used by the analysis notebooks and the paper come from this setup.
  - Precision/quantisation per model must match the list above and stay fixed across all three studies.

- **Mac setup**
  - Used primarily for **development and smoke tests**:
    - Debugging prompts and pipelines.
    - Running the metrics and plotting notebooks on small subsets.
  - May use:
    - **Full precision (float32/float16)** for smaller models (PsyLLM‑8B, Qwen3‑8B) where memory allows.
    - Heavier quantisation if needed for quick tests.
  - **Mac runs are not used as sources of final reported metrics** in the dissertation; they are for engineering convenience only.

## 3. Device selection in local helpers

The local PsyLLM helper script (`psy-llm-local/infer.py`) follows this policy:

- Respects `PSY_DEVICE` (`cuda | mps | cpu | mlx`) when set.
- Otherwise auto‑detects: **CUDA > MPS > CPU**.
- Prints an explicit message if `PSY_DEVICE=mlx` is requested, since this script uses PyTorch/transformers and MLX requires a separate conversion path (see `Assignment 2/docs/psyllm_setup.md`).

This keeps local Mac experiments aligned with the documented `PSY_DEVICE` semantics without changing the canonical Uni‑side configurations.


