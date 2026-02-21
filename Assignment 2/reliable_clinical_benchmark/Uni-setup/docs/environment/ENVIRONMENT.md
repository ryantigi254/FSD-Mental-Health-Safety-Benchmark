# Environment Setup (Uni-setup)

Uni-setup uses **two** Python environments in practice:

- **`mh-llm-benchmark-env`**: the **general** benchmark environment for running Studies A/B/C, metrics, and pytest.
  - Installs pinned packages from `Uni-setup/requirements.txt` (notably `transformers==4.40.1` and `pydantic<2.0`).
  - Used for LM Studio runners and evaluation pipelines.

- **`mh-llm-local-env`**: a **local HF inference** environment used when running **local PyTorch model runners** that require a more modern `transformers` stack (e.g., Qwen3-style `enable_thinking` chat templating).
  - This was used for the Piaget local run to avoid dependency clashes with the benchmark pins.

Both environments should be isolated (no base Anaconda writes, no user-site bleed).

## `mh-llm-benchmark-env` (general)

Create and use this for **evaluation** and **tests**:

```powershell
cd "Assignment 2\reliable_clinical_benchmark\Uni-setup"
conda create -n mh-llm-benchmark-env python=3.10 -y
conda activate mh-llm-benchmark-env   # adjust if Anaconda setup differs

pip install -r requirements.txt
# spaCy model via scispaCy S3 (matches spaCy 3.6.1)
python -m pip install --no-deps https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
python -m spacy validate
```

### What runs in this env

- `scripts/evaluation/run_evaluation.py` (Studies A/B/C; generate-only and from-cache)
- pytest:

```powershell
$Env:PYTHONPATH="src"
pytest tests/unit -v
pytest tests/integration -v
```

## `mh-llm-local-env` (local HF inference)

Use this when you want to run **local HF runners** that load weights from `Uni-setup/models/` (Piaget, Psyche-R1, Psych_Qwen_32B, PsyLLM).

### Setup

```powershell
cd "Assignment 2\reliable_clinical_benchmark\Uni-setup"
conda create -n mh-llm-local-env python=3.10 -y
conda activate mh-llm-local-env   # adjust if Anaconda setup differs

# Install PyTorch with CUDA support (REQUIRED for GPU inference)
# Check your CUDA version with: nvidia-smi
# For CUDA 12.1 (most common): use cu121
# For CUDA 11.8: use cu118
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies with latest transformers (needed for modern chat templates)
# bitsandbytes is REQUIRED for quantised models (e.g., Psych_Qwen_32B with 4-bit quantisation)
pip install transformers accelerate bitsandbytes

# Install vLLM for OpenAI-compatible local model serving
pip install vllm
pip install uvloop  # High-performance async event loop (required by vLLM on Linux/Mac)

# Install other requirements (may upgrade transformers beyond pinned version)
pip install -r requirements.txt --upgrade transformers

# For models requiring TensorFlow (e.g., some local runners)
# pip install tensorflow>=2.13.0  # Only if needed for specific models
```

**Important:** If you install PyTorch without the CUDA index URL, you'll get the CPU-only build (`torch-2.x.x+cpu`), which will not detect your GPU. Always use the CUDA-specific index URL matching your CUDA version (check with `nvidia-smi`).

### What runs in this env

- **Local HF model runners**: `piaget_local`, `psyche_r1_local`, `psych_qwen_local`, `psyllm_gml_local`
- **vLLM runners**: `psyllm_gml_vllm`, `piaget_vllm`, `psyche_r1_vllm`, `psych_qwen_vllm` (served via `python -m vllm.entrypoints.openai.api_server`)
- **Generation scripts**: `hf-local-scripts/run_study_*_generate_only.py` (when using local models)
- Models that require:
  - Modern `transformers` versions (for Qwen3-style `enable_thinking` chat templating)
  - **`bitsandbytes`** for quantised inference (required for Psych_Qwen_32B 4-bit/8-bit quantisation on limited VRAM)
  - **`vllm`** for OpenAI-compatible local model serving
  - Higher TensorFlow versions (if needed by specific model implementations)

### Key points

- **Keep it separate from `mh-llm-benchmark-env`** so the pinned `transformers==4.38.2` does not block newer chat-template features.
- **PyTorch with CUDA support is required** for GPU inference. Install using the CUDA-specific index URL (e.g., `--index-url https://download.pytorch.org/whl/cu121` for CUDA 12.1). Without this, PyTorch will be CPU-only and models won't detect your GPU.
- **`bitsandbytes` is required** for running quantised models (e.g., Psych_Qwen_32B with `quantization="4bit"`). Without it, you'll get `PackageNotFoundError: No package metadata was found for bitsandbytes`.
- Prefer running pip via the env python and disable user-site packages (`PYTHONNOUSERSITE=1`) to avoid package bleed.
- This environment was specifically created for Piaget and other local models that need a more modern dependency stack.

## `mh-llm-hf-vllm-env` (vLLM-backed local servers)

Use this environment when you want to serve the HF-local models via **vLLM** and talk to them through an OpenAI-compatible HTTP API (rather than loading weights directly inside Uni-setup).

### Setup

```powershell
cd "Assignment 2\reliable_clinical_benchmark\Uni-setup"

conda create -n mh-llm-hf-vllm-env python=3.10 -y
conda activate mh-llm-hf-vllm-env   # adjust if Anaconda setup differs

# Install vLLM and core HF deps.
# Pick the CUDA build that matches your driver (see vLLM docs + `nvidia-smi`):
pip install "vllm[cu121]" transformers accelerate safetensors
```

If you are on CPU-only, you can omit the CUDA extra and install `vllm` without `[cuXXX]`, but throughput will be substantially lower.

### What runs in this env

- vLLM OpenAI-compatible servers for:
  - `GMLHUHE/PsyLLM-8B`
  - `gustavecortal/Piaget-8B`
  - `MindIntLab/Psyche-R1`
  - `Compumacy/Psych_Qwen_32B`
- Study A benchmark runs that talk to these servers via new `*_vllm` runners.

### Example vLLM server commands

Run one model server at a time when benchmarking:

```powershell
conda activate mh-llm-hf-vllm-env

# PsyLLM-8B on port 8101
python -m vllm.entrypoints.openai.api_server `
  --model GMLHUHE/PsyLLM-8B `
  --host 0.0.0.0 `
  --port 8101 `
  --gpu-memory-utilization 0.9 `
  --max-num-seqs 8

# Piaget-8B on port 8102
python -m vllm.entrypoints.openai.api_server `
  --model gustavecortal/Piaget-8B `
  --host 0.0.0.0 `
  --port 8102 `
  --gpu-memory-utilization 0.9 `
  --max-num-seqs 8
```

Adjust `--max-num-seqs` per benchmark run (e.g. 2, 4, 8, 12) and ports as needed. Uni-setup will connect to these via new vLLM-backed `ModelRunner` classes.

## Local weight storage

Local weights live under `Uni-setup/models/` and must remain **untracked** (gitignored).

## Runtime caches / offload directories

Some local runners may create runtime directories (e.g., `offload/psyche_r1`) during generation. These are safe to delete; they will be re-created when needed.
