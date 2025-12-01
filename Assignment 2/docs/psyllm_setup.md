# Model Download, Setup, and Weight Conversion Process

This guide captures two parallel workflows for running PsyLLM locally:

1. **Raw PyTorch/Transformers inference** – mirrors the FastAPI provider in `system-sandbox/backend/app/services/llm/providers/psy_llm.py` so you can serve the model without MLX.
2. **MLX conversion** – keeps the Apple Silicon-optimized flow documented previously.

Follow the path that matches your hardware; you can keep both copies (`models/PsyLLM/` for PyTorch, `models/PsyLLM-mlx/` for MLX) if you switch between devices.

---

## A. Running PsyLLM via PyTorch/Transformers (no MLX)

> Applies to `Assignment 2/system-sandbox` (or whatever folder you cloned the sandbox into). All commands assume you run them from that root unless noted.

### Step A1 – Install missing dependencies

Your `requirements.txt` currently lacks the PsyLLM stack. Install Torch + Transformers manually inside the backend venv:

```bash
# CUDA 11.8 build (recommended for NVIDIA GPUs)
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate

# CPU-only fallback (slow, but works if you lack CUDA)
pip install torch transformers accelerate
```

If you rely on AutoGPTQ/bitsandbytes later, add those after confirming GPU compatibility.

### Step A2 – Prepare the local models directory

The provider resolves `../../../../../models/PsyLLM` relative to `backend/app/services/llm/providers/psy_llm.py`. Mirror that expectation by creating the folder at the sandbox root:

```bash
mkdir -p models/PsyLLM
```

On Windows PowerShell from `system-sandbox`, run:

```powershell
New-Item -ItemType Directory -Force models\PsyLLM
```

### Step A3 – Download weights with `huggingface-cli`

Install the CLI and authenticate (needed if `GMLHUHE/PsyLLM` is gated):

```bash
pip install huggingface_hub
huggingface-cli login  # paste your HF token if prompted

huggingface-cli download GMLHUHE/PsyLLM \
  --local-dir models/PsyLLM \
  --local-dir-use-symlinks False
```

You should now see files such as `config.json`, `tokenizer.json`, `model.safetensors.index.json`, and `model-0000x-of-00004.safetensors` beneath `models/PsyLLM/`.

### Step A4 – Configure the backend `.env`

Create or edit `system-sandbox/backend/.env` so the FastAPI service points at PsyLLM:

```ini
PROVIDER=psy

# Valid devices: cuda | mps | cpu. Do NOT set mlx here – that path is only for the MLX exporter.
PSY_DEVICE=cuda

# Optional overrides
PSY_MAX_NEW_TOKENS=1024
# PSY_MODEL_PATH=C:/absolute/path/to/system-sandbox/models/PsyLLM
# PSY_DTYPE=float16  # change to bfloat16 or float32 if your hardware requires it
```

If you lack a GPU, set `PSY_DEVICE=cpu` and expect slower generations.

### Step A5 – Launch and verify

```bash
cd backend
uvicorn app.main:app --reload
```

The startup logs should show `Loading PsyLLM from …/models/PsyLLM using AutoModelForCausalLM`. If you hit CUDA OOM, reduce `PSY_MAX_NEW_TOKENS`, switch `PSY_DTYPE` to `bfloat16`, or enable 8-bit loading by editing `psy_llm.py` to pass `load_in_8bit=True` (requires `bitsandbytes`).

---

## B. MLX Weight Conversion (Apple Silicon)

### 1. Original Model Download (PyTorch/Hugging Face Format)

The PsyLLM model was downloaded from Hugging Face using the repository `GMLHUHE/PsyLLM`. This is the original PyTorch format stored in `Prototypes/v1/models/PsyLLM/`.

**Download command:**

```bash
mkdir -p Prototypes/v1/models
cd Prototypes/v1/models
huggingface-cli download GMLHUHE/PsyLLM --local-dir-use-symlinks False --local-dir PsyLLM
```

**What was downloaded:**

- 4 split safetensors files (`model-00001-of-00004.safetensors` through `model-00004-of-00004.safetensors`)
- Tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`)
- Model configuration (`config.json`, `generation_config.json`)
- Index file (`model.safetensors.index.json`)

This format works with PyTorch/MPS/CUDA but not directly with Apple's MLX framework.

## 2. MLX Environment Setup

Before conversion, an MLX conda environment was created:

```bash
conda create -y -n mlx-env python=3.11
conda run -n mlx-env python -m pip install mlx mlx-lm --break-system-packages
```

This installs:
- `mlx`: Apple's MLX framework for efficient ML on Apple Silicon
- `mlx-lm`: MLX's language model utilities, including the conversion tool

## 3. Weight Conversion Process

The conversion script (`scripts/convert_psyllm_to_mlx.py`) uses `mlx_lm.convert()` to transform the Hugging Face weights into MLX format.

**Conversion command:**

```bash
conda run -n mlx-env python scripts/convert_psyllm_to_mlx.py \
  --source Prototypes/v1/models/PsyLLM \
  --output Prototypes/v1/models/PsyLLM-mlx \
  --force \
  --quantize
```

**What happens during conversion:**

- The 4 split safetensors files are merged into a single `model.safetensors` file
- Weights are converted from PyTorch format to MLX's native format
- Optional quantization (`--quantize` flag) reduces weights to 4-bit precision (smaller, faster, but slight quality trade-off)
- Tokenizer files are copied/adapted for MLX compatibility
- A `chat_template.jinja` file is generated for MLX's chat template system

**Result:** The converted model is stored in `Prototypes/v1/models/PsyLLM-mlx/` with:
- Single `model.safetensors` file (instead of 4 split files)
- MLX-compatible tokenizer and config files
- `chat_template.jinja` for chat formatting

## 4. Why Both Formats Exist

- **`PsyLLM/` (PyTorch)**: Original format for PyTorch/MPS/CUDA backends
- **`PsyLLM-mlx/` (MLX)**: Converted format optimized for Apple Silicon, offering better performance on M-series Macs

The sandbox system uses the MLX version (`PSY_DEVICE=mlx`) for faster inference on Apple hardware, while the main prototype can use either format depending on the `PSY_DEVICE` setting in the environment configuration.

When running the Assignment 2 benchmark, point the study configs at either folder via the `model_path` field depending on your device.
