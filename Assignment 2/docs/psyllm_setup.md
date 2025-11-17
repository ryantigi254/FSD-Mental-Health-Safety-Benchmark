# Model Download, Setup, and Weight Conversion Process

This guide documents the complete process for downloading PsyLLM from Hugging Face and converting it to MLX format for efficient inference on Apple Silicon. This enables the CSY3055 benchmark to run on both PyTorch and Apple MLX backends.

## 1. Original Model Download (PyTorch/Hugging Face Format)

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
