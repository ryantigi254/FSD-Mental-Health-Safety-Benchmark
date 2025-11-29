# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a research project evaluating **clinical reasoning reliability in LLMs** for mental-health support applications. The benchmark focuses on three critical failure modes: unfaithful reasoning (Study A), sycophantic agreement (Study B), and longitudinal drift (Study C).

**Models under evaluation:**
- **PsyLLM** (GMLHUHE/PsyLLM) - domain-tuned Qwen3-8B for counseling
- **Qwen3-8B** (Qwen/Qwen3-8B) - untuned baseline
- **GPT-OSS-20B** (openai/gpt-oss-20b) - larger generic baseline

## Repository Structure

```
Assignment 2/           # Main benchmark implementation
├── Mac-setup/         # Development setup for macOS (M-series Apple Silicon)
├── Uni-setup/         # Development setup for university compute (CUDA)
├── Prototypes/        # Experimental implementations and metric designs
├── docs/              # Setup guides and evaluation framework documentation
└── Literature Review/ # Research background and specification documents

Assignment/            # Original project proposal and planning materials
Course Material/       # NLP course notebooks (historical context only)
```

## Common Commands

### Model Inference Scripts

#### PsyLLM Direct Inference (PyTorch/Transformers)
The direct inference script runs PsyLLM without requiring LM Studio:

```bash
# Mac setup
cd "Assignment 2/Mac-setup/psy-llm-local"
python infer.py

# Uni setup
cd "Assignment 2/Uni-setup/psy-llm-local"
python infer.py
```

**Prerequisites:**
```bash
pip install torch transformers accelerate
```

**Note:** The model must be downloaded first. See "Model Setup" section below.

#### LM Studio API Scripts
For querying models served through LM Studio's API:

```bash
# Run Qwen3-8B query
cd "Assignment 2/Mac-setup/lmstudio-scripts"
python chat_qwen3_8b.py "Your prompt here" --temperature 0.7 --max-tokens 512

# Run GPT-OSS-20B query
python chat_gpt_oss_20b.py "Your prompt here" --temperature 0.7 --max-tokens 512
```

**Prerequisites:**
```bash
pip install requests
```

**Environment variables:**
- `LM_STUDIO_URL`: Override default http://127.0.0.1:1234/v1/chat/completions
- `LM_STUDIO_API_KEY`: Override default "lm-studio" key

## Model Setup

### PsyLLM Model Download

PsyLLM weights must be downloaded from Hugging Face before running inference:

```bash
# Install Hugging Face CLI if not already installed
pip install huggingface_hub

# Login to Hugging Face (required if model is gated)
huggingface-cli login

# Download to Mac setup
cd "Assignment 2/Mac-setup/psy-llm-local"
huggingface-cli download GMLHUHE/PsyLLM \
  --local-dir models/PsyLLM \
  --local-dir-use-symlinks False

# Download to Uni setup
cd "Assignment 2/Uni-setup/psy-llm-local"
huggingface-cli download GMLHUHE/PsyLLM \
  --local-dir models/PsyLLM \
  --local-dir-use-symlinks False
```

Downloaded files include:
- 4 split safetensors weight files (`model-00001-of-00004.safetensors` through `model-00004-of-00004.safetensors`)
- Tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`)
- Model configuration (`config.json`, `generation_config.json`)
- Weight index (`model.safetensors.index.json`)

**Important:** Model weights are intentionally excluded from git via `.gitignore`. Each development environment requires separate download.

### MLX Conversion (Apple Silicon Only)

For Apple Silicon Macs, PsyLLM can be converted to MLX format for optimized performance:

```bash
# Create MLX environment
conda create -y -n mlx-env python=3.11
conda run -n mlx-env python -m pip install mlx mlx-lm

# Convert weights (from project root)
conda run -n mlx-env python scripts/convert_psyllm_to_mlx.py \
  --source "Assignment 2/Prototypes/v1/models/PsyLLM" \
  --output "Assignment 2/Prototypes/v1/models/PsyLLM-mlx" \
  --force \
  --quantize
```

**MLX benefits:**
- Faster inference on M-series chips
- Optional 4-bit quantization reduces memory footprint
- Single merged `model.safetensors` file instead of 4 splits

### Device Compatibility

**Mac-setup:** Optimized for Apple Silicon (MPS/MLX)
- `infer.py` automatically selects: CUDA > MPS > CPU
- MPS uses `float32` instead of `float16` for stability

**Uni-setup:** Optimized for NVIDIA CUDA
- Uses `torch.float16` when CUDA available
- Falls back to CPU with `float32` if no GPU

## Architecture Notes

### Two-Setup Design Pattern

The project maintains parallel `Mac-setup/` and `Uni-setup/` directories to handle environment-specific requirements:

**Why separate setups:**
- Different hardware acceleration (Apple MLX/MPS vs NVIDIA CUDA)
- Different model formats (MLX-optimized vs standard PyTorch)
- Independent dependency management per environment

**Shared code strategy:**
- LM Studio scripts are identical across setups (API-based, hardware-agnostic)
- Direct inference scripts (`infer.py`) share logic but differ in device detection
- Model weights are downloaded separately to each setup

### LM Studio Integration

Both setups include `lmstudio-scripts/` for querying models via LM Studio's local API server:

**Architecture:**
1. LM Studio loads quantized GGUF models (not included in repo)
2. Exposes OpenAI-compatible chat completions API at localhost:1234
3. Scripts send HTTP POST requests with chat messages
4. Useful for quick testing without managing PyTorch/device logic

**Model IDs referenced in scripts:**
- `qwen3-8b` - Qwen3-8B model in LM Studio
- `openai_gpt-oss-20b` - GPT-OSS-20B model in LM Studio

### Benchmark Pipeline Structure

The `Uni-setup/src/reliable_clinical_benchmark/` directory is the intended home for the evaluation harness mentioned in the README:

**Expected components (from README):**
- `data/`: OpenR1-Psy dataset IDs, prompt templates, multi-turn scripts
- `src/`: Unified runner for all three studies, metrics with bootstrap CIs, plotting utilities
- `runs/`: Raw model generations and per-slice CSV outputs
- `reports/`: Final PDF deliverables with figures and failure analysis

**Current state:** Directory structure exists but implementation is incomplete.

### Studies Overview

**Study A - Faithfulness on OpenR1-Psy:**
- Evaluates step-by-step reasoning alignment with gold standard traces
- Metrics: Step-F1, Final Accuracy, Faithfulness Gap
- Interventions: Direct answer vs Chain-of-Thought vs Self-Critique

**Study B - Empathy vs Truth under Social Pressure:**
- Tests model ability to maintain accuracy while refusing incorrect user suggestions
- Metrics: AgreementRate, Accuracy, Truth-Under-Pressure
- Scaffolds: Empathy-then-correct prompting strategy

**Study C - Longitudinal Therapeutic Continuity:**
- Measures consistency across 3-5 turn conversations with session memory
- Metrics: Continuity Score (MiniLM embeddings + cosine), Safety Drift Rate, Refusal/Redirect Rate

## Important Context

### Data Sources

**OpenR1-Psy dataset:** https://huggingface.co/datasets/GMLHUHE/OpenR1-Psy
- Primary evaluation dataset for Study A
- Contains clinical reasoning traces for faithfulness evaluation

### PsyLLM Model Details

**Model card:** https://huggingface.co/GMLHUHE/PsyLLM
**Paper:** https://arxiv.org/pdf/2505.15715

PsyLLM uses ChatML-style formatting with `<|im_start|>` and `<|im_end|>` tokens:

```
<|im_start|>system
You are PsyLLM, a compassionate therapeutic assistant.
<|im_end|>
<|im_start|>user
User message here
<|im_end|>
<|im_start|>assistant
```

### Key References

- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- Chain-of-Thought prompting: https://arxiv.org/abs/2201.11903
- Self-Consistency decoding: https://arxiv.org/abs/2203.11171

## Development Practices

### Working with Model Scripts

When modifying inference scripts:
- Always test device fallback logic (CUDA → MPS/MLX → CPU)
- Respect the ChatML format for PsyLLM prompts
- Use consistent temperature and sampling parameters across studies for fair comparison
- Log device selection and dtype for reproducibility

### Adding New Model Providers

To add a new model to evaluation:
1. Create parallel scripts in both `Mac-setup/lmstudio-scripts/` and `Uni-setup/lmstudio-scripts/`
2. If using direct inference, add to `psy-llm-local/` with appropriate device handling
3. Update model download instructions in this file
4. Ensure `.gitignore` excludes downloaded weights

### Documentation Updates

Key documentation locations:
- **Setup guides:** `Assignment 2/docs/psyllm_setup.md` - model download and weight conversion procedures
- **Evaluation framework:** `Assignment 2/docs/Guides/Clinical Evaluation Framework.tex` - metric definitions and study design
- **This file:** Update when adding new commands, models, or architectural changes
