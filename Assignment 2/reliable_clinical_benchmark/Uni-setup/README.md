# Uni Setup (Windows, LM Studio + local HF)

End-to-end steps for the Uni environment (Windows, x64) to mirror the Mac flow: create an isolated conda env, ensure spaCy/scispaCy are available, run generations via LM Studio or local HF models, then score from cache.

## 1) Prerequisites

- Anaconda/Miniconda on PATH
- Git
- LM Studio 0.3x (server reachable at `http://127.0.0.1:1234`)
- Hugging Face token (for HF downloads and gated models)
- Models placed under `Uni-setup/models/` (already gitignored)

## Models (8) used in this benchmark run

- **PsyLLM (domain expert)**: [`GMLHUHE/PsyLLM`](https://huggingface.co/GMLHUHE/PsyLLM)
- **Qwen3‑8B (untuned baseline)**: [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B)
- **GPT‑OSS‑20B (generalist baseline)**: [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b)
- **QwQ‑32B (general‑purpose reasoning model)**: [`Qwen/QwQ-32B-Preview`](https://huggingface.co/Qwen/QwQ-32B-Preview)
- **DeepSeek‑R1‑14B (distilled reasoning model)**: [`deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) — paper: [`DeepSeek_R1.pdf`](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)
- **Piaget‑8B (clinical reasoning baseline; local HF runner)**: [`gustavecortal/Piaget-8B`](https://huggingface.co/gustavecortal/Piaget-8B)
- **Psyche‑R1 (psychological reasoning)**: [`MindIntLab/Psyche-R1`](https://huggingface.co/MindIntLab/Psyche-R1) — paper: [`arXiv:2508.10848`](https://arxiv.org/pdf/2508.10848)
- **Psych_Qwen_32B (large psych model)**: [`Compumacy/Psych_Qwen_32B`](https://huggingface.co/Compumacy/Psych_Qwen_32B) — typically run 4‑bit quantised on 24GB VRAM (local weights)

## 2) Create the environments (conda)

Uni-setup uses **two** Python environments:

### `mh-llm-benchmark-env` (general benchmark environment)

Use this for **LM Studio runners**, **evaluation pipelines**, and **tests**:

```powershell
cd "Assignment 2\reliable_clinical_benchmark\Uni-setup"
conda create -n mh-llm-benchmark-env python=3.10 -y
conda activate mh-llm-benchmark-env   # Use your conda activation command if 'conda activate' is not on PATH

pip install -r requirements.txt
# spaCy model via scispaCy S3 (matches spaCy 3.6.1)
python -m pip install --no-deps https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
python -m spacy validate
```

**What runs in this env:**
- LM Studio model runners (qwen3_lmstudio, gpt_oss, qwq, deepseek_r1_lmstudio)
- Evaluation pipelines (`scripts/run_evaluation.py`)
- Metric calculations
- Pytest unit and integration tests

### `mh-llm-local-env` (local HF inference environment)

Use this for **local PyTorch model runners** that require a more modern `transformers` stack (e.g., Piaget, Psyche-R1, Psych_Qwen_32B, PsyLLM local):

```powershell
cd "Assignment 2\reliable_clinical_benchmark\Uni-setup"
conda create -n mh-llm-local-env python=3.10 -y
conda activate mh-llm-local-env   # Use your conda activation command if 'conda activate' is not on PATH

# Install PyTorch with CUDA support (REQUIRED for GPU inference)
# Check your CUDA version with: nvidia-smi
# For CUDA 12.1 (most common): use cu121
# For CUDA 11.8: use cu118
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install latest transformers and dependencies for local HF models
pip install transformers accelerate bitsandbytes
# Install package dependencies (but may need newer transformers than requirements.txt pins)
pip install -r requirements.txt --upgrade transformers
```

**Important:** PyTorch must be installed with CUDA support for GPU inference. If you install PyTorch without the CUDA index URL, you'll get the CPU-only build (`torch-2.x.x+cpu`), which will not detect your GPU. Always use the CUDA-specific index URL matching your CUDA version (check with `nvidia-smi`).

**Key points:**
- **Separate from `mh-llm-benchmark-env`** to avoid dependency conflicts
- **PyTorch with CUDA support is required** for GPU inference. Install using the CUDA-specific index URL (e.g., `--index-url https://download.pytorch.org/whl/cu121` for CUDA 12.1). Without this, PyTorch will be CPU-only and models won't detect your GPU.
- The benchmark env pins `transformers==4.38.2`, which may block newer chat-template features
- Local models (Piaget, Psyche-R1, etc.) need modern `transformers` for Qwen3-style `enable_thinking` chat templating
- Use `PYTHONNOUSERSITE=1` to avoid package bleed between environments

**What runs in this env:**
- Local HF model runners: `piaget_local`, `psyche_r1_local`, `psych_qwen_local`, `psyllm_gml_local`
- Generation scripts: `hf-local-scripts/run_study_*_generate_only.py` (for local models)

**Note:** If `en_core_sci_sm` is unavailable, the S3 link above is the tested path for v0.5.4. NER-dependent tests auto-skip if the model is missing.

For detailed environment setup, see `docs/environment/ENVIRONMENT.md`.

## 3) Hugging Face auth (CLI)

```powershell
huggingface-cli login --token YOUR_HF_TOKEN
```

## 4) LM Studio (local endpoints)

- Load your model (e.g., `gpt-oss-20b`) and keep server running on `127.0.0.1:1234`.
- Ensure “Enable CORS”, “Allow per-request MCPs”, and “Just-in-Time Model Loading” are on; allow long JIT timeouts for big models.

## 5) Quick sanity chat (LM Studio client scripts)

```powershell
cd lmstudio-scripts
python chat_gpt_oss_20b.py "Test prompt"  # or chat_qwq_32b.py etc.
```

## 6) Study A generation then metrics (example LM Studio model)

```powershell
cd ..
$env:PYTHONPATH="src"
$EP="http://127.0.0.1:1234/v1"
$MODEL="gpt-oss-20b"

# Generate only (writes cache)
python -c "from reliable_clinical_benchmark.pipelines.study_a import run_study_a; from reliable_clinical_benchmark.models.psyllm import PsyLLMRunner; model_id='$MODEL'; ep='$EP'; m=PsyLLMRunner(model_name=model_id, api_base=ep); run_study_a(model=m, data_dir='data/openr1_psy_splits', output_dir='results', model_name=model_id, generate_only=True, cache_out=f'results/{model_id}/study_a_generations.jsonl')"

# Score from cache (no model calls)
python -c "from reliable_clinical_benchmark.pipelines.study_a import run_study_a; from reliable_clinical_benchmark.models.psyllm import PsyLLMRunner; model_id='$MODEL'; ep='$EP'; m=PsyLLMRunner(model_name=model_id, api_base=ep); run_study_a(model=m, data_dir='data/openr1_psy_splits', output_dir='results', model_name=model_id, from_cache=f'results/{model_id}/study_a_generations.jsonl')"
```

Adjust `max-samples`/`max-cases` flags in the pipeline calls if you want smaller probes first.

## 7) Study B and C (LM Studio)

```powershell
# Study B (sycophancy), skip NLI for speed unless you have it
python -c "from reliable_clinical_benchmark.pipelines.study_b import run_study_b; from reliable_clinical_benchmark.models.psyllm import PsyLLMRunner; model_id='$MODEL'; ep='$EP'; m=PsyLLMRunner(model_name=model_id, api_base=ep); run_study_b(model=m, data_dir='data/openr1_psy_splits', output_dir='results', model_name=model_id, max_samples=10, use_nli=False)"

# Study C (drift)
python -c "from reliable_clinical_benchmark.pipelines.study_c import run_study_c; from reliable_clinical_benchmark.models.psyllm import PsyLLMRunner; model_id='$MODEL'; ep='$EP'; m=PsyLLMRunner(model_name=model_id, api_base=ep); run_study_c(model=m, data_dir='data/openr1_psy_splits', output_dir='results', model_name=model_id, max_cases=3, use_nli=False)"
```

### Using Local HF Models

For **local HF runners** (Piaget, Psyche-R1, Psych_Qwen_32B, PsyLLM), use the **`mh-llm-local-env`** environment:

```powershell
# Activate local environment
conda activate mh-llm-local-env
$Env:PYTHONNOUSERSITE="1"
$Env:PYTHONPATH="src"

# Run generation scripts (see docs/studies/*/study_*_commands.md for model-specific commands)
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id piaget_local
python hf-local-scripts\run_study_b_generate_only.py --model-id piaget_local
python hf-local-scripts\run_study_c_generate_only.py --model-id piaget_local
```

**Important:** Keep local and benchmark environments separate to avoid dependency conflicts. The local environment needs newer `transformers` versions for modern chat templates.

## 8) Tests

```powershell
pytest tests/unit -v
pytest tests/integration -v
```

Unit split invariants will guard against accidental data edits; integration tests skip if dependencies (datasets/NER/NLI) are absent.

## 9) Troubleshooting

- Connection refused: ensure LM Studio server running on port 1234 and model loaded.
- spaCy/scispaCy wheels: use Python 3.10 with conda; install `en_core_sci_sm` via `python -m spacy download en_core_sci_sm`.
- Large HF downloads: use `huggingface-cli download ... --resume-download --local-dir-use-symlinks False`. Keep weights under `models/` (gitignored).

## 10) Running large local models on limited VRAM (e.g., 32B on 24GB)

### What we used

For **Psych_Qwen_32B** on a **24GB VRAM** GPU, we used **`quantization="4bit"`** in the local runner.

**Important**: 
- **PyTorch with CUDA support** must be installed (see setup instructions above). Without CUDA-enabled PyTorch, models won't detect your GPU.
- Quantisation requires **`bitsandbytes`** to be installed in `mh-llm-local-env`. If you get `PackageNotFoundError: No package metadata was found for bitsandbytes`, install it:

```powershell
conda activate mh-llm-local-env
pip install bitsandbytes
```

### Customising quantisation (optional)

The local runner supports a `quantization=` argument so you can pick what fits your hardware/use case (e.g. `"4bit"` or `"8bit"`). **`bitsandbytes` must be installed** for any quantisation mode.

Example (PowerShell, using `mh-llm-local-env`):

```powershell
cd "Assignment 2\reliable_clinical_benchmark\Uni-setup"
$Env:PYTHONNOUSERSITE="1"
$Env:PYTHONPATH="src"
$py="python"  # Ensure your conda env is active

# NOTE: do not run during a GPU-heavy Study A job.
& $py -c "from reliable_clinical_benchmark.models.psych_qwen_local import PsychQwen32BLocalRunner; \
m=PsychQwen32BLocalRunner(quantization='4bit'); \
print('runner_ready', m.model_name)"
```
