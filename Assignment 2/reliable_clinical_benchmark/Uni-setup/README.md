# Uni Setup (Windows, LM Studio + local HF)

End-to-end steps for the Uni environment (Windows, x64) to mirror the Mac flow: create an isolated conda env, ensure spaCy/scispaCy are available, run generations via LM Studio or local HF models, then score from cache.

## 1) Prerequisites
- Anaconda/Miniconda on PATH
- Git
- LM Studio 0.3x (server reachable at `http://127.0.0.1:1234`)
- Hugging Face token (for HF downloads and gated models)
- Models placed under `Uni-setup/models/` (already gitignored)

## 2) Create the env (conda, same name as Mac)
```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
conda create -n mh-llm-benchmark-env python=3.10 -y
& "D:\Anaconda3\Scripts\activate" mh-llm-benchmark-env   # adjust path if Anaconda elsewhere

pip install -r requirements.txt
# spaCy model via scispaCy S3 (matches spaCy 3.6.1)
python -m pip install --no-deps https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz
python -m spacy validate
```
If `en_core_sci_sm` is unavailable, the above S3 link is the tested path for v0.5.3. NER-dependent tests auto-skip if the model is missing.

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
python -c "from pipelines.study_a import run_study_a; from models.psyllm import PsyLLMRunner; m=PsyLLMRunner(model_name='$MODEL', api_base='$EP'); run_study_a(model=m, data_dir='data/openr1_psy_splits', output_dir='results', model_name='$MODEL', generate_only=True, cache_out=f'results/{'$MODEL'}/study_a_generations.jsonl')"

# Score from cache (no model calls)
python -c "from pipelines.study_a import run_study_a; from models.psyllm import PsyLLMRunner; m=PsyLLMRunner(model_name='$MODEL', api_base='$EP'); run_study_a(model=m, data_dir='data/openr1_psy_splits', output_dir='results', model_name='$MODEL', from_cache=f'results/{'$MODEL'}/study_a_generations.jsonl')"
```
Adjust `max-samples`/`max-cases` flags in the pipeline calls if you want smaller probes first.

## 7) Study B and C (LM Studio)
```powershell
# Study B (sycophancy), skip NLI for speed unless you have it
python -c "from pipelines.study_b import run_study_b; from models.psyllm import PsyLLMRunner; m=PsyLLMRunner(model_name='$MODEL', api_base='$EP'); run_study_b(model=m, data_dir='data/openr1_psy_splits', output_dir='results', model_name='$MODEL', max_samples=10, use_nli=False)"

# Study C (drift)
python -c "from pipelines.study_c import run_study_c; from models.psyllm import PsyLLMRunner; m=PsyLLMRunner(model_name='$MODEL', api_base='$EP'); run_study_c(model=m, data_dir='data/openr1_psy_splits', output_dir='results', model_name='$MODEL', max_cases=3, use_nli=False)"
```
Switch to the HF-local runners (`hf-local-scripts/`) if you prefer to call downloaded PyTorch models directly instead of LM Studio.

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

### Why quantization is needed
- **32B in bf16** is ~64GB just for weights → cannot fit on a 24GB GPU.
- **8-bit** is ~32GB for weights → still usually too big for 24GB.
- **4-bit** is typically required; any remainder can be **offloaded to CPU RAM**.

### Psych_Qwen_32B local runner (quant + CPU offload)
The repo includes a local runner that loads from `models/Psych_Qwen_32B`:
- `reliable_clinical_benchmark.models.psych_qwen_local.PsychQwen32BLocalRunner`

It supports:
- **`quantization="4bit"`** (recommended for 24GB VRAM)
- **`quantization="8bit"`** (may still OOM on 24GB)
- **`max_memory={0:"22GiB","cpu":"48GiB"}`** to cap GPU usage and allow CPU offload
- **`offload_folder="offload/psych_qwen_32b"`** for stable offloading

Example (PowerShell):
```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
$Env:PYTHONNOUSERSITE="1"
$Env:PYTHONPATH="src"
$py="C:\Users\22837352\.conda\envs\mh-llm-benchmark-env\python.exe"

# NOTE: do not run during a GPU-heavy Study A job.
& $py -c "from reliable_clinical_benchmark.models.psych_qwen_local import PsychQwen32BLocalRunner; \
m=PsychQwen32BLocalRunner(quantization='4bit'); \
print('runner_ready', m.model_name)"
```

### Windows note (bitsandbytes)
4-bit/8-bit loading uses `bitsandbytes`. On Windows this can be finicky depending on CUDA/toolchain.
If `bitsandbytes` fails to install/import, the fallback is to run quantized inference under **WSL2/Linux**.
