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

### What we used
For **Psych_Qwen_32B** on a **24GB VRAM** GPU, we used **`quantization="4bit"`** in the local runner.

### Customising quantization (optional)
The local runner supports a `quantization=` argument so you can pick what fits your hardware/use case (e.g. `"4bit"` or `"8bit"`).

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
