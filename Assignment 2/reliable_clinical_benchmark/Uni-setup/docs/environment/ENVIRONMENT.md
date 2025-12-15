# Environment Setup (Uni-setup)

Uni-setup uses **two** Python environments in practice:

- **`mh-llm-benchmark-env`**: the **general** benchmark environment for running Studies A/B/C, metrics, and pytest.
  - Installs pinned packages from `Uni-setup/requirements.txt` (notably `transformers==4.38.2`).
  - Used for LM Studio runners and evaluation pipelines.

- **`mh-llm-local-env`**: a **local HF inference** environment used when running **local PyTorch model runners** that require a more modern `transformers` stack (e.g., Qwen3-style `enable_thinking` chat templating).
  - This was used for the Piaget local run to avoid dependency clashes with the benchmark pins.

Both environments should be isolated (no base Anaconda writes, no user-site bleed).

## `mh-llm-benchmark-env` (general)

Create and use this for **evaluation** and **tests**:

```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
conda create -n mh-llm-benchmark-env python=3.10 -y
& "D:\Anaconda3\Scripts\activate" mh-llm-benchmark-env   # adjust if Anaconda elsewhere

pip install -r requirements.txt
# spaCy model via scispaCy S3 (matches spaCy 3.6.1)
python -m pip install --no-deps https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz
python -m spacy validate
```

### What runs in this env

- `scripts/run_evaluation.py` (Studies A/B/C; generate-only and from-cache)
- pytest:

```powershell
$Env:PYTHONPATH="src"
pytest tests/unit -v
pytest tests/integration -v
```

## `mh-llm-local-env` (local HF inference)

Use this when you want to run **local HF runners** that load weights from `Uni-setup/models/` (Piaget, Psyche-R1, Psych_Qwen_32B, PsyLLM).

Key points:

- Keep it separate from `mh-llm-benchmark-env` so the pinned `transformers==4.38.2` does not block newer chat-template features.
- Prefer running pip via the env python and disable user-site packages (`PYTHONNOUSERSITE=1`) to avoid package bleed.

## Local weight storage

Local weights live under `Uni-setup/models/` and must remain **untracked** (gitignored).

## Runtime caches / offload directories

Some local runners may create runtime directories (e.g., `offload/psyche_r1`) during generation. These are safe to delete; they will be re-created when needed.

