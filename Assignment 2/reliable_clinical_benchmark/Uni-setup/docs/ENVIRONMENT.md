# Environment Setup (Uni-setup)

## Single Shared Environment per Setup

This benchmark uses **one Python 3.11 virtual environment per OS-specific setup** (`Mac-setup`, `Uni-setup`).  
That single env is used for all three studies (A, B, C) and for the unit/integration tests.

## Create the Uni virtual environment

From the repo root on the Uni machine, mirror the Mac setup: **one Python 3.11 venv, then install torch, spaCy/scispaCy, then the rest via requirements**:

```bash
cd "Assignment 2/reliable_clinical_benchmark/Uni-setup"

# 1) Create a dedicated 3.11 env for the benchmark
python3.11 -m venv .mh-llm-benchmark-env
source .mh-llm-benchmark-env/bin/activate    # Windows PowerShell: .\.mh-llm-benchmark-env\Scripts\Activate.ps1

# 2) Upgrade pip inside the venv
python -m pip install --upgrade pip

# 3) Install core heavy deps first
python -m pip install "torch==2.2.1"

# 4) Install spaCy + scispaCy (library + small clinical model)
python -m pip install "spacy==3.6.1" "scispacy==0.5.3"
python -m pip install \
  https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz

# 5) Install the remaining dependencies
python -m pip install -r requirements.txt
```

If `python3.11` is not on PATH, use `python` or `py -3.11` as appropriate for your Uni setup, but always from **inside** the venv; do not run `pip` against the system Python to avoid PEP‑668 “externally managed” errors.

## Required Dependencies

The benchmark environment for Uni uses:

- Python 3.11.x
- PyTorch 2.2.1
- Transformers 4.38.2
- spaCy 3.6.1
- scispaCy 0.5.3 (`scispacy` lib + `en_core_sci_sm` model)
- SciPy 1.10.1
- DeBERTa-v3 NLI model (auto-downloaded on first use)
- Sentence-transformers 2.5.1
- The rest of the packages listed in `requirements.txt`

These pins mirror the Mac environment for cross-machine reproducibility.

## Running Studies

All three studies share the same environment:

```bash
cd "Assignment 2/reliable_clinical_benchmark/Uni-setup"
source .mh-llm-benchmark-env/bin/activate

# Run all studies
python scripts/run_evaluation.py --study A
python scripts/run_evaluation.py --study B
python scripts/run_evaluation.py --study C
```

## Remote API Setup (HF / GPT-OSS)

Uni-setup typically runs remote baselines (QwQ, DeepSeek-R1, Qwen3, GPT-OSS) rather than local LM Studio.

1. Create a `.env` file in the Uni-setup root (or copy from `.env.example`):

```bash
# Hugging Face API Key (required for QwQ, DeepSeek-R1, Qwen3)
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# GPT-OSS API Key (if/when available)
GPT_OSS_API_KEY=your_gpt_oss_api_key_here
```

2. The model runners in `src/reliable_clinical_benchmark/models/*.py` will read these keys via `python-dotenv`.

## Verification

With the env activated:

```bash
python -c "
import sys
print(f'Python: {sys.version}')

try:
    import torch
    print(f'✓ PyTorch: {torch.__version__}')
    import transformers
    print(f'✓ Transformers: {transformers.__version__}')
    import spacy
    print(f'✓ spaCy: {spacy.__version__}')

    # Try loading scispaCy model
    nlp = spacy.load('en_core_sci_sm')
    print('✓ scispaCy medical model: loaded')

    print('\n✅ Environment setup complete!')
except ImportError as e:
    print(f'\n❌ Missing dependency: {e}')
    sys.exit(1)
"
```

## Publication Reproducibility

Suggested wording for the methods section (Uni machine):

```
All Uni-side experiments were conducted using a Python 3.11 virtual environment
(".mh-llm-benchmark-env") with dependencies specified in requirements.txt
(including PyTorch 2.2.1, Transformers 4.38.2, spaCy 3.6.1, SciPy 1.10.1, and scispaCy 0.5.3
with the en_core_sci_sm clinical NER model), mirroring the Mac-setup configuration.
```


