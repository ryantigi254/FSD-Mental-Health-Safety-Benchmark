# Environment Setup (Mac-setup)

## Single Shared Environment per Setup

This benchmark uses **one Python 3.11 virtual environment per OS-specific setup** (`Mac-setup`, `Uni-setup`).  
That single env is used for all three studies (A, B, C) and for the unit/integration tests.

## Create the Mac virtual environment

From the repo root, **always work inside a Python 3.11 virtualenv** (to avoid the Homebrew “externally managed” errors and the Python 3.13 C‑API breakage you saw earlier):

```bash
cd "Assignment 2/reliable_clinical_benchmark/Mac-setup"

# 1) Create a dedicated 3.11 env for the benchmark
python3.11 -m venv .mh-llm-benchmark-env
source .mh-llm-benchmark-env/bin/activate

# 2) Upgrade pip in the venv (NOT system-wide)
python -m pip install --upgrade pip

# 3) Install core heavy deps explicitly first
python -m pip install "torch==2.2.1"

# 4) Install spaCy + scispaCy (library + small clinical model)
python -m pip install "spacy==3.6.1" "scispacy==0.5.3"
python -m pip install \
  https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz

# 5) Install the rest of the stack from requirements.txt
python -m pip install -r requirements.txt
```

If you ever see `error: externally-managed-environment` from `pip`, you are **not** in the venv; re‑run `source .mh-llm-benchmark-env/bin/activate` and use `python -m pip ...` from there.

## Required Dependencies

The benchmark environment for Mac uses:

- Python 3.11.x
- PyTorch 2.2.1
- Transformers 4.38.2
- spaCy 3.6.1
- scispaCy 0.5.3 (`scispacy` lib + `en_core_sci_sm` model)
- SciPy 1.10.1
- DeBERTa-v3 NLI model (auto-downloaded on first use)
- Sentence-transformers 2.5.1
- The rest of the packages listed in `requirements.txt`

These pins match the publishable environment described in the dissertation.

## Running Studies

All three studies share the same environment:

```bash
cd "Assignment 2/reliable_clinical_benchmark/Mac-setup"
source .mh-llm-benchmark-env/bin/activate

# Run all studies
python scripts/run_evaluation.py --study A
python scripts/run_evaluation.py --study B
python scripts/run_evaluation.py --study C
```

## LM Studio Setup (for PsyLLM)

1. Download and install LM Studio from https://lmstudio.ai/
2. Download the PsyLLM-8B model (`GMLHUHE/PsyLLM`) in LM Studio **or** via the local `psy-llm-local` helper.
3. Load the model in LM Studio.
4. Enable "Local Server" (default: http://localhost:1234).
5. Verify LM Studio is live:

```bash
curl http://localhost:1234/v1/models
```

`PsyLLMRunner` in `src/reliable_clinical_benchmark/models/psyllm.py` will then talk to this LM Studio server.

## API Keys

Create a `.env` file in the Mac-setup root (or use the provided `.env.example` as a template):

```bash
# Hugging Face API Key (required for QwQ, DeepSeek-R1, Qwen3)
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# GPT-OSS API Key (if/when available)
GPT_OSS_API_KEY=your_gpt_oss_api_key_here
```

These keys are read by the remote model runners via `python-dotenv`.

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

Suggested wording for the methods section:

```
All experiments were conducted on macOS using a Python 3.11 virtual environment
(".mh-llm-benchmark-env") with dependencies specified in requirements.txt
(including PyTorch 2.2.1, Transformers 4.38.2, spaCy 3.6.1, SciPy 1.10.1, and scispaCy 0.5.3
with the en_core_sci_sm clinical NER model).
```


