# Environment Setup

## Single Shared Environment

This project uses **one virtual environment** for all three studies (A, B, C).

## Setup

```bash
cd Assignment\ 2/Uni-setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz
```

## Required Dependencies

The benchmark requires:

- Python 3.9+
- PyTorch 2.2.1
- Transformers 4.38.2
- spaCy 3.7.4
- scispaCy 0.5.3 (medical NER model)
- DeBERTa-v3 NLI model (auto-downloaded on first use)

## Running Studies

All studies use the same environment:

```bash
source venv/bin/activate
python scripts/run_evaluation.py --study A
python scripts/run_evaluation.py --study B
python scripts/run_evaluation.py --study C
```

## LM Studio Setup (for PsyLLM)

1. Download and install LM Studio from https://lmstudio.ai/
2. Download PsyLLM-8B model
3. Load model in LM Studio
4. Enable "Local Server" (default: http://localhost:1234)
5. Verify: `curl http://localhost:1234/v1/models`

## API Keys

Create a `.env` file in the project root:

```bash
# Hugging Face API Key (required for QwQ, DeepSeek-R1, Qwen3)
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# GPT-OSS API Key (if/when available)
GPT_OSS_API_KEY=your_gpt_oss_api_key_here
```

## Verification

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
    print(f'✓ scispaCy medical model: loaded')
    
    print('\n✅ Environment setup complete!')
except ImportError as e:
    print(f'\n❌ Missing dependency: {e}')
    sys.exit(1)
"
```

## Publication Reproducibility

Include in paper methods:

```
All experiments were conducted in a single Python 3.9.18 virtual environment
with dependencies specified in requirements.txt (PyTorch 2.2.1, Transformers 4.38.2,
spaCy 3.7.4, scispaCy 0.5.3).
```

