# Environment Setup (Mac, concise copy)

## Single shared environment

Maintain one Python 3.11 virtual environment for this setup. It runs Studies A, B, C plus the test suite.

## Create the virtual environment

```bash
cd "Assignment 2/reliable_clinical_benchmark/Mac-setup"

python3.11 -m venv .mh-llm-benchmark-env
source .mh-llm-benchmark-env/bin/activate

python -m pip install --upgrade pip
python -m pip install "torch==2.2.1"
python -m pip install "spacy==3.6.1" "scispacy==0.5.3"
python -m pip install \
  https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz
python -m pip install -r requirements.txt
```

If `pip` reports `externally-managed-environment`, reactivate the venv and retry.

## Dependencies

- Python 3.11.x
- PyTorch 2.2.1
- Transformers 4.38.2
- spaCy 3.6.1
- scispaCy 0.5.3 (`scispacy` + `en_core_sci_sm`)
- SciPy 1.10.1
- NLI model: `roberta-large-mnli`
- Sentence-transformers 2.5.1
- Remaining packages in `requirements.txt`

## Running studies

```bash
cd "Assignment 2/reliable_clinical_benchmark/Mac-setup"
source .mh-llm-benchmark-env/bin/activate

python scripts/run_evaluation.py --study A
python scripts/run_evaluation.py --study B
python scripts/run_evaluation.py --study C
```

## LM Studio (PsyLLM)

- Install LM Studio (https://lmstudio.ai/).
- Download PsyLLM-8B (`GMLHUHE/PsyLLM`) via LM Studio or the local helper.
- Enable “Local Server” (default http://localhost:1234) and verify with `curl http://localhost:1234/v1/models`.
- `PsyLLMRunner` in `src/reliable_clinical_benchmark/models/psyllm.py` targets this endpoint.

## API keys

Create `.env` in the Mac-setup root:

```bash
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
GPT_OSS_API_KEY=your_gpt_oss_api_key_here
```

## Verification snippet

```bash
python -c "
import sys
print(f'Python: {sys.version}')
try:
    import torch, transformers, spacy
    print(f'✓ PyTorch: {torch.__version__}')
    print(f'✓ Transformers: {transformers.__version__}')
    print(f'✓ spaCy: {spacy.__version__}')
    spacy.load('en_core_sci_sm')
    print('✓ scispaCy medical model: loaded')
    print('\\nEnvironment setup complete')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    sys.exit(1)
"
```

## Reproducibility note

```
Experiments ran on macOS in a Python 3.11 virtual environment (".mh-llm-benchmark-env") with dependencies pinned in requirements.txt, including PyTorch 2.2.1, Transformers 4.38.2, spaCy 3.6.1, SciPy 1.10.1, and scispaCy 0.5.3 (en_core_sci_sm).
```


