# Environment Setup (Mac-setup)

## Overview

Single Mac environment for running the benchmark locally (smoke tests and small evaluations) on 8B-class models. Everything below is scoped to macOS; heavier 32B+ runs stay on the Uni setup.

## 1. Create the virtual environment

From the repo root:

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

## 2. Required Dependencies

The benchmark relies on the following core technology stack:

  * **Python:** 3.11.x
  * **PyTorch:** 2.2.1
  * **Transformers:** 4.38.2
  * **NLP/NER:** spaCy 3.6.1 with scispaCy 0.5.3 (`en_core_sci_sm` model)
  * **NLI Model:** `roberta-large-mnli` (auto-downloaded via Transformers)
  * **Sentence Embeddings:** Sentence-transformers 2.5.1
  * **Scientific Computing:** SciPy 1.10.1

## 3. Mac-local models

**Qwen3-8B (precision on-device)**
- Load `Qwen/Qwen2.5-8B-Instruct` into LM Studio with MLX enabled, start the Local Server (`http://localhost:1234/v1`), and set `model_name` to the LM Studio identifier (e.g., `qwen3-8b-mlx`) when instantiating `PsyLLMRunner`.
- If you prefer the Hugging Face Inference API instead of LM Studio, export `HUGGINGFACE_API_KEY` in `.env`; this keeps precision server-side and avoids local VRAM pressure.

**PsyLLM-8B (local LM Studio)**
- Load `GMLHUHE/PsyLLM` in LM Studio, keep the default server port, and run via `PsyLLMRunner` (uses `lmstudio_client`).
- Target full precision on MLX/MPS; drop to CPU only if necessary for debugging.

Models larger than 8B (QwQ-32B, DeepSeek-R1-14B, GPT-OSS-120B) are intentionally excluded here and live in the Uni setup documentation.

### LM Studio checklist
1. Install LM Studio (https://lmstudio.ai/).
2. Load the chosen model (Qwen3-8B or PsyLLM-8B) and enable the Local Server.
3. Verify the endpoint before running:
   ```bash
   curl http://localhost:1234/v1/models
   ```

### API keys (Mac scope)
Create `.env` in `Mac-setup` if you call the Hugging Face Inference API:
```bash
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
```

## 4. NLI Model Setup

The benchmark uses Natural Language Inference (NLI) for two advanced metrics:

  * **Study B:** Evidence Hallucination detection.
  * **Study C:** Knowledge Conflict detection (contradictions between turns).

### Model Selection

**Model:** `roberta-large-mnli`

This model was selected for the benchmark because:

1.  **Stability:** It is fully compatible with `transformers==4.38.2`, avoiding tokenizer conflicts present in newer architectures.
2.  **Performance:** It provides established, high-accuracy performance for entailment and contradiction detection tasks in clinical NLP contexts.
3.  **Ease of Use:** It integrates seamlessly with the pipeline, auto-downloading on first use.

**Label Mapping:**

  * `0`: Contradiction
  * `1`: Neutral
  * `2`: Entailment

### Usage

The NLI model is initialised automatically via the `NLIModel` utility:

```python
from utils.nli import NLIModel

nli = NLIModel()
result = nli.predict(
    premise="The patient has depression.",
    hypothesis="The patient is sad."
)
```

*The system automatically utilises GPU acceleration (CUDA/MPS) if available.*

## 5. Running studies (Mac)

```bash
cd "Assignment 2/reliable_clinical_benchmark/Mac-setup"
source .mh-llm-benchmark-env/bin/activate

# Example smoke runs on Mac
PYTHONPATH=src python scripts/run_evaluation.py --model qwen3-8b-mlx --study A --max-samples 5
PYTHONPATH=src python scripts/run_evaluation.py --model PsyLLM-8B --study B --max-samples 5
```

## 6. Verification Script

Run the following command to verify that all components—including the clinical NER and NLI models—are correctly installed and accessible:

```bash
python -c "
import sys
import torch
import transformers
import spacy
from utils.nli import NLIModel
print(f'Python: {sys.version}')
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ Transformers: {transformers.__version__}')
print(f'✓ spaCy: {spacy.__version__}')
try:
    # Verify Clinical Model
    nlp = spacy.load('en_core_sci_sm')
    print('✓ scispaCy medical model: loaded')

    # Verify NLI Model
    nli = NLIModel()
    result = nli.predict('The patient has depression.', 'The patient is sad.')
    print(f'✓ NLI model: loaded (Test Result: {result})')

    print('\\n✅ Environment setup complete!')
except Exception as e:
    print(f'\\n❌ Error: {e}')
    sys.exit(1)
"
```

## Publication reproducibility statement

> Mac runs use Python 3.11 in `.mh-llm-benchmark-env` with dependencies pinned in `requirements.txt` (PyTorch 2.2.1, Transformers 4.38.2, spaCy 3.6.1, SciPy 1.10.1, scispaCy 0.5.3 with `en_core_sci_sm`). LM Studio-backed local inference is limited to 8B-class models on MLX/MPS; larger precision runs are handled separately on the Uni setup.


