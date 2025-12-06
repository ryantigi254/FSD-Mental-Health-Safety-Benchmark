# Environment Setup (Mac-setup)

## Overview

This benchmark operates within a **single shared Python 3.11 virtual environment**. This unified environment is designed to handle all three studies (A, B, C), unit/integration tests, and model execution on macOS systems.

The specific version pins listed below match the publishable environment described in the associated research, ensuring reproducibility.

## 1. Create the Virtual Environment

To avoid conflicts with system-level Python packages or Homebrew-managed environments, all operations must be performed inside the dedicated virtual environment.

Run the following from the repository root:

```bash
cd "Assignment 2/reliable_clinical_benchmark/Mac-setup"

# 1) Create a dedicated Python 3.11 environment
python3.11 -m venv .mh-llm-benchmark-env
source .mh-llm-benchmark-env/bin/activate

# 2) Upgrade pip within the virtual environment
python -m pip install --upgrade pip

# 3) Install core heavy dependencies (PyTorch)
python -m pip install "torch==2.2.1"

# 4) Install spaCy and the specific clinical model (scispacy)
python -m pip install "spacy==3.6.1" "scispacy==0.5.3"
python -m pip install \
  https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz

# 5) Install remaining dependencies
python -m pip install -r requirements.txt
```

> **Note:** If you encounter an `error: externally-managed-environment`, ensure you have activated the environment using `source .mh-llm-benchmark-env/bin/activate`.

## 2. Required Dependencies

The benchmark relies on the following core technology stack:

  * **Python:** 3.11.x
  * **PyTorch:** 2.2.1
  * **Transformers:** 4.38.2
  * **NLP/NER:** spaCy 3.6.1 with scispaCy 0.5.3 (`en_core_sci_sm` model)
  * **NLI Model:** `roberta-large-mnli` (auto-downloaded via Transformers)
  * **Sentence Embeddings:** Sentence-transformers 2.5.1
  * **Scientific Computing:** SciPy 1.10.1

## 3. Model Configuration

### PsyLLM (Local Hugging Face Loading)

The **PsyLLM-8B** (`GMLHUHE/PsyLLM`) model is configured to run locally via the Hugging Face `transformers` library.

  * The model will be automatically downloaded to your local Hugging Face cache upon the first execution.
  * Ensure your machine has sufficient RAM/VRAM to load the 8B parameter model.

### LM Studio (For Compatible Local Models)

For other models utilising the local server interface:

1.  Download and install [LM Studio](https://lmstudio.ai/).
2.  Load the target model within the application.
3.  Start the **Local Server** (default: `http://localhost:1234`).
4.  The benchmark runners will communicate with the model via this local API endpoint.

### Remote API Keys

For models requiring remote API access (e.g., QwQ, DeepSeek-R1), create a `.env` file in the `Mac-setup` root directory:

```bash
# Hugging Face API Key
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# GPT-OSS API Key (if applicable)
GPT_OSS_API_KEY=your_gpt_oss_api_key_here
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

## 5. Running Studies

With the environment activated, you can execute the evaluation scripts for each study:

```bash
cd "Assignment 2/reliable_clinical_benchmark/Mac-setup"
source .mh-llm-benchmark-env/bin/activate

# Run Studies
python scripts/run_evaluation.py --study A
python scripts/run_evaluation.py --study B
python scripts/run_evaluation.py --study C
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

## Publication Reproducibility Statement

> All experiments were conducted on macOS using a Python 3.11 virtual environment (`.mh-llm-benchmark-env`) with dependencies specified in `requirements.txt`. Key components include PyTorch 2.2.1, Transformers 4.38.2, spaCy 3.6.1, and scispaCy 0.5.3 (utilising the `en_core_sci_sm` clinical NER model).


