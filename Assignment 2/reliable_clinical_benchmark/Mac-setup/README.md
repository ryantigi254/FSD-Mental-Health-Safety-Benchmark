# Setup

## Structure

- `psy-llm-local/` - Direct PyTorch inference script for PsyLLM
- `lmstudio-scripts/` - Scripts for querying models via LM Studio API

## PsyLLM Inference

1. Install dependencies:
```bash
pip install torch transformers accelerate
```

2. Download the PsyLLM model:
```bash
cd psy-llm-local/models/PsyLLM
huggingface-cli login  # if needed
huggingface-cli download GMLHUHE/PsyLLM --local-dir . --local-dir-use-symlinks False
```

3. Run inference:
```bash
cd psy-llm-local
python infer.py
```

## LM Studio Scripts

1. Install dependencies:
```bash
pip install requests
```

2. Start LM Studio server with your model loaded (default: http://127.0.0.1:1234)

3. Run a script:
```bash
cd lmstudio-scripts
python chat_gpt_oss_20b.py "Your prompt here"
# or
python chat_qwen3_8b.py "Your prompt here"
```

## Model IDs

- OpenAI GPT-OSS-20B: `openai_gpt-oss-20b`
- Qwen3-8B: `qwen3-8b`

Adjust the `MODEL_ID` in the scripts if your LM Studio model identifier differs.

## Study A generation and metrics (LM Studio)

Two-phase flow is supported with cache compaction/resume:

1) Generation-only (compacts cache to one row per id/mode, backs up the old file, retries only missing/error):
```bash
cd "Assignment 2/reliable_clinical_benchmark/Mac-setup"
source .mh-llm-benchmark-env/bin/activate
PYTHONPATH=src python - <<'PY'
from pipelines.study_a import run_study_a
from models.psyllm import PsyLLMRunner
model = PsyLLMRunner(model_name="qwen3-8b-mlx", api_base="http://localhost:1234/v1")
run_study_a(
    model=model,
    data_dir="data/openr1_psy_splits",
    output_dir="results",
    model_name="qwen3-8b-mlx",
    generate_only=True,
    cache_out="results/qwen3-8b-mlx/study_a_generations.jsonl",
)
PY
```
2) Metrics from cache (no model calls):
```bash
PYTHONPATH=src python - <<'PY'
from pipelines.study_a import run_study_a
from models.psyllm import PsyLLMRunner
dummy = PsyLLMRunner(model_name="qwen3-8b-mlx", api_base="http://localhost:1234/v1")
run_study_a(
    model=dummy,
    data_dir="data/openr1_psy_splits",
    output_dir="results",
    model_name="qwen3-8b-mlx",
    from_cache="results/qwen3-8b-mlx/study_a_generations.jsonl",
)
PY
```
Notes:
- Cache compaction keeps 600 rows (300 IDs × 2 modes), preferring status=ok, else latest error, and writes a timestamped backup before overwriting.
- Per-sample writes remain immediate; reruns append only new attempts for missing/error pairs.

