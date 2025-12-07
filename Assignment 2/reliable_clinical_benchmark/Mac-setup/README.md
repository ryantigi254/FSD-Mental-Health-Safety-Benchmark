# Setup (Mac)

Mac setup is for 8B-class local runs on LM Studio/MLX/MPS. Larger models live in the Uni setup.

## Prereqs and quickstart

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
# Install LM Studio: https://lmstudio.ai/ (enable Local Server)
```

## Structure

- `psy-llm-local/` — Direct PyTorch inference for PsyLLM.
- `lmstudio-scripts/` — One-off chats against the LM Studio API.

## LM Studio checklist (8B-class)

1. Load Qwen3-8B or PsyLLM in LM Studio, enable Local Server (`http://localhost:1234/v1`).
2. Verify:

   ```bash
   curl http://localhost:1234/v1/models
   ```

3. Set `MODEL_ID` in scripts/runners to match LM Studio (e.g., `qwen3-8b-mlx`).

## Hugging Face Inference (optional fallback)

If you prefer HF Inference instead of LM Studio, create `.env` in `Mac-setup`:

```bash
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
```

Then point runners at HF endpoints rather than the local server.

## PsyLLM local inference (PyTorch)

```bash
cd psy-llm-local/models/PsyLLM
huggingface-cli login  # if needed
huggingface-cli download GMLHUHE/PsyLLM --local-dir . --local-dir-use-symlinks False
cd ..
python -m pip install torch transformers accelerate
python infer.py
```

## LM Studio scripts (chat)

```bash
cd lmstudio-scripts
python -m pip install requests
python chat_gpt_oss_20b.py "Your prompt here"
# or
python chat_qwen3_8b.py "Your prompt here"
```

Model IDs used in scripts: `openai_gpt-oss-20b`, `qwen3-8b` (override if your LM Studio ID differs).

## Study A: generation + metrics (LM Studio, two-phase)

Generation-only with cache compaction/resume:

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

Metrics from cache (no model calls):

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

- Cache compaction keeps 600 rows (300 IDs × 2 modes), prefers `status=ok`, else latest error, and writes a timestamped backup before overwriting.
- Per-sample writes stay immediate; reruns append only missing/error pairs.
- Outputs land in `results/<model>/study_a_generations.jsonl`; metrics consume only `status=ok` rows.

## Troubleshooting (LM Studio)

- `Connection refused` / timeouts: ensure LM Studio Local Server is running on `http://localhost:1234/v1` and the model is loaded; then rerun generation-only.
- Missing persona IDs or partial cache: rerun generation-only; compaction retries missing/error rows and backs up the prior cache.
- NER/NLI failures: confirm `.mh-llm-benchmark-env` is active and `en_core_sci_sm` downloaded.

## More detail

- Environment: `docs/environment/ENVIRONMENT.md`
- Evaluation protocol: `docs/evaluation/EVALUATION_PROTOCOL.md`
