# Study A Generation Commands

## Overview

Study A evaluates **Faithfulness** - measuring whether model reasoning drives predictions or is merely post-hoc rationalisation.

**Scripts**: Individual model runners in `hf-local-scripts/` (e.g., `run_qwen3_lmstudio.py`, `run_psyllm_gml.py`)

**What it does**:
- Generates responses in two modes per sample:
  - **CoT mode** (`mode="cot"`): Chain-of-Thought reasoning with step-by-step analysis
  - **Direct mode** (`mode="direct"`): Immediate answer without explicit reasoning
- Each sample generates both modes (2 generations per sample)
- Writes to: `results/<model-id>/study_a_generations.jsonl`

**Note**: This is a standalone generation run. Metrics are calculated separately using `from_cache` mode or via `scripts/study_a/metrics/calculate_metrics.py`.

- Uses `run_study_a()` pipeline from `reliable_clinical_benchmark.pipelines.study_a`
- Each model has a dedicated script with `study-a` subcommand
- Supports `--generate-only` flag for generation-only mode (no metrics)

**Modularity**: Like Study B and Study A Bias, this study uses a separated execution model where raw data is decoupled from metrics calculation. This ensures scalability for large runs.

## Commands per Model

Run these from `Uni-setup/`:

### Qwen3-8B (LM Studio)
```powershell
python hf-local-scripts\run_qwen3_lmstudio.py study-a --generate-only
```

### QwQ-32B (LM Studio)
```powershell
python hf-local-scripts\run_qwen3_lmstudio.py study-a --generate-only --api-identifier QwQ-32B-GGUF --model-name qwq
```

**Note**: QwQ uses the same script as Qwen3 but with different API identifier. Alternatively, use the model factory approach (see below).

### DeepSeek-R1 (LM Studio distill)
```powershell
python hf-local-scripts\run_qwen3_lmstudio.py study-a --generate-only --api-identifier deepseek-r1-distill-qwen-14b --model-name deepseek-r1-lmstudio
```

**Note**: DeepSeek-R1 uses the same script pattern with different API identifier. Alternatively, use the model factory approach (see below).

### GPT-OSS-20B (LM Studio)
```powershell
python hf-local-scripts\run_qwen3_lmstudio.py study-a --generate-only --api-identifier gpt-oss-20b --model-name gpt-oss-20b
```

**Note**: GPT-OSS uses the same script pattern with different API identifier. Alternatively, use the model factory approach (see below).

### PsyLLM (HF local, GMLHUHE/PsyLLM)
```powershell
python hf-local-scripts\run_psyllm_gml.py study-a --generate-only
```

### Piaget-8B (HF local)
**Note**: `run_piaget_8b.py` is a simple prompt runner. For Study A, use the model factory approach (see below) or create a wrapper script similar to `run_psyche_r1.py`.

### Psyche-R1 (HF local)
```powershell
python hf-local-scripts\run_psyche_r1.py study-a --generate-only
```

### Psych-Qwen-32B (HF local, 4-bit)
```powershell
python hf-local-scripts\run_psych_qwen_32b.py study-a --generate-only
```

## Alternative: Using Model Factory

For models without dedicated scripts, you can use the model factory approach:

```python
from reliable_clinical_benchmark.models.factory import get_model_runner
from reliable_clinical_benchmark.models.base import GenerationConfig
from reliable_clinical_benchmark.pipelines.study_a import run_study_a

config = GenerationConfig(max_tokens=4096)
runner = get_model_runner("qwq", config)  # or "deepseek_r1_lmstudio", "gpt_oss", "piaget_local", etc.

run_study_a(
    model=runner,
    data_dir="data/openr1_psy_splits",
    output_dir="results",
    model_name="qwq",  # matches model_id
    generate_only=True,
    cache_out="results/qwq/study_a_generations.jsonl",
)
```

**Available model IDs**:
- LM Studio: `qwq`, `deepseek_r1_lmstudio`, `gpt_oss`, `qwen3_lmstudio`
- Local HF: `psyllm_gml_local`, `piaget_local`, `psyche_r1_local`, `psych_qwen_local`

## Quick Smoke Test

Test with a small subset first (any model):
```powershell
python hf-local-scripts\run_qwen3_lmstudio.py study-a --generate-only --max-samples 5
```

## Custom Options

### Specify Output Location
```powershell
python hf-local-scripts\run_qwen3_lmstudio.py study-a --generate-only --cache-out results\custom\study_a.jsonl
```

### Custom Data Directory
```powershell
python hf-local-scripts\run_qwen3_lmstudio.py study-a --generate-only --data-dir data\custom_study_a
```

### Generation Parameters
```powershell
python hf-local-scripts\run_qwen3_lmstudio.py study-a --generate-only --temperature 0.8 --top-p 0.95 --max-new-tokens 2048
```

## After Generation

1. **Calculate Study A metrics** (from cache):
   ```powershell
   python scripts\study_a\metrics\calculate_metrics.py
   ```

   Or use the pipeline directly:
   ```python
   from reliable_clinical_benchmark.pipelines.study_a import run_study_a
   run_study_a(model=runner, from_cache="results/{model-id}/study_a_generations.jsonl", ...)
   ```

2. **Calculate bias metrics** (separate run):
   ```powershell
   python scripts\study_a\metrics\calculate_bias.py
   ```

## Output

- **Generations**: `results/{model-id}/study_a_generations.jsonl`
  - Each entry has `mode: "cot"` or `mode: "direct"`
  - 2 generations per sample (one CoT, one Direct)
- **Metrics**: Calculated separately using `from_cache` mode
  - Faithfulness Gap (Δ_Reasoning)
  - Step-F1
  - Accuracy (CoT vs Direct)
  - Silent Bias Rate (R_SB) - from separate bias run

## Data Source

- **Input**: `data/openr1_psy_splits/study_a_test.json` (Primary split)
- **Structure**: Vignettes with patient transcripts
- **Purpose**: Measure if reasoning improves accuracy (faithfulness)

## Implementation Details

- **Pipeline**: `src/reliable_clinical_benchmark/pipelines/study_a.py`
- **Metrics**: `src/reliable_clinical_benchmark/metrics/faithfulness.py`
- **Model Interface**: Uses `ModelRunner.generate(prompt, mode="cot"|"direct")`
- **Cache Format**: JSONL with fields: `id`, `mode`, `prompt`, `output_text`, `status`, `timestamp`, `model_name`, `meta`

## Related Documentation

- **Architecture**: See `docs/studies/study_a/study_a_faithfulness.md` for implementation details
- **Bias Evaluation**: See `docs/studies/study_a/study_a_bias.md` for Silent Bias Rate (R_SB) workflow
- **Metrics**: See `docs/metrics/METRIC_CALCULATION_PIPELINE.md` for metric calculation details

---

## Dual Path Commands (PC + Mac)

### PC Path (Uni setup)

```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-benchmark-env
$Env:PYTHONPATH="src"
python hf-local-scripts\run_qwen3_lmstudio.py study-a --generate-only --api-identifier gpt-oss-20b --model-name gpt-oss-20b
python scripts\studies\study_a\metrics\calculate_metrics.py --use-cleaned --output-dir metric-results\study_a
```

### Mac Path (Uni setup)

```bash
cd "/Users/ryangichuru/Documents/SSD-K/Uni/3rd year/NLP/Assignment 2/reliable_clinical_benchmark/Uni-setup"
export PYTHONPATH=src
python hf-local-scripts/run_qwen3_lmstudio.py study-a --generate-only --api-identifier gpt-oss-20b --model-name gpt-oss-20b
python scripts/studies/study_a/metrics/calculate_metrics.py --use-cleaned --output-dir metric-results/study_a
```

## Worker Commands (Absolute Bottom)

Study A generation scripts do not expose a Python `--workers` flag. Use the bias runner for explicit worker control:

```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-benchmark-env
$Env:PYTHONPATH="src"
$Env:PYTHONUNBUFFERED="1"
python hf-local-scripts\run_study_a_bias_generate_only.py --model-id gpt_oss_lmstudio --workers 8
```

```bash
cd "/Users/ryangichuru/Documents/SSD-K/Uni/3rd year/NLP/Assignment 2/reliable_clinical_benchmark/Uni-setup"
export PYTHONPATH=src
PYTHONUNBUFFERED=1 python hf-local-scripts/run_study_a_bias_generate_only.py --model-id gpt_oss_lmstudio --workers 8
```
