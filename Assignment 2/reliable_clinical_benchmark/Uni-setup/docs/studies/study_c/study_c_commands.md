# Study C Generation Commands

## Overview

Study C evaluates **Longitudinal Drift** - measuring whether models maintain consistency and avoid drift over multi-turn conversations.

**File**: `hf-local-scripts/run_study_c_generate_only.py`

**What it does**:

- Validates Study C split **personas + IDs** (`validate_study_c_schema`) before generating
- Generates responses for each case and turn (summary + dialogue variants)
- Writes to: `results/<model-id>/study_c_generations.jsonl`

**Note**: This is a standalone generation run. Study C metrics are calculated by running the Study C pipeline (`run_study_c()`), e.g. via `scripts/run_evaluation.py --study C`.

## Continuity Score Gold Target Plans (Optional)

To compute Study C `continuity_score` reproducibly (no API / no external model), first populate gold target plans derived from OpenR1-Psy therapist reasoning:

```powershell
python scripts\study_c\gold_plans\populate_from_openr1.py --force
```

This writes:
- `data/study_c_gold/target_plans.json`

Then run Study C evaluation as normal; `continuity_score` will be included in `study_c_results.json` if at least one gold plan is available.

**Architecture**:

- Uses `run_study_c()` pipeline from `reliable_clinical_benchmark.pipelines.study_c`
- Each case has 10 turns, each turn generates 2 variants (summary + dialogue)
- Supports `--generate-only` flag for generation-only mode (no metrics)

## Environment Requirements

- **`mh-llm-benchmark-env`**: For LM Studio models (`qwq`, `deepseek_r1_lmstudio`, `gpt_oss`, `qwen3_lmstudio`, `psyllm`)
- **`mh-llm-local-env`**: For local HF models (`psyllm_gml_local`, `piaget_local`, `psyche_r1_local`, `psych_qwen_local`)

See `docs/environment/ENVIRONMENT.md` for setup instructions.

**Note**: Adjust the Anaconda path (`D:\Anaconda3\Scripts\activate`) if your installation lives elsewhere.

## Context Length Configuration

**IMPORTANT**: Study C uses 10 turns per case, which requires sufficient context length. **ALL models must be configured for 32,768 tokens (32K)** in LM Studio:

1. **Open LM Studio** → **My Models**
2. **Select your model** → Click the **⚙️ Settings** (gear icon)
3. **Set Context Length to 32,768** for ALL models:
   - **GPT-OSS-20B**: **32,768** tokens (must be increased from default 4,096)
   - **Qwen3-8B, QwQ-32B, DeepSeek-R1-14B, PsyLLM**: **32,768** tokens
   - **Local HF models**: Ensure model config supports 32K (most do by default)

4. **Reload the model** after changing context length

**Why this matters**: Study C accumulates conversation history across 10 turns (20 messages: user + assistant). All models in Study C support 32K context, allowing **full conversation history to be maintained without truncation**. The pipeline keeps the complete conversation history for all 10 turns, ensuring models have full context for consistent responses.

**Verification**: After setting context length, check LM Studio logs to confirm the model loaded with the correct context size.

## Workflow per Model

For each model:

1. **Activate environment** and set up paths
2. **Run unit tests** (shared extraction/metrics tests, once per environment)
3. **Run Study C smoke test** (model-specific, tiny `max_cases`)
4. **Run full Study C generation** (`run_study_c_generate_only.py`)

All commands below are run from:

```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
```

---

### PsyLLM (LM Studio)

**Environment**: `mh-llm-benchmark-env`

#### 1. Activate Environment

```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-benchmark-env
$Env:PYTHONNOUSERSITE="1"
$Env:PYTHONPATH="src"
```

#### 2. Unit Tests (Run Once – Shared Across All Models)

```powershell
pytest tests/unit/metrics/test_extraction.py -v
```

#### 3. Smoke Test (Study C)

```powershell
python src/tests/studies/study_c/lmstudio/test_study_c_gpt_oss.py
```

**Note**: PsyLLM (LM Studio) uses the same smoke test as GPT-OSS.

#### 4. Full Generation (Study C)

```powershell
python hf-local-scripts\run_study_c_generate_only.py --model-id gpt_oss
```

**Alternative (using full Python path, if activation fails):**

```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
$Env:PYTHONNOUSERSITE="1"
$Env:PYTHONPATH="src"
C:\Users\22837352\.conda\envs\mh-llm-benchmark-env\python.exe -m pytest tests/unit/metrics/test_extraction.py -v
C:\Users\22837352\.conda\envs\mh-llm-benchmark-env\python.exe src/tests/studies/study_c/lmstudio/test_study_c_gpt_oss.py
C:\Users\22837352\.conda\envs\mh-llm-benchmark-env\python.exe hf-local-scripts\run_study_c_generate_only.py --model-id gpt_oss
```

---

### PsyLLM (HF local, GMLHUHE/PsyLLM)

**Environment**: `mh-llm-local-env`

#### 1. Activate Environment

```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-local-env
$Env:PYTHONNOUSERSITE="1"
$Env:PYTHONPATH="src"
```

#### 2. Unit Tests (Run Once – Shared Across All Models)

```powershell
C:\Users\22837352\.conda\envs\mh-llm-local-env\python.exe -m pytest tests/unit/metrics/test_extraction.py -v
```

#### 3. Smoke Test (Study C)

```powershell
C:\Users\22837352\.conda\envs\mh-llm-local-env\python.exe src/tests/studies/study_c/models/test_study_c_psyllm_gml_local.py
```

#### 4. Full Generation (Study C)

```powershell
C:\Users\22837352\.conda\envs\mh-llm-local-env\python.exe hf-local-scripts\run_study_c_generate_only.py --model-id psyllm_gml_local
```

**Alternative (if activation works, you can use `python` directly):**

```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-local-env
$Env:PYTHONNOUSERSITE="1"
$Env:PYTHONPATH="src"
python -m pytest tests/unit/metrics/test_extraction.py -v
python src/tests/studies/study_c/models/test_study_c_psyllm_gml_local.py
python hf-local-scripts\run_study_c_generate_only.py --model-id psyllm_gml_local
```

---

### QwQ-32B (LM Studio)

**Environment**: `mh-llm-benchmark-env`

#### 1. Activate Environment

```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-benchmark-env
$Env:PYTHONNOUSERSITE="1"
$Env:PYTHONPATH="src"
```

#### 2. Unit Tests (Run Once – Shared Across All Models)

```powershell
pytest tests/unit/metrics/test_extraction.py -v
```

#### 3. Smoke Test (Study C)

```powershell
python src/tests/studies/study_c/lmstudio/test_study_c_qwq.py
```

#### 4. Full Generation (Study C)

```powershell
python hf-local-scripts\run_study_c_generate_only.py --model-id qwq
```

**Alternative (using full Python path, if activation fails):**

```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
$Env:PYTHONNOUSERSITE="1"
$Env:PYTHONPATH="src"
C:\Users\22837352\.conda\envs\mh-llm-benchmark-env\python.exe -m pytest tests/unit/metrics/test_extraction.py -v
C:\Users\22837352\.conda\envs\mh-llm-benchmark-env\python.exe src/tests/studies/study_c/lmstudio/test_study_c_qwq.py
C:\Users\22837352\.conda\envs\mh-llm-benchmark-env\python.exe hf-local-scripts\run_study_c_generate_only.py --model-id qwq
```

---

### DeepSeek-R1 (LM Studio distill)

**Environment**: `mh-llm-benchmark-env`

#### 1. Activate Environment

```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-benchmark-env
$Env:PYTHONNOUSERSITE="1"
$Env:PYTHONPATH="src"
```

#### 2. Unit Tests (Run Once – Shared Across All Models)

```powershell
pytest tests/unit/metrics/test_extraction.py -v
```

#### 3. Smoke Test (Study C)

```powershell
python src/tests/studies/study_c/lmstudio/test_study_c_deepseek_r1_lmstudio.py
```

#### 4. Full Generation (Study C)

```powershell
python hf-local-scripts\run_study_c_generate_only.py --model-id deepseek_r1_lmstudio
```

**Alternative (using full Python path, if activation fails):**

```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
$Env:PYTHONNOUSERSITE="1"
$Env:PYTHONPATH="src"
C:\Users\22837352\.conda\envs\mh-llm-benchmark-env\python.exe -m pytest tests/unit/metrics/test_extraction.py -v
C:\Users\22837352\.conda\envs\mh-llm-benchmark-env\python.exe src/tests/studies/study_c/lmstudio/test_study_c_deepseek_r1_lmstudio.py
C:\Users\22837352\.conda\envs\mh-llm-benchmark-env\python.exe hf-local-scripts\run_study_c_generate_only.py --model-id deepseek_r1_lmstudio
```

---

### GPT-OSS-20B (LM Studio)

**Environment**: `mh-llm-benchmark-env`

#### 1. Activate Environment

```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-benchmark-env
$Env:PYTHONNOUSERSITE="1"
$Env:PYTHONPATH="src"
```

#### 2. Unit Tests (Run Once – Shared Across All Models)

```powershell
pytest tests/unit/metrics/test_extraction.py -v
```

#### 3. Smoke Test (Study C)

```powershell
python src/tests/studies/study_c/lmstudio/test_study_c_gpt_oss.py
```

#### 4. Full Generation (Study C)

```powershell
python hf-local-scripts\run_study_c_generate_only.py --model-id gpt_oss
```

**Alternative (using full Python path, if activation fails):**

```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
$Env:PYTHONNOUSERSITE="1"
$Env:PYTHONPATH="src"
C:\Users\22837352\.conda\envs\mh-llm-benchmark-env\python.exe -m pytest tests/unit/metrics/test_extraction.py -v
C:\Users\22837352\.conda\envs\mh-llm-benchmark-env\python.exe src/tests/studies/study_c/lmstudio/test_study_c_gpt_oss.py
C:\Users\22837352\.conda\envs\mh-llm-benchmark-env\python.exe hf-local-scripts\run_study_c_generate_only.py --model-id gpt_oss
```

---

### Qwen3-8B (LM Studio)

**Environment**: `mh-llm-benchmark-env`

#### 1. Activate Environment

```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-benchmark-env
$Env:PYTHONNOUSERSITE="1"
$Env:PYTHONPATH="src"
```

#### 2. Unit Tests (Run Once – Shared Across All Models)

```powershell
pytest tests/unit/metrics/test_extraction.py -v
```

#### 3. Smoke Test (Study C)

```powershell
python src/tests/studies/study_c/lmstudio/test_study_c_qwen3_lmstudio.py
```

#### 4. Full Generation (Study C)

```powershell
python hf-local-scripts\run_study_c_generate_only.py --model-id qwen3_lmstudio
```

**Alternative (using full Python path, if activation fails):**

```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
$Env:PYTHONNOUSERSITE="1"
$Env:PYTHONPATH="src"
C:\Users\22837352\.conda\envs\mh-llm-benchmark-env\python.exe -m pytest tests/unit/metrics/test_extraction.py -v
C:\Users\22837352\.conda\envs\mh-llm-benchmark-env\python.exe src/tests/studies/study_c/lmstudio/test_study_c_qwen3_lmstudio.py
C:\Users\22837352\.conda\envs\mh-llm-benchmark-env\python.exe hf-local-scripts\run_study_c_generate_only.py --model-id qwen3_lmstudio
```

---

### Piaget-8B (HF local)

**Environment**: `mh-llm-local-env`

#### 1. Activate Environment

```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-local-env
$Env:PYTHONNOUSERSITE="1"
$Env:PYTHONPATH="src"
```

#### 2. Unit Tests (Run Once – Shared Across All Models)

```powershell
C:\Users\22837352\.conda\envs\mh-llm-local-env\python.exe -m pytest tests/unit/metrics/test_extraction.py -v
```

#### 3. Smoke Test (Study C)

```powershell
C:\Users\22837352\.conda\envs\mh-llm-local-env\python.exe src/tests/studies/study_c/models/test_study_c_piaget_local.py
```

#### 4. Full Generation (Study C)

```powershell
C:\Users\22837352\.conda\envs\mh-llm-local-env\python.exe hf-local-scripts\run_study_c_generate_only.py --model-id piaget_local
```

**Alternative (if activation works, you can use `python` directly):**

```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-local-env
$Env:PYTHONNOUSERSITE="1"
$Env:PYTHONPATH="src"
python -m pytest tests/unit/metrics/test_extraction.py -v
python src/tests/studies/study_c/models/test_study_c_piaget_local.py
python hf-local-scripts\run_study_c_generate_only.py --model-id piaget_local
```

---

### Psyche-R1 (HF local)

**Environment**: `mh-llm-local-env`

#### 1. Activate Environment

```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-local-env
$Env:PYTHONNOUSERSITE="1"
$Env:PYTHONPATH="src"
```

#### 2. Unit Tests (Run Once – Shared Across All Models)

```powershell
C:\Users\22837352\.conda\envs\mh-llm-local-env\python.exe -m pytest tests/unit/metrics/test_extraction.py -v
```

#### 3. Smoke Test (Study C)

```powershell
C:\Users\22837352\.conda\envs\mh-llm-local-env\python.exe src/tests/studies/study_c/models/test_study_c_psyche_r1_local.py
```

#### 4. Full Generation (Study C)

```powershell
C:\Users\22837352\.conda\envs\mh-llm-local-env\python.exe hf-local-scripts\run_study_c_generate_only.py --model-id psyche_r1_local
```

**Alternative (if activation works, you can use `python` directly):**

```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-local-env
$Env:PYTHONNOUSERSITE="1"
$Env:PYTHONPATH="src"
python -m pytest tests/unit/metrics/test_extraction.py -v
python src/tests/studies/study_c/models/test_study_c_psyche_r1_local.py
python hf-local-scripts\run_study_c_generate_only.py --model-id psyche_r1_local
```

---

### Psych-Qwen-32B (HF local, 4-bit)

**Environment**: `mh-llm-local-env`

#### 1. Activate Environment

```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-local-env
$Env:PYTHONNOUSERSITE="1"
$Env:PYTHONPATH="src"
```

#### 2. Unit Tests (Run Once – Shared Across All Models)

```powershell
C:\Users\22837352\.conda\envs\mh-llm-local-env\python.exe -m pytest tests/unit/metrics/test_extraction.py -v
```

#### 3. Smoke Test (Study C)

```powershell
C:\Users\22837352\.conda\envs\mh-llm-local-env\python.exe src/tests/studies/study_c/models/test_study_c_psych_qwen_local.py
```

#### 4. Full Generation (Study C)

```powershell
C:\Users\22837352\.conda\envs\mh-llm-local-env\python.exe hf-local-scripts\run_study_c_generate_only.py --model-id psych_qwen_local
```

**Alternative (if activation works, you can use `python` directly):**

```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-local-env
$Env:PYTHONNOUSERSITE="1"
$Env:PYTHONPATH="src"
python -m pytest tests/unit/metrics/test_extraction.py -v
python src/tests/studies/study_c/models/test_study_c_psych_qwen_local.py
python hf-local-scripts\run_study_c_generate_only.py --model-id psych_qwen_local
```

---

## Quick Smoke Test (Generic)

If you just want a fast pipeline-level smoke test using the shared helper:

```powershell
python src/tests/studies/study_c/test_study_c_generate_only.py --model-id qwen3_lmstudio --max-cases 1 --max-tokens 512
```

## After Generation

1. **Calculate Study C metrics** (from cache):

   ```powershell
   python scripts\run_evaluation.py --study c --study-c-from-cache results\{model-id}\study_c_generations.jsonl
   ```

   Or use the pipeline directly:

   ```python
   from reliable_clinical_benchmark.pipelines.study_c import run_study_c
   run_study_c(model=runner, from_cache="results/{model-id}/study_c_generations.jsonl", ...)
   ```

## Output

- **Generations**: `results/{model-id}/study_c_generations.jsonl`
  - Per case + per turn:
    - `variant: "summary"` (for Entity Recall Decay metric)
    - `variant: "dialogue"` (for Knowledge Conflict Rate metric)
- **Metrics**: Calculated separately using `from_cache` mode
- **Total Generations**: 600 (30 cases × 10 turns × 2 variants)

## Data Source

- **Input**: `data/openr1_psy_splits/study_c_test.json`
- **Structure**:
  - `cases`: Multi-turn conversations with `patient_summary`, `critical_entities`, `turns`, `metadata` (persona_id, source_openr1_ids)
- **Personas**: 10 personas (aisha, jamal, eleni, maya, sam, leo, priya, noor, tomas, kai)
- **Turns**: 10 turns per case

## Implementation Details

- **Pipeline**: `src/reliable_clinical_benchmark/pipelines/study_c.py`
- **Metrics**: `src/reliable_clinical_benchmark/metrics/drift.py`
- **Model Interface**: Uses `ModelRunner.generate(prompt, mode="default")` for both variants
- **Cache Format**: JSONL with fields: `id`, `case_id`, `turn_num`, `variant`, `prompt`, `response_text`, `conversation_text`, `status`, `timestamp`, `model_name`, `persona_id`, `meta`

## Gold Target Plans (Continuity Score)

**Continuity Score** requires gold target plans extracted from OpenR1-Psy. These are stored in `data/study_c_gold/target_plans.json`.

### Extracting Target Plans

To populate/refresh gold target plans:

```bash
python scripts/study_c/gold_plans/populate_from_openr1.py --force
```

This extracts plan-of-care summaries from OpenR1-Psy `counselor_think` (full conversation) using the same dataset used for Study A gold labels, ensuring objectivity and reproducibility.

**Current Coverage**: 30/30 cases have extracted plans (100%). All cases can have Continuity Score computed.

See `data/study_c_gold/README.md` for detailed documentation on the extraction process and reproducibility guarantees.

## Related Documentation

- **Architecture**: See `docs/studies/study_c/study_c_drift.md` for implementation details
- **Metrics**: See `docs/metrics/METRIC_CALCULATION_PIPELINE.md` for metric calculation details
- **Gold Plans**: See `data/study_c_gold/README.md` for target plan extraction and usage
