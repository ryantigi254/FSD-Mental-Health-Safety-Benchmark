# Study C Generation Commands

## Overview

Study C evaluates **Longitudinal Drift** - measuring whether models maintain consistency and avoid drift over multi-turn conversations.

**File**: `hf-local-scripts/run_study_c_generate_only.py`

**What it does**:
- Validates Study C split **personas + IDs** (`validate_study_c_schema`) before generating
- Generates responses for each case and turn (summary + dialogue variants)
- Writes to: `results/<model-id>/study_c_generations.jsonl`

**Note**: This is a standalone generation run. Metrics are calculated separately using `from_cache` mode.

**Architecture**:
- Uses `run_study_c()` pipeline from `reliable_clinical_benchmark.pipelines.study_c`
- Each case has 10 turns, each turn generates 2 variants (summary + dialogue)
- Supports `--generate-only` flag for generation-only mode (no metrics)

## Environment Requirements

- **`mh-llm-benchmark-env`**: For LM Studio models (`qwq`, `deepseek_r1_lmstudio`, `gpt_oss`, `qwen3_lmstudio`, `psyllm`)
- **`mh-llm-local-env`**: For local HF models (`psyllm_gml_local`, `piaget_local`, `psyche_r1_local`, `psych_qwen_local`)

See `docs/environment/ENVIRONMENT.md` for setup instructions.

**Note**: Adjust the Anaconda path (`D:\Anaconda3\Scripts\activate`) if your installation lives elsewhere.

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
python hf-local-scripts\run_study_c_generate_only.py --model-id psyllm
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
pytest tests/unit/metrics/test_extraction.py -v
```

#### 3. Smoke Test (Study C)

```powershell
python src/tests/studies/study_c/models/test_study_c_psyllm_gml_local.py
```

#### 4. Full Generation (Study C)

```powershell
python hf-local-scripts\run_study_c_generate_only.py --model-id psyllm_gml_local
```

---

### QwQ-32B (LM Studio)

**Environment**: `mh-llm-benchmark-env`

#### 1. Activate Environment

```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-benchmark-env
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

---

### DeepSeek-R1 (LM Studio distill)

**Environment**: `mh-llm-benchmark-env`

#### 1. Activate Environment

```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-benchmark-env
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

---

### GPT-OSS-20B (LM Studio)

**Environment**: `mh-llm-benchmark-env`

#### 1. Activate Environment

```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-benchmark-env
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

---

### Qwen3-8B (LM Studio)

**Environment**: `mh-llm-benchmark-env`

#### 1. Activate Environment

```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-benchmark-env
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
pytest tests/unit/metrics/test_extraction.py -v
```

#### 3. Smoke Test (Study C)

```powershell
python src/tests/studies/study_c/models/test_study_c_piaget_local.py
```

#### 4. Full Generation (Study C)

```powershell
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
pytest tests/unit/metrics/test_extraction.py -v
```

#### 3. Smoke Test (Study C)

```powershell
python src/tests/studies/study_c/models/test_study_c_psyche_r1_local.py
```

#### 4. Full Generation (Study C)

```powershell
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
pytest tests/unit/metrics/test_extraction.py -v
```

#### 3. Smoke Test (Study C)

```powershell
python src/tests/studies/study_c/models/test_study_c_psych_qwen_local.py
```

#### 4. Full Generation (Study C)

```powershell
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

## Related Documentation

- **Architecture**: See `docs/studies/study_c/study_c_drift.md` for implementation details
- **Metrics**: See `docs/metrics/METRIC_CALCULATION_PIPELINE.md` for metric calculation details