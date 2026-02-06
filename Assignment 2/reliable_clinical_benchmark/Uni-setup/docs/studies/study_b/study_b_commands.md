# Study B Generation Commands

## Overview

Study B evaluates **Sycophancy** - measuring whether models abandon medical truth to agree with incorrect user opinions.

**Files**:
- `hf-local-scripts/run_study_b_generate_only.py` (Single-Turn)
- `hf-local-scripts/run_study_b_multi_turn_generate_only.py` (Multi-Turn)

**What they do**:
- Validate Study B personas and IDs before generating.
- `run_study_b_generate_only.py`: Generates Single-Turn Control + Injected variants (for P_Syc, H_Ev, and analysis-only Flip Rate).
- `run_study_b_multi_turn_generate_only.py`: Generates Multi-Turn iterative conversations with rolling context (for Turn of Flip metric).
- Use `model.chat()` for multi-turn conversations with proper context passing.
- Write to: `results/<model-id>/study_b_generations.jsonl` (Single) or `results/<model-id>/study_b_multi_turn_generations.jsonl` (Multi).

**Note**: These are standalone generation runs. Metrics are calculated separately using `from_cache` mode.

**Architecture**:
- Single-turn generation: `_generate_single_turn_study_b()` - uses `model.generate()`
- Multi-turn generation: `_generate_multi_turn_study_b()` - uses `model.chat()` with structured message history
- Rolling context: Each turn includes full conversation history (system + all previous user/assistant messages)

## Environment Requirements

- **`mh-llm-benchmark-env`**: For LM Studio models (`qwq`, `deepseek_r1_lmstudio`, `gpt_oss`, `qwen3_lmstudio`, `psyllm`)
- **`mh-llm-local-env`**: For local HF models (`psyllm_gml_local`, `piaget_local`, `psyche_r1_local`, `psych_qwen_local`)

See `docs/environment/ENVIRONMENT.md` for setup instructions.

**Note**: Adjust the Anaconda path (`D:\Anaconda3\Scripts\activate`) if your installation lives elsewhere.

## Workflow per Model

For each model:

1. **Activate environment** and set up paths  
2. **Run unit tests** (shared extraction/metrics tests, once per environment)  
3. **Run Study B smoke test** (model-specific, tiny `max_samples`)  
4. **Run full Study B generation** (`run_study_b_generate_only.py`)  

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

#### 3. Smoke Test (Study B)

```powershell
python src/tests/studies/study_b/lmstudio/test_study_b_gpt_oss.py
```

#### 4. Full Generation (Study B)

```powershell
python hf-local-scripts\run_study_b_generate_only.py --model-id psyllm
```

#### 5. Multi-Turn Generation (Study B)

```powershell
python hf-local-scripts\run_study_b_multi_turn_generate_only.py --model-id psyllm
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

#### 3. Smoke Test (Study B)

```powershell
python src/tests/studies/study_b/models/test_study_b_psyllm_gml_local.py
```

#### 4. Full Generation (Study B)

```powershell
python hf-local-scripts\run_study_b_generate_only.py --model-id psyllm_gml_local
```

#### 5. Multi-Turn Generation (Study B)

```powershell
python hf-local-scripts\run_study_b_multi_turn_generate_only.py --model-id psyllm_gml_local
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

#### 3. Smoke Test (Study B)

```powershell
python src/tests/studies/study_b/lmstudio/test_study_b_qwq.py
```

#### 4. Full Generation (Study B)

```powershell
python hf-local-scripts\run_study_b_generate_only.py --model-id qwq
```

#### 5. Multi-Turn Generation (Study B)

```powershell
python hf-local-scripts\run_study_b_multi_turn_generate_only.py --model-id qwq
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

#### 3. Smoke Test (Study B)

```powershell
python src/tests/studies/study_b/lmstudio/test_study_b_deepseek_r1_lmstudio.py
```

#### 4. Full Generation (Study B)

```powershell
python hf-local-scripts\run_study_b_generate_only.py --model-id deepseek_r1_lmstudio
```

#### 5. Multi-Turn Generation (Study B)

```powershell
python hf-local-scripts\run_study_b_multi_turn_generate_only.py --model-id deepseek_r1_lmstudio
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

#### 3. Smoke Test (Study B)

```powershell
python src/tests/studies/study_b/lmstudio/test_study_b_gpt_oss.py
```

#### 4. Full Generation (Study B)

```powershell
python hf-local-scripts\run_study_b_generate_only.py --model-id gpt_oss
```

#### 5. Multi-Turn Generation (Study B)

```powershell
python hf-local-scripts\run_study_b_multi_turn_generate_only.py --model-id gpt_oss
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

#### 3. Smoke Test (Study B)

```powershell
python src/tests/studies/study_b/lmstudio/test_study_b_qwen3_lmstudio.py
```

#### 4. Full Generation (Study B)

```powershell
python hf-local-scripts\run_study_b_generate_only.py --model-id qwen3_lmstudio
```

#### 5. Multi-Turn Generation (Study B)

```powershell
python hf-local-scripts\run_study_b_multi_turn_generate_only.py --model-id qwen3_lmstudio
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

#### 3. Smoke Test (Study B)

```powershell
python src/tests/studies/study_b/models/test_study_b_piaget_local.py
```

#### 4. Full Generation (Study B)

```powershell
python hf-local-scripts\run_study_b_generate_only.py --model-id piaget_local
```

#### 5. Multi-Turn Generation (Study B)

```powershell
python hf-local-scripts\run_study_b_multi_turn_generate_only.py --model-id piaget_local
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

#### 3. Smoke Test (Study B)

```powershell
python src/tests/studies/study_b/models/test_study_b_psyche_r1_local.py
```

#### 4. Full Generation (Study B)

```powershell
python hf-local-scripts\run_study_b_generate_only.py --model-id psyche_r1_local
```

#### 5. Multi-Turn Generation (Study B)

```powershell
python hf-local-scripts\run_study_b_multi_turn_generate_only.py --model-id psyche_r1_local
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

#### 3. Smoke Test (Study B)

```powershell
python src/tests/studies/study_b/models/test_study_b_psych_qwen_local.py
```

#### 4. Full Generation (Study B)

```powershell
python hf-local-scripts\run_study_b_generate_only.py --model-id psych_qwen_local
```

#### 5. Multi-Turn Generation (Study B)

```powershell
python hf-local-scripts\run_study_b_multi_turn_generate_only.py --model-id psych_qwen_local
```

---

## Quick Smoke Test (Generic)

If you just want a fast pipeline-level smoke test using the shared helper:

```powershell
python src/tests/studies/study_b/test_study_b_generate_only.py --model-id qwen3_lmstudio --max-samples 1 --max-tokens 512
```

## After Generation

1. **Calculate Study B metrics** (from cache):
   ```powershell
   python scripts\run_evaluation.py --study b --study-b-from-cache results\{model-id}\study_b_generations.jsonl
   ```

2. **Calculate Multi-Turn metrics** (Turn of Flip):
   ```powershell
   python scripts\run_evaluation.py --study b --study-b-from-cache results\{model-id}\study_b_multi_turn_generations.jsonl
   ```

   Or use the pipeline directly:
   ```python
   from reliable_clinical_benchmark.pipelines.study_b import run_study_b
   run_study_b(model=runner, from_cache="results/{model-id}/study_b_generations.jsonl", ...)
   ```

## Output

- **Generations**: `results/{model-id}/study_b_generations.jsonl`
  - Single-turn: `variant: "control"` and `variant: "injected"` (2 per sample)
  - Multi-turn: `variant: "multi_turn"` (1 per turn)
    - Includes `conversation_history` (structured) and `conversation_text` (backward compatible)
- **Metrics**: Calculated separately using `from_cache` mode
- **Total Generations**: 4050 (2000 samples × 2 + 50 multi-turn)
- **Implementation**: See `src/reliable_clinical_benchmark/pipelines/study_b.py` for single-turn/multi-turn separation

## Data Source

- **Input**:
  - Single-Turn: `data/openr1_psy_splits/study_b_test.json`
  - Multi-Turn: `data/openr1_psy_splits/study_b_multi_turn_test.json`
- **Structure**:
  - `study_b_test.json`: Flat list of items with `prompt`, `gold_answer`, `incorrect_opinion`, `metadata`.
  - `study_b_multi_turn_test.json`: Flat list of cases for Turn of Flip (ToF) metric.
- **Personas**: 40 personas (Active coverage for scaling).

---

## Dual Path Commands (PC + Mac)

### PC Path (Uni setup)

```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-benchmark-env
$Env:PYTHONPATH="src"
python hf-local-scripts\run_study_b_generate_only.py --model-id gpt_oss
python hf-local-scripts\run_study_b_multi_turn_generate_only.py --model-id gpt_oss
python scripts\studies\study_b\metrics\calculate_metrics.py --use-cleaned --output-dir metric-results\study_b
```

### Mac Path (Uni setup)

```bash
cd "/Users/ryangichuru/Documents/SSD-K/Uni/3rd year/NLP/Assignment 2/reliable_clinical_benchmark/Uni-setup"
export PYTHONPATH=src
python hf-local-scripts/run_study_b_generate_only.py --model-id gpt_oss
python hf-local-scripts/run_study_b_multi_turn_generate_only.py --model-id gpt_oss
python scripts/studies/study_b/metrics/calculate_metrics.py --use-cleaned --output-dir metric-results/study_b
```

## Worker Commands (Absolute Bottom)

Study B scripts do not expose a Python `--workers` flag. Worker concurrency is controlled in LM Studio (`Max Concurrent Predictions`).
