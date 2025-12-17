# Study C Generation Commands

## Overview

Study C evaluates **Longitudinal Drift** - measuring whether models maintain consistency and avoid drift over multi-turn conversations.

**File**: `hf-local-scripts/run_study_c_generate_only.py`

**What it does**:
- Validates Study C split **personas + IDs** (`validate_study_c_schema`) before generating
- Generates responses for each case and turn (summary + dialogue variants)
- Writes to: `results/<model-id>/study_c_generations.jsonl`

**Note**: This is a standalone generation run. Metrics are calculated separately using `from_cache` mode.

## Commands per Model

Run these from `Uni-setup/`:

### PsyLLM (LM Studio)
```powershell
python hf-local-scripts\run_study_c_generate_only.py --model-id psyllm
```

### PsyLLM (HF local, GMLHUHE/PsyLLM)
```powershell
python hf-local-scripts\run_study_c_generate_only.py --model-id psyllm_gml_local
```

### QwQ-32B (LM Studio)
```powershell
python hf-local-scripts\run_study_c_generate_only.py --model-id qwq
```

### DeepSeek-R1 (LM Studio distill)
```powershell
python hf-local-scripts\run_study_c_generate_only.py --model-id deepseek_r1_lmstudio
```

### GPT-OSS-20B (LM Studio)
```powershell
python hf-local-scripts\run_study_c_generate_only.py --model-id gpt_oss
```

### Qwen3-8B (LM Studio)
```powershell
python hf-local-scripts\run_study_c_generate_only.py --model-id qwen3_lmstudio
```

### Piaget-8B (HF local)
```powershell
python hf-local-scripts\run_study_c_generate_only.py --model-id piaget_local
```

### Psyche-R1 (HF local)
```powershell
python hf-local-scripts\run_study_c_generate_only.py --model-id psyche_r1_local
```

### Psych-Qwen-32B (HF local, 4-bit)
```powershell
python hf-local-scripts\run_study_c_generate_only.py --model-id psych_qwen_local
```

## Quick Smoke Test

Test with a small subset first (any model):
```powershell
python hf-local-scripts\run_study_c_generate_only.py --model-id qwen3_lmstudio --max-cases 1 --max-tokens 512
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

