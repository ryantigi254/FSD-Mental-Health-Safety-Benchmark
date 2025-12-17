# Study B Generation Commands

## Overview

Study B evaluates **Sycophancy** - measuring whether models abandon medical truth to agree with incorrect user opinions.

**File**: `hf-local-scripts/run_study_b_generate_only.py`

**What it does**:
- Validates Study B split **personas + IDs** (`validate_study_b_schema`) before generating
- Generates responses in two distinct modes:
  - **Single-turn**: Control + injected variants (for P_Syc, Flip Rate, H_Ev metrics)
  - **Multi-turn**: Iterative generation with rolling context (for Turn of Flip metric)
- Uses `model.chat()` for multi-turn conversations with proper context passing
- Writes to: `results/<model-id>/study_b_generations.jsonl`

**Note**: This is a standalone generation run. Metrics are calculated separately using `from_cache` mode.

**Architecture**:
- Single-turn generation: `_generate_single_turn_study_b()` - uses `model.generate()`
- Multi-turn generation: `_generate_multi_turn_study_b()` - uses `model.chat()` with structured message history
- Rolling context: Each turn includes full conversation history (system + all previous user/assistant messages)

## Commands per Model

Run these from `Uni-setup/`:

### PsyLLM (LM Studio)
```powershell
python hf-local-scripts\run_study_b_generate_only.py --model-id psyllm
```

### PsyLLM (HF local, GMLHUHE/PsyLLM)
```powershell
python hf-local-scripts\run_study_b_generate_only.py --model-id psyllm_gml_local
```

### QwQ-32B (LM Studio)
```powershell
python hf-local-scripts\run_study_b_generate_only.py --model-id qwq
```

### DeepSeek-R1 (LM Studio distill)
```powershell
python hf-local-scripts\run_study_b_generate_only.py --model-id deepseek_r1_lmstudio
```

### GPT-OSS-20B (LM Studio)
```powershell
python hf-local-scripts\run_study_b_generate_only.py --model-id gpt_oss
```

### Qwen3-8B (LM Studio)
```powershell
python hf-local-scripts\run_study_b_generate_only.py --model-id qwen3_lmstudio
```

### Piaget-8B (HF local)
```powershell
python hf-local-scripts\run_study_b_generate_only.py --model-id piaget_local
```

### Psyche-R1 (HF local)
```powershell
python hf-local-scripts\run_study_b_generate_only.py --model-id psyche_r1_local
```

### Psych-Qwen-32B (HF local, 4-bit)
```powershell
python hf-local-scripts\run_study_b_generate_only.py --model-id psych_qwen_local
```

## Quick Smoke Test

Test with a small subset first (any model):
```powershell
python hf-local-scripts\run_study_b_generate_only.py --model-id qwen3_lmstudio --max-samples 5 --max-tokens 512
```

## After Generation

1. **Calculate Study B metrics** (from cache):
   ```powershell
   python scripts\run_evaluation.py --study b --study-b-from-cache results\{model-id}\study_b_generations.jsonl
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
- **Total Generations**: 602 (276 samples × 2 + 50 multi-turn)
- **Implementation**: See `src/reliable_clinical_benchmark/pipelines/study_b.py` for single-turn/multi-turn separation

## Data Source

- **Input**: `data/openr1_psy_splits/study_b_test.json`
- **Structure**:
  - `samples`: Single-turn prompts with `prompt`, `gold_answer`, `incorrect_opinion`, `metadata` (persona_id, age)
  - `multi_turn_cases`: Multi-turn conversations for Turn of Flip (ToF) metric
- **Personas**: 10 personas (aisha, jamal, eleni, maya, sam, leo, priya, noor, tomas, kai)

