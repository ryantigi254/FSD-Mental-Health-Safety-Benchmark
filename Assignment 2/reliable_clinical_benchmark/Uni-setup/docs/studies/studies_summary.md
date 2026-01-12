# Benchmark Overview

This document provides a high-level summary of the mental health LLM benchmark, covering all three studies, models, and the complete evaluation workflow.

## Benchmark Structure

The benchmark evaluates LLMs across **three complementary studies** that measure different aspects of clinical reasoning:

### Study A: Faithfulness + Bias
**Purpose**: Measures reasoning utility and fairness

**Components**:
1. **Main Faithfulness Evaluation** (Primary)
   - **Faithfulness Gap (Δ_Reasoning)**: Measures if CoT reasoning actually drives the answer
   - **Step-F1**: Diagnostic metric for reasoning quality
   - **Accuracy**: Diagnosis correctness (CoT and Direct modes)
   - **Data**: 300 samples from OpenR1-Psy test split
   - **Modes**: CoT (Chain-of-Thought) and Direct (diagnosis only)

2. **Silent Bias Rate (R_SB)** (Supplementary)
   - Measures demographic bias without explicit mention in reasoning
   - **Data**: 58 adversarial bias cases across 7 dimensions (Race, Gender, Age, Housing, Substance Use, Obesity, Socioeconomic)
   - **Mode**: CoT only

**Outputs**:
- Main: `results/{model-id}/study_a_generations.jsonl` (600 lines: 300 samples × 2 modes)
- Bias: `processed/study_a_bias/{model-id}/study_a_bias_generations.jsonl` (58 lines)

**Documentation**: 
- [Study A Faithfulness](study_a/study_a_faithfulness.md)
- [Study A Bias](study_a/study_a_bias.md)
- [Generation Commands](study_a/study_a_generation_commands.md)

---

### Study B: Sycophancy
**Purpose**: Measures resistance to user pressure and incorrect opinions

**Metrics**:
- **Sycophancy Probability (P_Syc)**: Primary metric - agreement rate difference between injected vs control
- **Evidence Hallucination (H_Ev)**: Diagnostic metric - unsupported claims rate
- **Flip Rate**: Supplementary - rate of changing correct diagnosis to incorrect
- **Truthfulness of Feedback (ToF)**: Advanced - semantic similarity to ground truth

**Data**: 
- 276 single-turn samples (IDs: b_001 to b_276)
- 10 multi-turn cases
- **Total**: 602 generations (552 single-turn + 50 multi-turn)
- Uses 10 synthetic personas from `persona_registry_v2.json`

**Output**: `results/{model-id}/study_b_generations.jsonl`

**Documentation**:
- [Study B Sycophancy](study_b/study_b_sycophancy.md)
- [Generation Commands](study_b/study_b_commands.md)

---

### Study C: Longitudinal Drift
**Purpose**: Measures memory retention and consistency over long conversations

**Metrics**:
- **Entity Recall Decay**: Primary metric - recall of critical entities over turns
- **Knowledge Conflict Rate (K_Conflict)**: Diagnostic - contradiction rate between turns
- **Continuity Score**: Supplementary - semantic consistency of advice

**Data**:
- 30 multi-turn cases (IDs: c_001 to c_030)
- 10 turns per case, 2 variants (summary + dialogue)
- **Total**: 600 generations (30 cases × 10 turns × 2 variants)
- Uses same 10 personas as Study B

**Output**: `results/{model-id}/study_c_generations.jsonl`

**Documentation**:
- [Study C Drift](study_c/study_c_drift.md)
- [Generation Commands](study_c/study_c_commands.md)

---

## Models Evaluated

The benchmark supports **8 models** across three runner types:

### Local Hugging Face Runners (PyTorch, weights on disk)

Models loaded from `Uni-setup/models/` directory:

| Model | Runner Class | Model ID(s) | Notes |
|-------|-------------|-------------|-------|
| **PsyLLM** | `PsyLLMGMLLocalRunner` | `psyllm`, `psyllm_gml_local` | [GMLHUHE/PsyLLM](https://huggingface.co/GMLHUHE/PsyLLM) - domain expert, counselling-tuned |
| **Piaget-8B** | `Piaget8BLocalRunner` | `piaget_local`, `piaget-8b-local` | [gustavecortal/Piaget-8B](https://huggingface.co/gustavecortal/Piaget-8B) |
| **Psyche-R1** | `PsycheR1LocalRunner` | `psyche_r1_local`, `psyche-r1-local` | [MindIntLab/Psyche-R1](https://huggingface.co/MindIntLab/Psyche-R1) - psychological reasoning |
| **Psych_Qwen-32B** | `PsychQwen32BLocalRunner` | `psych_qwen_local`, `psych-qwen-32b-local` | [Compumacy/Psych_Qwen_32B](https://huggingface.co/Compumacy/Psych_Qwen_32B) - **4-bit quantised** (24GB VRAM) |

### LM Studio Runners (OpenAI-compatible API)

Models hosted locally via LM Studio server:

| Model | Runner Class | Model ID(s) | Notes |
|-------|-------------|-------------|-------|
| **QwQ-32B** | `QwQLMStudioRunner` | `qwq_lmstudio`, `qwq-32b-lmstudio` | [Qwen/QwQ-32B-Preview](https://huggingface.co/Qwen/QwQ-32B-Preview) (overview: https://qwenlm.github.io/blog/qwq-32b-preview/) |
| **DeepSeek-R1-14B** | `DeepSeekR1LMStudioRunner` | `deepseek_r1_lmstudio`, `deepseek-r1-lmstudio` | [deepseek-ai/DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) (report: https://arxiv.org/abs/2501.12948) |
| **GPT-OSS-20B** | `GPTOSSLMStudioRunner` | `gpt_oss_lmstudio`, `gpt-oss-20b` | Model card: https://openai.com/index/gpt-oss-model-card/ |
| **Qwen3-8B** | `Qwen3LMStudioRunner` | `qwen3_lmstudio`, `qwen3-8b-lmstudio` | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) (report: https://arxiv.org/abs/2505.09388) |

**Model Runner Documentation**: See [MODEL_RUNNERS.md](../models/MODEL_RUNNERS.md) for detailed implementation and interface.

---

## Complete Workflow

### 1. Environment Setup

Two conda environments are required:

- **`mh-llm-benchmark-env`**: For LM Studio runners, evaluation, tests
- **`mh-llm-local-env`**: For local PyTorch models (requires modern transformers)

See [Environment Setup](../environment/ENVIRONMENT.md) for details.

### 2. Generation (Per Model, Per Study)

**Study A Main**:
```bash
python hf-local-scripts/run_psyllm_gml.py  # Example for PsyLLM
# Or use study-specific scripts in hf-local-scripts/
```

**Study A Bias**:
```bash
python hf-local-scripts/run_study_a_bias_generate_only.py --model-id psyllm
```

**Study B**:
```bash
python hf-local-scripts/run_study_b_generate_only.py --model-id psyllm
```

**Study C**:
```bash
python hf-local-scripts/run_study_c_generate_only.py --model-id psyllm
```

**Output Locations**:
- Study A Main: `results/{model-id}/study_a_generations.jsonl`
- Study A Bias: `processed/study_a_bias/{model-id}/study_a_bias_generations.jsonl`
- Study B: `results/{model-id}/study_b_generations.jsonl`
- Study C: `results/{model-id}/study_c_generations.jsonl`

### 3. Metric Calculation

**Study A Metrics** (includes bias):
```bash
python scripts/study_a/metrics/calculate_metrics.py
# Output: metric-results/all_models_metrics.json
```

**Study B Metrics**:
```bash
python scripts/study_b/metrics/calculate_sycophancy.py
# Output: results/{model-id}/study_b_results.json
```

**Study C Metrics**:
```bash
python scripts/study_c/metrics/calculate_drift.py
# Output: results/{model-id}/study_c_results.json
```

### 4. Testing

**Smoke Tests** (quick verification):
- Study A: `src/tests/studies/study_a/models/generations/test_study_a_generation_*.py`
- Study A Bias: `src/tests/studies/study_a/models/bias/test_study_a_bias_*.py`
- Study B: `src/tests/studies/study_b/test_study_b_*.py`
- Study C: `src/tests/studies/study_c/test_study_c_*.py`

**Unit Tests**:
- Extraction logic: `src/tests/studies/study_a/extraction/test_*.py`
- Metrics: `src/tests/studies/study_a/metrics/test_*.py`

See [Testing Guide](../testing/TESTING_GUIDE.md) for comprehensive testing documentation.

---

## Data Sources

### Test Splits
- **Location**: `data/openr1_psy_splits/`
- **Source**: [OpenR1-Psy Dataset](https://huggingface.co/datasets/GMLHUHE/OpenR1-Psy)
- **Files**:
  - `study_a_test.json` - 300 samples for Study A
  - `study_b_test.json` - 276 samples for Study B
  - `study_c_test.json` - 30 cases for Study C

### Adversarial Bias Cases
- **Location**: `data/adversarial_bias/biased_vignettes.json`
- **Cases**: 58 adversarial prompts across 7 bias dimensions
- **Purpose**: Silent Bias Rate (R_SB) evaluation

### Gold Labels
- **Location**: `data/study_a_gold/`
- **Files**: `gold_diagnosis_labels.json`, `gold_labels_mapping.json`
- **Purpose**: Study A accuracy and faithfulness calculation

### Personas
- **Location**: `docs/patient template/personas.json`
- **Registry**: `data/persona_registry_v2.json`
- **Usage**: Studies B and C use 10 synthetic personas for multi-turn evaluation
- **Note**: All personas are AI-generated from a template - no real patient data

---

## Key Metrics Summary

| Study | Primary Metric | Target Threshold | Interpretation |
|-------|---------------|------------------|----------------|
| **A (Faithfulness)** | Faithfulness Gap (Δ_Reasoning) | > 0.10 | Functional reasoning vs decorative |
| **A (Bias)** | Silent Bias Rate (R_SB) | < 0.20 | Low demographic bias |
| **B (Sycophancy)** | Sycophancy Probability (P_Syc) | < 0.20 | Resistance to user pressure |
| **C (Drift)** | Entity Recall (at Turn 10) | > 0.70 | Memory retention |

---

## File Structure

```
Uni-setup/
├── hf-local-scripts/          # Generation scripts (per study, per model)
│   ├── run_study_a_bias_generate_only.py
│   ├── run_study_b_generate_only.py
│   └── run_study_c_generate_only.py
├── scripts/                   # Metric calculation scripts
│   ├── study_a/metrics/
│   ├── study_b/metrics/
│   └── study_c/metrics/
├── data/
│   ├── openr1_psy_splits/     # Test splits (A, B, C)
│   ├── adversarial_bias/      # Bias cases (58 prompts)
│   ├── study_a_gold/          # Gold labels for Study A
│   └── persona_registry_v2.json
├── results/                   # Generation outputs
│   └── {model-id}/
│       ├── study_a_generations.jsonl
│       ├── study_b_generations.jsonl
│       └── study_c_generations.jsonl
├── processed/                 # Processed outputs
│   └── study_a_bias/{model-id}/study_a_bias_generations.jsonl
├── metric-results/            # Calculated metrics
│   └── all_models_metrics.json
└── docs/                      # Documentation
    └── studies/
        ├── study_a/
        ├── study_b/
        ├── study_c/
        └── studies_summary.md   # This file
```

---

## Quick Reference

### For New Users
1. **Start Here**: [Main Documentation Index](../README.md)
2. **Environment**: [Environment Setup](../environment/ENVIRONMENT.md)
3. **Models**: [Model Runners](../models/MODEL_RUNNERS.md)
4. **Testing**: [Testing Guide](../testing/TESTING_GUIDE.md)

### For Running Evaluations
1. **Study A**: [Study A Generation Commands](study_a/study_a_generation_commands.md)
2. **Study A Bias**: [Study A Bias](study_a/study_a_bias.md)
3. **Study B**: [Study B Commands](study_b/study_b_commands.md)
4. **Study C**: [Study C Commands](study_c/study_c_commands.md)

### For Understanding Metrics
1. **Study A**: [Study A Faithfulness](study_a/study_a_faithfulness.md)
2. **Study B**: [Study B Sycophancy](study_b/study_b_sycophancy.md)
3. **Study C**: [Study C Drift](study_c/study_c_drift.md)
4. **All Metrics**: [Metrics Quick Reference](../metrics/QUICK_REFERENCE.md)

---

## Notes

- **Deterministic Splits**: All test splits use `random.seed(42)` for reproducibility
- **Local Models**: Local HF models default to `models/{ModelName}` directories
- **Quantisation**: Psych_Qwen-32B uses 4-bit quantisation by default (24GB VRAM requirement)
- **Smoke Tests**: Use `max_tokens=512` for quick verification; full generations use `max_tokens=8192`
- **No Real Patient Data**: All personas and test cases are synthetic or derived from public datasets
