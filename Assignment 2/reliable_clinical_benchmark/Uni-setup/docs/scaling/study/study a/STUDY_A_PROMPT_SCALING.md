# Study A: Prompt Scaling Guide

> **Purpose**: A comprehensive guide for scaling Study A prompts to meet statistical power requirements, using external psychological datasets as sources.

> [!IMPORTANT]
> **CRITICAL DATASET PROTOCOL**: To maintain the highest integrity and consistency, this study strictly uses **OpenR1-Psy** for all prompt generation and evaluative comparisons.
>
> **Rationale**:
> 1.  **Gold Standard Consistency**: OpenR1-Psy contains linked `gold_answer` and `gold_reasoning` fields that match our prompts. Switching datasets breaks this alignment.
> 2.  **Domain Consistency**: OpenR1-Psy focuses on *Clinical Reasoning*. Other datasets (e.g., PsychologicalReasoning-15k) focus on general psychology, introducing a domain shift confounder.
>
> **Action**: Do NOT introduce prompts from other datasets (like PsychologicalReasoning-15k) into the main evaluation splits. Use them ONLY for separate transfer learning verification if absolutely necessary.

---

## Benchmark Models (8)

| Model | Type | HuggingFace ID |
|-------|------|----------------|
| PsyLLM | Domain Expert | `GMLHUHE/PsyLLM` |
| Qwen3-8B | Untuned Baseline | `Qwen/Qwen3-8B` |
| GPT-OSS-20B | Generalist Baseline | `openai/gpt-oss-20b` |
| QwQ-32B | General Reasoning | `Qwen/QwQ-32B-Preview` |
| DeepSeek-R1-14B | Distilled Reasoning | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` |
| Piaget-8B | Clinical Reasoning | `gustavecortal/Piaget-8B` |
| Psyche-R1 | Psychological Reasoning | `MindIntLab/Psyche-R1` |
| Psych_Qwen_32B | Large Psych Model | `Compumacy/Psych_Qwen_32B` |

---

## 1. Current State & Scaling Requirements

### 1.1 Study A Faithfulness (Main)

| Metric | Current | Issue | Target |
|--------|---------|-------|--------|
| Samples per model | 180–300 | ~3× underpowered for ±2pp precision | 600–900 |
| CI width | 6–12 pp | Too wide for model comparison | ±2–3 pp |
| Total prompts (8 models) | ~1,920–2,400 | Insufficient | **4,800+** |

### 1.2 Study A Bias (Adversarial)

| Metric | Current | Issue | Target |
|--------|---------|-------|--------|
| Adversarial prompts | 58 | Only 7–14 biased outcomes per model | **200–300** |
| CI width | 42–72 pp | Essentially unusable | ±10 pp |
| Biased outcomes needed | 7–14 | 1 response changes rate by 14% | **100+ per model** |

> [!CAUTION]
> **Study A Bias is SEVERELY UNDERPOWERED** — the current 58 adversarial prompts produce only 7–14 biased outcomes per model, making statistical claims unreliable.

---

## 2. External Dataset Sources

### 2.1 Dataset Overview

| Dataset | Size | Format | Study A Use Case |
|---------|------|--------|-----------------|
| **OpenR1-Psy** | 19,302 dialogues | Multi-turn with reasoning traces | Clinical reasoning prompts |
| **Psych_data** | Large (largest synthetic) | Q&A JSON | Domain prompts with CBT patterns |
| **PsychologicalReasoning-15k** | 15,006 rows | Reasoning chains | Psychological reasoning test cases |
| **CPsyCoun** | Multi-turn | Counseling dialogues | Longitudinal case studies |
| **Cognitive Distortion Dataset** | Annotated | Text classification | Bias-related reasoning tests |
| **Empathy-Mental-Health** | 10k pairs | Post-response pairs | Empathic response evaluation |

### 2.2 Dataset Details

#### OpenR1-Psy (Primary Source)
- **HuggingFace**: `GMLHUHE/OpenR1-Psy`
- **Paper**: [arXiv:2505.15715](https://arxiv.org/pdf/2505.15715)
- **Size**: 19,302 dialogues, 125 MB
- **Key Features**:
  - Multi-turn psychological counseling interactions
  - Diagnostic & therapeutic reasoning traces
  - Clinically grounded (DSM/ICD standards)
  - Therapeutic diversity: CBT, ACT, psychodynamic, humanistic
- **Extraction Target**: 300–500 clinical reasoning prompts

#### Psych_data (Compumacy) -- **DEPRECATED FOR MAIN STUDY**
- **Status**: **Secondary / Transfer Test Only**
- **Reason**: Different prompt structure (Q&A vs Dialog) reduces consistency. Use only for out-of-distribution testing.

#### PsychologicalReasoning-15k -- **DEPRECATED FOR MAIN STUDY**
- **Status**: **Secondary / Transfer Test Only**
- **Reason**: **Domain Shift**. Focuses on general psychology knowledge, not clinical reasoning. Lacks explicit "Chain-of-Thought" structure found in OpenR1.

#### CPsyCoun -- **DEPRECATED FOR MAIN STUDY**
- **Status**: **Reference Only**
- **Reason**: Language barrier (Chinese) and reconstructed format.

#### Cognitive Distortion -- **RETAIN FOR BIAS ONLY**
- **Status**: **Approved for Adversarial Bias**
- **Reason**: Specific niche for Study A Bias (Adversarial) testing only.

#### Empathy-Mental-Health -- **DEPRECATED FOR MAIN STUDY**
- **Status**: **Secondary**
- **Reason**: Single-turn structure mismatch with OpenR1's multi-turn focus.

---

## 3. Extraction Pipelines

### 3.1 OpenR1-Psy Extraction

```python
"""
Extract clinical reasoning prompts from OpenR1-Psy for Study A faithfulness testing.
"""
from datasets import load_dataset
import json

# Load dataset
ds = load_dataset("GMLHUHE/OpenR1-Psy")

# Target conditions for Study A expansion
TARGET_CONDITIONS = [
    # Existing conditions (expand)
    "depression", "anxiety", "ptsd", "ocd", "bipolar",
    # New conditions (from gap analysis)
    "schizophrenia", "psychosis", "somatization", "adjustment",
    "dissociative", "body dysmorphic", "agoraphobia", "burnout",
    "personality", "eating disorder", "substance use"
]

# Filter for clinical reasoning content
def is_relevant(example):
    text = example.get('text', '') or example.get('dialogue', '')
    text_lower = text.lower()
    return any(cond in text_lower for cond in TARGET_CONDITIONS)

filtered = ds['train'].filter(is_relevant)

# Convert to Study A prompt format
def convert_to_study_a_prompt(example):
    """
    Transform OpenR1-Psy dialogue to Study A evaluation format.
    """
    dialogue = example.get('text', '') or example.get('dialogue', '')
    
    # Extract patient presentation (first turn)
    # Extract expected reasoning (reasoning trace)
    # Map to clinical reasoning task
    
    return {
        "id": f"openr1_{example.get('id', 'unknown')}",
        "prompt": dialogue,
        "expected_reasoning_steps": [],  # Extract from reasoning trace
        "condition_category": "extracted",
        "source": "OpenR1-Psy"
    }

# Export
expanded_prompts = [convert_to_study_a_prompt(ex) for ex in filtered]

# Save to Study A data directory
output_path = "data/study_a_gold/expanded_prompts_openr1.json"
with open(output_path, 'w') as f:
    json.dump(expanded_prompts, f, indent=2)

print(f"Extracted {len(expanded_prompts)} prompts from OpenR1-Psy")
```



---

## 4. Integration Strategy

### 4.1 Prompt Format Standardization

All extracted prompts must conform to Study A's expected format:

```json
{
  "id": "string (unique identifier)",
  "prompt": "string (clinical presentation or question)",
  "expected_output": "string (gold standard response pattern)",
  "reasoning_steps": ["step1", "step2", "..."],
  "condition": "string (DSM/ICD category)",
  "difficulty": "easy | medium | hard",
  "source": "string (original dataset)",
  "bias_features": ["feature1", "..."]  // Optional: for adversarial
}
```

### 4.2 Integration Checklist

- [ ] Download all datasets to `data/external_datasets/`
- [ ] Run extraction pipelines for each dataset
- [ ] Validate prompt format against Study A schema
- [ ] Merge with existing `study_a_gold` prompts
- [ ] Update prompt count in configuration
- [ ] Re-run sample size analysis
- [ ] Verify CI width improvement

### 4.3 Final Prompt Distribution (Target)

| Source | Faithfulness Prompts | Bias Prompts | Total |
|--------|---------------------|--------------|-------|
| Existing Study A | 180–300 | 58 | ~340 |
| OpenR1-Psy | 300–500 | 50 | ~450 |
| Psych_data | 200–300 | 30 | ~280 |
| PsychologicalReasoning-15k | 100–200 | 20 | ~160 |
| CPsyCoun (adapted) | 50–100 | 20 | ~80 |
| Cognitive Distortion | 20 | 80 | 100 |
| **Total** | **850–1,400** | **258** | **~1,410** |

**Per Model (8 models)**: ~600–900 faithfulness prompts + ~250+ bias prompts

---

## 5. Validation Requirements

### 5.1 Clinical Accuracy Check

All extracted prompts must be validated for:

- [ ] DSM-5/ICD-11 alignment
- [ ] Clinically accurate reasoning steps
- [ ] No stereotyping or harmful content
- [ ] Appropriate difficulty grading

### 5.2 Statistical Power Verification

After scaling, verify:

| Metric | Target | Verification Method |
|--------|--------|---------------------|
| Faithfulness CI width | <±3 pp | Calculate 95% CI after expansion |
| Bias biased outcomes | 100+ per model | Count biased classifications |
| Bias CI width | <±10 pp | Calculate 95% CI for bias rates |

---

## 6. References

### Papers
- Hu et al. (2025). Beyond Empathy: Integrating Diagnostic and Therapeutic Reasoning with LLMs. arXiv:2505.15715
- Psyche-R1 (2025). Towards Reliable Psychological LLMs. arXiv:2508.10848
- Empowering Psychotherapy with LLMs (2023). ACL Findings EMNLP
- Computational Approach to Empathy (2020). Sharma et al.
- CPsyCoun Framework (2024). ACL Findings
- Improving Language Models for Emotion Analysis (2024). CMCL

### Datasets
- `GMLHUHE/OpenR1-Psy` - HuggingFace
- `Compumacy/Psych_data` - HuggingFace
- `gustavecortal/PsychologicalReasoning-15k` - HuggingFace
- `CAS-SIAT-XinHai/CPsyCoun` - GitHub
- `behavioral-data/Empathy-Mental-Health` - GitHub
- Cognitive Distortion Detection - Kaggle

---

*Last Updated: 2026-01-30*
