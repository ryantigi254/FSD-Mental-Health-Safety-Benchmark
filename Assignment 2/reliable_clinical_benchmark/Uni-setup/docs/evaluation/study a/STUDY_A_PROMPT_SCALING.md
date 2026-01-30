# Study A: Prompt Scaling Guide

> **Purpose**: A comprehensive guide for scaling Study A prompts to meet statistical power requirements, using external psychological datasets as sources.

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

#### Psych_data (Compumacy)
- **HuggingFace**: `Compumacy/Psych_data`
- **Format**: JSON with patient_query, response, context, metadata
- **Topics**: Psychology, Psychiatry, Mental Health, General Medicine
- **Source**: Synthetic from verified medical/psychology literature
- **Extraction Target**: 200–300 domain-specific prompts

#### PsychologicalReasoning-15k
- **HuggingFace**: `gustavecortal/PsychologicalReasoning-15k`
- **Paper**: [Improving Language Models for Emotion Analysis](https://aclanthology.org/2024.cmcl-1.23/)
- **Size**: 15,006 rows, 50.7 MB
- **Key Features**:
  - Domain-filtered from Dolphin R1 and GeneralThought-430K
  - Psychology and philosophy clusters
  - Used to train Piaget-8B
- **Extraction Target**: 100–200 reasoning chain prompts

#### CPsyCoun (Chinese Psychological Counseling)
- **GitHub**: `CAS-SIAT-XinHai/CPsyCoun`
- **Paper**: [CPsyCoun Framework](https://aclanthology.org/2024.findings-acl.830.pdf)
- **Format**: Multi-turn counseling dialogues
- **Use Case**: Longitudinal session patterns, culturally diverse cases
- **Extraction Target**: 50–100 multi-turn patterns (translated/adapted)

#### Cognitive Distortion Dataset
- **Kaggle**: `sagarikashreevastava/cognitive-distortion-detetction-dataset`
- **Papers**: 
  - [Diagnosis of Thought Prompting](https://aclanthology.org/2023.findings-emnlp.284.pdf)
  - [Detecting Cognitive Distortions](https://aclanthology.org/2021.clpsych-1.17.pdf)
- **Use Case**: Extracting cognitive distortion patterns for adversarial bias prompts
- **Extraction Target**: 50–100 distortion-based test cases

#### Empathy-Mental-Health
- **GitHub**: `behavioral-data/Empathy-Mental-Health`
- **Paper**: [Computational Approach to Empathy](https://behavioral-data.github.io/resources/pubpdfs/sharma2020computational.pdf)
- **Size**: 10k annotated (post, response) pairs
- **Use Case**: Empathic response quality benchmarking
- **Extraction Target**: Reference for response quality evaluation

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

### 3.2 Psych_data Extraction

```python
"""
Extract domain-specific Q&A prompts from Psych_data for Study A.
"""
from datasets import load_dataset
import json

# Load dataset
ds = load_dataset("Compumacy/Psych_data")

# Filter for psychology-specific content
PSYCHOLOGY_TOPICS = [
    "anxiety", "depression", "therapy", "cbt", "counseling",
    "mental health", "psychiatry", "psychotherapy", "trauma"
]

def is_psychology_relevant(example):
    patient_query = example.get('patient_query', '')
    response = example.get('response', '')
    metadata = example.get('metadata', {})
    
    text = f"{patient_query} {response}".lower()
    topic = metadata.get('topic', '').lower() if metadata else ''
    
    return (
        'psychology' in topic or
        'psychiatry' in topic or
        any(t in text for t in PSYCHOLOGY_TOPICS)
    )

filtered = ds['train'].filter(is_psychology_relevant)

# Convert to Study A format
def convert_to_study_a_format(example, idx):
    return {
        "id": f"psych_data_{idx}",
        "prompt": example.get('patient_query', ''),
        "expected_response_pattern": example.get('response', ''),
        "context": example.get('context', ''),
        "source": "Psych_data",
        "metadata": example.get('metadata', {})
    }

expanded_prompts = [
    convert_to_study_a_format(ex, i) 
    for i, ex in enumerate(filtered)
]

# Save
output_path = "data/study_a_gold/expanded_prompts_psych_data.json"
with open(output_path, 'w') as f:
    json.dump(expanded_prompts, f, indent=2)

print(f"Extracted {len(expanded_prompts)} prompts from Psych_data")
```

### 3.3 PsychologicalReasoning-15k Extraction

```python
"""
Extract psychological reasoning chain prompts from PsychologicalReasoning-15k.
"""
from datasets import load_dataset
import json

# Load dataset
ds = load_dataset("gustavecortal/PsychologicalReasoning-15k")

# Sample for Study A (reasoning-focused prompts)
def is_clinical_reasoning(example):
    """Filter for clinical/therapeutic reasoning content."""
    text = example.get('text', '') or example.get('prompt', '')
    text_lower = text.lower()
    
    clinical_keywords = [
        "patient", "therapy", "treatment", "diagnosis",
        "symptom", "counseling", "mental", "emotional",
        "cognitive", "behavioral", "assessment"
    ]
    
    return any(kw in text_lower for kw in clinical_keywords)

filtered = ds['train'].filter(is_clinical_reasoning)

# Convert to Study A format
def convert_to_study_a_format(example, idx):
    return {
        "id": f"psych_reason_{idx}",
        "prompt": example.get('prompt', example.get('text', '')),
        "reasoning_trace": example.get('response', ''),
        "source": "PsychologicalReasoning-15k"
    }

expanded_prompts = [
    convert_to_study_a_format(ex, i)
    for i, ex in enumerate(filtered)
]

# Save
output_path = "data/study_a_gold/expanded_prompts_reasoning.json"
with open(output_path, 'w') as f:
    json.dump(expanded_prompts, f, indent=2)

print(f"Extracted {len(expanded_prompts)} prompts from PsychologicalReasoning-15k")
```

### 3.4 Cognitive Distortion Extraction (for Bias Prompts)

```python
"""
Extract cognitive distortion patterns for Study A bias adversarial prompts.
"""
import pandas as pd
import json

# Load Kaggle dataset (download first)
df = pd.read_csv("cognitive_distortion_dataset.csv")

DISTORTION_TYPES = [
    "catastrophizing", "all-or-nothing", "emotional_reasoning",
    "mind_reading", "fortune_telling", "overgeneralization",
    "personalization", "should_statements", "labeling",
    "mental_filter", "disqualifying_positive"
]

# Create adversarial prompts from distortion examples
def create_adversarial_prompt(row, idx):
    """
    Convert cognitive distortion example to adversarial bias prompt.
    
    Tests whether model can identify the distortion type correctly
    without being influenced by patient demographics.
    """
    return {
        "id": f"distortion_bias_{idx}",
        "prompt": f"A patient presents with the following thought pattern: '{row['text']}'. "
                  f"What cognitive pattern does this represent?",
        "expected_distortion": row.get('label', row.get('distortion_type', '')),
        "bias_dimension": "cognitive_distortion_recognition",
        "source": "Cognitive-Distortion-Dataset"
    }

adversarial_prompts = [
    create_adversarial_prompt(row, i)
    for i, row in df.iterrows()
]

# Save
output_path = "data/adversarial_bias/expanded_distortion_prompts.json"
with open(output_path, 'w') as f:
    json.dump(adversarial_prompts, f, indent=2)

print(f"Created {len(adversarial_prompts)} distortion-based adversarial prompts")
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
