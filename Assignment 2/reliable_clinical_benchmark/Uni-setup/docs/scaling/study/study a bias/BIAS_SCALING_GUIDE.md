# Bias Study Scaling Guide

> **Purpose**: This document provides a comprehensive framework for scaling Study A's bias evaluation by expanding personas, applying intersectional bias combinations, and integrating conditions from external psychological datasets.

---

## 1. Overview

### Current Setup
| Component | Count | Coverage |
|-----------|-------|----------|
| Bias Vignettes | 58 | 7 dimensions |
| Personas | 25 | 15+ conditions |
| Age Bands | 6 | 13-75 years |
| Risk Levels | 4 | low, passive, moderate, high |

### Target Scale
| Component | Target | Scaling Factor |
|-----------|--------|----------------|
| Bias Vignettes | 80-100 | 1.5-2× |
| Personas | 35-40 | 1.5× |
| Age Bands | 8 | +2 bands |
| Intersectional Cases | 20-30 | New category |

### Benchmark Models (8)

This scaling guide applies to the **8 models** in the clinical LLM benchmark:

| Model | Type | HuggingFace ID |
|-------|------|----------------|
| **PsyLLM** | Domain Expert | `GMLHUHE/PsyLLM` |
| **Qwen3-8B** | Untuned Baseline | `Qwen/Qwen3-8B` |
| **GPT-OSS-20B** | Generalist Baseline | `openai/gpt-oss-20b` |
| **QwQ-32B** | General Reasoning | `Qwen/QwQ-32B-Preview` |
| **DeepSeek-R1-14B** | Distilled Reasoning | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` |
| **Piaget-8B** | Clinical Reasoning | `gustavecortal/Piaget-8B` |
| **Psyche-R1** | Psychological Reasoning | `MindIntLab/Psyche-R1` |
| **Psych_Qwen_32B** | Large Psych Model | `Compumacy/Psych_Qwen_32B` |

---

## 2. Persona Scaling Strategy

### 2.1 Why Scale Personas for Bias Study?

Scaling personas is essential because:
1. **Statistical Power**: More diverse personas = narrower confidence intervals
2. **Intersectionality**: Real patients exhibit multiple bias-triggering characteristics
3. **Condition Coverage**: External datasets reveal clinical gaps in current personas
4. **Demographic Representation**: Better coverage of underrepresented groups

### 2.2 Combinatorial Approach

Rather than creating personas ad-hoc, use systematic combinations:

```
Scale Factor = Personas × Bias_Dimensions × Severity_Levels
             = 35 × 8 × 2 = 560 potential test cases
```

However, focus on **clinically meaningful combinations** (see Section 4).

### 2.3 Profile Areas to Expand

| Area | Current | Gap | Action |
|------|---------|-----|--------|
| **Age Bands** | 13-75 | Missing elderly (76+) and mid-late (56-65) | Add 2 bands |
| **Socioeconomic** | Implicit | Not explicitly modeled | Add income/employment fields |
| **Ethnicity** | Name-implied | Not explicit | Add ethnicity field to registry |
| **Comorbidities** | ~10% | Limited overlap | Increase to 25-30% |
| **Gender Identity** | 3 options | Limited trans/NB with condition-specific challenges | Add 2-3 personas |
| **Cultural Factors** | UK only | No immigration/language barriers | Add 3-4 personas |

---

## 3. External Dataset Integration

### 3.1 Dataset Sources and Conditions to Add

#### OpenR1-Psy Dataset
- **Source**: [HuggingFace - GMLHUHE/OpenR1-Psy](https://huggingface.co/datasets/GMLHUHE/OpenR1-Psy)
- **Size**: 19,302 dialogues
- **Key Features**: DSM/ICD-grounded diagnostic reasoning, CBT/ACT/psychodynamic therapy traces
- **Conditions to Extract**:
  - Schizophrenia spectrum disorders
  - Somatization patterns
  - Diagnostic reasoning examples

#### Compumacy/Psych_data
- **Source**: [HuggingFace - Compumacy/Psych_data](https://huggingface.co/datasets/Compumacy/Psych_data)
- **Format**: Synthetic Q&A from clinical literature
- **Conditions to Extract**:
  - Body dysmorphic disorder presentations
  - Burnout syndrome patterns
  - Adjustment disorder conversations

#### CPsyCoun Dataset
- **Source**: [GitHub - CAS-SIAT-XinHai/CPsyCoun](https://github.com/CAS-SIAT-XinHai/CPsyCoun)
- **Type**: Chinese psychological counseling (concepts transferable)
- **Conditions to Extract**:
  - Academic pressure variants
  - Family/relationship conflict presentations
  - Longitudinal session patterns

#### Cognitive Distortion Datasets
- **Source**: [Kaggle - Cognitive Distortion Detection](https://www.kaggle.com/datasets/sagarikashreevastava/cognitive-distortion-detetction-dataset/)
- **Use Case**: Add explicit distortion patterns to personas
- **Distortions to Model**:
  - Catastrophizing
  - All-or-nothing thinking
  - Emotional reasoning
  - Mind reading / Fortune telling

#### Empathy-Mental-Health
- **Source**: [GitHub - behavioral-data/Empathy-Mental-Health](https://github.com/behavioral-data/Empathy-Mental-Health)
- **Size**: 10k annotated (post, response) pairs
- **Use Case**: Reference for empathic response quality benchmarking

### 3.2 Extraction Pipeline

```python
# Pseudo-code for dataset extraction
from datasets import load_dataset

# Step 1: Load external dataset
dataset = load_dataset("GMLHUHE/OpenR1-Psy")

# Step 2: Filter for target conditions
target_conditions = [
    "schizophrenia", "somatization", "adjustment",
    "dissociative", "body dysmorphic", "phobia"
]

# Step 3: Extract dialogue patterns
filtered = dataset.filter(
    lambda x: any(cond in x['dialogue'].lower() for cond in target_conditions)
)

# Step 4: Convert to persona format
# See NEW_PERSONAS_IMPLEMENTATION.md for templates
```

---

## 4. Bias Dimension Combinations

### 4.1 Current Dimensions (7)

1. **Race**: black, white, asian, hispanic, middle eastern
2. **Gender**: female, male
3. **Age**: elderly, young, middle-aged
4. **Housing**: homeless, stably housed
5. **Substance Use**: heroin, alcohol, cannabis histories
6. **Obesity**: obese (BMI >30), normal weight
7. **Socioeconomic**: income level, occupation, insurance status

### 4.2 Recommended New Dimensions

| Dimension | Bias Features | Clinical Relevance |
|-----------|---------------|-------------------|
| **Religion/Culture** | Muslim, Jewish, Hindu, atheist | Treatment preferences, stigma |
| **Disability Status** | physical disability, chronic illness | Healthcare access bias |
| **Immigration** | refugee, undocumented, language barrier | System navigation barriers |

### 4.3 Intersectional Combinations (Priority Order)

#### Tier 1: High Clinical Evidence
| Combination | Example | Bias Pattern |
|-------------|---------|--------------|
| Race × Socioeconomic | Black + Low Income | Compounded access barriers |
| Gender × Age | Female + Elderly | Cardiac symptom dismissal |
| Mental Health × Substance | Depression + Alcohol history | Dual stigma in treatment |
| Housing × Substance | Homeless + Heroin history | Emergency care bias |

#### Tier 2: Moderate Evidence
| Combination | Example | Bias Pattern |
|-------------|---------|--------------|
| Obesity × Gender | Obese + Female | Weight-based dismissal + gender bias |
| Age × Socioeconomic | Elderly + Low Income | Ageism + resource limitations |
| Race × Gender | Black + Female | Intersectional healthcare disparities |

#### Tier 3: Exploratory
| Combination | Example | Bias Pattern |
|-------------|---------|--------------|
| Triple intersection | Asian + Female + Language barrier | Communication + cultural + gender |
| Contextual triggers | Middle Eastern + Post-9/11 setting | Implicit bias in threat assessment |

---

## 5. Implementation Checklist

### Phase 1: Persona Expansion
- [ ] Add 8-10 new personas covering missing conditions
- [ ] Add explicit socioeconomic fields to persona registry
- [ ] Add ethnicity and cultural background fields
- [ ] Create 2-3 comorbidity personas

### Phase 2: Vignette Generation
- [ ] Generate 20-30 intersectional bias vignettes
- [ ] Add Tier 1 combinations to existing vignettes
- [ ] Create paired cases for new dimensions

### Phase 3: Dataset Integration
- [ ] Download and filter OpenR1-Psy for target conditions
- [ ] Extract dialogue patterns for persona messages
- [ ] Validate against DSM-5 criteria

### Phase 4: Validation
- [ ] Review with clinical accuracy checklist
- [ ] Run power analysis on expanded dataset
- [ ] Document methodology for reproducibility

---

## 6. Quality Assurance

### Clinical Accuracy Checks
- [ ] All conditions align with DSM-5/ICD-11 criteria
- [ ] Risk levels appropriately assigned
- [ ] No stereotyping in persona descriptions
- [ ] Bias patterns reflect documented clinical disparities

### Statistical Requirements
- Minimum 30 cases per bias dimension for reliable metrics
- Intersectional cases: minimum 10 per combination
- Confidence intervals target: <10% at 95% CI

---

## Related Documents

- [NEW_PERSONAS_IMPLEMENTATION.md](./NEW_PERSONAS_IMPLEMENTATION.md) - Detailed persona profiles
- [INTERSECTIONAL_BIAS_VIGNETTES.md](./INTERSECTIONAL_BIAS_VIGNETTES.md) - Combined dimension cases
- [METHODOLOGY.md](./METHODOLOGY.md) - Formal methodology for academic report
- [BIAS_DIMENSIONS.md](../../data/adversarial_bias/BIAS_DIMENSIONS.md) - Current dimension details

---

*Last Updated: 2026-01-30*
