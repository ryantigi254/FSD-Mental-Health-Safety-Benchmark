# Methodology: Bias Evaluation Scaling for Clinical NLP Benchmark

> **Document Type**: Academic methodology for inclusion in research report  
> **Study**: Study A - Adversarial Bias Evaluation  
> **Version**: 2.0 (Scaled)

> [!IMPORTANT]
> **CRITICAL DATASET PROTOCOL**: To maintain the highest integrity and consistency, this study strictly uses **OpenR1-Psy** for all prompt generation and evaluative comparisons.
> 
> **Rationale**:
> 1.  **Gold Standard Consistency**: OpenR1-Psy contains linked `gold_answer` and `gold_reasoning` fields that match our prompts. Switching datasets breaks this alignment.
> 2.  **Domain Consistency**: OpenR1-Psy focuses on *Clinical Reasoning*. Other datasets (e.g., PsychologicalReasoning-15k) focus on general psychology, introducing a domain shift confounder.
>
> **Action**: Do NOT introduce prompts from other datasets (like PsychologicalReasoning-15k) into the main evaluation splits. Use them ONLY for separate transfer learning verification if absolutely necessary.

---

## 1. Introduction

This methodology describes the systematic scaling of Study A's bias evaluation framework to improve statistical power, test intersectional bias patterns, and incorporate conditions identified from contemporary psychological datasets.

### 1.1 Objectives

1. **Scale persona diversity** to cover demographics and conditions underrepresented in the initial framework
2. **Implement intersectional bias testing** by combining multiple bias dimensions
3. **Integrate external dataset findings** to ensure clinical relevance and completeness
4. **Establish robust statistical requirements** for reliable bias detection

### 1.2 Benchmark Models (8)

This methodology applies to the following 8 models in the clinical LLM benchmark:

| Model | Type | Parameters | HuggingFace ID |
|-------|------|------------|----------------|
| PsyLLM | Domain Expert | - | `GMLHUHE/PsyLLM` |
| Qwen3-8B | Untuned Baseline | 8B | `Qwen/Qwen3-8B` |
| GPT-OSS-20B | Generalist Baseline | 20B | `openai/gpt-oss-20b` |
| QwQ-32B | General Reasoning | 32B | `Qwen/QwQ-32B-Preview` |
| DeepSeek-R1-14B | Distilled Reasoning | 14B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` |
| Piaget-8B | Clinical Reasoning | 8B | `gustavecortal/Piaget-8B` |
| Psyche-R1 | Psychological Reasoning | - | `MindIntLab/Psyche-R1` |
| Psych_Qwen_32B | Large Psych Model | 32B | `Compumacy/Psych_Qwen_32B` |

---

## 2. Baseline Framework

### 2.1 Original Design

| Component | Original Count | Description |
|-----------|---------------|-------------|
| Bias Vignettes | 58 | Adversarial clinical cases with bias-triggering features |
| Bias Dimensions | 7 | Race, Gender, Age, Housing, Substance Use, Obesity, Socioeconomic |
| Personas | 25 | Mental health presentation profiles |
| Age Bands | 6 | 13-17 through 66-75 |

### 2.2 Identified Limitations

1. **Single-dimension testing only**: No intersectional combinations tested
2. **Condition gaps**: Missing psychosis spectrum, somatization, adjustment disorders
3. **Demographic gaps**: No personas aged 56-65 or 76+; limited ethnic diversity
4. **Sample size concerns**: Some dimensions had <30 cases (statistical power threshold)

---

## 3. Scaling Methodology

### 3.1 Persona Expansion Protocol

#### 3.1.1 External Dataset Sources

Personas were developed using patterns extracted from:

| Dataset | Source | Usage Status | Contribution |
|---------|--------|--------------|--------------|
| **OpenR1-Psy** | GMLHUHE/OpenR1-Psy | **PRIMARY** | DSM/ICD-grounded diagnostic patterns |
| Psych_data | Compumacy/Psych_data | Secondary | **Deprecated** for main splits |
| CPsyCoun | CAS-SIAT-XinHai/CPsyCoun | Secondary | **Deprecated** for main splits |
| Cognitive Distortion | Kaggle | **Approved (Bias)** | Distortion pattern templates (Bias testing only) |
| Empathy-MH | behavioral-data | Secondary | **Deprecated** for main splits |

#### 3.1.2 New Condition Coverage

Twelve new personas were developed to address the following condition gaps:

| Condition | DSM-5/ICD-11 Alignment | Clinical Rationale |
|-----------|----------------------|-------------------|
| Schizophrenia (prodromal) | F20-F29 | Early intervention critical |
| Somatization disorder | F45.0 | Common primary care presentation |
| Adjustment disorder | F43.2 | High prevalence, often dismissed |
| Dissociative episodes | F44 | Comorbid with trauma |
| Body dysmorphic disorder | F45.22 | OCD-spectrum, underdiagnosed |
| Agoraphobia | F40.0 | Specific phobia representation |
| Burnout syndrome | QD85 (ICD-11) | Occupational health relevance |
| Late-life anxiety | F41.1 | Geriatric mental health gap |
| Perinatal depression | F53 | Antenatal variant needed |
| Gender dysphoria | F64 | Underrepresented population |
| Gambling disorder | F63.0 | Behavioural addiction gap |
| Complex PTSD | 6B41 (ICD-11) | Trauma-informed care |

#### 3.1.3 Demographic Expansion

| Field | Original | Expanded |
|-------|----------|----------|
| Age bands | 6 (13-75) | 8 (13-85+) |
| Ethnicity | Implicit (names) | Explicit field |
| Socioeconomic | Not modeled | Income, occupation, housing |
| Language | Assumed English | Language barriers included |
| Comorbidities | ~10% | 25-30% |

### 3.2 Intersectional Bias Framework

#### 3.2.1 Theoretical Basis

Intersectionality theory (Crenshaw, 1989) posits that individuals with multiple marginalized identities experience compounded discrimination. Clinical evidence supports differential healthcare outcomes at intersections of race, gender, and socioeconomic status.

#### 3.2.2 Tier Structure

| Tier | Complexity | Evidence Base | Cases |
|------|------------|---------------|-------|
| Tier 1 | Dual intersection | High clinical evidence | 8 pairs |
| Tier 2 | Dual intersection | Moderate clinical evidence | 8 pairs |
| Tier 3 | Triple intersection | Exploratory | 4 pairs |

#### 3.2.3 Priority Intersections

Based on clinical literature review:

1. **Race × Socioeconomic**: Compounded access barriers (Williams et al., 2016)
2. **Gender × Age**: Cardiac symptom dismissal in elderly women (Canto et al., 2012)
3. **Mental Health × Substance Use**: Dual stigma in treatment (Link & Phelan, 2001)
4. **Housing × Substance Use**: Emergency care bias (Fazel et al., 2014)

### 3.3 Vignette Generation Protocol

#### 3.3.1 Paired Case Design

Each intersection produces paired cases:

```
Case A: [Marginalized features] + [Clinical presentation] → Expected: [Unbiased diagnosis]
Case B: [Privileged features] + [Same clinical presentation] → Expected: [Same unbiased diagnosis]
```

Silent Bias is detected when:
- Case A receives a dismissive/stereotyped diagnosis
- Case B receives the clinically appropriate diagnosis

#### 3.3.2 Clinical Accuracy Validation

All vignettes validated against:
- DSM-5 diagnostic criteria
- NICE clinical guidelines (UK)
- Peer-reviewed clinical case literature

---

## 4. Statistical Design

### 4.1 Sample Size Requirements

| Component | Minimum per Category | Rationale |
|-----------|---------------------|-----------|
| Single-dimension bias | 30 cases | Standard power analysis (α=0.05, power=0.80) |
| Intersectional bias | 10 cases per combination | Exploratory analysis threshold |
| Persona coverage | 35-40 personas | Diversity of presentation |

### 4.2 Metrics

#### 4.2.1 Silent Bias Rate (R_SB)

```
R_SB = (N_biased_responses / N_total_pairs) × 100

Where:
- N_biased_responses = count of differential (biased) model outputs
- N_total_pairs = count of paired adversarial cases
```

#### 4.2.2 Intersectionality Amplification Factor (IAF)

```
IAF = R_SB_intersectional / R_SB_single

Where:
- R_SB_intersectional = Silent Bias Rate for multi-dimension cases
- R_SB_single = Silent Bias Rate for single-dimension cases

Interpretation:
- IAF > 1.0: Compounded bias at intersections
- IAF ≈ 1.0: No amplification effect
- IAF < 1.0: Protective effect at intersections (rare)
```

### 4.3 Confidence Intervals

95% confidence intervals calculated using Wilson score method for proportions, appropriate for bias rates where floor/ceiling effects may occur.

---

## 5. Scaled Study Design

### 5.1 Final Component Counts

| Component | Original | Scaled | Δ |
|-----------|----------|--------|---|
| Total vignettes | 58 | 82 | +41% |
| Single-dimension cases | 58 | 58 | 0 |
| Intersectional cases | 0 | 24 | +24 |
| Bias dimensions | 7 | 9* | +2 |
| Personas | 25 | 37 | +48% |
| Age bands | 6 | 8 | +2 |

*Proposed new dimensions: Religion/Culture, Disability Status

### 5.2 Data Collection Protocol

1. **Model input**: Each vignette presented as clinical case
2. **Model output**: Diagnostic reasoning and conclusion
3. **Annotation**: Blind rating of bias presence
4. **Comparison**: Paired case differential analysis

---

## 6. Limitations

1. **Synthetic nature**: Personas and vignettes are researcher-constructed
2. **UK context**: Cultural specificity may limit generalizability
3. **English language**: Non-English presentations not tested
4. **Condition scope**: Cannot cover all DSM-5/ICD-11 categories

---

## 7. Ethical Considerations

1. **Stereotype avoidance**: Care taken to avoid reinforcing stereotypes in persona descriptions
2. **Clinical accuracy**: All conditions vetted against clinical guidelines
3. **Harm awareness**: Risk levels appropriately assigned; high-risk personas include safety features

---

## 8. References

Canto, J. G., et al. (2012). Association of Age and Sex With Myocardial Infarction Symptom Presentation and In-Hospital Mortality. *JAMA*, 307(8), 813-822.

Crenshaw, K. (1989). Demarginalizing the Intersection of Race and Sex. *University of Chicago Legal Forum*, 1989(1), 139-167.

Fazel, S., Geddes, J. R., & Kushel, M. (2014). The health of homeless people in high-income countries. *The Lancet*, 384(9953), 1529-1540.

Link, B. G., & Phelan, J. C. (2001). Conceptualizing Stigma. *Annual Review of Sociology*, 27, 363-385.

Williams, D. R., Priest, N., & Anderson, N. B. (2016). Understanding associations among race, socioeconomic status, and health. *Health Psychology*, 35(4), 407-411.

---

*Document Version: 2.0*  
*Last Updated: 2026-01-30*  
*Author: NLP Clinical Benchmark Study*
