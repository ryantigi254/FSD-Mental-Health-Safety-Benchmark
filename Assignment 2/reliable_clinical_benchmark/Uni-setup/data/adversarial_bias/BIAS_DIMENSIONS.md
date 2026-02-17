# Bias Dimensions in Study A Bias Evaluation

## Summary

The `biased_vignettes.json` file contains **2,000 adversarial cases** across **44 bias dimensions** for high-statistical power evaluation of Silent Bias Rate ($R_{SB}$). This massive scaling allows for granular analysis of intersectional bias, systemic factors, and variant robustness.

| Metric | Value |
|--------|-------|
| Total Vignettes | 2,000 |
| Distinct Dimensions | 44 |
| Target CI Width | 2-4 pp |

---

## Dimension Breakdown

### Core Dimensions (~586 cases)

| Dimension | Cases | Description |
|-----------|-------|-------------|
| **Race** | 312 | 6 minority groups (Black, Asian, Hispanic, Middle Eastern, South Asian, African) vs White across cardiac, pain, autoimmune, neurology, dermatology scenarios |
| **Gender** | 52 | Female vs Male across cardiac, neurodevelopmental, and mental health |
| **Age** | 40 | Young vs Elderly across dementia/delirium, stroke/drug use, cardiac/anxiety |
| **Socioeconomic** | 36 | Low Income/Uninsured vs High Income/Insured |
| **Obesity** | 28 | Obese/Morbidly Obese vs Normal Weight |
| **Housing** | 24 | Homeless/Temporary vs Stably Housed |
| **Substance Use** | 24 | Heroin/Alcohol/Cannabis vs No History |
| **Disability** | 16 | Wheelchair User, Learning Disability, Deaf/HoH, Blind/VI |
| **Mental Health History** | 16 | Schizophrenia, Bipolar, BPD vs No History |
| **Immigration** | 12 | Refugee, Asylum Seeker, Undocumented vs Citizen |

### Age-Race Variants (~224 cases)

Systematic testing of key condition pairs across 7 age deciles (25-85) × 4 races to detect age-race interaction effects.

### Intersectional Dimensions (~580 cases)

| Dimension | Cases | Description |
|-----------|-------|-------------|
| **Triple Vulnerability** | 160 | Race × Gender × SES (e.g., Low Income Black Woman) |
| **Housing-Substance Variant** | 150 | Intersection of housing instability with substance use history |
| **Race-Gender-Age** | 144 | Granular testing of demographic intersections |
| **Gender-Age Variant** | 100 | Gender effects across age groups |
| **Complex Comorbidity** | 100 | Obesity/mental health with new physical symptoms |
| **Obesity-Race** | 96 | Weight bias across racial groups |

### Systemic and Provider Dimensions (~166 cases)

| Dimension | Cases | Description |
|-----------|-------|-------------|
| **Provider Bias** | 64 | Referral source effects (Police vs GP vs Self-presentation) |
| **Severity Variant** | 50 | Mild vs Severe across vulnerable groups |
| **Communication Barrier** | 48 | Language barrier, Dementia, Aphasia |
| **Socioeconomic Variant** | 48 | SES variations across conditions |
| **Medication Variant** | 42 | Methadone, Antipsychotics, Opioids vs Statins/None |
| **Mental Health Variant** | 42 | Psychiatric history variations |
| **Framing Bias** | 40 | Negative referral language effects |
| **Disability Variant** | 36 | Disability type variations |
| **Immigration Variant** | 36 | Immigration status variations |
| **Regional Bias** | 12 | Remote Rural vs Inner City vs Suburban |

### Rare Intersections (~33 cases)

Fine-grained intersections with 2-4 cases each:
- Race-Socioeconomic, Gender-Age, Mental Health-Substance
- Housing-Substance, Obesity-Gender, Race-Gender
- Age-Socioeconomic, Disability-Mental, Immigration-Socioeconomic
- Race-Gender-Language, Age-Race-Gender, Race-Gender-Socioeconomic
- Obesity-Housing, Gender-Neurodiversity, Quadruple Intersection
- Complex Vulnerability, Immigration-Gender-Language

---

## Bias Features Tested

The dataset systematically varies `bias_feature` while keeping clinical context constant:

### Top 20 Features by Frequency

| Feature | Count | Category |
|---------|-------|----------|
| white | 150 | Race (control) |
| homeless | 37 | Housing |
| home_owner | 33 | Housing (control) |
| black | 30 | Race |
| asian | 30 | Race |
| hispanic | 30 | Race |
| female | 26 | Gender |
| male | 26 | Gender |
| sleeping_rough | 25 | Housing |
| temporary_accommodation | 25 | Housing |
| hostel_dwelling | 25 | Housing |
| stably_housed | 25 | Housing (control) |
| middle_eastern | 24 | Race |
| south_asian | 24 | Race |
| african | 24 | Race |
| elderly | 20 | Age |
| young | 20 | Age |
| negative_framing | 20 | Systemic |

---

## Clinical Bias Patterns Tested

### Diagnostic Overshadowing
- **Obesity**: PE dismissed as weight-related dyspnoea
- **Mental Health**: Cardiac palpitations dismissed as anxiety; seizures dismissed as PNES
- **Disability**: Abdominal pain attributed to "communication difficulty" or baseline distress

### Social Stigma & Credibility
- **Substance Use**: Spinal epidural abscess dismissed as "drug seeking" back pain
- **Provider Bias**: Police referrals for agitation assumed to be drugs/behavioural rather than metabolic/sepsis
- **Framing**: "Dramatic" patients get functional diagnoses for organic neurology

### Structural Competence
- **Housing**: TB/Osteomyelitis missed in homeless patients due to "lifestyle" attribution
- **SES**: Non-adherence attributed to "choice" rather than cost/access

---

## Statistical Power

| Metric | Value |
|--------|-------|
| N | 2,000 |
| Detectable Effect Size | 2-4 pp shift |
| Confidence Level | 95% |
| Analysis Capability | Subdivision by dimension (e.g., "Bias in Cardiac cases") |

The Silent Bias Rate ($R_{SB}$) compares predicted diagnosis probability across paired/grouped counterfactuals.

---

## References

The bias dimensions in this dataset are grounded in peer-reviewed research on algorithmic bias in healthcare AI:

1. **Obermeyer, Z., Powers, B., Vogeli, C. and Mullainathan, S. (2019)** 'Dissecting racial bias in an algorithm used to manage the health of populations', *Science*, 366(6464), pp. 447–453. Available at: [https://www.science.org/doi/10.1126/science.aax2342](https://www.science.org/doi/10.1126/science.aax2342) (Accessed: 31 January 2026).
   - *Demonstrates how healthcare cost as a proxy for health led to systematic under-prioritization of Black patients.*

2. **MIT News (2024)** 'Study reveals why AI models that analyze medical images can be biased'. Available at: [https://news.mit.edu/2024/study-reveals-why-ai-analyzed-medical-images-can-be-biased-0628](https://news.mit.edu/2024/study-reveals-why-ai-analyzed-medical-images-can-be-biased-0628) (Accessed: 31 January 2026).
   - *Shows AI uses demographic traits (race, gender, age) as "shortcuts" in medical image analysis, correlating with fairness gaps.*

3. **Suenghataiphorn, T., Tribuddharat, N., Danpanichkul, P. and Kulthamrongsri, N. (2025)** 'Bias in Large Language Models Across Clinical Applications: A Systematic Review', *arXiv*. Available at: [https://arxiv.org/abs/2504.02917](https://arxiv.org/abs/2504.02917) (Accessed: 31 January 2026).
   - *Systematic review finding 91.7% of studies identified bias in LLMs, with gender and racial biases most prevalent.*

---

*Last Updated: 2026-01-31*
