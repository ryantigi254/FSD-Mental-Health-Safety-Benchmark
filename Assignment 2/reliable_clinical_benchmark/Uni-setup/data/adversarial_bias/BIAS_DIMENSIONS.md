# Bias Dimensions in Study A Bias Evaluation

## Summary

The `biased_vignettes.json` file has been scaled to **1,964 adversarial cases** across **15 bias dimensions** (expanded from 12 → 58 → 1,964 cases) to achieve **2-4pp confidence intervals** for high-statistical power evaluation of Silent Bias Rate ($R_{SB}$).

This massive scaling allows for granular analysis of intersectional bias, systemic factors, and variant robustness.

### Dimensions and Case Counts

#### 1. Core Dimensions (Expanded)
- **Race** (~336 cases): 
    - Systematic pairing of 6 minority groups (Black, Asian, Hispanic, Middle Eastern, South Asian, African) vs White.
    - Scenarios: Cardiac (ACS), Pain Management, Autoimmune, Neurology, Dermatology.
- **Gender** (~40 cases): Female vs Male across cardiac, neurodevelopmental, and mental health presentations.
- **Age** (~40 cases): Young vs Elderly across dementia/delirium, stroke/drug use, and cardiac/anxiety axes.
- **Socioeconomic Status (SES)** (~72 cases): Low Income/Uninsured vs High Income/Insured affecting adherence and resource assumptions.
- **Housing** (~72 cases): Homeless/Temporary vs Stably Housed affecting infection and wound care logic.
- **Substance Use** (~48 cases): Heroin/Alcohol/Cannabis vs No History affecting pain and acute abdomen management.
- **Obesity** (~56 cases): Obese/Morbidly Obese vs Normal Weight affecting respiratory and cardiac diagnostics.

#### 2. New Dimensions
- **Disability** (~72 cases): Wheelchair User, Learning Disability, Deaf/HoH, Blind/VI vs No Disability.
    - *Bias*: Diagnostic overshadowing and attribution of acute symptoms to disability.
- **Immigration Status** (~72 cases): Refugee, Asylum Seeker, Undocumented vs Citizen.
    - *Bias*: Trauma attribution (psychosomatisation) of physical symptoms.
- **Mental Health History** (~84 cases): Schizophrenia, Bipolar, BPD vs No History.
    - *Bias*: "Functional" labeling of organic pathology (Diagnostic Overshadowing).

#### 3. Systematic Variants
- **Medication Bias** (~42 cases): Patients on Methadone, Antipsychotics, Opioids vs Statins/None.
    - Tests if medication history triggers "drug-seeking" labels.
- **Severity Bias** (~50 cases): Mild vs Severe presentations across vulnerable groups.
    - Tests if severe symptoms are dismissed as "anxiety" in specific demographics (e.g., Black women).
- **Age Variants** (~224 cases): Systematic testing of key condition pairs across 7 age deciles (25-85) x 4 races.

#### 4. Complex Intersections
- **Triple Vulnerability** (~48 cases): Interaction of **Race x Gender x SES** (e.g., Low Income Black Woman).
    - Tests compound disadvantage and "weathering" effects models.
- **Comorbidity Bias** (~100 cases): Intersections of obesity/mental health with new physical symptoms.
    - Specific test for **Diagnostic Overshadowing**.
- **Communication Barriers** (~32 cases): Language barrier, Dementia, Aphasia affecting history reliability.
- **Race x Gender x Age** (~144 cases): Granular testing of "Elderly Hispanic Female" vs "Young White Male".

#### 5. Systemic Factors
- **Framing Bias** (~40 cases): Impact of negative referral language ("known to be dramatic") vs neutral framing.
- **Provider Bias** (~64 cases): Impact of referral source (Police vs GP vs Self-presentation).
- **Regional Access** (~12 cases): Remote Rural vs Inner City vs Suburban access bias.

## Bias Features Tested

The dataset systematically varies the `bias_feature` field while keeping the `prompt` clinical core constant (or logically equivalent):

- **Demographics**: `black_female`, `elderly_hispanic`, `low_income_white`
- **Clinical Status**: `obese_bmi_45`, `history_schizophrenia`, `on_methadone`
- **Social Determinants**: `homeless`, `refugee`, `manual_labourer`
- **Systemic**: `police_referral`, `negative_framing_letter`, `remote_rural_address`

## Clinical Bias Patterns Tested

### Diagnostic Overshadowing
- **Obesity**: PE dismissed as weight-related dyspnoea.
- **Mental Health**: Cardiac palpitations dismissed as anxiety; seizures dismissed as PNES.
- **Disability**: Abdominal pain attribute to "communication difficulty" or baseline distress.

### Social Stigma & Credibility
- **Substance Use**: Spinal epidural abscess dismissed as "drug seeking" back pain.
- **Provider Bias**: Police referrals for agitation assumed to be drugs/behavioural rather than metabolic/sepsis.
- **Framing**: "Dramatic" patients get functional diagnoses for organic neurology.

### Structural Competence
- **Housing**: TB/Osteomyelitis missed in homeless patients due to "lifestyle" attribution.
- **SES**: Non-adherence attributed to "choice" rather than cost/access.

## Notes on Evaluation
- **Metric**: Silent Bias Rate ($R_{SB}$) compares predicted diagnosis probability across paired/grouped counterfactuals.
- **Statistical Power**: With N≈2000, we can detect bias shifts of 2-4% with 95% confidence, enabling subdivision analysis (e.g., "Bias in Cardiac cases" vs "Bias in Pain cases").
