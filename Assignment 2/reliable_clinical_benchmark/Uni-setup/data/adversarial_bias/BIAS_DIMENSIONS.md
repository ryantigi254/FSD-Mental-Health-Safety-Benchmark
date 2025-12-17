# Bias Dimensions in Study A Bias Evaluation

## Summary

The `biased_vignettes.json` file contains **58 adversarial cases** across **7 bias dimensions** (expanded from 12 cases to meet the tex file requirement of 50 base + 8 buffer = 58 total cases per model).

### Dimensions and Case Counts

1. **Race** (16 cases)
   - Black vs White pairs (4 cases): chest pain, diabetes complications
   - Asian vs White pairs (2 cases): cardiac symptoms, autoimmune conditions
   - Hispanic vs White pairs (2 cases): pain management, language barriers
   - Middle Eastern vs White pairs (2 cases): neurological symptoms
   - Additional race-based scenarios (6 cases)

2. **Gender** (10 cases)
   - Female vs Male pairs: cardiac symptoms (under-diagnosis in women), ADHD/autism (under-diagnosis in women), eating disorders (under-diagnosis in men), mental health presentations

3. **Age** (8 cases)
   - Elderly vs Young pairs: UTI confusion vs dementia, stroke vs drug use assumptions, memory loss presentations, cardiac symptoms

4. **Housing** (6 cases)
   - Homeless vs Stably Housed pairs: abdominal pain, infections, wound care, tuberculosis risk

5. **Substance Use** (6 cases)
   - History vs No History pairs: heroin use (spinal infections), alcohol dependence (pancreatitis), cannabis use (hyperemesis syndrome)

6. **Obesity** (6 cases - NEW DIMENSION)
   - Obese vs Normal Weight pairs: pulmonary embolism (weight bias), heart failure, joint pain (inflammatory vs mechanical)

7. **Socioeconomic** (6 cases - NEW DIMENSION)
   - Low Income vs High Income: medication adherence, diabetes management
   - Manual Labourer vs Office Worker: back pain assessment
   - Uninsured vs Insured: cardiac care access and delay

## Bias Features

The bias features tested include:
- **Race**: `black`, `white`, `asian`, `hispanic`, `middle eastern`
- **Gender**: `female`, `male`
- **Age**: `elderly`, `young`, `middle-aged`
- **Housing**: `homeless`, `stably housed`
- **Substance Use**: `history of heroin use`, `no history of substance use`, `history of alcohol dependence`, `no history of alcohol use`, `regular cannabis use`, `no cannabis use`
- **Obesity**: `obese`, `normal weight` (BMI-based)
- **Socioeconomic**: `low income`, `high income`, `manual labourer`, `office worker`, `uninsured`, `insured`

## Clinical Bias Patterns Tested

### Race-Based Biases
- Cardiac symptoms: under-diagnosis of ACS in Black/Asian patients
- Pain management: cultural stoicism assumptions for Hispanic patients
- Language barriers: communication issues affecting diagnosis
- Autoimmune conditions: under-recognition in Asian patients

### Gender-Based Biases
- Cardiac disease: women's symptoms attributed to anxiety
- Neurodevelopmental disorders: ADHD/autism under-diagnosis in women
- Eating disorders: anorexia under-diagnosis in men
- Mental health: different diagnostic thresholds by gender

### Age-Based Biases
- UTI in elderly: confusion attributed to dementia
- Stroke in young: symptoms attributed to drug use
- Memory loss: depression vs dementia assumptions
- Cardiac symptoms: age-related decline assumptions

### Housing-Based Biases
- Infections: TB under-diagnosis in homeless patients
- Wound care: self-neglect vs medical conditions
- Abdominal pain: alcohol-related assumptions

### Substance Use Biases
- Spinal infections: drug-seeking vs medical emergency
- Pancreatitis: alcohol assumption vs gallstone pathology
- Hyperemesis: cannabis attribution vs other causes

### Obesity-Based Biases (NEW)
- Pulmonary embolism: weight-related dyspnoea assumption
- Heart failure: obesity complications vs cardiac pathology
- Joint pain: mechanical strain vs inflammatory disease

### Socioeconomic Biases (NEW)
- Medication adherence: non-compliance vs cost barriers
- Back pain: malingering vs legitimate pathology
- Care access: delay attribution vs insurance status

## Notes

- Each dimension has paired cases where the same clinical presentation receives different diagnoses based on the bias feature
- The evaluation measures **Silent Bias Rate (R_SB)** by comparing model outputs across these paired cases
- Cases are designed to test whether models exhibit demographic or social bias in their diagnostic reasoning
- All 58 cases are required per the tex file specification (50 base + 8 buffer)
- Each case = one prompt (not multiple prompts per case)
