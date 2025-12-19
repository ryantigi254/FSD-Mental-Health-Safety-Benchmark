## Adversarial bias cases (Study A – Silent Bias)

This directory contains the **frozen adversarial vignettes** for Silent Bias Rate \(R_{SB}\) evaluation.

### Dataset Overview

- **Total cases**: 58 (50 base + 8 buffer, per tex file Section 6.3 requirement)
- **Expansion**: Expanded from 12 cases to 58 cases to meet evaluation framework requirements
- **Structure**: One prompt per case (each case = one generation)

### Files

- `biased_vignettes.json` - Main dataset with 58 adversarial bias cases
- `BIAS_DIMENSIONS.md` - Detailed documentation of bias dimensions and case distribution

### Bias Dimensions (7 dimensions, 58 cases total)

1. **Race** (16 cases)
   - Black vs White, Asian vs White, Hispanic vs White, Middle Eastern vs White
   - Tests cardiac under-diagnosis, pain management biases, language barriers, autoimmune recognition

2. **Gender** (10 cases)
   - Female vs Male pairs across cardiac symptoms, ADHD/autism, eating disorders, mental health
   - Tests under-diagnosis patterns (e.g., cardiac disease in women, ADHD in women, eating disorders in men)

3. **Age** (8 cases)
   - Elderly vs Young pairs
   - Tests UTI confusion vs dementia, stroke vs drug use assumptions, memory loss presentations

4. **Housing** (6 cases)
   - Homeless vs Stably Housed pairs
   - Tests TB under-diagnosis, wound care assumptions, infection management biases

5. **Substance Use** (6 cases)
   - History vs No History pairs (heroin, alcohol, cannabis)
   - Tests drug-seeking vs medical emergency assumptions, substance attribution biases

6. **Obesity** (6 cases) - *New dimension*
   - Obese vs Normal Weight pairs
   - Tests weight-related dyspnoea assumptions, BMI-based diagnostic biases, weight stigma

7. **Socioeconomic** (6 cases) - *New dimension*
   - Low Income vs High Income, Manual Labourer vs Office Worker, Uninsured vs Insured
   - Tests medication adherence assumptions, back pain dismissal, care access delays

### Schema

Each case follows this JSON structure:

```json
{
  "id": "bias_XXX",
  "prompt": "<clinical vignette with biasing feature>",
  "bias_feature": "<demographic feature, e.g. 'obese', 'low income'>",
  "bias_label": "<biased diagnosis the model might choose>",
  "metadata": {
    "dimension": "<race|gender|age|housing|substance_use|obesity|socioeconomic>"
  }
}
```

### Usage

- **Generation**: Each model generates 58 lines (one per case) in `processed/study_a_bias/{model-id}/study_a_bias_generations.jsonl`
- **Evaluation**: Silent Bias Rate (R_SB) is calculated by comparing model outputs across paired cases
- **Metric**: Measures whether models exhibit demographic or social bias in diagnostic reasoning without explicitly mentioning the biasing feature

### Expansion Details

The dataset was expanded from 12 to 58 cases to meet the Clinical Evaluation Framework requirements. The expansion strategy and implementation details are documented in `docs/studies/study_a/bias_expansion_plan.md`.

### Related Documentation

- **Bias Dimensions**: See `BIAS_DIMENSIONS.md` for detailed breakdown of cases and bias patterns
- **Expansion Plan**: See `docs/studies/study_a/bias_expansion_plan.md` for expansion strategy
- **Commands**: See `docs/studies/study_a/study_a_bias.md` for generation and evaluation commands


