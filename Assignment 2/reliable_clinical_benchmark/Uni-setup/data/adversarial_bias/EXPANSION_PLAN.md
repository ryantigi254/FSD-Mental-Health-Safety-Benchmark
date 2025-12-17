# Bias Cases Expansion Plan: 12 → 58 Cases

## Current Status

- **Current cases**: 12
- **Required cases**: 58 (50 base + 8 buffer, per tex file Section 6.3)
- **Gap**: Need 46 additional cases

## Structure Confirmation

✅ **One prompt per case** (not multiple prompts)
- Each case in `biased_vignettes.json` = one generation
- The tex file pseudocode confirms: `model.generate_with_reasoning(case.prompt)` - single prompt per case

✅ **Personas are separate**
- Personas in `Assignment 2/docs/personas/` are for **Study B (Sycophancy)**, not Study A (Bias)
- Bias cases use simple vignettes, not persona-based conversations

## Expansion Strategy

### Current Dimensions (12 cases)
1. **Race** (4 cases): black/white pairs
2. **Gender** (2 cases): female/male pairs
3. **Age** (2 cases): elderly/young pairs
4. **Housing** (2 cases): homeless/stably housed pairs
5. **Substance Use** (2 cases): heroin use/no history pairs

### Target Distribution (58 cases)

| Dimension | Current | Target | New Cases Needed |
|-----------|---------|--------|------------------|
| Race | 4 | 16 | +12 |
| Gender | 2 | 10 | +8 |
| Age | 2 | 8 | +6 |
| Housing | 2 | 6 | +4 |
| Substance Use | 2 | 6 | +4 |
| **Obesity** (NEW) | 0 | 6 | +6 |
| **Socioeconomic** (NEW) | 0 | 6 | +6 |
| **Total** | **12** | **58** | **+46** |

## New Cases to Generate

### Race (12 new cases)
- Asian vs White (e.g., pain management "stoicism" bias)
- Hispanic vs White (language barrier bias)
- Middle Eastern vs White (cultural bias)
- Additional black/white pairs (different clinical scenarios)

### Gender (8 new cases)
- Male vs Female cardiac symptoms (women under-diagnosed)
- Female vs Male autism/ADHD (under-diagnosis in women)
- Male vs Female eating disorders (men under-diagnosed)
- Additional pairs for different conditions

### Age (6 new cases)
- Elderly: UTI presenting as confusion vs Dementia
- Young: Stroke vs Drug use assumption
- Middle-aged vs Elderly: different treatment assumptions
- Additional age-based scenarios

### Housing (4 new cases)
- Additional homeless vs stably housed pairs
- Different clinical presentations

### Substance Use (4 new cases)
- Alcohol history: Liver failure vs Pancreatitis
- Cannabis use: CHS vs Gastroenteritis
- Additional substance-related biases

### Obesity (6 new cases - NEW DIMENSION)
- Obese vs Thin: Shortness of breath → "Lose weight" vs "Pulmonary Embolism"
- BMI-based diagnostic assumptions
- Weight stigma in clinical decision-making

### Socioeconomic (6 new cases - NEW DIMENSION)
- Low Income: Non-compliance vs Cost issues
- Manual Laborer: Back pain dismissal vs proper diagnosis
- Insurance status bias
- Education level assumptions

## Implementation Notes

1. **Keep existing 12 cases** - they're already validated
2. **Add 46 new cases** following the same JSON schema
3. **Maintain paired structure** where applicable (e.g., black/white, male/female)
4. **Each case = one prompt** - no multiple prompts per case
5. **Personas remain separate** - do not use persona data for bias cases

## JSON Schema (Unchanged)

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

## Next Steps

1. Generate 46 new cases following the schema above
2. Add to `biased_vignettes.json` (IDs: bias_013 through bias_058)
3. Update `BIAS_DIMENSIONS.md` to reflect new dimensions
4. Re-run bias generations for all models (will now generate 58 lines per model)

