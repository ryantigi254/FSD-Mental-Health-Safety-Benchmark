# Setup Summary: Study A Bias + Split Verification

## ✅ Completed Tasks

### 1. Study A Bias Evaluation Setup

**Created Files:**
- `hf-local-scripts/run_study_a_bias_generate_only.py` - Bias generation script
- `scripts/study_a/metrics/calculate_bias.py` - Bias metric calculation
- `docs/study_a_bias_setup.md` - Documentation

**Integration:**
- Updated `scripts/study_a/metrics/calculate_metrics.py` to merge bias metrics
- Bias metrics now included in `all_models_metrics.json`

**Workflow:**
1. Generate: `python hf-local-scripts/run_study_a_bias_generate_only.py --model-id <model>`
2. Calculate: `python scripts/study_a/metrics/calculate_bias.py`
3. Main metrics script automatically merges bias results

### 2. Split Verification

**Study B:**
- ✅ 276 samples (IDs: b_001 to b_276)
- ✅ Age in metadata (v2 structure)
- ✅ Persona IDs present
- ✅ 10 multi-turn cases
- ✅ Deterministic (uses random.seed(42))

**Study C:**
- ✅ 30 cases (IDs: c_001 to c_030)
- ✅ Persona IDs present
- ✅ Deterministic structure

**Generation Counts:**
- Study B: 276 × 2 = 552 single-turn + 50 multi-turn = **602 total** (2 over 600 target)
- Study C: 30 × 10 × 2 = **600 total** (exact target)

### 3. Persona Registry

**Available Personas:** 25 total in `persona_registry_v2.json`
- Currently using: 10 personas (aisha, jamal, eleni, maya, sam, leo, priya, noor, tomas, kai)
- Additional personas available if needed: chloe, danny, derek, diana, fiona, grace, jordan, marcus, margaret, olivia, rashid, rowan, victor, zara, zoe

**Note:** Current 10 personas are sufficient for Studies B and C. Additional personas can be added if you need more diversity or larger sample sizes.

## File Structure

```
Uni-setup/
├── hf-local-scripts/
│   ├── run_study_a_bias_generate_only.py  [NEW]
│   ├── run_study_b_generate_only.py
│   └── run_study_c_generate_only.py
├── scripts/
│   └── study_a/
│       └── metrics/
│           ├── calculate_metrics.py  [UPDATED - merges bias]
│           └── calculate_bias.py     [NEW]
├── data/
│   ├── openr1_psy_splits/
│   │   ├── study_b_test.json  [276 samples, v2 structure]
│   │   └── study_c_test.json  [30 cases]
│   └── adversarial_bias/
│       └── biased_vignettes.json
└── docs/
    └── study_a_bias_setup.md  [NEW]
```

## Next Steps

1. **Run bias generation** for your models:
   ```bash
   python hf-local-scripts/run_study_a_bias_generate_only.py --model-id qwq-lmstudio
   ```

2. **Calculate bias metrics**:
   ```bash
   python scripts/study_a/metrics/calculate_bias.py
   ```

3. **Calculate all Study A metrics** (includes bias):
   ```bash
   python scripts/study_a/metrics/calculate_metrics.py
   ```

## Verification

All splits are:
- ✅ Deterministic (reproducible with same seed)
- ✅ Have sequential IDs
- ✅ Include v2 structure (age in metadata for Study B)
- ✅ Have persona IDs for stratified analysis

