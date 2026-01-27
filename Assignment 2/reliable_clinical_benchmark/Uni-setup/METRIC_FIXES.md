# Critical Metric Fixes - Supervisor Feedback Response

## Issues Identified

### 1. Study A (Faithfulness): Negative Gap & Zero Step-F1
**Problem**: Models output reasoning in `<think>...</think>` tags, but extraction only looked for `REASONING:` markers.

**Fix Applied**: Updated `extract_reasoning_steps()` in `metrics/faithfulness.py` to:
- First check for `<think>...</think>` XML tags (DeepSeek-R1, QwQ style)
- Fall back to `REASONING:...DIAGNOSIS:` markers
- Final fallback: everything before `DIAGNOSIS:` marker

**Impact**: Step-F1 scores should now be non-zero for models using XML thinking tags.

### 2. Study B (Sycophancy): 0.0 Flip Rate for All Models
**Problem**: Agreement detection was too strict, missing nuanced agreements like "You raise a valid point" or "That's a possibility".

**Fix Applied**: Enhanced `is_agreement()` in `metrics/sycophancy.py` to:
- Add nuanced agreement phrases ("you raise a valid point", "that makes sense", etc.)
- Check for contradiction markers after agreement phrases ("I agree, but...")
- Improve context-aware detection when bias label is present

**Impact**: Flip rates should now be non-zero, reflecting actual model sycophancy.

## Next Steps

### Immediate Action Required (15-30 minutes)

1. **Re-run metric calculations** (NOT model generations - those are fine):
   ```bash
   # From Uni-setup directory
   python scripts/studies/study_a/metrics/calculate_metrics.py
   python scripts/studies/study_b/metrics/calculate_metrics.py
   python scripts/studies/study_c/metrics/calculate_metrics.py
   ```

2. **Verify results**:
   - Check that Step-F1 scores are now > 0.0
   - Check that Flip Rates are now > 0.0 (but still reasonable)
   - Verify Faithfulness Gaps are no longer all negative

3. **Update notebooks** with new results if needed.

### Bootstrap Confidence Intervals

Bootstrap CI calculation already exists in `utils/stats.py` and is partially used. Ensure all metrics report CIs in the final results JSON files.

## Files Modified

- `src/reliable_clinical_benchmark/metrics/faithfulness.py`: Fixed reasoning extraction
- `src/reliable_clinical_benchmark/metrics/sycophancy.py`: Improved agreement detection

## Expected Outcome

After re-running metrics:
- **Step-F1**: Should increase from ~0.01 to realistic values (0.2-0.6 range expected)
- **Flip Rate**: Should increase from 0.0 to non-zero (0.05-0.30 range expected for most models)
- **Faithfulness Gap**: May still be negative for some models, but should be less extreme

These fixes address the supervisor's concerns about measurement errors vs. actual model failures.
