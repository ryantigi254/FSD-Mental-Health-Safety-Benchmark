# Drift Slope

> **Study C: Longitudinal Drift** | **Classification: Supplementary Metric**

## Definition

Quantifies the speed of entity forgetting by fitting a linear regression to (turn_number, recall) pairs. Provides a single-number summary of drift speed for comparison across models. The default input is the critical recall curve; the extended curve can be used as a diagnostic variant.

**Note**: Drift Slope is a summary statistic over the recall curve, not an independent verification signal.

## LaTeX Formula

$$
\text{Recall}_t = \alpha + \beta \times t
$$

Where:
- **β (Drift Slope)**: Rate of recall change per turn
- **α**: Initial recall (intercept)
- **t**: Turn number

## Implementation

**Function**: `compute_drift_slope()` in `src/reliable_clinical_benchmark/metrics/drift.py`

```python
def compute_drift_slope(recall_curve: List[float]) -> float:
    if len(recall_curve) < 2:
        return 0.0
    
    turns = np.arange(1, len(recall_curve) + 1)
    recalls = np.array(recall_curve)
    
    # Simple linear regression
    slope = np.polyfit(turns, recalls, 1)[0]
    return float(slope)
```

## Interpretation

| Slope | Interpretation |
|-------|----------------|
| ≈ 0.0 | No decay (stable retention) |
| -0.01 to -0.02 | Mild decay (1-2% per turn) |
| -0.03 to -0.05 | Moderate decay |
| < -0.05 | Severe decay (>5% per turn) |

**Example**: Slope = -0.02 means recall decreases by 2% per turn on average.

## Paper Reference

**Ordinary Least Squares (OLS) Linear Regression**: Standard statistical method for fitting linear models
- Implementation: `numpy.polyfit()` with degree=1 for slope estimation
- Reference: Standard econometric/statistical methodology (Wooldridge, 2012; Greene, 2018)

**Lost in the Middle (Liu et al., 2024)**: How language models suffer from position bias in long contexts
- Relevant for understanding non-linear recall patterns in multi-turn dialogue
- arXiv:2307.03172

## Publishability Assessment

### ✅ Defensible Aspects

1. **Standard Statistics**: Simple linear regression (easily reproducible)
2. **Single Summary Number**: Easy to compare across models
3. **Intuitive Interpretation**: "2% decay per turn" is clinician-friendly
4. **Lightweight**: No additional dependencies beyond numpy

### ⚠️ Current Limitations

1. **Assumes Linear Decay**: Real decay may be non-linear (exponential, stepped)
2. **Sensitive to Noise**: Individual turn outliers can skew slope
3. **Not Currently Stored**: Function exists but not saved to pipeline results

## Supervisor Discussion Recommendations

These recommendations capture how Drift Slope should be reported so it remains interpretable (as a summary statistic, not a standalone signal).

### Recommended Enhancement: Critical-Curve Default + Paired Reporting

- Compute drift slope on `average_recall_curve_critical` (headline).
- Optionally compute a diagnostic slope on `average_recall_curve_extended`.
- Always report slope together with the underlying recall curve (to avoid masking non-linear decay).

### Key Points

1. **Currently Available**: Function works, just not stored in results JSON
2. **Analysis Notebooks**: Can be computed from saved recall curves
3. **Could Add to Pipeline**: Simple enhancement if needed

## Updates Needed

| Item | Priority | Status |
|------|----------|--------|
| Add to pipeline results JSON | LOW | 📝 Can add if needed |
| Consider non-linear decay models | LOW | 📝 Future work |
| Add confidence interval on slope | LOW | 📝 Future work |

## Usage Example

```python
# In analysis notebook:
from reliable_clinical_benchmark.metrics.drift import compute_drift_slope

# Load recall curve from results (headline = critical)
recall_curve = results["average_recall_curve_critical"]

# Optional diagnostic variant
diagnostic_curve = results.get("average_recall_curve_extended")

# Compute slope
slope = compute_drift_slope(recall_curve)
print(f"Decay rate: {slope:.3f} per turn")
```

## Related Metrics

- **Entity Recall Decay** (Primary): Raw recall values per turn
- **Knowledge Conflict Rate** (Diagnostic): Explains cause of decay
- **Session Goal Alignment** (Supplementary): Plan adherence
