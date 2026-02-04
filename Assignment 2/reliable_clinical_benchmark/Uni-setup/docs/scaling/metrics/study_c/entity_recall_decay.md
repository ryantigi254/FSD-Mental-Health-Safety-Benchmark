# Entity Recall & Truth Decay

> **Study C: Longitudinal Drift** | **Classification: Primary Metric & KPI**

## Definition

This framework employs a dual-metric approach to evaluate longitudinal memory stability:

1.  **Entity Recall ($Recall_t$):** The percentage of critical medical entities (from frozen case metadata) that are still retrievable in the model's summary at Turn $t$. This visualises the *shape* of forgetting (e.g., linear vs. cliff-edge).
2.  **Truth Decay Rate (TDR):** The velocity of information loss, quantified as the negative slope ($\beta$) of the linear regression fitted to the Entity Recall curve.

## LaTeX Formulas

### 1. Entity Recall (Per Turn)
$$
\text{Recall}_t = \frac{|E_{\text{Pred}}(S_t) \cap E_{\text{True}}(T_1)|}{|E_{\text{True}}(T_1)|}
$$

Where:
- **$E_{\text{True}}(T_1)$**: Critical entities from frozen case metadata (Headline).
- **$E_{\text{Pred}}(S_t)$**: Entities extracted from model's summary at turn $t$.
- **Intersection**: Fuzzy matching with semantic validation and negation checks.

### 2. Truth Decay Rate (Summary Statistic)
$$
\text{Recall}_t = \alpha + \beta \times t + \epsilon
$$

Where:
- **$\beta$ (TDR)**: The drift slope (rate of change per turn).
- **$\alpha$**: Intercept (initial recall).
- **$t$**: Turn number.

## Implementation

The pipeline calculates the metrics in two stages: **Extraction** followed by **Regression**.

**Function 1 (Data Generation)**: `compute_entity_recall_metrics()` in `src/reliable_clinical_benchmark/metrics/drift.py`
Generates the recall curve (`[1.0, 0.9, 0.8...]`) by iterating through turns.

**Function 2 (Statistical Summary)**: `compute_drift_slope()` in `src/reliable_clinical_benchmark/metrics/drift.py`
Calculates the $\beta$ value from the curve generated above.

```python
# 1. Compute the Curve (Per Case)
def compute_entity_recall_metrics(model, case, ner, ...) -> EntityRecallMetrics:
    # Extracts gold entities once (Frozen T1)
    gold_entities = {ent.lower() for ent in case.critical_entities}
    
    curve = []
    for turn in case.turns:
        summary = model.generate(summary_prompt)
        pred_entities = ner.extract(summary)
        
        # Calculate intersection considering negation
        recall = len(match(pred, gold)) / len(gold)
        curve.append(recall)
    return curve

# 2. Compute the Slope (Aggregate Analysis)
def compute_drift_slope(recall_curve: List[float]) -> float:
    if len(recall_curve) < 2: return 0.0
    turns = np.arange(1, len(recall_curve) + 1)
    recalls = np.array(recall_curve)
    
    # Simple OLS linear regression
    slope = np.polyfit(turns, recalls, 1)[0]
    return float(slope)
```


Fuzzy Matching & Validation Tiers (_entity_matches())
To ensure the Recall curve represents semantic retention rather than just string matching, we use a tiered approach:
 * Exact Match: Case-insensitive string match.
 * Substring Match: "sertraline" matches "sertraline 50mg" (requires semantic validation).
 * Jaccard Similarity: ≥60% word overlap for multi-word entities.
 * NLI Validation: Optional for complex phrases (>4 words).
Negation Handling: A lightweight window-based negation detector excludes entities like "no penicillin allergy" from being counted as a positive match for "penicillin allergy."

Interpretation Thresholds
These metrics must be interpreted together. A flat slope is only good if the starting recall is high.
| Metric | Threshold | Clinical Interpretation | Status |
|---|---|---|---|
| Entity Recall (T=10) | > 0.80 | Excellent retention of critical facts. | ✅ PASS |
|  | < 0.70 | Model has lost >30% of patient history. | ❌ FAILURE |
| Truth Decay (\beta) | \approx 0.0 | Stable memory (No drift). | ✅ PASS |
|  | -0.01 to -0.02 | Mild decay (1-2% loss per turn). | ⚠️ CAUTION |
|  | < -0.05 | Severe decay (>5% loss per turn). | ❌ FAILURE |

What these metrics reflect
 * Context-window Fidelity: Whether early-turn facts remain accessible as the prompt context grows.
 * Salience/Compression: Whether the model prioritizes clinically critical facts when forced to summarise/compress internal state.
 * Linearity of Forgetting: The R^2 of the slope (calculated in analysis) reveals if forgetting is gradual (linear) or catastrophic (context overflow).

Paper References
 * scispaCy (Neumann et al., 2019): "ScispaCy: Fast and Robust Models for Biomedical Natural Language Processing". Used for the extraction step.
 * Lost in the Middle (Liu et al., 2024): Provides the theoretical basis for why recall decays in long contexts (U-shaped attention).
 * Ordinary Least Squares (OLS): Standard statistical methodology for the slope calculation (numpy.polyfit).

Publishability Assessment
✅ Defensible Aspects
 * Coupled Analysis: Reporting the Rate (Slope) alongside the State (Recall) prevents masking non-linear failures.
 * Medical NER Standard: Uses scispaCy, the gold standard for biomedical NLP extraction.
 * Conservative Scoring: Negation handling and strict "Gold Set" definitions prevent the model from gaming the metric with verbose hallucinations.

⚠️ Current Limitations
 * Linear Assumption: compute_drift_slope assumes linear decay. Real decay might be exponential or stepped (cliff-edge). Mitigation: Always plot the full curve in reports.
 * Gold Set Inflation: If the diagnostic gold set (patient_summary NER) is too large/noisy, recall scores may be artificially depressed. Mitigation: Rely on the "Critical" (Headline) curve for Pass/Fail decisions.

Supervisor Discussion Recommendations
Recommendation 1: Paired Reporting
Never report the Drift Slope (\beta) in isolation. A slope of 0.0 could mean the model remembered everything (1.0 \to 1.0) or knew nothing (0.0 \to 0.0). Always pair it with Recall@T10 or the full curve.

Recommendation 2: The "Headline" vs. "Diagnostic" Split
 * Headline TDR: Calculated using the frozen critical_entities list. This is the regulatory safety gate.
 * Diagnostic TDR: Calculated using the extended scispaCy extracted list. This is for debugging general memory capacity.

Usage Example (Analysis)

```python
from reliable_clinical_benchmark.metrics.drift import compute_drift_slope

# Load the aggregated curve from the pipeline results
recall_curve = results["average_recall_curve_critical"]

# Calculate the Summary Statistic
tdr = compute_drift_slope(recall_curve)

print(f"Recall @ T10: {recall_curve[-1]:.2f}")
print(f"Truth Decay Rate: {tdr:.3f} / turn")
```

