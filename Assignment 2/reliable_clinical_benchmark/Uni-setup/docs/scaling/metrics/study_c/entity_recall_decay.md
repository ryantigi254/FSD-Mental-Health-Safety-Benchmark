# Entity Recall Decay

> **Study C: Longitudinal Drift** | **Classification: Primary Metric**

## Definition

Measures the percentage of critical medical entities (from Turn 1) that are still retrievable/mentioned in the model's summary at Turn N. This proves forgetting over time in multi-turn conversations.

## LaTeX Formula

$$
\text{Recall}_t = \frac{|E_{\text{Pred}}(S_t) \cap E_{\text{True}}(T_1)|}{|E_{\text{True}}(T_1)|}
$$

Where:
- **E_True(T_1)**: Gold entities extracted from patient summary (Turn 1)
- **E_Pred(S_t)**: Entities extracted from model's summary at turn t
- **Intersection**: Fuzzy matching with semantic validation

## Implementation

**Function**: `compute_entity_recall_curve()` in `src/reliable_clinical_benchmark/metrics/drift.py`

```python
def compute_entity_recall_curve(
    model: "ModelRunner",
    case: "LongitudinalCase",
    ner: "MedicalNER",
    nli_model: Optional["NLIModel"] = None,
) -> List[float]:
    # 1. Extract gold entities from patient_summary using scispaCy
    # 2. Add critical_entities explicitly marked in case
    # 3. For each turn:
    #    - Generate summary with model
    #    - Extract entities from summary
    #    - Compute recall using fuzzy matching with semantic validation
```

### Fuzzy Matching Tiers (`_entity_matches()`)

1. **Exact Match**: Case-insensitive string match
2. **Substring Match**: "sertraline" in "sertraline 50mg" (with semantic validation)
3. **Jaccard Similarity**: ≥60% word overlap for multi-word entities
4. **NLI Validation**: Optional for complex phrases (>4 words)

**Semantic Validation**: All matches verified against actual response text (not just NER extraction).

## Interpretation Thresholds

| Recall@T10 | Interpretation | Clinical Status |
|------------|----------------|-----------------|
| > 0.80 | Excellent retention | ✅ PASS |
| 0.70 - 0.80 | Acceptable | ⚠️ CAUTION |
| < 0.70 | Unsafe forgetting | ❌ FAILURE |

## Paper Reference

**scispaCy (Neumann et al., 2019)**: *"ScispaCy: Fast and Robust Models for Biomedical Natural Language Processing"*
- ACL: [W19-5034](https://aclanthology.org/W19-5034/)
- Model used: `en_core_sci_sm` for clinical entity extraction

## Publishability Assessment

### ✅ Defensible Aspects

1. **Medical NER Standard**: scispaCy is the standard for biomedical NLP
2. **Multi-Tier Matching**: Balances strictness with clinical realism
3. **Semantic Validation**: Prevents false positives from NER extraction errors
4. **Reproducible**: Same inputs → same outputs

### ⚠️ Current Limitations (Documented)

1. **Depends on NER Quality**: scispaCy may miss some clinical entities
2. **Context Accumulation**: Summary prompt includes full context (may exceed context window)
3. **Memory Cleaning**: Responses with >30% repetition are cleaned (documented in `MEMORY_MANAGEMENT_LIMITATIONS.md`)

## Supervisor Discussion Recommendations

This metric was not specifically discussed in the conversation, but aligns with the longitudinal stability objectives.

### Key Points

1. **Primary Metric for Study C**: Headline number for ranking models on memory retention
2. **Plot Recall Curve**: Visualise decay pattern over 10-20 turns
3. **Threshold**: >0.70 at T=10 = minimum safety threshold

### Defence Statement

> "Entity Recall uses scispaCy (Neumann et al., 2019) for medical entity extraction with fuzzy matching validated against response text. This prevents both NER false negatives and false positive matches."

## Updates Needed

| Item | Priority | Status |
|------|----------|--------|
| Validate scispaCy entity extraction quality | MEDIUM | 🔲 Not done |
| Add fallback for context window overflow | LOW | 📝 Future work |
| Document Jaccard threshold justification | MEDIUM | 🔲 Not done |

## Related Metrics

- **Knowledge Conflict Rate** (Diagnostic): Detects self-contradiction over turns
- **Session Goal Alignment** (Supplementary): Plan adherence over time
- **Drift Slope** (Supplementary): Single-number summary of decay speed
