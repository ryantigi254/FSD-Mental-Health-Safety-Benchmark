# Entity Recall Decay

> **Study C: Longitudinal Drift** | **Classification: Primary Metric**

## Definition

Measures the percentage of critical medical entities (from frozen case metadata) that are still retrievable/mentioned in the model's summary at Turn N. This proves forgetting over time in multi-turn conversations. An extended diagnostic variant adds filtered scispaCy entities from the patient summary.

## LaTeX Formula

$$
\text{Recall}_t = \frac{|E_{\text{Pred}}(S_t) \cap E_{\text{True}}(T_1)|}{|E_{\text{True}}(T_1)|}
$$

Where:
- **E_True(T_1)**: Critical entities from frozen case metadata (headline metric)
- **E_True^+**: Critical entities ∪ filtered scispaCy entities from patient summary (diagnostic)
- **E_Pred(S_t)**: Entities extracted from model's summary at turn t
- **Intersection**: Fuzzy matching with semantic validation and negation checks

## Implementation

**Function**: `compute_entity_recall_metrics()` in `src/reliable_clinical_benchmark/metrics/drift.py` (A+B curves with precision + hallucinated rate). `compute_entity_recall_curve()` remains as a headline alias (critical curve).

```python
def compute_entity_recall_metrics(
    model: "ModelRunner",
    case: "LongitudinalCase",
    ner: "MedicalNER",
    nli_model: Optional["NLIModel"] = None,
) -> EntityRecallMetrics:
    # 1. Gold (headline): critical_entities
    # 2. Gold (diagnostic): critical_entities ∪ filtered NER(patient_summary)
    # 3. Per turn: generate summary → extract entities
    # 4. Compute recall, precision, hallucinated rate (negation-aware)
```

### Fuzzy Matching Tiers (`_entity_matches()`)

1. **Exact Match**: Case-insensitive string match
2. **Substring Match**: "sertraline" in "sertraline 50mg" (with semantic validation)
3. **Jaccard Similarity**: ≥60% word overlap for multi-word entities
4. **NLI Validation**: Optional for complex phrases (>4 words)

**Semantic Validation**: All matches verified against actual response text (not just NER extraction). Negated mentions are excluded from recall/precision.

**Negation Handling**: Negated mentions (e.g., "no penicillin allergy") are excluded using a short negation window.

## Outputs

- **Headline**: `recall_curve_critical` (gold = `critical_entities` only)
- **Diagnostic**: `recall_curve_extended` (critical + filtered NER from `patient_summary`)
- **Quality controls**: precision, F1, and hallucinated-entity rate curves for both critical and extended sets

## Interpretation Thresholds

| Recall@T10 | Interpretation | Clinical Status |
|------------|----------------|-----------------|
| > 0.80 | Excellent retention | ✅ PASS |
| 0.70 - 0.80 | Acceptable | ⚠️ CAUTION |
| < 0.70 | Unsafe forgetting | ❌ FAILURE |

## What the decay curve reflects

1. **Context-window fidelity**: whether early-turn facts remain accessible as the prompt grows.
2. **Salience/compression choices**: whether the model preserves clinically critical facts when summarising.
3. **Extractor noise**: scispaCy + fuzzy matching limits (tracked via manual audit).

## Closest conceptual relatives (evaluation framing)

- **Dialogue State Tracking evaluation**: slot precision/recall/F1 and joint goal accuracy in DST challenge evaluations (Williams et al., 2016) — https://doi.org/10.5087/dad.2016.301
- **Long-term dialogue memory evaluation**: F1/overlap versus human references (Recursively Summarising, 2023) — https://arxiv.org/abs/2308.15022
- **Very long-term memory benchmarks**: QA F1 / retrieval accuracy and fact-based summarisation scoring (LoCoMo, 2024) — https://arxiv.org/abs/2402.17753
- **Atomic-fact factuality framing**: fact decomposition for fine-grained summary correctness (FActScore, 2023) — https://arxiv.org/abs/2305.14251

## Known failure modes

1. **Polarity/negation errors**: entity mentioned but negated (e.g., “no penicillin allergy”) can still evade simple matching without explicit checks.
2. **Synonym/dose normalisation gaps**: “sertraline 50 mg” vs “sertraline 50mg”, “MDD” vs “major depressive disorder”.
3. **Gold set inflation**: generic scispaCy spans dilute recall when `E_true^+` is broad.
4. **Summary style variance**: verbose summaries inflate recall, terse summaries depress it.

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
2. **Requires scispaCy**: evaluation exits if `en_core_sci_sm` is unavailable
3. **Context Accumulation**: Summary prompt includes full context (may exceed context window)
4. **Memory Cleaning**: Responses with >30% repetition are cleaned (documented in `MEMORY_MANAGEMENT_LIMITATIONS.md`)

## Supervisor Discussion Recommendations

From the metric refinement review:

### Recommendation 1: Freeze the headline gold set to curated `critical_entities` (A1)

Problem: using NER spans from Turn 1 as the headline gold set makes the metric noisy (generic spans inflate/deflate recall and turn the metric into a NER-quality proxy).

Recommendation:
- **Headline metric** uses **only** the curated, frozen `critical_entities` list.
- NER-derived entities from the patient summary are kept as a **diagnostic extended set** (`E_True^+`) rather than the headline.

Implemented as:

```python
gold_entities_critical = {ent.lower() for ent in case.critical_entities}
gold_entities_extended = gold_entities_critical | _filter_entities(
    ner.extract_clinical_entities(case.patient_summary)
)
```

### Recommendation 2: Negation-aware exclusion window (B1)

Problem: polarity errors (e.g., "no penicillin allergy") count as recalled unless explicitly blocked.

Recommendation: apply a lightweight token-window negation rule that treats an entity as negated when a negation cue occurs within a short window immediately before the entity mention.

Implemented as (from `metrics/drift.py`):

```python
_NEGATION_TERMS = {"no", "not", "denies", "without", "never", "none", "denied"}
_NEGATION_WINDOW_TOKENS = 5

def _is_negated(entity: str, text: str) -> bool:
    entity_tokens = _tokenise_text(entity)
    text_tokens = _tokenise_text(text)
    for idx in range(len(text_tokens) - len(entity_tokens) + 1):
        if text_tokens[idx : idx + len(entity_tokens)] == entity_tokens:
            window_tokens = text_tokens[max(0, idx - _NEGATION_WINDOW_TOKENS) : idx]
            if any(token in _NEGATION_TERMS for token in window_tokens):
                return True
    return False
```

Effect: negated gold entities do **not** contribute to recall, and negated predicted entities do **not** contribute to true positives.

### Recommendation 3: Add precision/F1 and hallucinated-entity rate curves (C)

Problem: recall alone is easy to game (verbose summaries can mention lots of entities, including incorrect ones).

Recommendation: report complementary per-turn diagnostics:
- **Precision**: of predicted entities, how many match the gold set.
- **F1**: harmonic mean of precision/recall.
- **Hallucinated-entity rate**: fraction of extracted entities that do not match the gold set.

Implemented as (from `_compute_entity_set_metrics()`):

```python
recall = matched_gold_count / len(gold_entities) if gold_entities else 0.0
precision = matched_predicted_count / len(predicted_entities) if predicted_entities else 0.0
f1_score = (2.0 * precision * recall) / (precision + recall) if (precision > 0.0 or recall > 0.0) else 0.0
hallucinated_rate = (
    (len(predicted_entities) - matched_predicted_count) / len(predicted_entities)
    if predicted_entities
    else 0.0
)
```

### Supervisor-facing reporting (what to show)

1. **Recall@T10 (critical curve)** as the primary stability gate.
2. **Recall curve (critical + extended)** to show decay shape.
3. **Precision/F1/hallucinated curves** to show whether recall is supported by non-hallucinatory entity mentions.

### Citations

- Neumann et al. (2019) scispaCy: https://aclanthology.org/W19-5034/
- Chapman et al. (2001) NegEx: https://pubmed.ncbi.nlm.nih.gov/11376540/
- Harkema et al. (2009) ConText: https://pubmed.ncbi.nlm.nih.gov/19834171/

## Updates Needed

| Item | Priority | Status |
|------|----------|--------|
| Validate scispaCy entity extraction quality | MEDIUM | 🔲 Not done |
| Add fallback for context window overflow | LOW | 📝 Future work |
| Document Jaccard threshold justification | MEDIUM | 🔲 Not done |
| Report precision/hallucinated curves alongside recall | MEDIUM | ✅ Implemented |

## Related Metrics

- **Knowledge Conflict Rate** (Diagnostic): Detects self-contradiction over turns
- **Session Goal Alignment** (Supplementary): Plan adherence over time
- **Drift Slope** (Supplementary): Single-number summary of decay speed
