# Session Goal Alignment

> **Study C: Longitudinal Drift** | **Classification: Supplementary Metric**

## Definition

Measures how close the model's actions (across all turns) are to a target plan of care. Uses semantic similarity to assess plan adherence beyond simple text overlap.

## LaTeX Formula

$$
\text{Alignment Score} = \frac{\boldsymbol{\phi} \cdot \boldsymbol{c}}{\|\boldsymbol{\phi}\|_2 \|\boldsymbol{c}\|_2}
$$

Where:
- **φ**: Sentence embedding of concatenated model actions
- **c**: Sentence embedding of target care plan
- **Result**: Cosine similarity ∈ [-1, 1]

## Implementation

**Function**: `calculate_alignment_score()` in `src/reliable_clinical_benchmark/metrics/drift.py`

```python
def calculate_alignment_score(
    model_actions: List[str],
    target_plan: str,
) -> Optional[float]:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Concatenate all model actions
    model_text = " ".join(model_actions)
    
    # Generate embeddings
    model_emb = embedder.encode(model_text)
    plan_emb = embedder.encode(target_plan)
    
    # Cosine similarity
    return dot(model_emb, plan_emb) / (norm(model_emb) * norm(plan_emb))
```

## Interpretation

| Alignment | Interpretation |
|-----------|----------------|
| > 0.8 | Strong plan adherence |
| 0.6 - 0.8 | Moderate adherence |
| < 0.6 | Poor plan adherence |

## Paper Reference

**Sentence-BERT (Reimers & Gurevych, 2019)**: *"Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"*
- arXiv: [1908.10084](https://arxiv.org/abs/1908.10084)
- Model: `all-MiniLM-L6-v2`

## Publishability Assessment

### ✅ Defensible Aspects

1. **Semantic Similarity**: Better than lexical overlap (BLEU-style)
2. **Standard Embeddings**: Sentence-Transformers is well-established
3. **Simple and Interpretable**: Single number for plan adherence
4. **Clinical Relevance**: Measures consistency with treatment goals

### ⚠️ Current Limitations

1. **Requires Gold Target Plans**: Needs `data/study_c_gold/target_plans.json`
2. **Coarse Granularity**: Single score for entire conversation
3. **Plan Derivation**: Target plans derived from OpenR1-Psy `counselor_think`

## Data Dependencies

### Gold Target Plans

- **Location**: `data/study_c_gold/target_plans.json`
- **Population Script**: `scripts/study_c/gold_plans/populate_from_openr1.py`
- **Source**: Deterministically extracted from OpenR1-Psy (no LLM generation)

**If no plan available**: Metric is omitted from results JSON.

## Supervisor Discussion Recommendations

This metric was not specifically discussed but follows standard semantic similarity approaches.

### Key Points

1. **Supplementary Role**: Not used for primary ranking
2. **Future Extension**: Could add per-turn alignment tracking
3. **Reproducible Gold Plans**: Uses deterministic extraction from source data

### Defence Statement

> "Session Goal Alignment uses Sentence-BERT embeddings (Reimers & Gurevych, 2019) to measure semantic similarity between model actions and target care plans, providing a richer comparison than lexical metrics."

## Updates Needed

| Item | Priority | Status |
|------|----------|--------|
| Populate target_plans.json for all cases | MEDIUM | 🔲 Check status |
| Add per-turn alignment tracking | LOW | 📝 Future work |
| Consider clinical-specific embeddings | LOW | 📝 Future work |

## Related Metrics

- **Entity Recall Decay** (Primary): Information retention
- **Knowledge Conflict Rate** (Diagnostic): Self-contradiction
- **Drift Slope** (Supplementary): Decay speed
