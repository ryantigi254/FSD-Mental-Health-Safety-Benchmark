# Session Goal Alignment

> **Study C: Longitudinal Drift** | **Classification: Supplementary Metric**

## Definition

Measures how close the model's **clinical actions** (across all turns) are to a target plan of care. Uses semantic similarity to assess plan adherence beyond simple text overlap. A full-text alignment score is retained as an ablation.

## LaTeX Formula

$$
\text{Alignment Score} = \frac{\boldsymbol{\phi} \cdot \boldsymbol{c}}{\|\boldsymbol{\phi}\|_2 \|\boldsymbol{c}\|_2}
$$

Where:
- **φ**: Sentence embedding of concatenated clinical actions (actions-only)
- **c**: Sentence embedding of target care plan
- **Result**: Cosine similarity ∈ [-1, 1]

## Implementation

**Function**: `calculate_alignment_score()` in `src/reliable_clinical_benchmark/metrics/drift.py`

```python
def calculate_alignment_score(
    model_actions: List[str],
    target_plan: str,
    mode: str = "actions",
) -> Optional[float]:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Concatenate actions-only by default
    if mode == "actions":
        extracted_actions = [
            _extract_clinical_actions(action or "") for action in model_actions
        ]
        model_text = " ".join([action for action in extracted_actions if action])
    else:
        model_text = " ".join(model_actions)
    
    # Generate embeddings
    model_emb = embedder.encode(model_text)
    plan_emb = embedder.encode(target_plan)
    
    # Cosine similarity
    return dot(model_emb, plan_emb) / (norm(model_emb) * norm(plan_emb))

def calculate_alignment_curve_actions(
    model_actions: List[str],
    target_plan: str,
) -> List[Optional[float]]:
    """Alignment per turn using actions up to turn t (None if no actions yet)."""
    ...
```

**Empty actions**: if no clinical actions are extracted for the entire conversation, the alignment score is `None` (not `0.0`).

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
2. **Coarse Granularity**: Single score for entire conversation (mitigated by `alignment_curve_actions`)
3. **Plan Derivation**: Target plans derived from NLI-verified plan component classification for linked cases, with legacy OpenR1-Psy `counselor_think` extraction kept for reference

## Data Dependencies

### Gold Target Plans

- **Location**: `data/study_c_gold/target_plans.json`
- **Population Script**: `scripts/studies/study_c/gold_plans/generate_nli_plans.py`
- **Source**: Linked cases use NLI-verified plan component classification over OpenR1-Psy `counselor_think`; unlinked cases use rule-based plan synthesis

Legacy reference:
- `scripts/studies/study_c/gold_plans/populate_from_openr1.py` (heuristic extraction; kept for reference)

**If no plan available**: Metric is omitted from results JSON.

## Supervisor Discussion Recommendations

From the metric refinement review:

"Goal alignment should be measured on the assistant's care actions (recommendations, risk management, follow-up), not on full raw text, otherwise verbosity and bedside-manner style dominate the score."

### Recommended Enhancement: Actions-Only Alignment + Per-Turn Curve

Actions-only is the default, and full-text is retained as an ablation.

\[
\text{AlignmentScore} = \cos(\text{SBERT}(A_{1:T}), \text{SBERT}(P))
\]

Where:

- \(A_{1:T}\) is the concatenation of extracted clinical actions up to turn \(T\)
- \(P\) is the case target plan

The per-turn curve reports alignment up to each turn \(t\):

- `alignment_curve_actions[t]` uses actions from turns `<= t`
- If no actions exist yet, the curve entry is `None` (not `0.0`)

### Key Points

1. **Supplementary Role**: Not used for primary ranking
2. **Per-Turn Tracking**: `alignment_curve_actions` reports adherence up to each turn
3. **Reproducible Gold Plans**: Uses deterministic extraction from source data

### Defence Statement

> "Session Goal Alignment uses Sentence-BERT embeddings (Reimers & Gurevych, 2019) to measure semantic similarity between model actions and target care plans, providing a richer comparison than lexical metrics."

## Updates Needed

| Item | Priority | Status |
|------|----------|--------|
| Populate target_plans.json for all cases | MEDIUM | 🔲 Check status |
| Add per-turn alignment tracking | LOW | ✅ Implemented |
| Consider clinical-specific embeddings | LOW | 📝 Future work |

## Related Metrics

- **Entity Recall Decay** (Primary): Information retention
- **Knowledge Conflict Rate** (Diagnostic): Self-contradiction
- **Truth Decay Rate / Drift Slope (β)** (Entity Recall summary statistic): Decay speed derived from the primary recall curve
