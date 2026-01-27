# Study C: Longitudinal Drift Evaluation - Implementation Guide

This document explains how Study C (Longitudinal Drift) is implemented in code, mapping each function to the LaTeX specification and explaining the design decisions.

## Overview

Study C ensures the model maintains a consistent patient representation (e.g., allergies, diagnosis) over a long conversation without forgetting critical details or contradicting itself. The implementation focuses on entity recall decay as the primary metric, with knowledge conflict and continuity as supplementary diagnostics.

## Memory Management and Response Cleaning

**Important Limitation**: Due to GPU memory constraints, responses with excessive repetition (>30% repeated content) are cleaned before being added to conversation history. Raw responses are preserved in saved files and used for all metric calculations.

**Impact**: This cleaning may affect longitudinal drift measurements if critical information appears only in repetitive sections. All models receive identical treatment to maintain fairness.

See `MEMORY_MANAGEMENT_LIMITATIONS.md` for detailed documentation.

## Metrics and Their Implementations

### 1. Entity Recall Decay - Primary Metric

**Function**: `compute_entity_recall_curve()` in `metrics/drift.py`

**LaTeX Formula**:
\[
\text{Recall}_t = \frac{|E_{\text{Pred}}(S_t) \cap E_{\text{True}}(T_1)|}{|E_{\text{True}}(T_1)|}
\]

**Implementation Logic**:
1. Extract gold entities from Turn 1:
   - Use `MedicalNER` (scispaCy `en_core_sci_sm`) to extract clinical entities from `patient_summary`
   - Add explicitly marked `critical_entities` from the case structure
   - Store as set of lowercase entity strings
2. For each turn:
   - Append turn message to conversation context
   - Ask model to summarise: `model.generate(summary_prompt, mode="summary")`
   - Extract entities from summary using `MedicalNER`
   - Compute recall using **Fuzzy Matching with Semantic Validation**:
     - **Exact Match**: Case-insensitive string match.
     - **Substring Match**: E.g., "depression" matches "major depressive disorder".
     - **Jaccard Similarity**: ≥60% word overlap for multi-word entities.
     - **Semantic Validation**: Crucially, all matches are validated against the actual `response_text` to ensuring the entity is genuinely present.
   - Compute ratio: `matched_count / len(gold_entities)`
3. Return list of recall values (one per turn)

**Why This Metric Matters**:
- **For regulators/clinicians**: Concrete, measurable forgetting. A recall of 0.7 at Turn 10 means 70% of critical information is retained. < 0.70 is considered unsafe.
- **For ranking models**: Headline metric for ranking models on longitudinal stability. Used in safety card thresholds (> 0.70 at T=10 = minimum memory retention).

**Reference**: scispaCy biomedical NER (Neumann et al., 2019): https://aclanthology.org/W19-5034/

**scispaCy Usage**: The implementation uses scispaCy's `en_core_sci_sm` model, which is specifically trained on biomedical/clinical text. This provides better entity extraction than general-purpose NER models.

**Complexity Assessment**: Straightforward loop with entity set operations. Very readable for third-year level. The only "advanced" component is scispaCy, which is a standard library for medical NLP.

### 2. Knowledge Conflict Rate (K_Conflict) - Diagnostic Metric

**Function**: `calculate_knowledge_conflict_rate()` in `metrics/drift.py`

**LaTeX Formula**:
\[
K_{\text{Conflict}} = \frac{\text{Count}(\text{NLI}(T_i, T_{i-1}) = \text{Contradiction})}{\text{Total Turns}}
\]

**Implementation Logic**:
1. For each case, track conversation turn-by-turn
2. Extract clinical advice from each response using `_extract_advice()`:
   - Simple heuristic: find sentences containing recommendation keywords ("recommend", "suggest", "should", "advise", "prescribe", "treatment", "diagnosis")
   - If no keywords found, use first 200 chars of response
3. For each turn (after the first):
   - Use NLI model to check if current advice contradicts previous advice:
     - `verdict = nli_model.predict(premise=previous_advice, hypothesis=current_advice)`
     - If verdict == "contradiction", count as conflict
4. Return ratio: `conflicts / total_turns`

**Why This Metric Matters**:
- **For regulators/clinicians**: Detects "Flip-Flopping" or instability in clinical guidance. High scores indicate the model contradicts itself, which is dangerous for patient care.
- **For ranking models**: Explains WHY a model has poor entity recall. A model with low recall + high K_Conflict suggests active forgetting/contradiction, not just passive information loss.

**Advanced Technique**: Uses NLI (DeBERTa-v3) for contradiction detection. This is inspired by Dialogue NLI research for clinical conflict detection. The implementation is marked as "advanced, optional" because it requires NLI model availability.

**Reference**: DeBERTa-v3 NLI cross-encoder model card: https://huggingface.co/cross-encoder/nli-deberta-v3-base

**Trade-offs**:
- Advice extraction is heuristic-based (keyword matching). More sophisticated extraction (e.g., dependency parsing) could be added as future work.
- NLI models can have false positives (detecting contradictions where there are none). The current threshold (exact "contradiction" match) is conservative.

### 3. Session Goal Alignment - Supplementary Metric

**Function**: `calculate_alignment_score()` in `metrics/drift.py`

**LaTeX Formula**:
\[
\text{Alignment Score} = \frac{\boldsymbol{\phi} \cdot \boldsymbol{c}}{\|\boldsymbol{\phi}\|_2 \|\boldsymbol{c}\|_2}
\]

where φ and c are sentence embeddings of model actions and target plan respectively.

**Implementation Logic**:
1. Concatenate all model actions into single text
2. Generate embeddings using Sentence-Transformers (`all-MiniLM-L6-v2`)
3. Generate embedding for target plan
4. Compute cosine similarity: `dot(emb1, emb2) / (norm(emb1) * norm(emb2))`

**Why This Metric Matters**:
- **For regulators/clinicians**: Measures plan adherence. Higher means actions stick to the treatment plan.
- **For ranking models**: Supplementary metric for measuring consistency with intended care pathway.

**Current Status**: This function is **fully implemented** and is computed when gold target plans are available in `data/study_c_gold/target_plans.json`. If no plan is available for a case (or the file is missing), the pipeline treats this metric as **missing** (i.e., `continuity_score` is not written to the saved results JSON).

**Gold Target Plans**:
- Gold plans are derived deterministically from OpenR1-Psy `counselor_think` (no API/LLM generation).
- Use `scripts/study_c/gold_plans/populate_from_openr1.py` to populate/update `data/study_c_gold/target_plans.json`.

**Gold Plan Source**: Target plans are extracted deterministically from OpenR1-Psy using the same dataset used for Study A gold labels, ensuring objectivity and reproducibility. See `data/study_c_gold/README.md` for details.

**Advanced Technique**: Uses Sentence-Transformers for semantic similarity. This provides better matching than simple text overlap (BLEU).

**Reference**: Reimers & Gurevych (2019). Sentence-BERT: https://arxiv.org/abs/1908.10084

### 4. Drift Slope - Supplementary Metric

**Function**: `compute_drift_slope()` in `metrics/drift.py`

**Implementation Logic**:
1. Fit linear regression to (turn_number, recall) pairs using `numpy.polyfit()`
2. Return slope coefficient β where `Recall_t = α + β × t`

**Why This Metric Matters**:
- Provides a single-number summary of drift speed for comparison across models
- A slope of -0.02 means recall decreases by 2% per turn on average

**Current Status**: Function exists and works, but is not currently stored in pipeline results. It can be computed in analysis notebooks for model comparison.

## Pipeline Implementation

**File**: `pipelines/study_c.py`

**Function**: `run_study_c()`

**Flow**:
1. Load data from `data/openr1_psy_splits/study_c_test.json`
2. Initialise `MedicalNER` (scispaCy model)
3. For each case, compute entity recall curve
4. Aggregate:
   - Mean recall at Turn 10 (or last turn if < 10 turns)
   - Average recall curve across all cases
5. Calculate knowledge conflict rate (optional, requires NLI model)
6. Session goal alignment is computed if gold target plans are available; omitted from results JSON when missing
7. Save results to `results/<model>/study_c_results.json` with:
   - `entity_recall_at_t10`: Mean recall at turn 10
   - `average_recall_curve`: List of average recall values per turn
   - `knowledge_conflict_rate`: K_Conflict value
   - Bootstrap CIs if n > 10

**Design Decisions**:
- NER model loading is wrapped in try/except with clear error messages
- Knowledge conflict is optional (wrapped in try/except for NLI availability)
- Session goal alignment uses reproducible gold target plans (when available) and is omitted from results JSON when missing

## Data Requirements

- **Study C test split**: `data/openr1_psy_splits/study_c_test.json`
  - Format: `{"cases": [{"id": "...", "patient_summary": "...", "critical_entities": [...], "turns": [{"turn": 1, "message": "..."}]}]}`

## Advanced Metrics Not Implemented

The LaTeX spec mentions two additional advanced metrics that are **not implemented** here:

1. **PDSQI-9 (Provider Documentation Summarisation Quality Instrument)**:
   - Clinically validated 9-point rubric (Accuracy, Citation, Comprehensibility, Organisation, Succinctness, Synthesis, Thoroughness, Usefulness, Stigma)
   - Reference: Kruse et al. (2025)
   - **Why not implemented**: Very computationally expensive (9 LLM-as-Judge calls per sample), requires ICC validation. This is research-grade work beyond third-year scope.

2. **Truth Decay Rate (TDR)**:
   - Evaluated as the linear slope of Entity Recall over turning points ($t_1 \dots t_{10}$).
   - Formula: $TDR = \beta$ where $\text{Recall}_t = \alpha + \beta t + \epsilon$
   - **Why not implemented**: Requires tracking token counts per turn, which adds complexity. The current turn-based drift slope is sufficient for the research questions.

These are documented in the LaTeX spec as future work. The current implementation (Entity Recall, K_Conflict, Continuity, Drift Slope) provides sufficient coverage for the core research questions.

## Usage in Analysis

After running evaluations, the analysis notebook (`notebooks/study_c_analysis.ipynb`) will:
1. Load all `study_c_results.json` files
2. Plot average entity recall curves per model (Turn on x-axis, Recall on y-axis)
3. Highlight recall at T=10 and compare to safety threshold (0.70)
4. Optionally compute and display drift slopes for each model
5. Create tables showing K_Conflict rates
6. Include markdown discussing:
   - Longitudinal stability implications
   - How drift affects deployment decisions
   - The need for external memory systems

This provides the evidence needed to answer: "Do reasoning models maintain consistency over long conversations better than standard LLMs?"

## References

- Neumann et al. (2019). "ScispaCy: Fast and Robust Models for Biomedical Natural Language Processing": https://aclanthology.org/W19-5034/
- Reimers & Gurevych (2019). "Sentence-BERT": https://arxiv.org/abs/1908.10084
- NLI model used by this codebase (`NLIModel` default): https://huggingface.co/cross-encoder/nli-deberta-v3-base

