# Metric Defensibility & Verification Protocol

To ensure all clinical safety metrics are defensible for final reporting and supervisor review, we implement a **Double Verification** strategy. This framework moves beyond simple heuristics by ensembling rule-based logic with semantic (NLI/Embedding) validation.

## Metric Refinement Methodology

Our approach to selecting and refining metrics follows a strict **four-stage validation process**. We do not blindly implement metrics; we critically evaluate them for clinical safety contexts.

1.  **Source Identification**:
    *   **Input**: Review definitions in `Metrics and Evaluation.tex` and supervisor guidelines.
    *   **Action**: Identify the core concept (e.g., "Faithfulness", "Sycophancy").

2.  **Academic Alignment**:
    *   **Input**: Read the primary source paper (e.g., Wei et al., Chang et al., Turpin et al.).
    *   **Action**: Verify if the original metric (often built for general NLP) is robust enough for *clinical safety*.

3.  **Gap Analysis (The "Single Point of Failure" Check)**:
    *   **Question**: *"Can a model cheat this metric?"* or *"Can simple noise trigger a false positive?"*
    *   **Outcome**: Identify weaknesses (e.g., Regex can be tricked by "I don't agree", simple NER misses synonyms).

4.  **Double Verification Implementation**:
    *   **Action**: Layer a secondary, semantic validation method on top of the primary heuristic.
    *   **Result**: The "Double Verification" strategies detailed below (e.g., NLI Ensemble, Semantic Overlap).

---

## 1. Study A: Faithfulness (The "Logic-Check")

**Double Verification Strategy**: *Syntactic Extraction + Semantic Alignment*

| Layer | Method | Purpose |
|-------|--------|---------|
| **Primary** | Regex Step Extraction | Isolates reasoning blocks using `REASONING:` and `DIAGNOSIS:` markers. |
| **Secondary** | **Step-F1 (Semantic)** | Compares the *content* of extracted steps against Gold reasoning using token-level overlap. |
| **Validation** | Manual Reasoning Audit | 50-sample human check to verify the extracted logic matches the final answer. |

**Defensibility**: This proves the model arrived at the correct diagnosis through valid clinical reasoning, rather than "unguided" accuracy.

---

## 2. Study B: Sycophancy (The "Pressure-Check")

**Double Verification Strategy**: *Heuristic Detection + NLI Ensemble*

| Layer | Method | Purpose |
|-------|--------|---------|
| **Primary** | Multi-Tier `is_agreement()` | 4-tier regex system for phrases like "I agree" or "You are correct." |
| **Secondary** | **NLI Entailment (DeBERTa)** | Verifies if the model's response *semantically entails* the user's incorrect hypothesis. |
| **Validation** | P_Syc Baseline Shift | Comparing Control vs. Injected runs to isolate the "Influence of Pressure." |

**Defensibility**: Distinguishes between "polite conversational fillers" and genuine "clinical capitulation" where medical truth is abandoned.

---

## 3. Study C: Longitudinal Drift (The "Memory-Check")

**Double Verification Strategy**: *NER Extraction + Semantic Occurrence Validation*

| Layer | Method | Purpose |
|-------|--------|---------|
| **Primary** | scispaCy NER (`sci_sm`) | Extracts clinical entities (medications, symptoms) from multi-turn summaries. |
| **Secondary** | **Fuzzy Semantic Match** | Cross-validates that extracted entities actually exist in the response text via Jaccard overlap. |
| **Validation** | Recall Decay Slope | Statistical linear regression to prove "Forgetting" follows a significant trend. |

**Defensibility**: Prevents false positive recall counts caused by NER errors and ensures the model is meaningfully retaining patient data over turns.

---

## 4. Universal Statistical Rigour

To maintain academic and clinical standards, the following "Double Checks" are applied globally:

1.  **Bootstrap Confidence Intervals**: 95% CI reported for all headline metrics (P_Syc, Recall@T10) to ensure results aren't just noise.
2.  **Asymmetry Testing (T3 Style)**: For Sycophancy, we check **Bad Flip Rate** vs. **Good Flip Rate**. High Bad/Low Good proves malignant sycophancy.
3.  **FactScore-Lite**: For Hallucination, we use Scispacy + NLI to approximate the standard FactScore methodology used in state-of-the-art research.

---

## Metric Defensibility Reference Matrix

| Metric | Threat to Validity | Double Verification Fix |
|--------|---------------------|--------------------------|
| **Faithfulness Gap** | CoT might be "post-hoc" junk | Step-F1 verifies reasoning quality. |
| **P_Syc** | "Agreeing to disagree" | NLI verifies adopting the wrong stance. |
| **Entity Recall** | NER extraction noise | Semantic presence validation. |

---

## 5. Metrics Pending Double Verification (Future Work)

The following metrics currently rely on **Single-Point Verification**. While defensible for this stage of research, they represent opportunities for future enhancement to meet the full "Double Verification" standard.

| Metric | Current Method (Single-Point) | Proposed Double Verification |
|--------|-------------------------------|------------------------------|
| **Silent Bias Rate** | **Correlation Check**: Associations between demographics and diagnosis. | **Causal Intervention**: Flip the demographic variable and measure the *change* in diagnosis (Counterfactual Testing). |
| **Turn of Flip** | **First Failure Point**: The turn number where the model yields. | **Sentiment Analysis**: Verify the *tone* of the flip (was it a polite correction or a submissive apology?). |
| **Knowledge Conflict** | **NLI Contradiction**: Direct logical conflict check. | **Consistency Score**: Cross-reference with external knowledge bases (RAG verification). |
| **Session Goal Alignment** | **Cosine Similarity**: Embedding distance to target plan. | **Action Classification**: Train a classifier to label specific clinical actions (e.g., "Prescribed Meds", "Ordered Tests"). |
