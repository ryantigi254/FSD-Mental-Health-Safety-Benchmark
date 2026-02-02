# Metric Defensibility & Verification Protocol

I verify all clinical safety metrics to ensure they are defensible for final reporting and supervisor review. This framework moves beyond simple heuristics by ensembling rule-based logic with semantic (NLI/Embedding) validation.

## Metric Refinement Methodology

I follow a strict four-stage validation process when selecting and refining metrics. I do not blindly implement metrics; I critically evaluate them for clinical safety contexts.

1. **Source Identification**:
   * **Input**: I review definitions in `Metrics and Evaluation.tex` and supervisor guidelines.
   * **Action**: I identify the core concept (e.g., "Faithfulness", "Sycophancy").

2. **Academic Alignment**:
   * **Input**: I read the primary source paper (e.g., Wei et al., Chang et al., Turpin et al.).
   * **Action**: I verify if the original metric (often built for general NLP) is robust enough for *clinical safety*.

3. **Gap Analysis (The "Single Point of Failure" Check)**:
   * **Question**: *"Can a model cheat this metric?"* or *"Can simple noise trigger a false positive?"*
   * **Outcome**: I identify weaknesses (e.g., Regex can be tricked by "I don't agree", simple NER misses synonyms).

4. **Double Verification Implementation**:
   * **Action**: I layer a secondary, semantic validation method on top of the primary heuristic.
   * **Result**: The "Double Verification" strategies detailed below (e.g., NLI Ensemble, Semantic Overlap).

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

I apply the following "Double Checks" globally to maintain academic and clinical standards:

1. **Bootstrap Confidence Intervals**: I report 95% CI for all headline metrics (P_Syc, Recall@T10) to ensure results are not just noise.
2. **Asymmetry Testing (T3 Style)**: For Sycophancy, I check **Bad Flip Rate** vs. **Good Flip Rate**. High Bad/Low Good proves malignant sycophancy.
3. **FactScore-Lite**: For Hallucination, I use Scispacy + NLI to approximate the FactScore methodology from Min et al. (2023).

---

## Metric Defensibility Reference Matrix

| Metric | Threat to Validity | Double Verification Fix |
|--------|---------------------|--------------------------|
| **Faithfulness Gap** | CoT might be "post-hoc" junk | Step-F1 verifies reasoning quality. |
| **P_Syc** | "Agreeing to disagree" | NLI verifies adopting the wrong stance. |
| **Entity Recall** | NER extraction noise | Semantic presence validation. |

---

## Metric-to-Citation Matrix

| Metric | Study | Primary Citation | Verification Method |
|--------|-------|------------------|---------------------|
| **Faithfulness Gap** | A | Lanham et al. (2023) | Regex extraction + Step-F1 semantic alignment |
| **Step-F1** | A | DeYoung et al. (2019) - ERASER | Token-level reasoning overlap |
| **Silent Bias Rate** | A | Turpin et al. (2023) | Correlation check (causal intervention proposed) |
| **Sycophancy Probability** | B | Wei et al. (2023) | Regex + NLI entailment ensemble |
| **Flip Rate** | B | Chang et al. (2026) - T3 | Bad vs Good flip asymmetry testing |
| **Evidence Hallucination** | B | Min et al. (2023) - FactScore; RAGAS (Es et al., 2023) | ScispaCy SVO extraction + NLI verification |
| **Turn of Flip** | B | Multi-turn dialogue safety | First failure point tracking |
| **Entity Recall Decay** | C | Neumann et al. (2019) - scispaCy | NER extraction + Jaccard semantic validation |
| **Knowledge Conflict Rate** | C | He et al. (2020) - DeBERTa; Welleck et al. (2019) - Dialogue NLI | DeBERTa-v3 NLI contradiction detection |
| **Session Goal Alignment** | C | Reimers & Gurevych (2019) - Sentence-BERT | Cosine similarity to target plan |
| **Drift Slope** | C | Standard OLS regression | Linear regression on recall curves |

---

## 5. Metrics Pending Double Verification (Future Work)

The following metrics currently rely on **Single-Point Verification**. While defensible for this stage of research, they represent opportunities for future enhancement to meet the full "Double Verification" standard.

| Metric | Current Method (Single-Point) | Proposed Double Verification |
|--------|-------------------------------|------------------------------|
| **Silent Bias Rate** | **Correlation Check**: Associations between demographics and diagnosis. | **Causal Intervention**: Flip the demographic variable and measure the *change* in diagnosis (Counterfactual Testing). |
| **Turn of Flip** | **First Failure Point**: The turn number where the model yields. | **Sentiment Analysis**: Verify the *tone* of the flip (was it a polite correction or a submissive apology?). |
| **Knowledge Conflict** | **NLI Contradiction**: Direct logical conflict check. | **Consistency Score**: Cross-reference with external knowledge bases (RAG verification). |
| **Session Goal Alignment** | **Cosine Similarity**: Embedding distance to target plan. | **Action Classification**: Train a classifier to label specific clinical actions (e.g., "Prescribed Meds", "Ordered Tests"). |
