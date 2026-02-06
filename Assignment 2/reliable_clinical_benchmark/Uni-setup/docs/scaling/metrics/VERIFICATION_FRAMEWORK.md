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

Metric docs:

1. [Faithfulness Gap](study_a/faithfulness_gap.md)
2. [Step-F1](study_a/step_f1.md)
3. [Silent Bias Rate](study_a/silent_bias_rate.md)

Relevant study docs:

1. [Study A: Faithfulness](../../studies/study_a/study_a_faithfulness.md)
2. [Study A: Bias](../../studies/study_a/study_a_bias.md)
3. [Study A: Generation Commands](../../studies/study_a/study_a_generation_commands.md)

### Study A — Metric 1: Faithfulness Gap (Δ_Reasoning)

**Double Verification Strategy**: *Early Answering + Output Integrity Checks*

| Layer | Method | Purpose |
|-------|--------|---------|
| **Primary** | Early Answering (CoT vs Direct accuracy gap) | Measures whether visible reasoning tokens causally improve accuracy. |
| **Secondary** | Direct-mode reasoning suppression audit (manual spot-check; `<think>` / `REASONING:` absent in direct) | Ensures the “Early” arm is actually suppressing reasoning (prevents a fake Δ). |
| **Validation** | Bootstrap CI over Δ + small manual audit of direct-mode generations | Confirms the gap is statistically stable and not driven by formatting leakage. |

**Defensibility**: Δ is a black-box proxy for causal faithfulness: if the model cannot “use” reasoning to improve accuracy, its reasoning is decorative.

### Study A — Metric 2: Step-F1

**Double Verification Strategy**: *Deterministic Step Matching + Conservative Scoring*

| Layer | Method | Purpose |
|-------|--------|---------|
| **Primary** | Token overlap Step-F1 (Dice coefficient with threshold) | Scores how well the model’s extracted steps match gold steps. |
| **Secondary** | One-to-one greedy matching + normalisation (lowercase/punctuation/whitespace) | Prevents double counting and makes matching reproducible across runs. |
| **Validation** | Manual audit on a small sample of matched/unmatched step pairs | Confirms the threshold behaves sensibly in clinical phrasing edge cases. |

**Defensibility**: Step-F1 is a deterministic “clinical terminology aware” similarity baseline; it checks reasoning content quality, not just answer correctness.

### Study A — Metric 3: Silent Bias Rate (R_SB)

**Double Verification Strategy**: *Heuristic Detection + Semantic Mention Verification*

| Layer | Method | Purpose |
|-------|--------|---------|
| **Primary** | Biased answer detection + “no mention of bias feature” keyword check | Flags “sneaky bias” where the model uses a protected attribute but does not disclose it. |
| **Secondary** | NLI entailment check for implicit bias mention (optional / planned upgrade) | Catches euphemisms / implicit references that keyword matching misses. |
| **Validation** | Counterfactual feature flip or targeted manual review on a small subset | Confirms the biased feature is causally relevant rather than a spurious correlation. |

**Defensibility**: Silent bias follows Turpin-style unfaithful explanation detection but replaces the manual-only bottleneck with a scalable entailment proxy.

---

## 2. Study B: Sycophancy (The "Pressure-Check")

Metric docs:

1. [Sycophancy Probability](study_b/sycophancy_probability.md)
2. [Flip Rate](study_b/sycophancy_probability.md#flip-rate-clinical-harm-rate)
3. [Evidence Hallucination](study_b/evidence_hallucination.md)

Additional metric docs:

1. [Turn of Flip](study_b/turn_of_flip.md)

Relevant study docs:

1. [Study B: Sycophancy](../../studies/study_b/study_b_sycophancy.md)
2. [Study B: Multi-turn](../../studies/study_b/study_b_multi_turn.md)
3. [Study B: Commands](../../studies/study_b/study_b_commands.md)

### Study B — Metric 1: Sycophancy Probability (P_Syc)

**Double Verification Strategy**: *Heuristic Agreement Detection + NLI Entailment Ensemble*

| Layer | Method | Purpose |
|-------|--------|---------|
| **Primary** | `is_agreement()` multi-tier heuristic | Detects explicit and implicit agreement patterns quickly. |
| **Secondary** | NLI entailment check (OR-ensemble with heuristic; optional / planned upgrade) | Disambiguates polite language from true adoption of an incorrect clinical belief. |
| **Validation** | Control vs Injected baseline shift + manual precision/recall audit on a small sample | Confirms the classifier is measuring “pressure influence” and not generic politeness. |

**Defensibility**: The differential (Control vs Injected) design isolates the causal effect of user pressure.

### Study B — Metric 2: Flip Rate (Clinical Harm Rate)

**Double Verification Strategy**: *Gold-label Correctness + Asymmetry Decomposition*

| Layer | Method | Purpose |
|-------|--------|---------|
| **Primary** | Bad flip detection (Correct_control → Incorrect_injected) against gold label | Quantifies direct clinical harm from sycophancy. |
| **Secondary** | T3-style decomposition (Bad flips vs Good flips; optional / planned upgrade) | Separates “sycophantic harm” from general instability (high flip rates both directions). |
| **Validation** | Bootstrap CI over flip rates | Ensures rates are statistically stable and comparable across models. |

**Defensibility**: Unlike raw agreement, flip rate is a harm-grounded metric tied to gold truth.

**Computation note**: Flip Rate is computed in analysis notebooks from cached control/injected generations. It is not emitted by the Study B pipeline JSON.

### Study B — Metric 3: Evidence Hallucination (H_Ev)

**Double Verification Strategy**: *Deterministic Claim Extraction + NLI Grounding*

| Layer | Method | Purpose |
|-------|--------|---------|
| **Primary** | Deterministic atomic claim extraction (scispaCy dependency parsing / SVO; optional / planned upgrade) | Creates a reproducible set of checkable claims from the model’s reasoning. |
| **Secondary** | NLI entailment verification against the vignette | Flags unsupported (hallucinated) claims rather than surface-level disagreement. |
| **Validation** | Bootstrap CI over mean H_Ev + sanity bounds (0 ≤ H_Ev ≤ 1) | Confirms stability and prevents artefacts from degenerate “no-claims” cases. |

**Defensibility**: This distinguishes “polite agreement” from clinically dangerous confabulation.

---

## 3. Study C: Longitudinal Drift (The "Memory-Check")

Metric docs:

1. [Entity Recall Decay](study_c/entity_recall_decay.md)
2. [Knowledge Conflict Rate](study_c/knowledge_conflict_rate.md)
3. [Session Goal Alignment](study_c/session_goal_alignment.md)

Relevant study docs:

1. [Study C: Drift](../../studies/study_c/study_c_drift.md)
2. [Study C: Commands](../../studies/study_c/study_c_commands.md)

### Study C — Metric 1: Entity Recall Decay

**Primary metric**: Entity Recall curve with `Recall@T10` as the headline thresholded endpoint.

**Summary statistic**: Truth Decay Rate / Drift Slope (β), computed from the same recall curve (`numpy.polyfit`, degree=1).

**Double Verification Strategy**: *Medical NER + Match Validation*

| Layer | Method | Purpose |
|-------|--------|---------|
| **Primary** | scispaCy NER (`en_core_sci_sm`) + critical-entities gold set | Extracts clinical entities from model summaries and anchors the headline gold set to frozen metadata. |
| **Secondary** | Multi-tier matching + semantic presence validation + negation checks (exact/substring/Jaccard; optional NLI) | Prevents false positives from NER artefacts, over-permissive fuzzy matching, and polarity errors. |
| **Validation** | Cross-check full recall curve + Recall@T10 thresholding + Truth Decay Rate/Drift Slope (β) summary + precision/hallucinated rate curves + sampled manual audit (TP/FP/FN by entity class) | Ensures decay is consistent across representations and not inflated by extraction noise. |

**Defensibility**: Recall uses a conservative matching stack and validates “presence in text”, not just “found by NER”.

**What it measures (scope)**:
- **Context-window fidelity**: whether early-turn facts remain accessible as the prompt grows.
- **Salience/compression choices**: whether the model preserves clinically critical facts when summarising.
- **Extractor noise**: scispaCy + fuzzy matching limits (tracked via manual audit).

### Study C — Metric 2: Knowledge Conflict Rate (K_Conflict)

**Double Verification Strategy**: *Advice Extraction + Contradiction Classification*

| Layer | Method | Purpose |
|-------|--------|---------|
| **Primary** | Shared clinical action extraction (`_extract_advice()` via `_extract_clinical_actions`) | Isolates the actionable clinical guidance from each turn without fallback snippets. |
| **Secondary** | NLI contradiction detection (DeBERTa-v3) between adjacent turns | Flags explicit self-contradiction in guidance rather than mere topic drift. |
| **Validation** | Conservative counting (only “contradiction” verdict) + bootstrap CI over K_Conflict | Reduces false positives and ensures comparisons are stable across models. |

**Defensibility**: This is a direct safety signal: flip-flopping guidance is riskier than passive forgetting.

### Study C — Metric 3: Session Goal Alignment

**Double Verification Strategy**: *Deterministic Target Plans + Embedding Similarity Validation*

| Layer | Method | Purpose |
|-------|--------|---------|
| **Primary** | Deterministic target plan construction per case (Study C `target_plans.json`) | Ensures a stable, reproducible plan-of-care reference for each case. |
| **Secondary** | Sentence-embedding cosine similarity (SBERT / MiniLM) over actions-only text, with full-text ablation | Scores semantic plan adherence beyond surface-level string overlap while reducing style bias. |
| **Validation** | Bootstrap CI over mean alignment score + per-turn alignment curve (actions up to turn t) + stratified reporting by plan provenance (linked vs generated) | Confirms stability and highlights when alignment begins to drift. |

**Defensibility**: I treat plan extraction/generation as a deterministic pre-processing step, and only then compute alignment. This keeps the metric black-box and reproducible.

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

## Metric Citations (Per Study)

### Study A

| Metric | Primary Citation (Source Link) |
|--------|-------------------------------|
| **Faithfulness Gap** | Lanham et al. (2023) — https://arxiv.org/abs/2307.13702 |
| **Step-F1** | DeYoung et al. (2019) — ERASER: https://arxiv.org/abs/1911.03429 |
| **Silent Bias Rate** | Turpin et al. (2023) — https://arxiv.org/abs/2305.04388 |

### Study B

| Metric | Primary Citation (Source Link) |
|--------|-------------------------------|
| **Sycophancy Probability** | Wei et al. (2023) — https://arxiv.org/abs/2308.03958 |
| **Flip Rate** | Chang et al. / T3-aligned bad flip framing: https://arxiv.org/abs/2601.08258 |
| **Evidence Hallucination** | Min et al. (2023) — FActScore: https://arxiv.org/abs/2305.14251; RAGAS docs: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness; RAGAS paper (Es et al., 2023): https://arxiv.org/abs/2309.15217 |

### Study C

| Metric | Primary Citation (Source Link) |
|--------|-------------------------------|
| **Entity Recall Decay** | Neumann et al. (2019) — scispaCy: https://aclanthology.org/W19-5034/ |
| **Knowledge Conflict Rate** | He et al. (2020) — DeBERTa: https://arxiv.org/abs/2006.03654; Welleck et al. (2019) — Dialogue NLI: https://aclanthology.org/P19-1363/ |
| **Drift Slope** | OLS linear regression (standard method; implementation uses `numpy.polyfit`) |

**Closest conceptual relatives (evaluation framing; not the same metric)**:
- Dialogue State Tracking evaluation (slot precision/recall/F1, joint goal accuracy): Williams et al. (2016) — https://doi.org/10.5087/dad.2016.301
- Long-term dialogue memory evaluation with F1/overlap vs human references: https://arxiv.org/abs/2308.15022
- Very long-term memory benchmarks using QA F1 / retrieval accuracy and fact-based summarisation scoring (LoCoMo): https://arxiv.org/abs/2402.17753
- Atomic-fact factuality framing (FActScore) as a fine-grained proxy for summary correctness: https://arxiv.org/abs/2305.14251

---

## 5. Metrics Pending Double Verification (Future Work)

The following metrics currently rely on **Single-Point Verification**. While defensible for this stage of research, they represent opportunities for future enhancement to meet the full "Double Verification" standard.

| Metric | Current Method (Single-Point) | Proposed Double Verification |
|--------|-------------------------------|------------------------------|
| **Silent Bias Rate** | **Correlation Check**: Associations between demographics and diagnosis. | **Causal Intervention**: Flip the demographic variable and measure the *change* in diagnosis (Counterfactual Testing). |
| **Turn of Flip** | **First Failure Point**: The turn number where the model yields. | **Sentiment Analysis**: Verify the *tone* of the flip (was it a polite correction or a submissive apology?). |
| **Knowledge Conflict** | **NLI Contradiction**: Direct logical conflict check. | **Consistency Score**: Cross-reference with external knowledge bases (RAG verification). |
| **Session Goal Alignment** | **Cosine Similarity**: Embedding distance to target plan. | **Action Classification**: Train a classifier to label specific clinical actions (e.g., "Prescribed Meds", "Ordered Tests"). |
