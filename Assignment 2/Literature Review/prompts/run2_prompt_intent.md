# Run 2 – Prompt & Research Intent Record

**Role Spec:** Principal NLP Researcher/Engineer building a clinical LLM evaluation harness.

**Prompt Summary:**
- Bridge theory-to-execution for three studies (Faithfulness, Sycophancy, Longitudinal Drift).
- For each study:
  1. Summarize dominant methodologies beyond the key paper.
  2. Provide 2–3 metrics (Gold standard + alternatives/novel derivatives).
  3. Supply implementation logic (math derivations / Python pseudocode).
- Deliverable format: "Researcher’s Implementation Guide" with Theory, Metrics, and Code Specs per study.

**Plan / Objectives:**
1. **Faithfulness Study (Study A)**
   - Reference Lanham (2023) for Early Answering, plus Turpin (2023) biasing features, Wang (2022) self-consistency, CC-SHAP, Ragas Faithfulness.
   - Metrics to extract:
     - Metric A: Faithfulness Gap (Lanham) – \(Acc_{CoT} - Acc_{Early}\).
     - Metric B: Silent Bias Rate (Turpin) – ratio of biased outputs that omit bias mention.
     - Metric C: Token Attribution Alignment (SHAP/NLI-based) – cosine similarity between cited evidence and attribution weights.
   - Implementation: show Pythonic logic for Early Answering loop and derived metrics (e.g., using truncation runs, SHAP integration).

2. **Sycophancy Study (Study B)**
   - Base on Wei (2023) for opinion injection; supplement with Sharma (persona-based agreements), Fanous (SycEval challenge types), Pandey/Beacon (logit detection), Evidence hallucination detection (NLI/Ragas), Truth Decay (probability shifts).
   - Metrics:
     - Metric A: Sycophancy Probability (Wei) – agreement probability difference.
     - Metric B: Log-prob shift (Beacon) – \(\log p(agree) - \log p(correct)\).
     - Metric C: Evidence Hallucination Rate – unsupported claims / total claims.
   - Implementation: prompt templates (control vs injected), probability extraction logic, NLI hallucination checker.

3. **Longitudinal Drift Study (Study C)**
   - Follow Kruse (2025) for PDSQI-9 concept but replace with automated proxies: entity tracking, contextual integrity, judge LLM scoring, NLI contradictions.
   - Metrics:
     - Metric A: Entity Recall/Drift Rate – recall of canonical entities across turns.
     - Metric B: Knowledge Conflict Score – NLI-based contradiction frequency not grounded in new evidence.
     - Metric C: Judge LLM Consistency Score – automated rubric for accuracy/citations.
   - Implementation: stepwise flow for entity extraction (scispaCy), NLI checks between sequential summaries, judge prompts.

**Outcome Goal:**
Produce a Researcher’s Implementation Guide capturing the above, serving as documentation of prior intent and guiding harness development.
