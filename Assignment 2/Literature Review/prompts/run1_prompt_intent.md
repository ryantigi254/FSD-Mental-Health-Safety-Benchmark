# Run 1 – Prompt & Literature Search Intent

**Role:** Rigorous NLP researcher curating literature for “Reliable Clinical Reasoning in LLMs.”

**Prompt Summary:**
- Deliver a prioritized shortlist (20 papers) covering Faithfulness, Sycophancy, Mental-Health Benchmarks, and Longitudinal Drift.
- Apply strict filters:
  - Year ≥ 2022 (LLM era). Exception: seminal rationale paper ERASER (2020).
  - Venues: ACL/EMNLP/NAACL, NeurIPS/ICLR, Nature Medicine, Lancet Digital Health, or high-impact arXiv (2024–2025).
  - Must relate to reasoning, alignment failures, or clinical safety.
- Produce output as CSV with columns: Citation, Year, Bucket, Metric_Used, Key_Finding, Rationale_for_Selection.

**Bucket Targets:**
- Bucket A – Faithfulness & Reasoning: 6 papers (metrics like Faithfulness Gap, CoT Utility, Silent Bias).
- Bucket B – Sycophancy & Alignment Faking: 5 papers (opinion injection, alignment faking, scheming).
- Bucket C – Mental Health LLM Benchmarks: 5 papers (PsyLLM, Med-PaLM, safety audits, clinical datasets).
- Bucket D – Longitudinal/Memory Evaluation: 4 papers (multi-turn consistency, truth decay, contextual integrity).

**Execution Plan:**
1. Query literature per bucket with venue/year filters.
2. Apply hard filters (venue, year, topic match) and keep only top-tier studies.
3. Extract for each paper: metric names (e.g., Faithfulness Gap, Truth Decay Rate), key empirical result, and rationale for inclusion in clinical audit.
4. Populate the CSV shortlist meeting bucket quotas.

**Intent:** Provide a durable record of the search strategy for Run 1 so later documentation shows how the 20-paper shortlist aligns with project goals.
