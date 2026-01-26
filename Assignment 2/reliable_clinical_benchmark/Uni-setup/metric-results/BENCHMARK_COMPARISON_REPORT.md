# Benchmark Comparison & Analysis Report

## 1. Overview
This report compares the findings of our Reliable Clinical Benchmark (Study A/B/C) against the published capabilities and external benchmarks of the models evaluated. The goal is to contextualize our results: Do "Domain Experts" actually outperform "Reasoning Models" in mental health, or do they fail in specific ways?

## 2. Model Landscape
We evaluated three categories of models:

| Model | Type | Claims/Notes | External Benchmarks (MMLU/MedQA) |
|---|---|---|---|
| **DeepSeek-R1-Distill** | Reasoning (SOTA) | Distilled from R1-671B. Claims high MMLU/Math/Code. | MMLU: ~80%+, Code: High. Known for "Aha!" moments. |
| **Qwen3-8B** (Untuned) | General Baseline | Strong dense model. | MMLU: ~74.7% (Unquantized). |
| **QWQ-32B-Preview** | Reasoning (Beta) | Experimental reasoning model from Alibaba. | Strong math/reasoning but beta stability issues. |
| **PsyLLM** | Domain Expert | Fine-tuned for psychological counseling. | Evaluated on internal counseling benchmarks. No MMLU/MedQA published. |
| **Psyche-R1** | Domain Reasoning | R1 distilled + Psych fine-tune. | Claims results comparable to R1-671B on Psych benchmarks (PCEB). |
| **Psych-Qwen-32B** | Domain Expert | Large 32B clinical fine-tune. | Built on Qwen-32B (Strong coding/math base). |
| **GPT-OSS-20B** | General Baseline | Open weights, older architecture? | MMLU: ~85.3 (High Reasoning). |
| **Piaget-8B** | Domain Expert | Clinical fine-tune (WAIS/Action). | Focus on action/clinical execution. |

## 3. Findings vs. Expectations

### A. The "Reasoning Tax" (Study A)
**Expectation**: Reasoning models (DeepSeek, QWQ, Psyche-R1) should excel at complex diagnosis by "thinking through" the symptoms.
**Reality**: Most reasoning models showed a **negative Faithfulness Gap** (CoT < Early Accuracy).
*   **DeepSeek-R1**: -0.190 Gap.
*   **QWQ**: -0.145 Gap.
*   **Analysis**: This contradicts the general trend where CoT boosts performance on Math/Logic (GSM8K). In clinical diagnosis, "over-thinking" allows models to rationalize incorrect hallucinations or drift into refusal modes (Refusal Rate for some reasoning models was higher or they produced vast text without a clear label).
*   **Exception**: `psyche-r1-local` had a small gap (-0.020) and high raw accuracy, suggesting *domain-tuned* reasoning is better than *general* reasoning for this task.

### B. Domain vs. Generalist (Study C - Drift)
**Expectation**: Domain experts should maintain better clinical context over time.
**Reality**: **Confirmed**.
*   **PsyLLM** (Domain Expert) achieved **0.715 Recall @ Turn 10** (Highest).
*   **DeepSeek-R1** (General Reasoning) dropped to **0.366** (Lowest).
*   **Insight**: General reasoning models ("Reasoners") consume their context window with long "thought" chains, potentially pushing out the original patient details (Turn 1) faster or losing focus. Domain experts (PsyLLM) trained on counseling transcripts likely learn to track patient history better.

### C. Bias & Sycophancy (Study B)
**Expectation**: Specialized models might be safer?
**Reality**: **Mixed/Danger**.
*   **Psyche-R1**: **0.714 Silent Bias Rate** (Highest). This is critical. Converting a reasoning model to a domain model might hide biases inside the "thought" process or make it confident in stereotypes.
*   **DeepSeek-R1**: **-0.162 Sycophancy Probability** (Best). The "Reasoning" capability makes the model stubborn/independent, preventing it from agreeing with user misconceptions. Ideally, we want this *Independence* but without the *Accuracy Loss* seen in Study A.

## 4. Conclusion: Is "Mental Health Fine-tuning" Better?
**Yes, but with caveats.**

1.  **For Long Interactions (Therapy/Counseling)**: **YES**. `PsyLLM` and `Psych-Qwen` are superior at maintaining context (Drift/Recall) compared to raw reasoning models.
2.  **For Diagnostic Accuracy**: **NO**. Current fine-tunes (except maybe Psyche-R1) struggle with strict diagnostic tasks (Faithfulness Gap). They revert to "counselor speak" (vague, supportive) rather than "clinician speak" (precise diagnosis).
3.  **For Safety**: **WARNING**. `Psyche-R1` shows that fine-tuning can introduce severe hidden biases (0.714 rate). General reasoners (DeepSeek) are more robust against simple sycophancy but less accurate in diagnosis.

## 5. Recommendations
*   **Hybrid Approach**: Use **DeepSeek-R1** (or QWQ) for distinct, single-turn diagnostic validation (checking logic), but use **PsyLLM** for the actual conversational interface (patient history).
*   **Bias Mitigation**: Immediate retraining requirement for `Psyche-R1` to address the 71% silent bias rate.
*   **Prompting**: Reasoning models need constraints to prevent "over-thinking" symptoms into obscurity.

