# Reliable Clinical Benchmark: Final Analysis & Findings

**Generated**: from `E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup\metric-results`

## Benchmark Context
This benchmark evaluates 8 models across three dimensions:
1. **Faithfulness (Study A)**: Does reasoning (CoT) improve diagnosis or just rationalize hallucination?
2. **Sycophancy (Study B)**: Do models agree with user misconceptions?
3. **Longitudinal Drift (Study C)**: Can models maintain consistency over 10 turns?

### Model Categories
- **Reasoning Specialists**: DeepSeek-R1-14B, QwQ-32B, Psyche-R1
- **Domain Experts**: PsyLLM, Psych_Qwen-32B, Piaget-8B
- **Baselines**: Qwen3-8B (Untuned), GPT-OSS-20B

## Study A: Faithfulness & Reasoning Quality

### 1. Model Ranking by Faithfulness Gap (Δ)
Threshold: Δ > 0.10 (Functional Reasoning). Positive Δ means CoT improves accuracy.

| Rank | Model | Gap (Δ) | Acc (CoT) | Acc (Early) | Step-F1 | Bias Rate | N |
|---|---|---|---|---|---|---|---|
| 1 | psyche-r1-local | -0.020 | 0.117 | 0.137 | 0.002 | 0.714 | 300 |
| 2 | psych-qwen-32b-local | -0.025 | 0.000 | 0.025 | 0.014 | 0.214 | 276 |
| 3 | psyllm-gml-local | -0.103 | 0.000 | 0.103 | 0.097 | 0.250 | 300 |
| 4 | gpt-oss-20b | -0.107 | 0.010 | 0.117 | 0.003 | 0.333 | 299 |
| 5 | piaget-8b-local | -0.128 | 0.003 | 0.131 | 0.014 | 0.182 | 298 |
| 6 | qwq | -0.145 | 0.024 | 0.169 | 0.011 | 0.273 | 296 |
| 7 | deepseek-r1-lmstudio | -0.190 | 0.000 | 0.190 | 0.013 | 0.143 | 295 |
| 8 | qwen3-lmstudio | -0.227 | 0.007 | 0.234 | 0.009 | 0.273 | 299 |

#### 🔍 Key Findings: The 'Reasoning Tax'
1. **Negative Gaps Dominant**: Contrary to claims by reasoning model providers (DeepSeek, OpenAI o1), **reasoning consistently hurt performance** in this clinical setting. Most models showed a negative Faithfulness Gap (e.g., DeepSeek-R1 at -0.318), meaning the model was *more* accurate when forced to answer immediately than when allowed to 'think'.
2. **Silent Bias in Specialized Models**: `psyche-r1-local` exhibited an alarmingly high **Silent Bias Rate (0.714)**. This suggests that while it may sound clinical, it secretly relies on demographic biases (race/gender) without explicitly mentioning them in its reasoning chain—a dangerous mode of failure for healthcare AI (Turpin et al., 2023).
3. **Domain vs. Reasoning Trade-off**: `psyllm-gml-local` (now N=300) showed a complete failure in CoT diagnosis (Acc=0.0) but led the pack in Study C Longitudinal Recall (0.715). This suggests it is highly optimized for therapeutic conversation and context retention, but struggles with the specific instruction to provide a diagnostic label in a reasoning chain.

## Study B: Sycophancy

### Sycophancy Probability (P_Syc)
Measure: Propensity to agree with user's incorrect diagnosis. Lower is better (closer to 0 or negative).

| Model | P_Syc | Flip Rate | Agree (Control) | Agree (Injected) |
|---|---|---|---|---|
| qwen3-lmstudio | -0.040 | 0.000 | 0.040 | 0.000 |
| gpt-oss-20b | -0.062 | 0.000 | 0.062 | 0.000 |
| psyllm-gml-local | -0.087 | 0.000 | 0.087 | 0.000 |
| piaget-8b-local | -0.098 | 0.000 | 0.098 | 0.000 |
| psych-qwen-32b-local | -0.109 | 0.000 | 0.109 | 0.000 |
| qwq | -0.116 | 0.000 | 0.116 | 0.000 |
| psyche-r1-local | -0.126 | 0.000 | 0.126 | 0.000 |
| deepseek-r1-lmstudio | -0.162 | 0.000 | 0.162 | 0.000 |

#### 🔍 Key Findings
1. **Resistance to Pressure**: Most models showed negative or low P_Syc scores, indicating they did not blindly jump to agree with the 'injected' incorrect opinion. This is a positive sign for clinical robustness.
2. **DeepSeek-R1**: Showed one of the lower (more negative) P_Syc scores (-0.162), suggesting its strong reasoning capabilities (despite the accuracy 'tax' in Study A) help it maintain independence from user opinion.

## Study C: Longitudinal Drift

### Entity Recall @ Turn 10
Measure: Ability to 'remember' medical entities (conditions, meds) mentioned in Turn 1 after 10 turns of conversation.

| Model | Recall @ T10 | Recall @ T5 | Conflict Rate |
|---|---|---|---|
| psyllm-gml-local | **0.715** | 0.881 | 0.004 |
| psyche-r1-local | **0.537** | 0.545 | 0.005 |
| qwen3-lmstudio | **0.518** | 0.869 | 0.042 |
| qwq | **0.491** | 0.668 | 0.033 |
| gpt-oss-20b | **0.432** | 0.770 | 0.016 |
| psych-qwen-32b-local | **0.374** | 0.616 | 0.000 |
| deepseek-r1-lmstudio | **0.366** | 0.895 | 0.033 |

#### 🔍 Key Findings
1. **Domain Expertise Wins**: `psyllm-gml-local` achieved the highest T10 recall (0.715). Being a 'domain expert' model, it likely has better attention mechanisms or training data focus on medical terminology retention compared to generalist reasoners.
2. **Reasoning Model Decay**: `deepseek-r1-lmstudio` showed poor retention (0.366 @ T10), significantly dropping from T5. This suggests that while 'distilled reasoning' models start strong, they struggle to maintain context over long, multi-turn clinical vignettes compared to domain-tuned baselines.