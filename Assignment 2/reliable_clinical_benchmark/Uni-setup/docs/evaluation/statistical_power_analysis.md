# Statistical Power Analysis & Sample Size Adequacy Review

## Executive Summary

This document provides a comprehensive analysis of the confidence intervals, sample sizes, and statistical power across Studies A, B, and C of the clinical LLM benchmark. The goal is to determine whether the current experimental design provides sufficient statistical confidence to make robust scientific claims about model performance.

---

## Current Sample Size Overview

### Benchmark Models (8)

| Model | Type | HuggingFace ID |
|-------|------|----------------|
| PsyLLM | Domain Expert | `GMLHUHE/PsyLLM` |
| Qwen3-8B | Untuned Baseline | `Qwen/Qwen3-8B` |
| GPT-OSS-20B | Generalist Baseline | `openai/gpt-oss-20b` |
| QwQ-32B | General Reasoning | `Qwen/QwQ-32B-Preview` |
| DeepSeek-R1-14B | Distilled Reasoning | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` |
| Piaget-8B | Clinical Reasoning | `gustavecortal/Piaget-8B` |
| Psyche-R1 | Psychological Reasoning | `MindIntLab/Psyche-R1` |
| Psych_Qwen_32B | Large Psych Model | `Compumacy/Psych_Qwen_32B` |

### Sample Sizes per Study

| Study | Metric | Samples per Model | Total Prompts (8 models) |
|-------|--------|-------------------|--------------------------|
| **Study A** (Faithfulness) | acc_cot, acc_early, step_f1 | 180–300 | ~1,920–2,400 |
| **Study A** (Bias) | silent_bias_rate | 58 adversarial prompts | ~464 |
| **Study B** (Sycophancy) | sycophancy_probability, flip_rate | 276 usable pairs | ~2,208 |
| **Study C** (Multi-turn) | entity_recall, knowledge_conflict | 30 cases × 10 turns | ~2,400 |

**Estimated Total Prompts Across All Studies: ~5,000–7,500 (8 models)**

---

## Study A: Faithfulness & Reasoning Analysis

### Current Confidence Intervals

#### Accuracy Metrics (acc_cot, acc_early)

| Model | acc_cot | 95% CI Width | acc_early | 95% CI Width |
|-------|---------|--------------|-----------|--------------|
| deepseek-r1-distill-qwen-7b | 2.13% | ±1.59% | 7.09% | ±3.05% |
| deepseek-r1-lmstudio | 8.93% | ±3.21% | 7.86% | ±3.21% |
| gpt-oss-20b | 9.96% | ±3.64% | 3.83% | ±2.30% |
| qwen3-lmstudio | 12.15% | ±3.65% | 6.25% | ±2.78% |
| psyche-r1-local | 11.64% | ±4.23% | 19.58% | ±5.82% |

#### Faithfulness Gap

| Model | Faithfulness Gap | 95% CI | CI Width |
|-------|------------------|--------|----------|
| deepseek-r1-distill-qwen-7b | -4.96% | [-8.16%, -2.13%] | 6.03 pp |
| psyche-r1-local | -7.94% | [-14.29%, -2.12%] | **12.17 pp** |
| qwen3-lmstudio | +5.90% | [+1.39%, +10.76%] | 9.37 pp |

### Statistical Assessment

> [!WARNING]
> **CI Width Concern**: Several models show confidence intervals spanning 6–12 percentage points, which is quite wide for accuracy metrics. This introduces uncertainty when comparing models.

#### Sample Size Calculation for Accuracy

For a binary outcome (correct/incorrect) with α=0.05 and desired margin of error (E):

```
n = (Z² × p × (1-p)) / E²
```

For p=0.10 (10% accuracy), E=0.02 (±2 percentage points), Z=1.96:
- **Required n ≈ 864 samples per model**

**Current: 180–300 samples** → **~3× underpowered for precision target**

### Expected CI Improvements After Scaling

| Metric | Current (180–300 samples) | After Scaling (600–900 samples) | Improvement |
|--------|--------------------------|--------------------------------|-------------|
| Accuracy CI Width | 3–6 pp | **1.5–3 pp** | ~50% narrower |
| Faithfulness Gap CI | 6–12 pp | **3–6 pp** | ~50% narrower |
| Model Ranking Confidence | Moderate overlap | Clear separation | Definitive rankings |

---

## Study A Bias: Silent Bias Rate Analysis

### Critical Issue: Extremely Small Denominators

| Model | n_biased_outcomes | n_silent | Silent Bias Rate | 95% CI |
|-------|-------------------|----------|------------------|--------|
| psyche-r1-local | 7 | 5 | 71.43% | [28.57%, 100%] |
| gpt-oss-20b | 12 | 4 | 33.33% | [8.33%, 58.33%] |
| psych-qwen-32b-local | 14 | 3 | 21.43% | [0%, 42.86%] |
| piaget-8b-local | 11 | 2 | 18.18% | [0%, 45.45%] |

> [!CAUTION]
> **SEVERELY UNDERPOWERED**: The bias analysis is based on only **7–14 biased outcomes per model**. This is far below acceptable sample sizes for detecting bias effects.

### Key Problems:

1. **Confidence intervals span 42–72 percentage points** — essentially unusable
2. **Most CIs include 0%** — cannot definitively claim bias exists
3. **n=7 biased outcomes** for psyche-r1 means 1 different response changes the rate by **14.3%**

### Minimum Sample Size for Bias Detection

For detecting a bias rate of 20% with ±5% precision:
```
n = (1.96² × 0.20 × 0.80) / 0.05² = 246 biased outcomes per model
```

**Current: 7–14 biased outcomes** → **~18–35× underpowered**

### Recommendation

| Current State | Recommended Target | Scaling Factor |
|---------------|-------------------|----------------|
| 58 adversarial prompts | **300–500 adversarial prompts** | 5–9× |
| 7–14 biased outcomes | **100+ biased outcomes per model** | 7–14× |

### Expected CI Improvements After Scaling

| Metric | Current CI Width | After Scaling (300+ prompts) | Improvement |
|--------|-----------------|------------------------------|-------------|
| Silent Bias Rate | 42–72 pp (unusable) | **8–12 pp** | ~80% narrower |
| Biased Outcomes | 7–14 per model | 100+ per model | Statistically valid |
| Point Estimate Stability | ±14% per response change | ±1–2% per response change | Scientific rigor |

---

## Study B: Sycophancy Analysis

### Confidence Interval Analysis

| Model | Sycophancy Prob. | 95% CI | CI Width |
|-------|------------------|--------|----------|
| qwen3-lmstudio | 3.99% | [1.81%, 6.16%] | 4.35 pp |
| gpt-oss-20b | 6.16% | [3.62%, 9.42%] | 5.80 pp |
| deepseek-r1-lmstudio | 16.67% | [12.32%, 21.01%] | 8.69 pp |
| psyche-r1-local | 12.68% | [8.70%, 17.03%] | 8.33 pp |

### Flip Rate Problem

| All Models | Flip Rate | 95% CI |
|------------|-----------|--------|
| All | **0.00%** | [0.00%, 0.00%] |

> [!IMPORTANT]
> **Zero Flip Rate Across All Models**: This is either a genuine finding (models never flip) OR the multi-turn protocol isn't applying sufficient pressure.

### Statistical Assessment

With **276 usable pairs per model**, the sycophancy analysis is **reasonably powered**:
- CI widths of 4–9 pp are acceptable for comparative purposes
- Can distinguish between high-sycophancy (15%+) and low-sycophancy (<5%) models

### Zero Events Issue

The flip_rate = 0.0 for all models creates an interpretive challenge:
- Either: Models are genuinely robust (good finding)
- Or: The 5-turn protocol is insufficient pressure

**Recommendation**: Test with **10 turns** and **stronger pushback intensity** to verify robustness.

### Expected CI Improvements After Scaling

| Metric | Current (276 pairs) | After Scaling (400–500 pairs) | Improvement |
|--------|---------------------|------------------------------|-------------|
| Sycophancy Prob. CI | 4–9 pp | **3–5 pp** | ~40% narrower |
| Flip Rate Validity | Uncertain (5 turns) | Validated (10 turns + stronger pressure) | Protocol confidence |
| Model Differentiation | Can distinguish >10pp differences | Can distinguish >5pp differences | Finer granularity |

---

## Study C: Multi-Turn Coherence & Memory

### Current Design

- **30 cases per model**
- **10 turns per case**
- **~300 total turn-level observations** (but nested structure)

### Confidence Interval Analysis

| Model | Entity Recall T10 | 95% CI | CI Width |
|-------|-------------------|--------|----------|
| psyllm-gml-local | 20.57% | [16.9%, 24.2%] | 7.3 pp |
| gpt-oss-20b | 25.92% | [23.23%, 29.08%] | 5.85 pp |
| qwq | 33.20% | [29.61%, 36.65%] | 7.04 pp |
| psych-qwen-32b-local | 49.65% | [46.31%, 52.80%] | 6.49 pp |

### Knowledge Conflict Rate

| Model | Conflict Rate | 95% CI | Contradictions Found |
|-------|---------------|--------|---------------------|
| gpt-oss-20b | 0.37% | [0%, 1.11%] | 1 |
| psych-qwen-32b-local | 0.37% | [0%, 1.11%] | 1 |
| All others | 0.00% | [0%, 0%] | 0 |

> [!WARNING]
> **Only 0–1 contradictions detected** across ~270 turn observations per model. This is either:
> - Excellent model consistency (genuine finding)
> - Insufficient cases to trigger the behavior
> - Detection methodology too strict

### Statistical Power Assessment

For 30 cases:
- Standard error for recall ≈ σ/√30 ≈ 0.18σ
- With σ ≈ 0.20, SE ≈ 3.6 percentage points

**30 cases provides marginal power** — can detect large effects (>10 pp) but may miss moderate differences.

### Multi-Turn Scaling Recommendations

| Current State | Minimum for Robustness | Ideal for Publication |
|---------------|------------------------|----------------------|
| 30 cases | **50–75 cases** | **100+ cases** |
| 10 turns | **20 turns** (realistic session) | 20 turns |

### Expected CI Improvements After Scaling

| Metric | Current (30 cases × 10 turns) | After Scaling (75 cases × 20 turns) | Improvement |
|--------|-------------------------------|-------------------------------------|-------------|
| Entity Recall CI | 5–7 pp | **2–4 pp** | ~50% narrower |
| Total Observations | ~300 per model | ~1,500 per model | 5× more data |
| Trend Detection | T10 only | T10, T15, T20 measurements | Decay curve analysis |
| Session Realism | Below typical therapy | Matches CPsyCoun 75th percentile | Clinically valid |

---

## Gold Standard Comparison Adequacy

### Study A Gold Standard

The study compares model outputs against clinical reasoning gold standards from the dataset.

| Assessment Factor | Current State | Concern Level |
|-------------------|---------------|---------------|
| **n_samples** | 180–300 | 🟡 Moderate (acceptable for ranking, insufficient for precision) |
| **Step F1 variance** | Very low (0.01–0.11 range) | 🟢 Consistent measurement |
| **Accuracy variance** | High (2%–20% range) | 🟢 Natural variation |

**Verdict**: Gold standard comparison sample size is **marginally adequate** for model ranking but **insufficient for precise performance estimates**.

---

## Persona & Bias Protocol Analysis

### Current Design Issues

1. **Only 58 adversarial prompts** — insufficient diversity
2. **Bias types may not be comprehensive** — need to verify coverage
3. **Silent bias detection requires more biased outcomes** — currently 7–14 per model

### Recommended Persona/Bias Scaling

| Component | Current | Recommended | Rationale |
|-----------|---------|-------------|-----------|
| **Adversarial prompts** | 58 | **200–300** | Need 100+ biased outcomes per model |
| **Bias categories** | Unknown | **5+ distinct types** | Ensure comprehensive coverage |
| **Persona variations** | Unknown | **3–5 per bias type** | Control for persona effects |

---

## Summary of Scaling Recommendations

### Immediate Priority (Essential for Scientific Claims)

| Study | Current Issue | Required Action | Effort Level |
|-------|---------------|-----------------|--------------|
| **Study A Bias** | n=7–14 biased outcomes | Scale adversarial prompts 5–10× | 🔴 HIGH |
| **Study C** | n=30 cases per model | Scale to 75–100 cases | 🟠 MEDIUM |

### Secondary Priority (Improve Precision)

| Study | Current Issue | Required Action | Effort Level |
|-------|---------------|-----------------|--------------|
| **Study A Main** | ±3–6 pp CI width | Scale prompts 2–3× (to ~500–600) | 🟠 MEDIUM |
| **Study B** | Zero flip rate observed | Test 10–15 turn subset | 🟢 LOW |

### Nice to Have (Publication-Ready)

| Study | Enhancement | Target |
|-------|-------------|--------|
| **Study A** | 900+ prompts per model | ±2 pp precision |
| **Study C** | 100+ cases per model | ±4 pp on entity recall |

---

## Statistical Confidence Quantification

### What You Can Confidently Claim Now

| Claim | Confidence | Evidence |
|-------|------------|----------|
| Model X has higher sycophancy than Model Y (if >5 pp gap) | **HIGH** | ~276 samples, narrow CIs |
| Models show very low flip rates (<1%) | **MODERATE** | Zero observed, but could be floor effect |
| Entity recall declines over conversation turns | **HIGH** | Consistent pattern across all models |
| Model X is more faithful than Model Y (large gap) | **MODERATE** | Wide CIs reduce precision |else | **Model X shows silent bias** | **LOW** | n=7–14 is far too small |

### What You Cannot Confidently Claim

1. **Precise bias rates** — CIs span 40–70 percentage points
2. **Small faithfulness differences** (<5 pp between models)
3. **Knowledge conflict rates** — 0–1 observations insufficient
4. **Whether flip_rate is truly 0% or just very low**

---

## Revised Sample Size Targets for Scientific Rigor

### Power Analysis Summary

| Metric | Effect Size to Detect | Required n per Model | Current n | Gap |
|--------|----------------------|---------------------|-----------|-----|
| Accuracy (±2 pp) | Small | 900 | 280 | 3× |
| Accuracy (±5 pp) | Moderate | 150 | 280 | ✅ |
| Sycophancy (±3 pp) | Small | 500 | 276 | 2× |
| Silent Bias (±10 pp) | Moderate | 100 biased | 10 | 10× |
| Entity Recall (±5 pp) | Moderate | 60 cases | 30 | 2× |

---

## Conclusion & Action Items

### Overall Assessment

| Aspect | Current Rating | Notes |
|--------|----------------|-------|
| **Study A (Main)** | 🟡 **Adequate** | Sufficient for model ranking, insufficient for precise benchmarking |
| **Study A (Bias)** | 🔴 **Inadequate** | Must scale 5–10× before drawing scientific conclusions |
| **Study B** | 🟢 **Good** | Well-powered for sycophancy detection |
| **Study C** | 🟡 **Marginal** | Scale to 75+ cases for robustness |

### Immediate Recommendations

1. **Do NOT make strong claims about silent bias rates** — current sample sizes make these estimates unreliable
2. **Report CIs alongside point estimates** in all publications
3. **Acknowledge limitations** in scope of bias testing
4. **Prioritize Study A Bias scaling** if resources allow
5. **Consider Study C as exploratory** pending additional cases

### For Publication-Quality Results

Target **~18,000 total prompts** distributed as (8 models):
- Study A Main: 4,800 (600 per model × 8)
- Study A Bias: 2,400 (300 adversarial × 8 models)
- Study B: 4,000 (500 pairs per model × 8)
- Study C: 6,400 (100 cases × 10 turns × 8 models)

This represents approximately **3× your current scale**.

---

## Study-Specific Scaling Protocols

### Study B vs Study C: Key Differences

| Study | What It Tests | Turn Requirements |
|-------|--------------|-------------------|
| **Study B** | Sycophancy/flip resistance | **5–10 turns** (short pressure test) |
| **Study C** | Long-term coherence/memory | **20 turns** (realistic therapy session) |

> [!IMPORTANT]
> **Study B (Sycophancy)** tests whether models flip their position when pressured — this is fundamentally a **short-term resistance test**, not a long-term memory test.

### Study B Scaling Focus

For Study B, scaling should focus on **scenario volume and pressure intensity**, NOT longer sessions:

| Scaling Target | Rationale |
|----------------|-----------|
| More test pairs (276 → 400–500) | Narrower CIs |
| More personas (25 → 37) | Wider condition coverage |
| More diverse pressure scenarios | Different disagreement types |
| Stronger pushback intensity | Test 0% flip rate validity |

**NOT** longer multi-turn sessions — sycophancy can be reliably measured in 5–10 turns.

### Addressing the 0% Flip Rate

The 0% flip rate issue is better addressed by:
1. **More aggressive disagreement language** in existing turns
2. **Different pressure types** (emotional, authority-based, persistent)
3. **More test scenarios** — not more turns per scenario

---

## Incremental Scaling Implementation

### Which Studies Can Scale Without Full Re-run?

| Study | Scaling Type | Can Append Incrementally? | Effort |
|-------|-------------|---------------------------|--------|
| **Study A Bias** | Add vignettes to JSON | ✅ YES | LOW |
| **Study A Faith** | Add gold prompts to JSON | ✅ YES | LOW |
| **Study B** | 5→10 turns + personas | ❌ Re-run needed | MEDIUM |
| **Study C** | 10→20 turns + personas | ❌ Re-run needed | HIGH |

### Why Multi-Turn Studies Need Full Re-run

1. **Turn count changes** affect the entire conversation structure
2. **Context accumulates** across turns — can't extend existing sessions
3. **Safer to regenerate** with consistent protocol across all cases

### Priority Order for Scaling

| Priority | Task | Data Files to Update | Impact |
|----------|------|---------------------|--------|
| 1️⃣ | **Study A Bias** | `biased_vignettes.json` | HIGH (currently unusable CIs) |
| 2️⃣ | **Study A Faith** | `gold_diagnosis_labels.json` | MEDIUM |
| 3️⃣ | **Study B re-run** | Persona splits + test pairs | MEDIUM |
| 4️⃣ | **Study C re-run** | Persona splits (20 turns) | HIGH |

---

## Related Documentation

- [study a/STUDY_A_PROMPT_SCALING.md](./study%20a/STUDY_A_PROMPT_SCALING.md) - Faithfulness prompt scaling
- [study a bias/](./study%20a%20bias/) - Bias scaling documentation
- [study b/STUDY_B_SCALING.md](./study%20b/STUDY_B_SCALING.md) - Sycophancy scaling (10 turns)
- [study c/STUDY_C_SCALING.md](./study%20c/STUDY_C_SCALING.md) - Coherence scaling (20 turns)

---

*Last Updated: 2026-01-30*
