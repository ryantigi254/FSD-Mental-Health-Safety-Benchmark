# Scaling Overview

This document summarises the benchmark scaling documentation and links the key guides used to expand sample sizes, persona coverage, and multi-turn robustness across all studies.

## Scaling Structure

The scaling work is organised into **study-level scaling guides**, a **metrics verification track**, and a **statistical power review**.

### Study A: Faithfulness Prompt Scaling
**Purpose**: Scale Study A main prompts while preserving OpenR1-Psy alignment and gold-label consistency.

**Documentation**:
- [Study A Prompt Scaling Guide](study/study%20a/STUDY_A_PROMPT_SCALING.md)

### Study A Bias: Adversarial and Intersectional Expansion
**Purpose**: Scale silent-bias evaluation with broader personas, richer intersectional cases, and improved demographic coverage.

**Documentation**:
- [Bias Scaling Guide](study/study%20a%20bias/BIAS_SCALING_GUIDE.md)
- [Bias Methodology](study/study%20a%20bias/METHODOLOGY.md)
- [Intersectional Bias Vignettes](study/study%20a%20bias/INTERSECTIONAL_BIAS_VIGNETTES.md)
- [New Personas Implementation](study/study%20a%20bias/NEW_PERSONAS_IMPLEMENTATION.md)

### Study B: Sycophancy Scaling
**Purpose**: Scale single-turn sycophancy testing to improve confidence intervals and persona diversity.

**Documentation**:
- [Study B Scaling Guide](study/study%20b/STUDY_B_SCALING.md)

### Study B Multi-Turn: Pressure Progression Scaling
**Purpose**: Define a publication-ready multi-turn pressure protocol with realistic session lengths and clustered statistical treatment.

**Documentation**:
- [Study B Multi-Turn Scaling Guide](study/study%20b%20multi-turn/STUDY_B_MULTI_TURN_SCALING.md)

### Study C: Longitudinal Drift Scaling
**Purpose**: Scale case count, turn depth, and persona breadth for stronger long-horizon memory and consistency analysis.

**Documentation**:
- [Study C Scaling Guide](study/study%20c/STUDY_C_SCALING.md)

### Cross-Study Statistical Adequacy
**Purpose**: Quantify power, confidence interval width, and sample-size sufficiency across Studies A/B/C.

**Documentation**:
- [Statistical Power Analysis](statistical_power_analysis.md)

---

## Scale Targets Snapshot

| Area | Focus | Scaling Outcome |
|------|-------|-----------------|
| Study A (Main) | Faithfulness precision | Increased prompt volume target for tighter CIs |
| Study A (Bias) | Fairness stress testing | Expanded vignettes/personas and intersectional scenarios |
| Study B (Single-turn) | Sycophancy precision | Larger sample pool with broader persona coverage |
| Study B (Multi-turn) | Pressure robustness | Realistic ~20-turn protocol and variant-controlled pressure styles |
| Study C | Longitudinal consistency | Increased cases and turns for more stable drift estimates |
| Cross-study | Statistical defensibility | CI and adequacy analysis documented for publication-readiness |

---

## File Structure

```text
scaling/
|-- scaling_summary.md                # This file
|-- statistical_power_analysis.md
|-- study/
|   |-- study a/
|   |   `-- STUDY_A_PROMPT_SCALING.md
|   |-- study a bias/
|   |   |-- BIAS_SCALING_GUIDE.md
|   |   |-- INTERSECTIONAL_BIAS_VIGNETTES.md
|   |   |-- METHODOLOGY.md
|   |   `-- NEW_PERSONAS_IMPLEMENTATION.md
|   |-- study b/
|   |   `-- STUDY_B_SCALING.md
|   |-- study b multi-turn/
|   |   `-- STUDY_B_MULTI_TURN_SCALING.md
|   `-- study c/
|       `-- STUDY_C_SCALING.md
`-- metrics/
    |-- README.md
    |-- metrics_summary.md
    `-- VERIFICATION_FRAMEWORK.md
```

---

## Quick Reference

### For Scaling Planning
1. Start with [Statistical Power Analysis](statistical_power_analysis.md)
2. Choose the study guide under `study/`
3. Align metric validation in [Metrics Summary](metrics/metrics_summary.md)

### For Metric Defensibility
1. Review [Metrics Summary](metrics/metrics_summary.md)
2. Apply [Metric Verification Protocol](metrics/VERIFICATION_FRAMEWORK.md)

---

## Notes

- Study-level guides retain **OpenR1-Psy-first** constraints for main split integrity.
- Multi-turn scaling treats turns as clustered observations, not independent samples.
- Metric-level publishability and verification requirements are tracked under `scaling/metrics/`.
