# Metrics Scaling Overview

This document provides a high-level summary of metric definitions, classification, verification expectations, and documentation layout for the scaling track.

## Metric Structure

Metrics are grouped by study and split into **primary**, **diagnostic**, and **supplementary/advanced** roles.

### Study A: Faithfulness + Bias
**Purpose**: Verify functional reasoning quality and fairness under adversarial demographic conditions.

**Metrics**:
- **Faithfulness Gap (Primary)**: [faithfulness_gap.md](study_a/faithfulness_gap.md)
- **Step-F1 (Diagnostic)**: [step_f1.md](study_a/step_f1.md)
- **Silent Bias Rate (Supplementary)**: [silent_bias_rate.md](study_a/silent_bias_rate.md)

### Study B: Sycophancy
**Purpose**: Measure resistance to incorrect user pressure and track harm-relevant failure dynamics.

**Metrics**:
- **Sycophancy Probability (Primary)**: [sycophancy_probability.md](study_b/sycophancy_probability.md)
- **Evidence Hallucination (Diagnostic)**: [evidence_hallucination.md](study_b/evidence_hallucination.md)
- **Turn of Flip (Advanced)**: [turn_of_flip.md](study_b/turn_of_flip.md)

### Study C: Longitudinal Drift
**Purpose**: Track memory retention, contradiction behaviour, and session-level plan adherence over longer conversations.

**Metrics**:
- **Entity Recall & Truth Decay (Primary)**: [entity_recall_decay.md](study_c/entity_recall_decay.md)
- **Knowledge Conflict Rate (Diagnostic)**: [knowledge_conflict_rate.md](study_c/knowledge_conflict_rate.md)
- **Session Goal Alignment (Supplementary)**: [session_goal_alignment.md](study_c/session_goal_alignment.md)

---

## Verification and Publishability

### Verification Protocol
- [Metric Verification Protocol (Double Verification)](VERIFICATION_FRAMEWORK.md)

### Current Publishability Signal
- Study A metrics are structurally defensible with established references and implementations.
- Study B primary and diagnostic metrics should continue hardening around agreement detection and claim-level evidence checks.
- Study C metrics are defensible when entity extraction quality and contradiction checks remain stable at scale.

---

## Implementation Mapping

| Study | Primary implementation path | Companion metrics path |
|------|------------------------------|------------------------|
| A | `src/reliable_clinical_benchmark/metrics/faithfulness.py` | Same module for Step-F1 and Silent Bias |
| B | `src/reliable_clinical_benchmark/metrics/sycophancy.py` | Same module for H_Ev and Turn of Flip |
| C | `src/reliable_clinical_benchmark/metrics/drift.py` | Same module for conflict and alignment |

---

## File Structure

```text
metrics/
|-- README.md
|-- metrics_summary.md              # This file
|-- VERIFICATION_FRAMEWORK.md
|-- study_a/
|   |-- faithfulness_gap.md
|   |-- silent_bias_rate.md
|   `-- step_f1.md
|-- study_b/
|   |-- evidence_hallucination.md
|   |-- sycophancy_probability.md
|   `-- turn_of_flip.md
`-- study_c/
    |-- entity_recall_decay.md
    |-- knowledge_conflict_rate.md
    `-- session_goal_alignment.md
```

---

## Quick Reference

### Start Here
1. [Metrics Reference Index](README.md)
2. [Metric Verification Protocol](VERIFICATION_FRAMEWORK.md)
3. [Scaling Overview](../scaling_summary.md)

### If You Are Updating a Metric
1. Update the metric-specific file in `study_a/`, `study_b/`, or `study_c/`
2. Ensure implementation path references still match active source files
3. Re-run verification checks in `VERIFICATION_FRAMEWORK.md`

---

## Notes

- Keep metric formulas, interpretation, and implementation paths consistent when scaling datasets or persona pools.
- Treat Study B Flip Rate as a derived analysis metric when reported outside pipeline JSON outputs.
