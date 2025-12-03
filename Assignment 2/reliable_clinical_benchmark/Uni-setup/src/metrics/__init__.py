"""Evaluation metrics for Studies A, B, and C."""

from .faithfulness import (
    calculate_faithfulness_gap,
    calculate_step_f1,
    calculate_silent_bias_rate,
    FaithfulnessResult,
)
from .sycophancy import (
    calculate_sycophancy_probability,
    calculate_flip_rate,
    evidence_hallucination_score,
    calculate_turn_of_flip,
    SycophancyResult,
)
from .drift import (
    compute_entity_recall_curve,
    calculate_knowledge_conflict_rate,
    calculate_continuity_score,
    DriftResult,
)

__all__ = [
    "calculate_faithfulness_gap",
    "calculate_step_f1",
    "calculate_silent_bias_rate",
    "FaithfulnessResult",
    "calculate_sycophancy_probability",
    "calculate_flip_rate",
    "evidence_hallucination_score",
    "calculate_turn_of_flip",
    "SycophancyResult",
    "compute_entity_recall_curve",
    "calculate_knowledge_conflict_rate",
    "calculate_continuity_score",
    "DriftResult",
]


