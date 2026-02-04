"""Utility modules for NLI, NER, statistics, and logging."""

from .logging_config import setup_logging
from .stats import bootstrap_confidence_interval
try:
    from .ner import MedicalNER
except Exception:
    MedicalNER = None
from .nli import NLIModel
from .plan_components import (
    PlanComponent,
    DEFAULT_PLAN_COMPONENTS,
    classify_plan_components,
    render_plan_from_components,
    extract_recommendation_candidates,
    nli_filter_candidates,
)

__all__ = [
    "setup_logging",
    "bootstrap_confidence_interval",
    "NLIModel",
    "PlanComponent",
    "DEFAULT_PLAN_COMPONENTS",
    "classify_plan_components",
    "render_plan_from_components",
    "extract_recommendation_candidates",
    "nli_filter_candidates",
]

if MedicalNER is not None:
    __all__.insert(2, "MedicalNER")
