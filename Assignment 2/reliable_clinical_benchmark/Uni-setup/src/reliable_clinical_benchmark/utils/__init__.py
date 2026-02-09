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
from .worker_runtime import (
    is_lmstudio_runner,
    resolve_worker_count,
    append_jsonl_with_retry,
    iter_threaded_results,
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
    "is_lmstudio_runner",
    "resolve_worker_count",
    "append_jsonl_with_retry",
    "iter_threaded_results",
]

if MedicalNER is not None:
    __all__.insert(2, "MedicalNER")
