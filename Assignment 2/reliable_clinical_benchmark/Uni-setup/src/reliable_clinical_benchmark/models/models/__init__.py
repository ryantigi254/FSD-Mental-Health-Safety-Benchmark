"""
Thin wrapper to expose ``reliable_clinical_benchmark.models`` as a forwarder
to the canonical top-level ``models`` package (used by tests and pipeline imports).
"""

from models.base import GenerationConfig, ModelRunner  # noqa: F401
from models import *  # noqa: F401,F403

