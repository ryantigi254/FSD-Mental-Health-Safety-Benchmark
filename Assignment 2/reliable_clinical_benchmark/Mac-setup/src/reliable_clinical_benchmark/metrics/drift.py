"""
Thin wrapper that forwards imports to the canonical top-level ``metrics.drift``
module so that imports of
``reliable_clinical_benchmark.metrics.drift`` keep working.
"""

from metrics.drift import *  # noqa: F401,F403
from metrics.drift import _extract_advice  # noqa: F401


