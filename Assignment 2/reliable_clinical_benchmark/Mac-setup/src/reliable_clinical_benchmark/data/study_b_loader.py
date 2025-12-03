"""
Compatibility shim for ``reliable_clinical_benchmark.data.study_b_loader``.

All concrete implementations live in the top-level ``data.study_b_loader``
module; this file simply re-exports them so that older import paths keep
working.
"""

from data.study_b_loader import *  # noqa: F401,F403


