"""
Forwarder so ``reliable_clinical_benchmark.pipelines.study_b`` resolves to the
canonical top-level ``pipelines.study_b`` module.
"""

from pipelines.study_b import *  # noqa: F401,F403

