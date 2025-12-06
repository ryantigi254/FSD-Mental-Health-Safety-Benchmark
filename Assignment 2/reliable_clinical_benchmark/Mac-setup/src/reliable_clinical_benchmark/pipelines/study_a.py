"""
Forwarder so ``reliable_clinical_benchmark.pipelines.study_a`` resolves to the
canonical top-level ``pipelines.study_a`` module.
"""

from pipelines.study_a import *  # noqa: F401,F403

