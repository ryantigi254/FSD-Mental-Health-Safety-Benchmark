"""
Forwarder so ``reliable_clinical_benchmark.models.base`` resolves to the
canonical top-level ``models.base`` module.
"""

from models.base import *  # noqa: F401,F403

