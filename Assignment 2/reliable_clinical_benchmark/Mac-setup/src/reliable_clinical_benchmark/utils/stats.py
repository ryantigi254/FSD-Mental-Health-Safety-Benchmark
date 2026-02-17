"""Statistical utilities for confidence intervals."""

import numpy as np
from typing import List, Tuple, Callable
import logging

logger = logging.getLogger(__name__)


def bootstrap_confidence_interval(
    data: List[float],
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    statistic_fn: Callable = np.mean,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.

    Non-parametric bootstrap CI (Efron & Tibshirani, 1993) to provide
    error bars on metrics for publication-quality results. This technique
    resamples the data with replacement to estimate the sampling distribution
    without assuming a specific parametric form.

    Reference: Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap.
    """
    if not data:
        return 0.0, 0.0, 0.0

    data = np.array(data)
    n = len(data)

    point_estimate = statistic_fn(data)

    bootstrap_stats = []
    for _ in range(n_iterations):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_fn(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_stats, lower_percentile)
    upper_bound = np.percentile(bootstrap_stats, upper_percentile)

    logger.info(
        f"Bootstrap CI ({confidence_level*100}%): "
        f"{point_estimate:.3f} [{lower_bound:.3f}, {upper_bound:.3f}]"
    )

    return point_estimate, lower_bound, upper_bound

