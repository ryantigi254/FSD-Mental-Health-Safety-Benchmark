"""Statistical utilities for metric calculation."""

import numpy as np
from typing import List, Callable, Tuple, Any

def compute_bootstrap_ci(
    data: List[Any],
    statistic: Callable[[List[Any]], float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Compute Bootstrap Confidence Interval.
    
    Args:
        data: List of items to resample.
        statistic: Function that takes a list of items and returns a float score.
        n_resamples: Number of bootstrap iterations.
        alpha: Significance level (default 0.05 for 95% CI).
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    if not data:
        return 0.0, 0.0
        
    rng = np.random.RandomState(seed)
    n = len(data)
    scores = []
    
    for _ in range(n_resamples):
        indices = rng.randint(0, n, n)
        resample = [data[i] for i in indices]
        score = statistic(resample)
        scores.append(score)
        
    scores = np.array(scores)
    lower = np.percentile(scores, 100 * (alpha / 2))
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    
    return float(lower), float(upper)
