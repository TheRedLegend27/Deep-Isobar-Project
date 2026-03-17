"""Normal distribution probability engine for Deep Isobar.

Provides threshold-based probability calculations using the normal (Gaussian)
distribution.  These functions are the numeric core of the probability surface
and serve as the fallback when KDE is not available.

Canonical interface (from INTERFACES.md)::

    probability_ge_normal(mean_f, std_f, threshold_f) -> float  # P(T >= threshold)
    probability_le_normal(mean_f, std_f, threshold_f) -> float  # P(T <= threshold)

Both functions require ``std_f > 0`` and return values in ``[0.0, 1.0]``.
"""

from __future__ import annotations

import logging

from scipy.stats import norm

logger = logging.getLogger(__name__)


def probability_ge_normal(mean_f: float, std_f: float, threshold_f: float) -> float:
    """Compute P(T >= threshold) under a normal distribution.

    Args:
        mean_f: Distribution mean in °F.
        std_f: Distribution standard deviation in °F.  Must be > 0.
        threshold_f: Temperature threshold in °F.

    Returns:
        Probability in ``[0.0, 1.0]``.

    Raises:
        ValueError: If *std_f* is not positive.

    Example::

        prob = probability_ge_normal(mean_f=75.0, std_f=5.0, threshold_f=80.0)
        # → P(T >= 80 | μ=75, σ=5) ≈ 0.159
    """
    if std_f <= 0:
        raise ValueError(f"std_f must be positive, got {std_f}")

    prob = float(max(0.0, min(1.0, 1.0 - norm.cdf(threshold_f, loc=mean_f, scale=std_f))))
    logger.debug(
        "probability_ge_normal  mean=%.1f std=%.2f threshold=%.1f → %.4f",
        mean_f, std_f, threshold_f, prob,
    )
    return prob


def probability_le_normal(mean_f: float, std_f: float, threshold_f: float) -> float:
    """Compute P(T <= threshold) under a normal distribution.

    Args:
        mean_f: Distribution mean in °F.
        std_f: Distribution standard deviation in °F.  Must be > 0.
        threshold_f: Temperature threshold in °F.

    Returns:
        Probability in ``[0.0, 1.0]``.

    Raises:
        ValueError: If *std_f* is not positive.

    Example::

        prob = probability_le_normal(mean_f=75.0, std_f=5.0, threshold_f=70.0)
        # → P(T <= 70 | μ=75, σ=5) ≈ 0.159
    """
    if std_f <= 0:
        raise ValueError(f"std_f must be positive, got {std_f}")

    prob = float(max(0.0, min(1.0, norm.cdf(threshold_f, loc=mean_f, scale=std_f))))
    logger.debug(
        "probability_le_normal  mean=%.1f std=%.2f threshold=%.1f → %.4f",
        mean_f, std_f, threshold_f, prob,
    )
    return prob
