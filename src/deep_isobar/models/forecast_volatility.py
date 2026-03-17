"""Forecast Volatility — ensemble spread metrics for Deep Isobar.

Measures the spread and uncertainty across a set of model forecast values.
For the MVP, :func:`compute_forecast_volatility_score` equals the sample
standard deviation.  This is clearly documented so future enhancements
(e.g. interquartile range, weighted spread) can be added without breaking
the interface.

All functions require a non-empty list and raise :exc:`ValueError` otherwise.

Typical usage::

    from deep_isobar.models.forecast_volatility import (
        compute_forecast_std,
        compute_forecast_variance,
        compute_forecast_volatility_score,
    )

    values = [72.0, 74.5, 71.0, 76.2, 73.8]
    std   = compute_forecast_std(values)           # sample std
    var   = compute_forecast_variance(values)      # sample variance
    score = compute_forecast_volatility_score(values)  # == std for MVP
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_forecast_std(forecast_values_f: list[float]) -> float:
    """Compute the sample standard deviation of forecast temperatures.

    Uses Bessel's correction (``ddof=1``).  Returns ``0.0`` for a
    single-element list (no spread possible).

    Args:
        forecast_values_f: Forecast temperatures in °F.  Must be non-empty.

    Returns:
        Sample standard deviation as a float.

    Raises:
        ValueError: If the list is empty.
    """
    if not forecast_values_f:
        raise ValueError("forecast_values_f cannot be empty")

    if len(forecast_values_f) == 1:
        logger.debug("compute_forecast_std: single value, returning 0.0")
        return 0.0

    std_val = float(np.std(forecast_values_f, ddof=1))
    logger.debug("compute_forecast_std: n=%d std=%.4f", len(forecast_values_f), std_val)
    return std_val


def compute_forecast_variance(forecast_values_f: list[float]) -> float:
    """Compute the sample variance of forecast temperatures.

    Uses Bessel's correction (``ddof=1``).  Returns ``0.0`` for a
    single-element list.

    Args:
        forecast_values_f: Forecast temperatures in °F.  Must be non-empty.

    Returns:
        Sample variance as a float.

    Raises:
        ValueError: If the list is empty.
    """
    if not forecast_values_f:
        raise ValueError("forecast_values_f cannot be empty")

    if len(forecast_values_f) == 1:
        logger.debug("compute_forecast_variance: single value, returning 0.0")
        return 0.0

    var_val = float(np.var(forecast_values_f, ddof=1))
    logger.debug("compute_forecast_variance: n=%d var=%.4f", len(forecast_values_f), var_val)
    return var_val


def compute_forecast_volatility_score(forecast_values_f: list[float]) -> float:
    """Compute a volatility score for ensemble forecast spread.

    **MVP behaviour**: returns the sample standard deviation (identical to
    :func:`compute_forecast_std`).  Future versions may incorporate
    additional factors such as run-to-run shift history or model
    disagreement weighting.

    Args:
        forecast_values_f: Forecast temperatures in °F.  Must be non-empty.

    Returns:
        Volatility score as a float.

    Raises:
        ValueError: If the list is empty.
    """
    if not forecast_values_f:
        raise ValueError("forecast_values_f cannot be empty")

    score = compute_forecast_std(forecast_values_f)
    logger.debug("compute_forecast_volatility_score: n=%d score=%.4f", len(forecast_values_f), score)
    return score
