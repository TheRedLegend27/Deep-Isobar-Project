"""KDE Temperature Distribution — smoothed forecast distribution for Deep Isobar.

Builds a Kernel Density Estimate over ensemble forecast values and serialises
the distribution metadata into a
:class:`~deep_isobar.core.types.TemperatureDistributionSnapshot`.

The KDE bandwidth parameter (from :class:`~deep_isobar.core.types.CityProfile`)
is treated as a **multiplier** applied to scipy's default Scott bandwidth.
A value of ``1.0`` means no adjustment; ``1.4`` widens the kernel by 40 %.

Requires at least 2 forecast points (scipy constraint for gaussian_kde).

Typical usage::

    from deep_isobar.models.kde_temperature_distribution import (
        build_kde_distribution,
        build_distribution_snapshot,
    )

    kde = build_kde_distribution([72.0, 74.5, 71.0, 76.2], bandwidth=1.2)
    snapshot = build_distribution_snapshot(
        city="Chicago",
        target_date=date(2026, 7, 4),
        metric="high_temp_f",
        forecast_values_f=[72.0, 74.5, 71.0, 76.2],
        kde_bandwidth=1.2,
        min_temp_f=60.0,
        max_temp_f=95.0,
    )
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

import numpy as np
from scipy.stats import gaussian_kde

from deep_isobar.core.types import TemperatureDistributionSnapshot

logger = logging.getLogger(__name__)


def build_kde_distribution(
    forecast_values_f: list[float],
    bandwidth: float | None = None,
) -> gaussian_kde:
    """Build a KDE over forecast temperature values.

    Args:
        forecast_values_f: List of forecast temperatures in °F.
            Must contain at least 2 values.
        bandwidth: Optional bandwidth multiplier applied to scipy's Scott
            factor.  ``1.0`` leaves the default unchanged; ``1.4`` widens
            the kernel by 40 %.  Must be positive if provided.

    Returns:
        A fitted :class:`scipy.stats.gaussian_kde` object.

    Raises:
        ValueError: If fewer than 2 forecast values are provided, or if
            *bandwidth* is not positive.
    """
    if len(forecast_values_f) < 2:
        raise ValueError(
            f"At least 2 forecast points are required for KDE, "
            f"got {len(forecast_values_f)}"
        )

    if bandwidth is not None and bandwidth <= 0:
        raise ValueError(f"bandwidth must be positive, got {bandwidth}")

    data = np.array(forecast_values_f, dtype=float)
    kde = gaussian_kde(data)

    if bandwidth is not None:
        # Multiply the auto-detected Scott factor by the city-specific multiplier.
        kde.set_bandwidth(bw_method=kde.factor * bandwidth)
        logger.debug(
            "build_kde_distribution  n=%d bandwidth_multiplier=%.2f "
            "effective_factor=%.4f",
            len(forecast_values_f), bandwidth, kde.factor,
        )
    else:
        logger.debug(
            "build_kde_distribution  n=%d default_factor=%.4f",
            len(forecast_values_f), kde.factor,
        )

    return kde


def build_distribution_snapshot(
    city: str,
    target_date: date,
    metric: str,
    forecast_values_f: list[float],
    kde_bandwidth: float,
    min_temp_f: float,
    max_temp_f: float,
) -> TemperatureDistributionSnapshot:
    """Create a serialisable snapshot of the KDE distribution metadata.

    The snapshot preserves all inputs needed to reproduce the distribution
    at a later time (see :data:`~deep_isobar.core.types.TemperatureDistributionSnapshot`).

    Args:
        city: Canonical city name (e.g. ``"Chicago"``).
        target_date: Date the distribution applies to.
        metric: ``"high_temp_f"`` or ``"low_temp_f"``.
        forecast_values_f: Source forecast values used to fit the KDE.
        kde_bandwidth: Bandwidth multiplier stored for reproducibility.
        min_temp_f: Lower support bound in °F.
        max_temp_f: Upper support bound in °F.

    Returns:
        A populated :class:`~deep_isobar.core.types.TemperatureDistributionSnapshot`.
    """
    snapshot = TemperatureDistributionSnapshot(
        city=city,
        target_date=target_date,
        metric=metric,
        distribution_time_utc=datetime.now(timezone.utc),
        source_forecasts_f=list(forecast_values_f),
        kde_bandwidth=kde_bandwidth,
        min_temp_f=min_temp_f,
        max_temp_f=max_temp_f,
        methodology="gaussian_kde",
    )

    logger.info(
        "build_distribution_snapshot  city=%s target=%s metric=%s "
        "n_forecasts=%d bandwidth=%.2f range=[%.1f, %.1f]",
        city, target_date, metric,
        len(forecast_values_f), kde_bandwidth, min_temp_f, max_temp_f,
    )
    return snapshot
