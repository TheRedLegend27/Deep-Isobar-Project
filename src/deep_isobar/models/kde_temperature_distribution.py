from datetime import date, datetime

import numpy as np
from scipy.stats import gaussian_kde

from deep_isobar.core.types import TemperatureDistributionSnapshot


def build_kde_distribution(
    forecast_values_f: list[float],
    bandwidth: float | None = None,
) -> gaussian_kde:
    if len(forecast_values_f) < 2:
        raise ValueError("At least 2 forecast points are required for KDE")

    data = np.array(forecast_values_f, dtype=float)
    kde = gaussian_kde(data)

    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("bandwidth must be positive")

    if bandwidth is not None:
        kde.set_bandwidth(bw_method=kde.factor * bandwidth)

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
    return TemperatureDistributionSnapshot(
        city=city,
        target_date=target_date,
        metric=metric,
        distribution_time_utc=datetime.utcnow(),
        source_forecasts_f=forecast_values_f,
        kde_bandwidth=kde_bandwidth,
        min_temp_f=min_temp_f,
        max_temp_f=max_temp_f,
        methodology="gaussian_kde",
    )