"""Forecast Generation — per-model point forecasts for Deep Isobar.

Creates :class:`~deep_isobar.core.types.ForecastPoint` instances for
individual weather model runs.  The first version uses deterministic
stub values (hash-based) so that outputs are reproducible while the
production data pipeline is built out.

Dependencies:
    - ``deep_isobar.core.types.ForecastPoint``
    - ``deep_isobar.data.city_universe.get_city_profile``

Typical usage::

    from datetime import date, datetime, timezone
    from deep_isobar.models.forecast_generation import (
        fetch_model_forecast,
        fetch_forecasts_for_city,
        register_forecast_run,
    )

    # Single model forecast
    point = fetch_model_forecast(
        city="Chicago",
        station_id="KORD",
        model_name="GFS",
        run_time_utc=datetime(2026, 3, 16, 12, tzinfo=timezone.utc),
        target_date=date(2026, 3, 17),
        metric="high_temp_f",
    )

    # All models for a city
    forecasts = fetch_forecasts_for_city(
        city="Chicago",
        target_date=date(2026, 3, 17),
        metric="high_temp_f",
        model_names=["GFS", "ECMWF", "NAM"],
    )

    # Register a run
    run_info = register_forecast_run(
        model_name="GFS",
        run_time_utc=datetime(2026, 3, 16, 12, tzinfo=timezone.utc),
        cycle_label="12z",
        source_name="NOAA",
    )
"""

from __future__ import annotations

import hashlib
import logging
from datetime import date, datetime, timezone
from typing import Final

from deep_isobar.core.types import ForecastPoint
from deep_isobar.data.city_universe import get_city_profile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_METRICS: Final[set[str]] = {"high_temp_f", "low_temp_f"}

# Stub forecast baseline and range (Fahrenheit).
_STUB_BASE_TEMP_F: Final[float] = 55.0
_STUB_RANGE_F: Final[float] = 40.0  # values will span base ± range/2


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_metric(metric: str) -> None:
    """Raise ``ValueError`` if *metric* is not in :data:`VALID_METRICS`.

    Args:
        metric: The metric string to validate.

    Raises:
        ValueError: If *metric* is not ``"high_temp_f"`` or ``"low_temp_f"``.
    """
    if metric not in VALID_METRICS:
        raise ValueError(
            f"Invalid metric: '{metric}'. "
            f"Must be one of {sorted(VALID_METRICS)}."
        )


def _compute_lead_hours(run_time_utc: datetime, target_date: date) -> int:
    """Compute lead hours between a model run and the target date.

    Lead hours are measured from *run_time_utc* to midnight UTC at the
    **start** of *target_date*.  A negative result (run after the target
    day begins) is clamped to ``0``.

    Args:
        run_time_utc: Timestamp of the model run.
        target_date: The forecast target date.

    Returns:
        Non-negative integer lead hours.
    """
    target_midnight = datetime(
        target_date.year, target_date.month, target_date.day,
        tzinfo=timezone.utc,
    )
    # Ensure run_time_utc is tz-aware for subtraction
    run_aware = (
        run_time_utc
        if run_time_utc.tzinfo is not None
        else run_time_utc.replace(tzinfo=timezone.utc)
    )
    delta = target_midnight - run_aware
    hours = int(delta.total_seconds() // 3600)
    return max(hours, 0)


def _compute_stub_forecast_value(
    city: str,
    model_name: str,
    target_date: date,
    metric: str,
) -> float:
    """Return a deterministic stub forecast value.

    The value is derived from a SHA-256 hash of the input fields so that
    different combinations yield different—but fully reproducible—
    temperatures.  Values fall in the range
    ``[_STUB_BASE_TEMP_F, _STUB_BASE_TEMP_F + _STUB_RANGE_F]``.

    Args:
        city: City name.
        model_name: Weather model identifier.
        target_date: Forecast target date.
        metric: Metric key (``high_temp_f`` or ``low_temp_f``).

    Returns:
        A deterministic float temperature in Fahrenheit.
    """
    seed = f"{city}|{model_name}|{target_date.isoformat()}|{metric}"
    digest = hashlib.sha256(seed.encode()).hexdigest()
    # Use the first 8 hex chars → 32-bit integer → normalised to [0, 1)
    fraction = int(digest[:8], 16) / 0xFFFFFFFF
    value = _STUB_BASE_TEMP_F + fraction * _STUB_RANGE_F
    return round(value, 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_model_forecast(
    city: str,
    station_id: str,
    model_name: str,
    run_time_utc: datetime,
    target_date: date,
    metric: str,
) -> ForecastPoint:
    """Create a single-model point forecast for a city.

    In the current stub implementation the forecast value is computed
    deterministically from the input parameters (see
    :func:`_compute_stub_forecast_value`).

    Args:
        city: City name (e.g. ``"Chicago"``).
        station_id: NWS station identifier (e.g. ``"KORD"``).
        model_name: Weather model name (e.g. ``"GFS"``).
        run_time_utc: UTC timestamp of the model run.
        target_date: Date the forecast applies to.
        metric: ``"high_temp_f"`` or ``"low_temp_f"``.

    Returns:
        A populated :class:`~deep_isobar.core.types.ForecastPoint`.

    Raises:
        ValueError: If *metric* is not a recognised value.
    """
    _validate_metric(metric)

    forecast_value = _compute_stub_forecast_value(
        city, model_name, target_date, metric,
    )
    lead_hours = _compute_lead_hours(run_time_utc, target_date)

    point = ForecastPoint(
        city=city,
        station_id=station_id,
        model_name=model_name,
        run_time_utc=run_time_utc,
        target_date=target_date,
        metric=metric,
        forecast_value_f=forecast_value,
        lead_hours=lead_hours,
        source_name=model_name,
    )

    logger.debug(
        "fetch_model_forecast  city=%s model=%s target=%s metric=%s → %.1f°F",
        city, model_name, target_date, metric, forecast_value,
    )
    return point


def fetch_forecasts_for_city(
    city: str,
    target_date: date,
    metric: str,
    model_names: list[str],
) -> list[ForecastPoint]:
    """Fetch point forecasts from every model for a single city.

    Uses :func:`~deep_isobar.data.city_universe.get_city_profile` to
    resolve the station ID, then calls :func:`fetch_model_forecast`
    for each model in *model_names*.

    Args:
        city: City name.
        target_date: Date the forecasts apply to.
        metric: ``"high_temp_f"`` or ``"low_temp_f"``.
        model_names: List of model identifiers.  Must not be empty.

    Returns:
        List of :class:`~deep_isobar.core.types.ForecastPoint` — one per
        model.

    Raises:
        ValueError: If *metric* is invalid or *model_names* is empty.
        KeyError: If *city* is not found in the city universe.
    """
    _validate_metric(metric)

    if not model_names:
        raise ValueError("model_names must not be empty.")

    profile = get_city_profile(city)
    run_time = datetime.now(timezone.utc)

    logger.info(
        "fetch_forecasts_for_city  city=%s target=%s metric=%s models=%s",
        city, target_date, metric, model_names,
    )

    return [
        fetch_model_forecast(
            city=city,
            station_id=profile.station_id,
            model_name=name,
            run_time_utc=run_time,
            target_date=target_date,
            metric=metric,
        )
        for name in model_names
    ]


def register_forecast_run(
    model_name: str,
    run_time_utc: datetime,
    cycle_label: str,
    source_name: str,
) -> dict:
    """Register metadata about an ingested forecast model run.

    Returns a dictionary suitable for logging or persisting to a
    forecast run registry.

    Args:
        model_name: Weather model identifier (e.g. ``"GFS"``).
        run_time_utc: UTC timestamp of the model run.
        cycle_label: Human-readable cycle label (e.g. ``"12z"``).
        source_name: Data source name (e.g. ``"NOAA"``).

    Returns:
        Dictionary with keys ``model_name``, ``run_time_utc``,
        ``cycle_label``, ``source_name``, ``ingest_status``, and
        ``registered_at_utc``.
    """
    registered_at = datetime.now(timezone.utc)

    record = {
        "model_name": model_name,
        "run_time_utc": run_time_utc.isoformat(),
        "cycle_label": cycle_label,
        "source_name": source_name,
        "ingest_status": "complete",
        "registered_at_utc": registered_at.isoformat(),
    }

    logger.info(
        "register_forecast_run  model=%s cycle=%s source=%s",
        model_name, cycle_label, source_name,
    )
    return record