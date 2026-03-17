"""Forecast Generation — per-model point forecasts for Deep Isobar.

Produces :class:`~deep_isobar.core.types.ForecastPoint` instances by ingesting
real forecast data from the **Open-Meteo** API (free, no API key required).
Falls back to deterministic stub values when live ingestion is unavailable.

Live-mode data source
---------------------
Open-Meteo (https://open-meteo.com) provides GFS and ECMWF daily forecasts
directly.  NAM is proxied through Open-Meteo's ``best_match`` composite,
which selects the highest-resolution available model for the location — in the
US this is typically HRRR/RAP/GFS, a reasonable NAM substitute.

Daily aggregation
-----------------
Open-Meteo's ``temperature_2m_max`` / ``temperature_2m_min`` daily variables
are aggregated in the **city's local timezone**.  This matches how the NWS
computes daily highs and lows for settlement purposes.

Fallback mode
-------------
Live ingestion is bypassed and stub values are returned when **any** of the
following is true:

- The environment variable ``FORECAST_STUB_MODE`` is set to ``1``, ``true``,
  or ``yes``.
- The city has no entry in the internal coordinates table (``_CITY_COORDS``).
- The model name has no Open-Meteo mapping (``_OPEN_METEO_MODELS``).
- The Open-Meteo HTTP request fails for any reason.
- The target date is outside the 16-day forecast horizon.

Fallback values are deterministic (SHA-256 hash of the inputs) and are
logged clearly so they are never mistaken for real data.

Extending to new cities or models
----------------------------------
1. Add the city to ``_CITY_COORDS`` (lat, lon, IANA timezone).
2. Add the model to ``_OPEN_METEO_MODELS`` with its Open-Meteo model code.
   See https://open-meteo.com/en/docs for the full list of supported models.

Typical usage::

    from datetime import date, datetime, timezone
    from deep_isobar.models.forecast_generation import (
        fetch_model_forecast,
        fetch_forecasts_for_city,
        register_forecast_run,
    )

    # Single model forecast (live if available, otherwise stub)
    point = fetch_model_forecast(
        city="Chicago",
        station_id="KORD",
        model_name="GFS",
        run_time_utc=datetime(2026, 3, 16, 12, tzinfo=timezone.utc),
        target_date=date(2026, 3, 17),
        metric="high_temp_f",
    )
    print(point.forecast_value_f, point.source_name)

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
        source_name="open-meteo",
    )
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import date, datetime, timezone
from typing import Final

import requests

from deep_isobar.core.types import ForecastPoint
from deep_isobar.data.city_universe import get_city_profile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_METRICS: Final[set[str]] = {"high_temp_f", "low_temp_f"}

# Stub fallback parameters
_STUB_BASE_TEMP_F: Final[float] = 55.0
_STUB_RANGE_F: Final[float] = 40.0  # stub values span [base, base + range]

# HTTP
_OPEN_METEO_BASE_URL: Final[str] = "https://api.open-meteo.com/v1/forecast"
_HTTP_TIMEOUT_SECONDS: Final[int] = 10

# Open-Meteo model codes.
# NAM is not a distinct Open-Meteo model; 'best_match' selects the
# highest-resolution NOAA composite available for the location (HRRR/RAP/GFS),
# which closely mirrors NAM guidance for US cities.
_OPEN_METEO_MODELS: Final[dict[str, str]] = {
    "GFS": "gfs_seamless",
    "ECMWF": "ecmwf_ifs04",
    "NAM": "best_match",
}

# City coordinates (lat, lon, IANA timezone) used to query Open-Meteo.
# Coordinates are centred near the NWS settlement station for each city.
# Keys must match the 'city' field in config/cities.yaml exactly.
_CITY_COORDS: Final[dict[str, tuple[float, float, str]]] = {
    "Chicago":  (41.78, -87.75,  "America/Chicago"),
    "New York": (40.63, -73.78,  "America/New_York"),
    "Phoenix":  (33.43, -112.01, "America/Phoenix"),
    "Denver":   (39.86, -104.67, "America/Denver"),
    "Dallas":   (32.90, -97.04,  "America/Chicago"),
}

# Source name strings written into ForecastPoint.source_name.
_SOURCE_LIVE: Final[str] = "open-meteo"
_SOURCE_STUB: Final[str] = "stub"

# Environment variable: set to "1", "true", or "yes" to bypass all live
# network calls and always return deterministic stub values.
_STUB_MODE_ENV_VAR: Final[str] = "FORECAST_STUB_MODE"


# ---------------------------------------------------------------------------
# Private helpers — kept private, exported only for testing
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
    fraction = int(digest[:8], 16) / 0xFFFFFFFF
    value = _STUB_BASE_TEMP_F + fraction * _STUB_RANGE_F
    return round(value, 1)


def _is_stub_mode() -> bool:
    """Return ``True`` if stub mode is forced via environment variable.

    Checks the ``FORECAST_STUB_MODE`` environment variable.  Accepted
    truthy values (case-insensitive): ``"1"``, ``"true"``, ``"yes"``.
    """
    return os.environ.get(_STUB_MODE_ENV_VAR, "").lower() in ("1", "true", "yes")


def _fetch_open_meteo(
    lat: float,
    lon: float,
    timezone_name: str,
    model_code: str,
    target_date: date,
    metric: str,
) -> float | None:
    """Fetch a single daily temperature value from the Open-Meteo forecast API.

    Requests ``temperature_2m_max`` (for ``high_temp_f``) or
    ``temperature_2m_min`` (for ``low_temp_f``) aggregated in the city's
    local timezone, which matches NWS daily high/low settlement periods.

    Args:
        lat: Latitude of the location.
        lon: Longitude of the location.
        timezone_name: IANA timezone string (e.g. ``"America/Chicago"``).
        model_code: Open-Meteo model identifier (e.g. ``"gfs_seamless"``).
        target_date: The calendar date whose value is needed.
        metric: ``"high_temp_f"`` or ``"low_temp_f"``.

    Returns:
        Temperature in Fahrenheit, or ``None`` if the target date is not
        present in the response or the value is missing.

    Raises:
        requests.RequestException: On HTTP / network errors.
        ValueError: If Open-Meteo returns an application-level error.
    """
    daily_var = (
        "temperature_2m_max" if metric == "high_temp_f" else "temperature_2m_min"
    )
    params: dict[str, str | int | float] = {
        "latitude": lat,
        "longitude": lon,
        "daily": daily_var,
        "temperature_unit": "fahrenheit",
        "timezone": timezone_name,
        "forecast_days": 16,
        "models": model_code,
    }

    resp = requests.get(
        _OPEN_METEO_BASE_URL,
        params=params,
        timeout=_HTTP_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    data: dict = resp.json()

    # Open-Meteo signals application errors with {"error": true, "reason": "..."}
    if data.get("error"):
        raise ValueError(f"Open-Meteo API error: {data.get('reason', 'unknown')}")

    daily = data.get("daily", {})
    dates: list[str] = daily.get("time", [])
    values: list[float | None] = daily.get(daily_var, [])

    target_str = target_date.isoformat()
    for d, v in zip(dates, values):
        if d == target_str:
            return round(float(v), 1) if v is not None else None

    # target_date not within the 16-day forecast horizon
    return None


def _fetch_live_value(
    city: str,
    model_name: str,
    target_date: date,
    metric: str,
) -> float | None:
    """Attempt a live forecast retrieval from Open-Meteo.

    Looks up city coordinates from ``_CITY_COORDS`` and the model code from
    ``_OPEN_METEO_MODELS``.  Returns ``None`` (without raising) for any of:

    - City not in ``_CITY_COORDS``
    - Model not in ``_OPEN_METEO_MODELS``
    - Any network or parse error
    - Target date outside the 16-day forecast horizon

    Args:
        city: City name.
        model_name: Weather model identifier.
        target_date: Forecast target date.
        metric: ``"high_temp_f"`` or ``"low_temp_f"``.

    Returns:
        Temperature in Fahrenheit, or ``None`` on any failure.
    """
    coords = _CITY_COORDS.get(city)
    if coords is None:
        logger.debug(
            "_fetch_live_value: city=%s not in coordinates table — skipping live fetch",
            city,
        )
        return None

    model_code = _OPEN_METEO_MODELS.get(model_name.upper())
    if model_code is None:
        logger.debug(
            "_fetch_live_value: model=%s has no Open-Meteo mapping — skipping live fetch",
            model_name,
        )
        return None

    lat, lon, tz = coords
    try:
        value = _fetch_open_meteo(lat, lon, tz, model_code, target_date, metric)
        if value is not None:
            logger.debug(
                "_fetch_live_value: live %.1f°F for model=%s city=%s target=%s metric=%s",
                value, model_name, city, target_date, metric,
            )
        else:
            logger.debug(
                "_fetch_live_value: Open-Meteo returned no value for "
                "model=%s city=%s target=%s (outside forecast horizon?)",
                model_name, city, target_date,
            )
        return value
    except Exception as exc:
        logger.warning(
            "_fetch_live_value: Open-Meteo request failed for model=%s city=%s — %s: %s",
            model_name, city, type(exc).__name__, exc,
        )
        return None


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
    """Fetch a single-model point forecast for a city.

    **Live mode** (default): queries the Open-Meteo forecast API for the
    daily high or low temperature.  Live mode requires the city to appear
    in ``_CITY_COORDS`` and the model to appear in ``_OPEN_METEO_MODELS``.

    **Stub mode** (fallback): returns a deterministic value derived from a
    SHA-256 hash of the inputs.  Stub mode is used when:

    - ``FORECAST_STUB_MODE=1`` (or ``true`` / ``yes``) is set in the
      environment.
    - Live ingestion fails for any reason.

    The ``source_name`` field of the returned :class:`ForecastPoint` is
    ``"open-meteo"`` for live data and ``"stub"`` for fallback data.

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
    lead_hours = _compute_lead_hours(run_time_utc, target_date)

    forecast_value: float
    source_name: str

    if _is_stub_mode():
        logger.info(
            "fetch_model_forecast: STUB_MODE active — "
            "returning deterministic fallback for model=%s city=%s target=%s metric=%s",
            model_name, city, target_date, metric,
        )
        forecast_value = _compute_stub_forecast_value(city, model_name, target_date, metric)
        source_name = _SOURCE_STUB
    else:
        live_value = _fetch_live_value(city, model_name, target_date, metric)
        if live_value is not None:
            forecast_value = live_value
            source_name = _SOURCE_LIVE
            logger.debug(
                "fetch_model_forecast: LIVE %.1f°F model=%s city=%s target=%s metric=%s",
                forecast_value, model_name, city, target_date, metric,
            )
        else:
            forecast_value = _compute_stub_forecast_value(city, model_name, target_date, metric)
            source_name = _SOURCE_STUB
            logger.info(
                "fetch_model_forecast: FALLBACK %.1f°F (live unavailable) "
                "model=%s city=%s target=%s metric=%s",
                forecast_value, model_name, city, target_date, metric,
            )

    return ForecastPoint(
        city=city,
        station_id=station_id,
        model_name=model_name,
        run_time_utc=run_time_utc,
        target_date=target_date,
        metric=metric,
        forecast_value_f=forecast_value,
        lead_hours=lead_hours,
        source_name=source_name,
    )


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

    Each model is fetched independently; a failure for one model falls back
    to a stub value without preventing the remaining models from being fetched.

    Args:
        city: City name.
        target_date: Date the forecasts apply to.
        metric: ``"high_temp_f"`` or ``"low_temp_f"``.
        model_names: List of model identifiers.  Must not be empty.

    Returns:
        List of :class:`~deep_isobar.core.types.ForecastPoint` — one per
        model, in the same order as *model_names*.

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
    forecast run registry (see ``forecast_run_registry`` in DATA_SCHEMA.md).

    Args:
        model_name: Weather model identifier (e.g. ``"GFS"``).
        run_time_utc: UTC timestamp of the model run.
        cycle_label: Human-readable cycle label (e.g. ``"12z"``).
        source_name: Data source name (e.g. ``"open-meteo"`` or ``"stub"``).

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
