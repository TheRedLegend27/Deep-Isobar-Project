"""Weather data ingestion module for Deep Isobar.

Provides functions for fetching, normalizing, and aggregating weather
station observations.  The current implementation uses a **deterministic
stub** for ``fetch_station_observations`` — no network calls are made.
This keeps the module testable and self-contained until a live data
source is wired in.

Canonical output columns
------------------------
*fetch / normalize*:
    ``timestamp_utc``, ``station_id``, ``source_name``, ``temperature_f``

*daily settlement*:
    ``city``, ``station_id``, ``target_date``,
    ``high_temp_f``, ``low_temp_f``, ``settlement_source``
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta, timezone
from typing import Final

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FETCH_COLUMNS: Final[list[str]] = [
    "timestamp_utc",
    "station_id",
    "source_name",
    "temperature_f",
]

_NORMALIZE_REQUIRED: Final[set[str]] = {
    "timestamp_utc",
    "station_id",
    "source_name",
}

_CANONICAL_COLUMNS: Final[list[str]] = [
    "timestamp_utc",
    "station_id",
    "source_name",
    "temperature_f",
]

_SETTLEMENT_OUTPUT: Final[list[str]] = [
    "city",
    "station_id",
    "target_date",
    "high_temp_f",
    "low_temp_f",
    "settlement_source",
]

# Stub tuning — baseline temp and amplitude for the sine‑wave generator.
_STUB_BASE_TEMP_F: Final[float] = 65.0
_STUB_AMPLITUDE_F: Final[float] = 15.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_station_observations(
    station_id: str,
    start_date: date,
    end_date: date,
    source_name: str = "NWS",
) -> pd.DataFrame:
    """Fetch weather observations for a station over a date range.

    .. note::

        **Stub implementation** — generates deterministic hourly rows using
        a sine‑wave temperature pattern.  Replace with a real API call when
        a live data source is available.

    Parameters
    ----------
    station_id:
        Weather station identifier (e.g. ``"KORD"``).
    start_date:
        Inclusive start date for the observation window.
    end_date:
        Inclusive end date for the observation window.
    source_name:
        Data source label.  Defaults to ``"NWS"``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``timestamp_utc``, ``station_id``,
        ``source_name``, and ``temperature_f``.

    Raises
    ------
    ValueError
        If *station_id* is empty/blank, *start_date* > *end_date*, or
        any argument has an unexpected type.
    """
    # ── Input validation ──────────────────────────────────────────────
    if not isinstance(station_id, str):
        raise ValueError(
            f"station_id must be a string, got {type(station_id).__name__}"
        )
    if not station_id.strip():
        raise ValueError("station_id must not be empty or blank")

    if not isinstance(start_date, date) or isinstance(start_date, datetime):
        raise ValueError(
            f"start_date must be a datetime.date, got {type(start_date).__name__}"
        )
    if not isinstance(end_date, date) or isinstance(end_date, datetime):
        raise ValueError(
            f"end_date must be a datetime.date, got {type(end_date).__name__}"
        )
    if start_date > end_date:
        raise ValueError(
            f"start_date ({start_date}) must not be after end_date ({end_date})"
        )

    if not isinstance(source_name, str) or not source_name.strip():
        raise ValueError("source_name must be a non-empty string")

    # ── Deterministic stub: hourly sine‑wave observations ─────────────
    logger.info(
        "STUB: generating observations for station=%s from %s to %s (source=%s)",
        station_id,
        start_date,
        end_date,
        source_name,
    )

    rows: list[dict] = []
    current_dt = datetime(
        start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc
    )
    final_dt = datetime(
        end_date.year, end_date.month, end_date.day, 23, 0, 0, tzinfo=timezone.utc
    )

    hour_index = 0
    while current_dt <= final_dt:
        # Sine wave peaking at 15:00 UTC, trough at 03:00 UTC.
        temp_f = _STUB_BASE_TEMP_F + _STUB_AMPLITUDE_F * math.sin(
            2 * math.pi * (current_dt.hour - 3) / 24
        )
        rows.append(
            {
                "timestamp_utc": current_dt.isoformat(),
                "station_id": station_id,
                "source_name": source_name,
                "temperature_f": round(temp_f, 2),
            }
        )
        current_dt += timedelta(hours=1)
        hour_index += 1

    df = pd.DataFrame(rows, columns=_FETCH_COLUMNS)
    logger.debug("Generated %d observation rows", len(df))
    return df


def normalize_weather_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw weather observations to the canonical schema.

    Ensures the output contains exactly the canonical columns
    (``timestamp_utc``, ``station_id``, ``source_name``, ``temperature_f``)
    with proper types.

    If ``temperature_f`` is absent but ``temperature_c`` is present the
    function converts Celsius → Fahrenheit automatically.

    Parameters
    ----------
    df:
        Raw observation DataFrame.

    Returns
    -------
    pd.DataFrame
        Normalized DataFrame with canonical columns.

    Raises
    ------
    ValueError
        If required columns are missing or the DataFrame is empty.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            f"Expected a pandas DataFrame, got {type(df).__name__}"
        )
    if df.empty:
        raise ValueError("Cannot normalize an empty DataFrame")

    missing = _NORMALIZE_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns for normalization: {sorted(missing)}"
        )

    result = df.copy()

    # Auto‑convert Celsius → Fahrenheit when temperature_f is absent.
    if "temperature_f" not in result.columns:
        if "temperature_c" in result.columns:
            logger.info("Converting temperature_c → temperature_f")
            result["temperature_f"] = result["temperature_c"] * 9.0 / 5.0 + 32.0
        else:
            raise ValueError(
                "DataFrame must contain 'temperature_f' or 'temperature_c'"
            )

    # Ensure timestamp_utc is proper datetime.
    result["timestamp_utc"] = pd.to_datetime(result["timestamp_utc"], utc=True)

    logger.debug("Normalized %d rows", len(result))
    return result[_CANONICAL_COLUMNS].copy()


def build_daily_settlement_observations(
    df: pd.DataFrame,
    city: str,
    station_id: str,
    settlement_source: str = "NWS",
) -> pd.DataFrame:
    """Aggregate hourly observations into daily settlement rows.

    Groups by calendar date and computes the daily high and low
    temperature.

    Parameters
    ----------
    df:
        Observations containing at least ``timestamp_utc`` and
        ``temperature_f``.
    city:
        Canonical city name (e.g. ``"Chicago"``).
    station_id:
        Station identifier (e.g. ``"KORD"``).
    settlement_source:
        Source label for the settlement.  Defaults to ``"NWS"``.

    Returns
    -------
    pd.DataFrame
        One row per day with columns ``city``, ``station_id``,
        ``target_date``, ``high_temp_f``, ``low_temp_f``,
        ``settlement_source``.

    Raises
    ------
    ValueError
        If required columns are missing, *city* or *station_id* are
        empty/blank, or the DataFrame is empty.
    """
    # ── Validate string arguments ─────────────────────────────────────
    if not isinstance(city, str) or not city.strip():
        raise ValueError("city must be a non-empty string")
    if not isinstance(station_id, str) or not station_id.strip():
        raise ValueError("station_id must be a non-empty string")
    if not isinstance(settlement_source, str) or not settlement_source.strip():
        raise ValueError("settlement_source must be a non-empty string")

    # ── Validate DataFrame ────────────────────────────────────────────
    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            f"Expected a pandas DataFrame, got {type(df).__name__}"
        )
    if df.empty:
        raise ValueError("Cannot build settlement observations from an empty DataFrame")

    if "temperature_f" not in df.columns:
        raise ValueError("temperature_f column is required")
    if "timestamp_utc" not in df.columns:
        raise ValueError("timestamp_utc column is required")

    # ── Aggregate ─────────────────────────────────────────────────────
    daily = df.copy()
    daily["target_date"] = pd.to_datetime(daily["timestamp_utc"]).dt.date

    grouped = (
        daily.groupby("target_date")["temperature_f"]
        .agg(["max", "min"])
        .reset_index()
    )
    grouped = grouped.rename(columns={"max": "high_temp_f", "min": "low_temp_f"})
    grouped["city"] = city
    grouped["station_id"] = station_id
    grouped["settlement_source"] = settlement_source

    result = grouped[_SETTLEMENT_OUTPUT].copy()
    logger.info(
        "Built %d daily settlement rows for city=%s station=%s",
        len(result),
        city,
        station_id,
    )
    return result