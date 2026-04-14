"""Historical NOAA settlement data ingestion via ACIS StnData API.

Fetches official daily high/low temperatures from the Applied Climate
Information System (ACIS) open API — the same source authoritative NWS
daily climate records used for Kalshi contract settlement.

ACIS endpoint: https://data.rcc-acis.org/StnData
- No API key required
- Accepts ICAO station codes directly (e.g. ``"KORD"``)
- Returns ``maxt`` (daily high °F) and ``mint`` (daily low °F) per day

Public interface::

    fetch_settlement_observations(city, start_date, end_date) -> pd.DataFrame
    save_settlement_parquet(df, output_path) -> Path

The returned DataFrame conforms to the ``daily_settlement_observation``
schema defined in ``docs/DATA_SCHEMA.md`` §5.

Usage::

    python -m deep_isobar.data.historical_noaa_ingest \\
        --city Chicago --start 2023-05-01 --end 2023-09-30 \\
        --out data/historical/settlement/chicago_2023.parquet

or::

    python src/deep_isobar/data/historical_noaa_ingest.py \\
        --city Chicago --start 2023-05-01 --end 2023-09-30
"""

from __future__ import annotations

import logging
import random
import time
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from deep_isobar.data.city_universe import get_city_profile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ACIS_URL = "https://data.rcc-acis.org/StnData"

_ACIS_ELEMENTS = [{"name": "maxt"}, {"name": "mint"}]

# ACIS special-value sentinels
_ACIS_MISSING = "M"
_ACIS_TRACE = "T"

# daily_settlement_observation schema columns (DATA_SCHEMA.md §5)
_OUTPUT_COLUMNS = [
    "city",
    "station_id",
    "target_date",
    "high_temp_f",
    "low_temp_f",
    "settlement_source",
    "settlement_report_id",
    "finalized_at_utc",
    "quality_flag",
]

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_acis_value(
    raw: str | int | float | None,
    field: str,
    date_str: str,
) -> float | None:
    """Convert an ACIS element value to float or None.

    ACIS special codes:
    - ``"M"``  — missing; returns ``None`` and logs a warning.
    - ``"T"``  — trace (precipitation only); returns ``None`` and logs a warning.
    - numeric string, int, or float — cast to ``float`` and returned.

    Args:
        raw: Raw value from ACIS response.
        field: Field name for log messages (e.g. ``"maxt"``).
        date_str: Date string for log messages (``"YYYY-MM-DD"``).

    Returns:
        Parsed float or ``None`` if the value represents missing data.
    """
    if raw is None or raw == _ACIS_MISSING:
        logger.warning("ACIS missing value for %s on %s (code M)", field, date_str)
        return None

    if raw == _ACIS_TRACE:
        logger.warning("ACIS trace value for %s on %s (code T) — treating as missing", field, date_str)
        return None

    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        logger.warning("ACIS unparseable value %r for %s on %s: %s", raw, field, date_str, exc)
        return None


def _post_with_backoff(
    url: str,
    payload: dict[str, Any],
    max_retries: int = 5,
    base_delay_s: float = 1.0,
    max_delay_s: float = 60.0,
) -> dict[str, Any]:
    """POST ``payload`` to ``url`` with exponential backoff and jitter.

    Retries on HTTP status codes 429, 500, 502, 503, 504 and on
    :class:`requests.ConnectionError` / :class:`requests.Timeout`.

    Delay formula::

        delay = min(base_delay_s * 2^attempt + random.uniform(0, 1), max_delay_s)

    Args:
        url: Target URL.
        payload: JSON-serialisable request body.
        max_retries: Total attempts before raising (default 5).
        base_delay_s: Base delay in seconds for the exponential series.
        max_delay_s: Maximum delay cap in seconds.

    Returns:
        Parsed JSON response as a dict.

    Raises:
        RuntimeError: After ``max_retries`` failed attempts.
    """
    _RETRY_STATUSES = {429, 500, 502, 503, 504}

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code in _RETRY_STATUSES:
                raise requests.HTTPError(
                    f"HTTP {resp.status_code}", response=resp
                )
            resp.raise_for_status()
            return resp.json()  # type: ignore[no-any-return]

        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"ACIS request failed after {max_retries} attempts: {exc}"
                ) from exc

            delay = min(base_delay_s * (2 ** attempt) + random.uniform(0, 1), max_delay_s)
            logger.warning(
                "ACIS attempt %d/%d failed (%s) — retrying in %.1fs",
                attempt + 1,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)

    # Unreachable — loop always raises or returns inside
    raise RuntimeError("Unexpected exit from retry loop")  # pragma: no cover


def _acis_request(
    station_id: str,
    sdate: date,
    edate: date,
) -> list[list]:
    """POST to ACIS StnData and return raw data rows.

    Each row is a list of the form ``["YYYY-MM-DD", maxt_raw, mint_raw]``
    where ``maxt_raw`` and ``mint_raw`` may be numeric strings, ``"M"``,
    or ``"T"``.

    Args:
        station_id: ICAO station code (e.g. ``"KORD"``).
        sdate: Inclusive start date.
        edate: Inclusive end date.

    Returns:
        List of ``[date_str, maxt_raw, mint_raw]`` rows from ACIS.

    Raises:
        ValueError: If ACIS returns an empty data list.
        RuntimeError: If the HTTP request fails after all retries.
    """
    payload: dict[str, Any] = {
        "sid": station_id,
        "sdate": sdate.isoformat(),
        "edate": edate.isoformat(),
        "elems": _ACIS_ELEMENTS,
    }

    logger.info(
        "Requesting ACIS StnData: station=%s  %s → %s",
        station_id,
        sdate.isoformat(),
        edate.isoformat(),
    )

    response = _post_with_backoff(_ACIS_URL, payload)

    data = response.get("data", [])
    if not data:
        raise ValueError(
            f"ACIS returned no data for station {station_id!r} "
            f"in range {sdate.isoformat()}–{edate.isoformat()}. "
            "Check that the station ID is valid and the date range is non-empty."
        )

    logger.info("ACIS returned %d daily rows", len(data))
    return data  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_settlement_observations(
    city: str,
    start_date: date,
    end_date: date,
    config_dir: str | None = None,
) -> pd.DataFrame:
    """Fetch daily high/low temperatures from ACIS for one city and date range.

    Looks up the ICAO ``station_id`` and ``settlement_source`` for *city*
    via :func:`~deep_isobar.data.city_universe.get_city_profile`, calls the
    ACIS StnData API with exponential backoff, and returns a DataFrame
    conforming to the ``daily_settlement_observation`` schema
    (``docs/DATA_SCHEMA.md`` §5).

    ACIS missing-value codes (``"M"``, ``"T"``) are stored as ``NaN`` with
    ``quality_flag = "missing"``.  Successfully parsed values receive
    ``quality_flag = "acis_verified"``.

    The columns ``settlement_report_id`` and ``finalized_at_utc`` are
    always ``None`` because ACIS does not provide them.

    Args:
        city: Exact city name as registered in ``config/cities.yaml``
            (e.g. ``"Chicago"``).  Case-sensitive.
        start_date: Inclusive start of the date range.
        end_date: Inclusive end of the date range.
        config_dir: Optional override for the config directory that
            contains ``cities.yaml``.  Defaults to
            ``<project_root>/config``.

    Returns:
        DataFrame with columns:

        - ``city`` (str)
        - ``station_id`` (str)
        - ``target_date`` (``datetime.date``)
        - ``high_temp_f`` (float, nullable)
        - ``low_temp_f`` (float, nullable)
        - ``settlement_source`` (str)
        - ``settlement_report_id`` (None)
        - ``finalized_at_utc`` (None)
        - ``quality_flag`` (str)

    Raises:
        KeyError: If *city* is not found in ``cities.yaml``.
        FileNotFoundError: If ``cities.yaml`` does not exist.
        ValueError: If ``start_date > end_date`` or ACIS returns no data.
        RuntimeError: If the ACIS request fails after all retries.

    Example::

        from datetime import date
        df = fetch_settlement_observations(
            city="Chicago",
            start_date=date(2023, 5, 1),
            end_date=date(2023, 9, 30),
        )
        print(df.shape)          # (153, 9)
        print(df.dtypes)
        print(df.head())
    """
    if start_date > end_date:
        raise ValueError(
            f"start_date must be <= end_date: "
            f"start={start_date.isoformat()}  end={end_date.isoformat()}"
        )

    profile = get_city_profile(city, config_dir=config_dir)
    # Use acis_station_id when set — it may differ from the METAR station_id
    # (e.g. Dallas: METAR=KDFW, ACIS/Kalshi settlement=CLIDFW).
    station_id = profile.acis_station_id if profile.acis_station_id else profile.station_id
    settlement_source = profile.settlement_source

    raw_rows = _acis_request(station_id, start_date, end_date)

    records: list[dict] = []
    for row in raw_rows:
        date_str: str = row[0]
        maxt_raw = row[1]
        mint_raw = row[2] if len(row) > 2 else None

        high_f = _parse_acis_value(maxt_raw, "maxt", date_str)
        low_f = _parse_acis_value(mint_raw, "mint", date_str)

        # quality_flag: "missing" if either value is absent, else "acis_verified"
        quality_flag = "missing" if (high_f is None or low_f is None) else "acis_verified"

        records.append(
            {
                "city": city,
                "station_id": station_id,
                "target_date": date.fromisoformat(date_str),
                "high_temp_f": high_f,
                "low_temp_f": low_f,
                "settlement_source": settlement_source,
                "settlement_report_id": None,
                "finalized_at_utc": None,
                "quality_flag": quality_flag,
            }
        )

    df = pd.DataFrame(records, columns=_OUTPUT_COLUMNS)

    missing_count = (df["quality_flag"] == "missing").sum()
    logger.info(
        "fetch_settlement_observations: city=%s  rows=%d  missing=%d",
        city,
        len(df),
        missing_count,
    )

    return df


def save_settlement_parquet(
    df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Write *df* to Parquet at *output_path*, creating parent dirs as needed.

    Args:
        df: DataFrame conforming to the ``daily_settlement_observation``
            schema.
        output_path: Destination file path.  Parent directories are
            created automatically.

    Returns:
        Resolved :class:`~pathlib.Path` of the written file.

    Raises:
        ValueError: If *df* is empty.

    Example::

        out = save_settlement_parquet(df, "data/historical/settlement/chicago_2023.parquet")
        print(f"Saved {len(df)} rows to {out}")
    """
    if df.empty:
        raise ValueError("Cannot save an empty DataFrame to Parquet.")

    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(path, index=False)
    logger.info("Saved %d rows to %s", len(df), path)

    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Fetch NOAA daily settlement temperatures via ACIS StnData API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--city", default="Chicago", help="City name (must match cities.yaml)")
    parser.add_argument("--start", default="2023-05-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2023-09-30", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--out",
        default="data/historical/settlement/chicago_2023.parquet",
        help="Output Parquet file path",
    )
    args = parser.parse_args()

    try:
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
    except ValueError as exc:
        print(f"ERROR: Invalid date format — {exc}", file=sys.stderr)
        sys.exit(1)

    df = fetch_settlement_observations(
        city=args.city,
        start_date=start,
        end_date=end,
    )

    out_path = save_settlement_parquet(df, args.out)

    print(f"\nSaved {len(df)} rows to {out_path}")
    print(f"Missing high_temp_f: {df['high_temp_f'].isna().sum()}")
    print(f"Missing low_temp_f:  {df['low_temp_f'].isna().sum()}")
    print()
    print(df.head(10).to_string(index=False))
