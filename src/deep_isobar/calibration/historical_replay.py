"""Historical forecast replay for bias profiling.

Replays archived GFS forecasts against NOAA settlement observations to produce
raw forecast errors and monthly bias profiles — without touching any market
data.  The output feeds the city calibration pipeline.

Public interface::

    run_historical_replay(city_profile, start_date, end_date, metric, cache_dir)
        -> pd.DataFrame  (raw errors, one row per date)

    build_monthly_profile(raw_errors_df)
        -> pd.DataFrame  (12-row monthly aggregation)

    save_profiles(station_id, raw_errors_df, monthly_profile_df, output_dir)

Usage::

    python -m deep_isobar.calibration.historical_replay \\
        --station KMDW --start 2020-01-01 --end 2024-12-31 --workers 4

WARNING — coordinate registration for new cities
-------------------------------------------------
``historical_forecast_ingest.py`` line 495 falls back to KMDW (Midway)
coordinates when a station is not present in ``_STATION_COORDS`` (defined
around line 87 of that file).  This fallback is *silent*: GFS data fetched
for an unregistered station will silently use wrong coordinates and produce
plausible-looking but incorrect forecasts.  Before adding any city other
than Chicago (KMDW/KORD), KJFK, KPHX, KDEN, or KDFW to this workflow,
confirm the station appears in ``_STATION_COORDS`` in
``src/deep_isobar/data/historical_forecast_ingest.py``.
Do NOT fix the fallback behaviour here; fix it in historical_forecast_ingest.py.
"""

from __future__ import annotations

import argparse
import calendar
import dataclasses
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from deep_isobar.core.types import CityProfile
from deep_isobar.data.city_universe import get_city_profile, load_city_profiles
from deep_isobar.data.historical_forecast_ingest import fetch_gfs_forecasts
from deep_isobar.data.historical_noaa_ingest import fetch_settlement_observations

# Import the reusable calibration primitives from the Chicago backtest driver.
# _run_window handles per-date forecast selection, ensemble construction, and
# error recording.  _derive_calibration_params computes global bias/variance
# from a collected error list — useful for whole-range diagnostics alongside
# the per-month build_monthly_profile output.
from deep_isobar.research.run_chicago_backtest import (
    _derive_calibration_params,
    _run_window,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Default output location for bias profile parquets.
_DEFAULT_PROFILE_DIR = _PROJECT_ROOT / "data" / "bias_profiles"

# Minimum ensemble std applied inside _run_window (must match the constant
# used there so _derive_calibration_params stays dimensionally consistent).
_MIN_ENSEMBLE_STD_F = 5.5

# GFS fetch defaults — same as historical_forecast_ingest defaults.
_DEFAULT_CYCLES: list[str] = ["00", "12"]
_DEFAULT_LEAD_DAYS: list[int] = [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _month_dates(year: int, month: int) -> list[date]:
    """Return every calendar date in ``year``/``month``."""
    _, n_days = calendar.monthrange(year, month)
    return [date(year, month, day) for day in range(1, n_days + 1)]


def _iter_months(start: date, end: date) -> list[tuple[int, int]]:
    """Return ``[(year, month), ...]`` for every month that overlaps [start, end]."""
    months: list[tuple[int, int]] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def _neutral_profile(base_profile: CityProfile) -> CityProfile:
    """Return a copy of *base_profile* with all bias corrections zeroed out.

    Mirrors Phase 1 of run_chicago_backtest.py exactly, with the addition of
    ``apr_cold_bias_adjustment_f`` so that seasonal adjustments are also
    suppressed for spring months.
    """
    return dataclasses.replace(
        base_profile,
        mean_bias_correction_f=0.0,
        variance_multiplier=1.0,
        heat_bias_adjustment_f=0.0,
        cold_bias_adjustment_f=0.0,
        apr_cold_bias_adjustment_f=0.0,
    )


def _fetch_month_gfs(
    city: str,
    year: int,
    month: int,
    cache_dir: str | None,
    config_dir: str | None,
) -> tuple[int, int, pd.DataFrame]:
    """Fetch GFS forecasts for one (year, month).  Returns (year, month, df).

    Exceptions from ``fetch_gfs_forecasts`` are intentionally not caught here;
    they propagate to the Future and are handled in the ``as_completed`` loop.
    """
    t0 = time.monotonic()
    df = fetch_gfs_forecasts(
        city=city,
        year=year,
        month=month,
        cycles=_DEFAULT_CYCLES,
        lead_days=_DEFAULT_LEAD_DAYS,
        cache_dir=cache_dir,
        config_dir=config_dir,
    )
    elapsed = time.monotonic() - t0
    logger.info("GFS fetch %d-%02d: %d rows in %.1f s", year, month, len(df), elapsed)
    return year, month, df


def _build_settlement_index(
    noaa_df: pd.DataFrame,
) -> dict[date, float]:
    """Return ``{target_date: high_temp_f}`` from a validated NOAA DataFrame.

    Rows with ``quality_flag == "missing"`` or NaN ``high_temp_f`` are
    excluded so that ``_run_window`` never receives a bad truth value.
    """
    good = noaa_df[
        (noaa_df["quality_flag"] != "missing")
        & noaa_df["high_temp_f"].notna()
    ].copy()
    # Normalise target_date to datetime.date
    if pd.api.types.is_datetime64_any_dtype(good["target_date"]):
        good["target_date"] = good["target_date"].dt.date
    return dict(zip(good["target_date"], good["high_temp_f"]))


def _empty_snapshots_df() -> pd.DataFrame:
    """Return an empty DataFrame shaped like a Kalshi snapshots parquet.

    _run_window requires a snapshots_df argument but never accesses it when
    ``date_thresholds`` is empty (the market evaluation branch is skipped).
    """
    return pd.DataFrame(
        columns=["timestamp_utc", "contract_id", "market_source", "best_bid", "best_ask"]
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_historical_replay(
    city_profile: CityProfile,
    start_date: date,
    end_date: date,
    metric: str = "high_temp_f",
    cache_dir: str | None = None,
    config_dir: str | None = None,
    workers: int = 4,
) -> pd.DataFrame:
    """Replay historical GFS forecasts and compute raw forecast errors.

    Fetches GFS data month-by-month (in parallel) and NOAA settlement data
    once for the full range, then evaluates each date using a neutral
    (uncorrected) city profile via ``_run_window``.

    NOTE: ``_run_window`` internally passes ``metric=METRIC`` where ``METRIC``
    is the module-level constant ``"high_temp_f"`` in run_chicago_backtest.py.
    Passing a non-default *metric* to this function will not change what
    ``_run_window`` uses internally.  This is a known limitation; extend the
    backtest driver's ``_run_window`` if multi-metric support is needed.

    Args:
        city_profile: Profile for the target city.  Loaded once by the caller
            via ``get_city_profile()`` — do not call inside a loop.
        start_date: Inclusive start of the replay window.
        end_date: Inclusive end of the replay window.
        metric: Weather metric to evaluate (default ``"high_temp_f"``).
            See note above regarding ``_run_window`` behaviour.
        cache_dir: Optional GFS byte-range cache directory passed through to
            ``fetch_gfs_forecasts``.
        config_dir: Optional config directory override for city/station YAML
            lookup.
        workers: Number of parallel threads for GFS month fetches (default 4).

    Returns:
        DataFrame with columns:
        ``date``, ``city_code``, ``station_id``, ``month``,
        ``raw_mean_f``, ``actual_f``, ``error_f``, ``raw_std_f``.
        Rows with missing GFS or NOAA data are excluded (see WARNING logs).
    """
    total_t0 = time.monotonic()
    city = city_profile.city
    city_code = city_profile.city_code
    station_id = city_profile.station_id

    neutral = _neutral_profile(city_profile)
    months = _iter_months(start_date, end_date)

    # ── Fetch NOAA settlement data (once, full range) ────────────────────────
    logger.info(
        "Fetching NOAA settlement data for %s  [%s → %s]",
        city, start_date, end_date,
    )
    try:
        noaa_df = fetch_settlement_observations(
            city=city,
            start_date=start_date,
            end_date=end_date,
            config_dir=config_dir,
        )
    except Exception:
        logger.exception("NOAA fetch failed — aborting replay")
        return pd.DataFrame(
            columns=["date", "city_code", "station_id", "month",
                     "raw_mean_f", "actual_f", "error_f", "raw_std_f"]
        )

    settlement_by_date = _build_settlement_index(noaa_df)
    logger.info("NOAA settlement index: %d usable dates", len(settlement_by_date))

    # ── Fetch GFS data per month (parallel) ─────────────────────────────────
    logger.info(
        "Fetching GFS forecasts for %d months using %d worker(s)",
        len(months), workers,
    )
    gfs_by_month: dict[tuple[int, int], pd.DataFrame] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _fetch_month_gfs,
                city,
                year,
                month,
                os.path.join(cache_dir, f"worker_{year}_{month:02d}") if cache_dir else None,
                config_dir,
            ): (year, month)
            for year, month in months
        }
        for future in as_completed(futures):
            try:
                year, month, df = future.result()
            except Exception:
                logger.error("Worker failed for %s", futures[future], exc_info=True)
                continue
            # Normalise target_date to datetime.date so _run_window comparisons work.
            if not df.empty and pd.api.types.is_datetime64_any_dtype(df["target_date"]):
                df["target_date"] = df["target_date"].dt.date
            if not df.empty and df["run_time_utc"].dt.tz is None:
                df["run_time_utc"] = df["run_time_utc"].dt.tz_localize("UTC")
            gfs_by_month[(year, month)] = df

    # Empty market-data stubs — _run_window skips the market evaluation branch
    # when date_thresholds is empty, so snapshots_df is never accessed.
    empty_snapshots = _empty_snapshots_df()
    empty_idx_map: dict[tuple, str] = {}
    empty_date_thresholds: dict[date, list[int]] = {}

    # ── Process each month serially through _run_window ─────────────────────
    all_error_records: list[dict[str, Any]] = []

    for year, month in months:
        month_t0 = time.monotonic()
        all_month_dates = _month_dates(year, month)

        # Clamp to the requested replay window.
        all_month_dates = [d for d in all_month_dates if start_date <= d <= end_date]

        # Warn for dates missing NOAA settlement observations.
        for d in all_month_dates:
            if d not in settlement_by_date:
                logger.warning(
                    "Skipping %s — no usable NOAA settlement observation "
                    "(missing or quality_flag='missing')", d
                )

        window_dates = [d for d in all_month_dates if d in settlement_by_date]
        if not window_dates:
            logger.warning(
                "%d-%02d: no usable settlement dates — skipping month entirely",
                year, month,
            )
            continue

        gfs_df = gfs_by_month.get((year, month), pd.DataFrame())
        if gfs_df.empty:
            logger.warning(
                "%d-%02d: GFS DataFrame is empty — skipping all %d date(s)",
                year, month, len(window_dates),
            )
            for d in window_dates:
                logger.warning("Skipping %s — GFS fetch returned no data for %d-%02d", d, year, month)
            continue

        # Identify which settlement dates have no GFS coverage and warn now,
        # before _run_window silently skips them at DEBUG level.
        gfs_dates = set(gfs_df["target_date"].unique())
        for d in window_dates:
            if d not in gfs_dates:
                logger.warning(
                    "Skipping %s — no GFS forecast records in %d-%02d parquet",
                    d, year, month,
                )

        _, month_error_records = _run_window(
            window_dates=window_dates,
            settlement_by_date=settlement_by_date,
            forecasts_df=gfs_df,
            snapshots_df=empty_snapshots,
            idx_map=empty_idx_map,
            date_thresholds=empty_date_thresholds,
            city_profile=neutral,
        )

        elapsed = time.monotonic() - month_t0
        logger.info(
            "%d-%02d: %d error records collected in %.1f s",
            year, month, len(month_error_records), elapsed,
        )
        all_error_records.extend(month_error_records)

    total_elapsed = time.monotonic() - total_t0
    logger.info(
        "Replay complete: %d total error records in %.1f s",
        len(all_error_records), total_elapsed,
    )

    if not all_error_records:
        logger.warning("No error records produced — returning empty DataFrame")
        return pd.DataFrame(
            columns=["date", "city_code", "station_id", "month",
                     "raw_mean_f", "actual_f", "error_f", "raw_std_f"]
        )

    # _run_window records: target_date, ensemble_mean_f, ensemble_std_f, true_high_f
    rows = []
    for rec in all_error_records:
        td: date = rec["target_date"]
        raw_mean_f: float = rec["ensemble_mean_f"]
        actual_f: float = rec["true_high_f"]
        raw_std_f: float = rec["ensemble_std_f"]
        rows.append({
            "date":       td,
            "city_code":  city_code,
            "station_id": station_id,
            "month":      td.month,
            "raw_mean_f": raw_mean_f,
            "actual_f":   actual_f,
            "error_f":    raw_mean_f - actual_f,
            "raw_std_f":  raw_std_f,
        })

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def build_monthly_profile(raw_errors_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw replay errors into a 12-row monthly bias profile.

    Uses the spec formula for variance_multiplier:
    ``std(actual_f) / mean(raw_std_f)`` per month — not the RMSE-based
    formula in ``_derive_calibration_params`` (which is suitable for a
    single global estimate).  Use ``_derive_calibration_params`` directly
    if you need the global RMSE-based estimate.

    Args:
        raw_errors_df: Output of ``run_historical_replay``.  Must contain
            columns ``month``, ``error_f``, ``actual_f``, ``raw_std_f``.

    Returns:
        DataFrame with columns:
        ``month``, ``mean_bias_f``, ``variance_multiplier``,
        ``sample_count``, ``last_updated``.
        Months with no data receive ``NaN`` for bias/variance fields.
    """
    if raw_errors_df.empty:
        logger.warning("build_monthly_profile: empty input — returning empty profile")
        return pd.DataFrame(
            columns=["month", "mean_bias_f", "variance_multiplier",
                     "sample_count", "last_updated"]
        )

    today = date.today()
    profile_rows: list[dict] = []

    for month in range(1, 13):
        group = raw_errors_df[raw_errors_df["month"] == month]
        n = len(group)
        if n == 0:
            profile_rows.append({
                "month":               month,
                "mean_bias_f":         float("nan"),
                "variance_multiplier": float("nan"),
                "sample_count":        0,
                "last_updated":        today,
            })
            continue

        # mean_bias_f: negate the mean error.
        # Positive means actual > model on average (model ran cold; correction adds warmth).
        mean_bias_f = -group["error_f"].mean()

        # variance_multiplier: actual spread / mean raw model spread.
        # ddof=1 is undefined for n=1; fall back to neutral (1.0) rather than NaN.
        std_actual = group["actual_f"].std(ddof=1) if n > 1 else 0.0
        mean_raw_std = group["raw_std_f"].clip(lower=_MIN_ENSEMBLE_STD_F).mean()
        variance_multiplier = std_actual / mean_raw_std if (mean_raw_std > 0 and n > 1) else 1.0

        profile_rows.append({
            "month":               month,
            "mean_bias_f":         mean_bias_f,
            "variance_multiplier": variance_multiplier,
            "sample_count":        n,
            "last_updated":        today,
        })

    return pd.DataFrame(profile_rows)


def save_profiles(
    station_id: str,
    raw_errors_df: pd.DataFrame,
    monthly_profile_df: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write raw errors and monthly profile DataFrames to parquet.

    Args:
        station_id: ICAO station identifier (e.g. ``"KMDW"``).
            Used as the filename prefix.
        raw_errors_df: Output of ``run_historical_replay``.
        monthly_profile_df: Output of ``build_monthly_profile``.
        output_dir: Directory for output files.  Defaults to
            ``data/bias_profiles/`` under the project root.
            Created if it does not exist.

    Returns:
        ``(raw_errors_path, monthly_profile_path)`` — resolved ``Path`` objects.
    """
    out = Path(output_dir) if output_dir is not None else _DEFAULT_PROFILE_DIR
    out.mkdir(parents=True, exist_ok=True)

    raw_path = out / f"{station_id}_raw_errors.parquet"
    profile_path = out / f"{station_id}_monthly_profile.parquet"

    raw_errors_df.to_parquet(raw_path, index=False)
    monthly_profile_df.to_parquet(profile_path, index=False)

    logger.info("Saved raw errors     → %s  (%d rows)", raw_path, len(raw_errors_df))
    logger.info("Saved monthly profile → %s  (%d rows)", profile_path, len(monthly_profile_df))

    return raw_path, profile_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Replay historical GFS forecasts against NOAA observations to build "
            "bias profiles for a given weather station."
        )
    )
    p.add_argument(
        "--station",
        required=True,
        help="ICAO station ID (e.g. KMDW).  Must match a city entry in cities.yaml.",
    )
    p.add_argument(
        "--start",
        required=True,
        type=date.fromisoformat,
        help="Replay start date (inclusive), YYYY-MM-DD.",
    )
    p.add_argument(
        "--end",
        required=True,
        type=date.fromisoformat,
        help="Replay end date (inclusive), YYYY-MM-DD.",
    )
    p.add_argument(
        "--metric",
        default="high_temp_f",
        help="Weather metric (default: high_temp_f).  See module docstring for limitations.",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="Optional GFS GRIB2 byte-range cache directory.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for parquet files (default: data/bias_profiles/).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel threads for GFS month fetches (default: 4).",
    )
    return p.parse_args()


def _find_city_for_station(station_id: str) -> str:
    """Look up the city name for a given ICAO station_id from cities.yaml."""
    profiles = load_city_profiles()
    for profile in profiles:
        if profile.station_id == station_id:
            return profile.city
    raise ValueError(
        f"Station '{station_id}' not found in cities.yaml.  "
        "Add the city entry or check the spelling."
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = _parse_args()

    # Resolve city name from station ID, then load profile once.
    city_name = _find_city_for_station(args.station)
    profile = get_city_profile(city_name)

    logger.info(
        "Historical replay: station=%s  city=%s  range=%s → %s  workers=%d",
        args.station, city_name, args.start, args.end, args.workers,
    )

    raw_errors = run_historical_replay(
        city_profile=profile,
        start_date=args.start,
        end_date=args.end,
        metric=args.metric,
        cache_dir=args.cache_dir,
        workers=args.workers,
    )

    monthly_profile = build_monthly_profile(raw_errors)

    # Log the global diagnostic from _derive_calibration_params alongside the
    # per-month output so operators can compare the two estimation approaches.
    if not raw_errors.empty:
        logger.info("── Global calibration diagnostics (_derive_calibration_params) ──")
        error_records_for_global = (
            raw_errors
            .rename(columns={
                "raw_mean_f": "ensemble_mean_f",
                "raw_std_f":  "ensemble_std_f",
                "actual_f":   "true_high_f",
            })[["ensemble_mean_f", "ensemble_std_f", "true_high_f"]]
            .to_dict("records")
        )
        _derive_calibration_params(error_records_for_global)

    logger.info("\n%s", monthly_profile.to_string(index=False))

    raw_path, profile_path = save_profiles(
        station_id=args.station,
        raw_errors_df=raw_errors,
        monthly_profile_df=monthly_profile,
        output_dir=args.output_dir,
    )

    print(f"\nRaw errors    : {raw_path}")
    print(f"Monthly profile: {profile_path}")
