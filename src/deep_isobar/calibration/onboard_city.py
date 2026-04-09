#!/usr/bin/env python3
"""onboard_city.py — one-command CLI to fully onboard a new weather city.

Phases (each timed):
  1. Validate station ID against NOAA ACIS API (live check)
  2. Fetch GFS history for --history-years (parallel across years, ThreadPoolExecutor)
  3. Fetch NOAA settlement history (preflight validation)
  4. Run historical_replay → raw errors + monthly bias profile
  5. Holdout backtest → RMSE, win rate, Sharpe (run_chicago_backtest.py logic)
  6. Print summary table
  7. Prompt: Approve and write to cities.yaml? [y/N]
     (skipped with --dry-run; bypassed with --auto-approve)
  8. Save data/bias_profiles/{station_id}_monthly_profile.parquet

Usage::

    python onboard_city.py --city Dallas --station KDFW \\
                           --lat 32.897 --lon -97.040 \\
                           --timezone America/Chicago \\
                           --history-years 5

    # Preview without writing:
    python onboard_city.py ... --dry-run

    # Fully automated (CI / scheduled jobs):
    python onboard_city.py ... --auto-approve

Notes
-----
- Station coordinates are injected into ``historical_forecast_ingest._STATION_COORDS``
  at runtime so GFS byte-range fetches land at the correct grid point.
  For stations already registered there (KMDW, KJFK, KPHX, KDEN, KDFW), the
  --lat / --lon values override the module defaults for this session only.
- The city is written to cities.yaml with ``active: false``.  Enable it manually
  once you are satisfied with the paper-trade performance.
- Phase 2 pre-populates the GRIB2 byte-range cache (data/historical/forecasts/.grib_cache).
  Phase 4 (historical_replay) re-fetches via the same path and hits the cache — no
  double download of GRIB2 data.
"""

from __future__ import annotations

import argparse
import logging
import math
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Path bootstrap — supports both:
#   python src/deep_isobar/calibration/onboard_city.py
#   python -m deep_isobar.calibration.onboard_city
# ---------------------------------------------------------------------------

_FILE_PATH = Path(__file__).resolve()
_PROJECT_ROOT = _FILE_PATH.parents[3]   # …/calibration/ → src/ → deep_isobar/ → project root
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

# ---------------------------------------------------------------------------
# Logging — configure before any project import so all loggers inherit it
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("onboard_city")

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from deep_isobar.core.types import CityProfile                                 # noqa: E402
from deep_isobar.data.historical_noaa_ingest import _acis_request              # noqa: E402
import deep_isobar.data.historical_forecast_ingest as _gfs_mod                 # noqa: E402
from deep_isobar.data.historical_forecast_ingest import fetch_gfs_forecasts    # noqa: E402
from deep_isobar.data.historical_noaa_ingest import fetch_settlement_observations  # noqa: E402
from deep_isobar.calibration.historical_replay import (                        # noqa: E402
    build_monthly_profile,
    run_historical_replay,
    save_profiles,
)
from deep_isobar.research.run_chicago_backtest import _derive_calibration_params  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CITIES_YAML = _PROJECT_ROOT / "config" / "cities.yaml"
_BIAS_PROFILES_DIR = _PROJECT_ROOT / "data" / "bias_profiles"

# Neutral defaults used while calibration params are not yet known.
_NEUTRAL_VARIANCE_MULT = 1.0
_NEUTRAL_MEAN_BIAS_F = 0.0
_DEFAULT_KDE_BANDWIDTH = 1.5
_DEFAULT_TAIL_MULT = 1.0

# Holdout: last N months are reserved for out-of-sample evaluation.
_HOLDOUT_MONTHS = 12

# Win-rate threshold (°F): a day is a "win" if the calibrated error is
# within this distance from the actual value.
_WIN_THRESHOLD_F = 5.0

# Maximum year-level GFS threads — tuned for the R520 server pair.
_MAX_GFS_WORKERS = 8

_MONTH_ABBR = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fully onboard a new city into Deep Isobar in one command.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--city",        required=True,
                   help="City display name matching future cities.yaml entry (e.g. Dallas)")
    p.add_argument("--station",     required=True,
                   help="ICAO station ID (e.g. KDFW)")
    p.add_argument("--lat",         required=True, type=float,
                   help="Station latitude in decimal degrees")
    p.add_argument("--lon",         required=True, type=float,
                   help="Station longitude in decimal degrees (negative for west)")
    p.add_argument("--timezone",    required=True,
                   help="IANA timezone name (e.g. America/Chicago)")
    p.add_argument("--history-years", dest="history_years", type=int, default=5,
                   help="Years of history to fetch and replay")
    p.add_argument("--dry-run",     dest="dry_run", action="store_true",
                   help="Run all phases but skip the cities.yaml write")
    p.add_argument("--auto-approve", dest="auto_approve", action="store_true",
                   help="Write to cities.yaml without an interactive prompt")
    p.add_argument("--cache-dir",   dest="cache_dir", default=None,
                   help="GFS GRIB2 byte-range cache directory "
                        "(default: data/historical/forecasts/.grib_cache)")
    p.add_argument("--workers",     type=int, default=_MAX_GFS_WORKERS,
                   help="Number of parallel threads for year-level GFS fetch")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _elapsed(t0: float) -> str:
    """Human-readable elapsed time string from a monotonic start."""
    return f"{time.monotonic() - t0:.1f}s"


def _city_code(city: str) -> str:
    """Derive a 3-character uppercase city code from the city name."""
    return city.split()[0][:3].upper()


def _gfs_lon360(lon: float) -> float:
    """Convert decimal longitude to the GFS 0–360 convention."""
    return 360.0 + lon if lon < 0.0 else lon


# ---------------------------------------------------------------------------
# Station coordinate registration
# ---------------------------------------------------------------------------


def _register_station_coords(station_id: str, lat: float, lon: float) -> None:
    """Inject station coordinates into the GFS ingest module's lookup dict.

    ``historical_forecast_ingest._STATION_COORDS`` is keyed by ICAO ID.
    Stations absent from that dict silently fall back to KMDW, producing
    plausible-looking but geographically incorrect GFS extractions.  This
    function patches the live dict for the duration of the process.

    Args:
        station_id: ICAO station ID (e.g. ``"KDFW"``).
        lat:        Decimal latitude.
        lon:        Decimal longitude (negative for western hemisphere).
    """
    lon_360 = _gfs_lon360(lon)
    _gfs_mod._STATION_COORDS[station_id] = (lat, lon_360)
    logger.info(
        "Registered %s in _STATION_COORDS: lat=%.4f  lon_360=%.4f",
        station_id, lat, lon_360,
    )


# ---------------------------------------------------------------------------
# Provisional city profile and temp config dir
# ---------------------------------------------------------------------------


def _build_provisional_profile(args: argparse.Namespace) -> CityProfile:
    """Construct a neutral CityProfile from CLI args.

    All bias / variance fields are set to neutral values (0.0 / 1.0).
    They will be overwritten in Phase 5 once the replay errors are known.
    """
    return CityProfile(
        city=args.city,
        city_code=_city_code(args.city),
        station_id=args.station,
        timezone=args.timezone,
        settlement_source="NWS",
        variance_multiplier=_NEUTRAL_VARIANCE_MULT,
        mean_bias_correction_f=_NEUTRAL_MEAN_BIAS_F,
        kde_bandwidth=_DEFAULT_KDE_BANDWIDTH,
        tail_multiplier=_DEFAULT_TAIL_MULT,
        model_weight_gfs=None,
        model_weight_ecmwf=None,
        model_weight_nam=None,
        heat_bias_adjustment_f=0.0,
        cold_bias_adjustment_f=0.0,
    )


def _write_temp_city_yaml(profile: CityProfile) -> Path:
    """Write a minimal cities.yaml with just the new city to a temp directory.

    Returns the temp directory path.  Pass it as ``config_dir`` to ingest
    functions so they can look up the city without touching the real
    ``config/cities.yaml``.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="deep_isobar_onboard_"))
    entry: dict[str, Any] = {
        "city":                   profile.city,
        "city_code":              profile.city_code,
        "station_id":             profile.station_id,
        "timezone":               profile.timezone,
        "settlement_source":      profile.settlement_source,
        "active":                 False,
        "variance_multiplier":    profile.variance_multiplier,
        "mean_bias_correction_f": profile.mean_bias_correction_f,
        "kde_bandwidth":          profile.kde_bandwidth,
        "tail_multiplier":        profile.tail_multiplier,
    }
    with (tmp_dir / "cities.yaml").open("w", encoding="utf-8") as fh:
        yaml.dump({"cities": [entry]}, fh, default_flow_style=False, sort_keys=False)
    logger.debug("Temp cities.yaml written to %s", tmp_dir)
    return tmp_dir


# ---------------------------------------------------------------------------
# Phase 1 — Station validation
# ---------------------------------------------------------------------------


def phase1_validate_station(station_id: str) -> None:
    """Verify the station exists in NOAA ACIS by fetching the last 7 days.

    Raises ``SystemExit`` with a descriptive message if the station is not
    found or the network request fails after all retries.
    """
    t0 = time.monotonic()
    end   = date.today() - timedelta(days=1)
    start = end - timedelta(days=6)
    logger.info(
        "Phase 1 — Validating station %s via NOAA ACIS  (%s → %s)",
        station_id, start, end,
    )
    try:
        rows = _acis_request(station_id, start, end)
    except ValueError as exc:
        raise SystemExit(
            f"\nERROR  Station '{station_id}' not found in NOAA ACIS.\n"
            f"       {exc}\n"
            "       Verify the ICAO ID at: https://www.ncdc.noaa.gov/cdo-web/"
        ) from exc
    except RuntimeError as exc:
        raise SystemExit(
            f"\nERROR  NOAA ACIS request failed for station '{station_id}'.\n"
            f"       {exc}\n"
            "       Check network connectivity and retry."
        ) from exc
    logger.info(
        "Phase 1 complete — station %s valid (%d rows)  [%s]",
        station_id, len(rows), _elapsed(t0),
    )


# ---------------------------------------------------------------------------
# Phase 2 — GFS history fetch (parallel across years)
# ---------------------------------------------------------------------------


def _fetch_year_gfs(
    city: str,
    year: int,
    config_dir: str,
    cache_dir: str | None,
) -> tuple[int, pd.DataFrame]:
    """Fetch all 12 months of GFS forecasts for *year*.

    Runs entirely within one thread; months are fetched serially so each
    byte-range download does not fight other threads for the same file.

    Returns ``(year, concatenated_df)`` so the caller can log per-year
    progress from ``as_completed``.
    """
    frames: list[pd.DataFrame] = []
    for month in range(1, 13):
        try:
            df = fetch_gfs_forecasts(
                city=city,
                year=year,
                month=month,
                config_dir=config_dir,
                cache_dir=cache_dir,
            )
            if not df.empty:
                frames.append(df)
        except Exception:
            logger.warning(
                "GFS fetch failed for %d-%02d — skipping month",
                year, month, exc_info=True
            )
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return year, combined


def phase2_gfs_fetch(
    profile: CityProfile,
    years: list[int],
    config_dir: str,
    cache_dir: str | None,
    workers: int,
) -> None:
    """Pre-fetch GFS history for all years in parallel to warm the GRIB2 cache.

    Uses ``ThreadPoolExecutor`` with one thread per year (up to ``workers``
    max) so all year downloads proceed concurrently.  Phase 4's
    ``run_historical_replay`` re-fetches the same data but hits the cache,
    turning network I/O into fast local reads.

    Args:
        profile:    Provisional city profile (needs city name + station).
        years:      Sorted list of calendar years to fetch.
        config_dir: Path to the temp ``cities.yaml`` directory.
        cache_dir:  GRIB2 byte-range cache root (``None`` → project default).
        workers:    Max parallel year threads.
    """
    t0 = time.monotonic()
    n_workers = min(workers, len(years))
    logger.info(
        "Phase 2 — Fetching GFS history for %s  years %d–%d  "
        "(%d year-thread(s), %d months each)",
        profile.city, min(years), max(years), n_workers, 12,
    )

    total_rows = 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {
            ex.submit(_fetch_year_gfs, profile.city, year, config_dir, cache_dir): year
            for year in years
        }
        for fut in as_completed(futures):
            year = futures[fut]
            try:
                fetched_year, df = fut.result()
                rows = len(df)
                total_rows += rows
                logger.info("  GFS year %d: %d rows cached", fetched_year, rows)
            except Exception:
                logger.exception("Year-level GFS fetch raised for year %d", year)

    logger.info(
        "Phase 2 complete — %d total GFS rows pre-cached  [%s]",
        total_rows, _elapsed(t0),
    )


# ---------------------------------------------------------------------------
# Phase 3 — NOAA settlement history (preflight validation)
# ---------------------------------------------------------------------------


def phase3_noaa_fetch(
    profile: CityProfile,
    start_date: date,
    end_date: date,
    config_dir: str,
) -> pd.DataFrame:
    """Fetch NOAA settlement history and return the DataFrame.

    Running this before Phase 4 surfaces missing / invalid station data
    early so the expensive historical replay is never started on bad input.
    Phase 4's ``run_historical_replay`` re-fetches the same data internally;
    the duplicate call is fast (one POST to ACIS StnData).

    Raises ``SystemExit`` if ACIS returns no usable data.
    """
    t0 = time.monotonic()
    logger.info(
        "Phase 3 — Fetching NOAA settlement history for %s  (%s → %s)",
        profile.city, start_date, end_date,
    )
    try:
        df = fetch_settlement_observations(
            city=profile.city,
            start_date=start_date,
            end_date=end_date,
            config_dir=config_dir,
        )
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(
            f"\nERROR  NOAA fetch failed for {profile.city} ({profile.station_id}).\n"
            f"       {exc}"
        ) from exc

    good = (df["quality_flag"] == "acis_verified").sum()
    if good == 0:
        raise SystemExit(
            f"\nERROR  NOAA returned 0 verified observations for {profile.city}.\n"
            "       The station may not have data for the requested date range."
        )

    logger.info(
        "Phase 3 complete — %d rows, %d verified  [%s]",
        len(df), good, _elapsed(t0),
    )
    return df


# ---------------------------------------------------------------------------
# Phase 4 — Historical replay + bias profile
# ---------------------------------------------------------------------------


def phase4_replay(
    profile: CityProfile,
    start_date: date,
    end_date: date,
    config_dir: str,
    cache_dir: str | None,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run ``historical_replay.run_historical_replay`` and build monthly profile.

    Phase 2's cache ensures GFS fetches inside this call are mostly disk
    reads.  NOAA is re-fetched (fast) from ACIS.

    Returns:
        ``(raw_errors_df, monthly_profile_df)``
    """
    t0 = time.monotonic()
    logger.info(
        "Phase 4 — Historical replay for %s  (%s → %s)  workers=%d",
        profile.city, start_date, end_date, workers,
    )
    raw_errors = run_historical_replay(
        city_profile=profile,
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
        config_dir=config_dir,
        workers=workers,
    )
    monthly = build_monthly_profile(raw_errors)
    logger.info(
        "Phase 4 complete — %d error records, %d monthly rows  [%s]",
        len(raw_errors), len(monthly), _elapsed(t0),
    )
    return raw_errors, monthly


# ---------------------------------------------------------------------------
# Phase 5 — Holdout backtest (run_chicago_backtest.py logic, parameterized)
# ---------------------------------------------------------------------------


def phase5_holdout(raw_errors: pd.DataFrame) -> dict[str, Any]:
    """Split raw errors into calibration / holdout windows and compute metrics.

    Split:
        Calibration — all dates except the last ``_HOLDOUT_MONTHS`` calendar months.
        Holdout     — the most recent ``_HOLDOUT_MONTHS`` calendar months.

    Calibration params (``mean_bias_correction_f``, ``variance_multiplier``) are
    derived from the calibration window using ``_derive_calibration_params`` from
    ``run_chicago_backtest.py`` (same function used in the Chicago / NYC backtests).

    Metrics computed on the holdout window:
        RMSE          — root-mean-squared error after applying calibration.
        Win rate      — fraction of dates where |calibrated_error| ≤ ``_WIN_THRESHOLD_F``
                        (5 °F = one effective-std floor), matching the boundary used
                        in the trading loop.
        Sharpe (raw)  — directional bet: for each date the model predicts whether the
                        high is above or below the holdout mean; +1 if correct, −1 if
                        wrong; Sharpe = mean(pnl) / std(pnl).

    Returns a dict with keys:
        calibration_start, calibration_end, holdout_start, holdout_end,
        calib_n, holdout_n,
        calib_rmse, holdout_rmse,
        win_rate, sharpe_raw,
        mean_bias_correction_f, variance_multiplier
    """
    t0 = time.monotonic()
    logger.info(
        "Phase 5 — Holdout backtest  (last %d months held out)", _HOLDOUT_MONTHS,
    )

    _empty: dict[str, Any] = {
        "calibration_start": None, "calibration_end": None,
        "holdout_start": None,     "holdout_end": None,
        "calib_n": 0,              "holdout_n": 0,
        "calib_rmse": float("nan"), "holdout_rmse": float("nan"),
        "win_rate": float("nan"),  "sharpe_raw": float("nan"),
        "mean_bias_correction_f": 0.0,
        "variance_multiplier": 1.0,
    }

    if raw_errors.empty:
        logger.warning("Phase 5: empty raw_errors — returning empty metrics")
        return _empty

    df = raw_errors.copy()
    if pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = df["date"].dt.date
    df = df.sort_values("date").reset_index(drop=True)

    # Cutoff: last calendar day that falls outside the holdout window.
    # Use exact month arithmetic, not timedelta(days=N*30) which drifts by ~5 days/year.
    last_date: date = df["date"].iloc[-1]
    cutoff_year  = last_date.year
    cutoff_month = last_date.month - _HOLDOUT_MONTHS
    while cutoff_month <= 0:
        cutoff_month += 12
        cutoff_year  -= 1
    cutoff = last_date.replace(year=cutoff_year, month=cutoff_month)

    calib_df   = df[df["date"] <= cutoff].copy()
    holdout_df = df[df["date"] > cutoff].copy()

    if calib_df.empty or holdout_df.empty:
        logger.warning(
            "Phase 5: insufficient data for split  "
            "(calib=%d, holdout=%d) — skipping",
            len(calib_df), len(holdout_df),
        )
        return _empty

    # ── Derive calibration params from calibration window ────────────────────
    calib_records = (
        calib_df
        .rename(columns={
            "raw_mean_f":  "ensemble_mean_f",
            "raw_std_f":   "ensemble_std_f",
            "actual_f":    "true_high_f",
        })[["ensemble_mean_f", "ensemble_std_f", "true_high_f"]]
        .to_dict("records")
    )
    logger.info("── _derive_calibration_params (calibration window) ──")
    result = _derive_calibration_params(calib_records)
    mean_bias_correction_f: float = float(result[0])
    variance_multiplier: float    = float(result[1])

    # ── Calibration window RMSE (raw, uncorrected) ───────────────────────────
    calib_rmse = math.sqrt((calib_df["error_f"] ** 2).mean())

    # ── Apply calibration to holdout window ──────────────────────────────────
    holdout_df["calibrated_mean_f"]  = holdout_df["raw_mean_f"] + mean_bias_correction_f
    holdout_df["calibrated_error_f"] = (
        holdout_df["calibrated_mean_f"] - holdout_df["actual_f"]
    )

    holdout_rmse = math.sqrt((holdout_df["calibrated_error_f"] ** 2).mean())

    # Win rate: |calibrated_error| ≤ _WIN_THRESHOLD_F
    wins     = (holdout_df["calibrated_error_f"].abs() <= _WIN_THRESHOLD_F).sum()
    win_rate = wins / len(holdout_df)

    # Sharpe: directional bet against holdout mean daily high
    mean_actual = holdout_df["actual_f"].mean()
    signals  = (holdout_df["calibrated_mean_f"] > mean_actual).astype(float) * 2 - 1
    outcomes = (holdout_df["actual_f"]           > mean_actual).astype(float) * 2 - 1
    pnl      = signals * outcomes   # +1 correct, −1 wrong
    pnl_std  = pnl.std()
    sharpe_raw = float(pnl.mean() / pnl_std) if pnl_std > 0 else 0.0

    logger.info(
        "Phase 5 complete — holdout_rmse=%.2f°F  win_rate=%.1f%%  "
        "sharpe_raw=%.3f  [%s]",
        holdout_rmse, win_rate * 100, sharpe_raw, _elapsed(t0),
    )

    return {
        "calibration_start": calib_df["date"].iloc[0],
        "calibration_end":   calib_df["date"].iloc[-1],
        "holdout_start":     holdout_df["date"].iloc[0],
        "holdout_end":       holdout_df["date"].iloc[-1],
        "calib_n":           len(calib_df),
        "holdout_n":         len(holdout_df),
        "calib_rmse":        calib_rmse,
        "holdout_rmse":      holdout_rmse,
        "win_rate":          win_rate,
        "sharpe_raw":        sharpe_raw,
        "mean_bias_correction_f": mean_bias_correction_f,
        "variance_multiplier":    variance_multiplier,
    }


# ---------------------------------------------------------------------------
# Summary table formatter
# ---------------------------------------------------------------------------


def _fmt_float(v: float, fmt: str = ".2f", suffix: str = "") -> str:
    return f"{v:{fmt}}{suffix}" if not math.isnan(v) else "N/A"


def _fmt_date(d: Any) -> str:
    return str(d) if d is not None else "N/A"


def format_summary(
    city: str,
    station_id: str,
    monthly: pd.DataFrame,
    holdout: dict[str, Any],
) -> str:
    """Return the multi-line summary table as a string."""
    W = 66
    EQ = "═" * W
    DA = "─" * W
    lines: list[str] = []

    lines += [
        "",
        EQ,
        f"  ONBOARDING SUMMARY  {city} ({station_id})".center(W),
        EQ,
    ]

    # ── Monthly bias profile ─────────────────────────────────────────────────
    lines += [
        "",
        "  Monthly Bias Profile",
        f"  {'Month':<6}  {'Mean Bias (°F)':>14}  {'Var Mult':>8}  {'Samples':>7}",
        "  " + DA[:58],
    ]
    for _, row in monthly.iterrows():
        m       = int(row["month"])
        name    = _MONTH_ABBR[m - 1]
        bias_f  = row["mean_bias_f"]
        vm      = row["variance_multiplier"]
        n       = int(row["sample_count"])
        bias_s  = f"{bias_f:+.3f}" if not math.isnan(bias_f) else "   N/A"
        vm_s    = f"{vm:.3f}"      if not math.isnan(vm)      else "   N/A"
        lines.append(f"  {name:<6}  {bias_s:>14}  {vm_s:>8}  {n:>7}")

    # ── Backtest statistics ───────────────────────────────────────────────────
    lines += [
        "",
        "  " + DA[:58],
        "  Backtest Statistics (run_chicago_backtest.py logic)",
        "  " + DA[:58],
        f"  Calibration window : "
        f"{_fmt_date(holdout['calibration_start'])} → "
        f"{_fmt_date(holdout['calibration_end'])}  "
        f"({holdout['calib_n']} days)",
        f"  Holdout window     : "
        f"{_fmt_date(holdout['holdout_start'])} → "
        f"{_fmt_date(holdout['holdout_end'])}  "
        f"({holdout['holdout_n']} days)",
        f"  Calibration RMSE   : {_fmt_float(holdout['calib_rmse'], '.2f', '°F')}",
        f"  Holdout RMSE       : {_fmt_float(holdout['holdout_rmse'], '.2f', '°F')}",
        f"  Win rate (±5 °F)   : {_fmt_float(holdout['win_rate'] * 100, '.1f', '%')}",
        f"  Sharpe (raw, dir.) : {_fmt_float(holdout['sharpe_raw'], '.3f')}",
        "",
        "  Derived calibration params (to be written to cities.yaml):",
        f"    mean_bias_correction_f : "
        f"{holdout['mean_bias_correction_f']:+.4f} °F",
        f"    variance_multiplier    : "
        f"{holdout['variance_multiplier']:.4f}",
        EQ,
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# cities.yaml helpers
# ---------------------------------------------------------------------------


def _city_exists(city_name: str) -> bool:
    """Return True if *city_name* already appears in config/cities.yaml."""
    if not _CITIES_YAML.exists():
        return False
    with _CITIES_YAML.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return any(c.get("city") == city_name for c in data.get("cities", []))


def _build_city_entry(args: argparse.Namespace, holdout: dict[str, Any]) -> dict[str, Any]:
    """Build the cities.yaml dict for the new city using derived calibration params."""
    return {
        "city":                   args.city,
        "city_code":              _city_code(args.city),
        "timezone":               args.timezone,
        "station_id":             args.station,
        "settlement_source":      "NWS",
        "active":                 False,
        "variance_multiplier":    round(holdout["variance_multiplier"], 4),
        "mean_bias_correction_f": round(holdout["mean_bias_correction_f"], 4),
        "kde_bandwidth":          _DEFAULT_KDE_BANDWIDTH,
        "tail_multiplier":        _DEFAULT_TAIL_MULT,
        "model_weight_gfs":       0.35,
        "model_weight_ecmwf":     0.45,
        "model_weight_nam":       0.20,
        "heat_bias_adjustment_f": 0.0,
        "cold_bias_adjustment_f": 0.0,
    }


def _entry_to_yaml_text(entry: dict[str, Any]) -> str:
    """Render a city entry dict as a YAML list-item block (2-space indent).

    Appends manually rather than using ``yaml.dump`` on the whole file so
    that existing entries and inline comments are preserved.
    """
    lines: list[str] = [f"  - city: {entry['city']}"]
    skip = {"city"}
    for key, val in entry.items():
        if key in skip:
            continue
        if isinstance(val, bool):
            lines.append(f"    {key}: {str(val).lower()}")
        elif isinstance(val, float):
            # Avoid scientific notation for small floats
            lines.append(f"    {key}: {val!r}")
        elif val is None:
            lines.append(f"    {key}: null")
        else:
            lines.append(f"    {key}: {val}")
    return "\n".join(lines) + "\n"


def _append_city_to_yaml(entry: dict[str, Any]) -> None:
    """Append *entry* to config/cities.yaml preserving all existing content."""
    block = "\n" + _entry_to_yaml_text(entry)
    with _CITIES_YAML.open("a", encoding="utf-8") as fh:
        fh.write(block)
    logger.info(
        "Appended %s to %s  (active: false)",
        entry["city"], _CITIES_YAML,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = _parse_args()
    total_t0 = time.monotonic()

    # Date range for history fetch
    end_date   = date.today() - timedelta(days=1)
    start_date = date(end_date.year - args.history_years, end_date.month, 1)
    years      = list(range(start_date.year, end_date.year + 1))

    logger.info(
        "Onboarding city=%r  station=%s  lat=%.4f  lon=%.4f  "
        "tz=%s  history=%d years  dry_run=%s  auto_approve=%s",
        args.city, args.station, args.lat, args.lon,
        args.timezone, args.history_years,
        args.dry_run, args.auto_approve,
    )
    logger.info("Date range: %s → %s  (%d calendar years)", start_date, end_date, len(years))

    # ── Phase 1: Validate station (live NOAA ACIS check) ─────────────────────
    phase1_validate_station(args.station)

    # ── Setup: patch GFS coords + build temp config dir ───────────────────────
    _register_station_coords(args.station, args.lat, args.lon)
    profile  = _build_provisional_profile(args)
    tmp_dir  = _write_temp_city_yaml(profile)
    cfg_dir  = str(tmp_dir)

    try:
        # ── Phase 2: GFS fetch (parallel across years, R520 ThreadPoolExecutor) ──
        phase2_gfs_fetch(
            profile=profile,
            years=years,
            config_dir=cfg_dir,
            cache_dir=args.cache_dir,
            workers=args.workers,
        )

        # ── Phase 3: NOAA settlement fetch (preflight validation) ─────────────
        phase3_noaa_fetch(
            profile=profile,
            start_date=start_date,
            end_date=end_date,
            config_dir=cfg_dir,
        )

        # ── Phase 4: Historical replay + monthly bias profile ─────────────────
        raw_errors, monthly = phase4_replay(
            profile=profile,
            start_date=start_date,
            end_date=end_date,
            config_dir=cfg_dir,
            cache_dir=args.cache_dir,
            workers=args.workers,
        )

        # ── Phase 5: Holdout backtest ─────────────────────────────────────────
        holdout = phase5_holdout(raw_errors)

    finally:
        # Always clean up the temp directory (yaml file only, not the cache)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Summary table ─────────────────────────────────────────────────────────
    summary = format_summary(args.city, args.station, monthly, holdout)
    print(summary)

    total_elapsed = time.monotonic() - total_t0
    logger.info("All 5 phases complete in %.1f s total", total_elapsed)

    # ── Save bias profile parquet (always, even in dry-run) ───────────────────
    if not raw_errors.empty:
        raw_path, profile_path = save_profiles(
            station_id=args.station,
            raw_errors_df=raw_errors,
            monthly_profile_df=monthly,
            output_dir=_BIAS_PROFILES_DIR,
        )
        print(f"  Bias profiles saved:")
        print(f"    Raw errors     → {raw_path}")
        print(f"    Monthly profile→ {profile_path}")
    else:
        logger.warning("No error records produced — bias profile parquets not saved.")

    # ── Write to cities.yaml ──────────────────────────────────────────────────
    if args.dry_run:
        logger.info("--dry-run: skipping cities.yaml write.")
        return 0

    if _city_exists(args.city):
        logger.warning(
            "%s already exists in cities.yaml — skipping append.", args.city,
        )
        return 0

    city_entry = _build_city_entry(args, holdout)

    print("─" * 66)
    print("  Preview — cities.yaml block to be appended:")
    print("─" * 66)
    print(_entry_to_yaml_text(city_entry))

    if args.auto_approve:
        logger.info("--auto-approve: writing without prompt.")
        approve = True
    else:
        try:
            ans = input("  Approve and write to cities.yaml? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            logger.info("Input cancelled — cities.yaml unchanged.")
            return 0
        approve = ans == "y"

    if approve:
        _append_city_to_yaml(city_entry)
        print(
            f"\n  Written to {_CITIES_YAML}\n"
            "  active: false — set active: true manually to enable paper trading."
        )
    else:
        logger.info("Not approved — cities.yaml unchanged.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
