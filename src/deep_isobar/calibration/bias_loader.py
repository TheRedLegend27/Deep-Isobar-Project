"""Bias profile loader and online updater for Deep Isobar calibration.

Manages per-station monthly bias profiles stored as parquet files under
``data/bias_profiles/``.  Provides graceful fallback to the locked
``cities.yaml`` parameters when a profile file has not been built yet.

Public interface::

    load_monthly_profile(station_id)
        -> dict[int, dict]          # keyed by month 1-12

    get_current_bias(station_id, month)
        -> tuple[float, float]      # (mean_bias_f, variance_multiplier)

    update_profile_after_settlement(station_id, date, raw_mean_f, actual_f)
        -> None

Thread-safety
-------------
Each station has a module-level ``threading.Lock`` that serialises concurrent
writes within the same process (e.g., multiple async trade-session coroutines).
Cross-process safety (R750 / R520 writing simultaneously) is handled via an
advisory ``.lock`` file in the bias_profiles directory using ``O_CREAT|O_EXCL``
exclusive creation, with a stale-lock timeout of 60 s.  All parquet writes go
through a temp-file + ``os.replace()`` dance so a crash mid-write never
corrupts the on-disk profile.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from deep_isobar.data.city_universe import load_city_profiles

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PROFILE_DIR = _PROJECT_ROOT / "data" / "bias_profiles"

# Minimum ensemble std used when variance_multiplier is recomputed from raw data
# (must match the constant in historical_replay.py / run_chicago_backtest.py).
_MIN_ENSEMBLE_STD_F = 5.5

# Advisory lock: abort if we cannot acquire within this many seconds.
_LOCK_TIMEOUT_S = 30.0
# Advisory lock: treat a lockfile older than this as stale (owner crashed).
_LOCK_STALE_S = 60.0
# Significant shift threshold for INFO logging.
_SIGNIFICANT_SHIFT_F = 0.5

# ── Per-station in-process threading locks ───────────────────────────────────

_station_locks: dict[str, threading.Lock] = {}
_locks_mutex = threading.Lock()


def _get_station_lock(station_id: str) -> threading.Lock:
    with _locks_mutex:
        if station_id not in _station_locks:
            _station_locks[station_id] = threading.Lock()
        return _station_locks[station_id]


# ── Cross-process advisory file lock ─────────────────────────────────────────


def _acquire_file_lock(profile_dir: Path, station_id: str) -> Path:
    """Create an exclusive ``.lock`` file; block until acquired or timeout.

    Returns the lock file path so the caller can release it.

    Raises:
        TimeoutError: If the lock cannot be acquired within ``_LOCK_TIMEOUT_S``.
    """
    lock_path = profile_dir / f"{station_id}.lock"
    deadline = time.monotonic() + _LOCK_TIMEOUT_S

    while time.monotonic() < deadline:
        # Remove stale lock if the owning process died.
        if lock_path.exists():
            try:
                age = time.time() - lock_path.stat().st_mtime
                if age > _LOCK_STALE_S:
                    logger.warning(
                        "Removing stale lock file %s (age %.0f s)", lock_path, age
                    )
                    lock_path.unlink(missing_ok=True)
            except OSError:
                pass

        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return lock_path
        except FileExistsError:
            time.sleep(0.05)

    raise TimeoutError(
        f"Could not acquire bias-profile lock for {station_id} "
        f"after {_LOCK_TIMEOUT_S:.0f} s"
    )


def _release_file_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink(missing_ok=True)
    except OSError:
        pass


# ── Atomic parquet write ──────────────────────────────────────────────────────


def _write_parquet_atomic(df: pd.DataFrame, target: Path) -> None:
    """Write *df* to *target* via a temp file and ``os.replace()``."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".parquet.tmp", dir=str(target.parent)
    )
    try:
        os.close(fd)
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, str(target))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ── cities.yaml fallback ──────────────────────────────────────────────────────


def _fallback_params(station_id: str) -> tuple[float, float]:
    """Return ``(mean_bias_correction_f, variance_multiplier)`` from cities.yaml.

    Returns ``(0.0, 1.0)`` if the station is not found (neutral correction).
    """
    try:
        profiles = load_city_profiles()
    except Exception:
        logger.exception(
            "Could not load cities.yaml for fallback — returning neutral params"
        )
        return 0.0, 1.0

    for p in profiles:
        if p.station_id == station_id:
            return p.mean_bias_correction_f, p.variance_multiplier

    logger.warning(
        "Station %s not found in cities.yaml — using neutral bias params (0.0, 1.0)",
        station_id,
    )
    return 0.0, 1.0


def _city_code_for_station(station_id: str) -> str:
    """Return the city_code for a station_id, or the station_id itself as fallback."""
    try:
        profiles = load_city_profiles()
        for p in profiles:
            if p.station_id == station_id:
                return p.city_code
    except Exception:
        pass
    return station_id


# ── Profile path helpers ──────────────────────────────────────────────────────


def _profile_path(station_id: str, profile_dir: Path | None = None) -> Path:
    d = profile_dir or _DEFAULT_PROFILE_DIR
    return d / f"{station_id}_monthly_profile.parquet"


def _raw_errors_path(station_id: str, profile_dir: Path | None = None) -> Path:
    d = profile_dir or _DEFAULT_PROFILE_DIR
    return d / f"{station_id}_raw_errors.parquet"


# ── Per-month profile recomputation ──────────────────────────────────────────


def _recompute_month(
    raw_df: pd.DataFrame,
    month: int,
    existing_variance_multiplier: float,
) -> dict[str, Any]:
    """Recompute bias stats for a single month from *raw_df*.

    ``variance_multiplier`` is only recalculated when enough rows have a valid
    ``raw_std_f`` value; otherwise *existing_variance_multiplier* is preserved.

    Returns a dict matching the monthly_profile schema.
    """
    group = raw_df[raw_df["month"] == month]
    n = len(group)

    if n == 0:
        return {
            "month": month,
            "mean_bias_f": float("nan"),
            "variance_multiplier": existing_variance_multiplier,
            "sample_count": 0,
            "last_updated": date.today(),
        }

    mean_bias_f = -group["error_f"].mean()

    # Only recompute variance_multiplier from matched pairs that have raw_std_f.
    # Using std_rows for both numerator and denominator keeps the populations consistent.
    std_rows = group.dropna(subset=["raw_std_f"])
    if len(std_rows) > 1:
        std_actual = std_rows["actual_f"].std(ddof=1)
        mean_raw_std = std_rows["raw_std_f"].clip(lower=_MIN_ENSEMBLE_STD_F).mean()
        variance_multiplier = (
            std_actual / mean_raw_std
            if (mean_raw_std > 0 and not math.isnan(std_actual))
            else existing_variance_multiplier
        )
    else:
        variance_multiplier = existing_variance_multiplier

    return {
        "month": month,
        "mean_bias_f": mean_bias_f,
        "variance_multiplier": variance_multiplier,
        "sample_count": n,
        "last_updated": date.today(),
    }


# ── Public API ────────────────────────────────────────────────────────────────


def load_monthly_profile(
    station_id: str,
    profile_dir: Path | None = None,
) -> dict[int, dict[str, Any]]:
    """Load the monthly bias profile for *station_id*.

    Args:
        station_id: ICAO station identifier (e.g. ``"KMDW"``).
        profile_dir: Override for the profile directory.  Defaults to
            ``data/bias_profiles/`` under the project root.

    Returns:
        Dict keyed by month integer (1–12).  Each value is a dict with keys
        ``mean_bias_f``, ``variance_multiplier``, ``sample_count``,
        ``last_updated``.  Returns an empty dict if the file does not exist.
    """
    path = _profile_path(station_id, profile_dir)
    if not path.exists():
        logger.debug("No monthly profile file for %s at %s", station_id, path)
        return {}

    try:
        df = pd.read_parquet(path)
    except Exception:
        logger.exception(
            "Failed to read monthly profile for %s from %s — returning empty",
            station_id, path,
        )
        return {}

    result: dict[int, dict[str, Any]] = {}
    for _, row in df.iterrows():
        m = int(row["month"])
        raw_count = row.get("sample_count", 0)
        result[m] = {
            "mean_bias_f":         row.get("mean_bias_f", float("nan")),
            "variance_multiplier": row.get("variance_multiplier", 1.0),
            "sample_count":        0 if (raw_count is None or (isinstance(raw_count, float) and math.isnan(raw_count))) else int(raw_count),
            "last_updated":        row.get("last_updated"),
        }
    return result


def get_current_bias(
    station_id: str,
    month: int,
    profile_dir: Path | None = None,
) -> tuple[float, float]:
    """Return ``(mean_bias_f, variance_multiplier)`` for *station_id* / *month*.

    Lookup order:
    1. Monthly profile parquet (per-month calibrated estimate).
    2. ``cities.yaml`` locked params (global, month-agnostic fallback).
    3. Neutral params ``(0.0, 1.0)`` if neither is available.

    This function never raises; all failures are logged and a safe value is
    returned so a live trading session is never interrupted.

    Args:
        station_id: ICAO station identifier.
        month: Calendar month integer (1–12).
        profile_dir: Override for the profile directory.

    Returns:
        ``(mean_bias_f, variance_multiplier)``
    """
    try:
        profile = load_monthly_profile(station_id, profile_dir)
        if profile:
            entry = profile.get(month)
            if entry is not None:
                mean_bias = entry["mean_bias_f"]
                variance_mult = entry["variance_multiplier"]
                # NaN in the profile means no data for this month; fall through.
                if not (math.isnan(mean_bias) or math.isnan(variance_mult)):
                    return float(mean_bias), float(variance_mult)
                logger.debug(
                    "Monthly profile for %s month %d has NaN values — falling back",
                    station_id, month,
                )
    except Exception:
        logger.exception(
            "Unexpected error reading monthly profile for %s month %d — falling back",
            station_id, month,
        )

    # Fallback: cities.yaml locked params.
    mean_bias_f, variance_multiplier = _fallback_params(station_id)
    logger.debug(
        "Using cities.yaml fallback for %s month %d: bias=%.4f vm=%.4f",
        station_id, month, mean_bias_f, variance_multiplier,
    )
    return mean_bias_f, variance_multiplier


def update_profile_after_settlement(
    station_id: str,
    settlement_date: date,
    raw_mean_f: float,
    actual_f: float,
    profile_dir: Path | None = None,
) -> None:
    """Append one settlement observation and recompute that month's bias stats.

    Steps (all under station-level thread + file lock):
    1. Append ``{date, station_id, month, raw_mean_f, actual_f, error_f}`` to
       ``{station_id}_raw_errors.parquet``.
    2. Load the current monthly profile (or create a skeleton if absent).
    3. Recompute *only* the affected month's row from the updated raw errors.
    4. Overwrite the monthly profile atomically.
    5. Log at INFO if ``mean_bias_f`` shifted by more than ``_SIGNIFICANT_SHIFT_F``.

    ``raw_std_f`` is not known at settlement time and is stored as ``NaN``; it
    is excluded from variance_multiplier calculations by ``_recompute_month``.

    Args:
        station_id: ICAO station identifier (e.g. ``"KMDW"``).
        settlement_date: The date the contract settled.
        raw_mean_f: Uncorrected ensemble mean forecast (°F).
        actual_f: Observed settlement temperature (°F).
        profile_dir: Override for the profile directory.
    """
    dir_ = Path(profile_dir) if profile_dir is not None else _DEFAULT_PROFILE_DIR
    dir_.mkdir(parents=True, exist_ok=True)

    with _get_station_lock(station_id):
        file_lock_path: Path | None = None
        try:
            # ── Acquire cross-process file lock ──────────────────────────────
            try:
                file_lock_path = _acquire_file_lock(dir_, station_id)
            except TimeoutError:
                logger.error(
                    "Could not acquire file lock for %s — skipping settlement update for %s",
                    station_id, settlement_date,
                )
                return

            month = settlement_date.month
            error_f = raw_mean_f - actual_f
            city_code = _city_code_for_station(station_id)

            new_row = pd.DataFrame([{
                "date":        settlement_date,
                "city_code":   city_code,
                "station_id":  station_id,
                "month":       month,
                "raw_mean_f":  raw_mean_f,
                "actual_f":    actual_f,
                "error_f":     error_f,
                "raw_std_f":   float("nan"),
            }])

            # ── 1. Append to raw errors ───────────────────────────────────────
            raw_path = _raw_errors_path(station_id, dir_)
            if raw_path.exists():
                try:
                    existing_raw = pd.read_parquet(raw_path)
                    # Deduplicate: drop any prior row for this exact date.
                    # Normalise to datetime.date for a robust comparison regardless
                    # of how parquet stored the column (date32 vs timestamp64).
                    existing_raw = existing_raw[
                        pd.to_datetime(existing_raw["date"]).dt.date != settlement_date
                    ]
                    raw_df = pd.concat([existing_raw, new_row], ignore_index=True)
                except Exception:
                    logger.exception(
                        "Failed to read existing raw errors for %s — starting fresh",
                        station_id,
                    )
                    raw_df = new_row.copy()
            else:
                raw_df = new_row.copy()

            raw_df = raw_df.sort_values("date").reset_index(drop=True)
            _write_parquet_atomic(raw_df, raw_path)

            # ── 2. Load current monthly profile ──────────────────────────────
            profile_dict = load_monthly_profile(station_id, dir_)

            # Capture previous mean_bias_f for significant-shift detection.
            prev_entry = profile_dict.get(month, {})
            prev_mean_bias = prev_entry.get("mean_bias_f", float("nan"))

            # Fall back to cities.yaml variance_multiplier if profile is empty/missing.
            fallback_vm = prev_entry.get("variance_multiplier", float("nan"))
            if not profile_dict or math.isnan(float(fallback_vm)):
                _, fallback_vm = _fallback_params(station_id)

            # ── 3. Recompute only the affected month ──────────────────────────
            updated_entry = _recompute_month(raw_df, month, fallback_vm)

            # ── 4. Merge into a full 12-row profile and write ─────────────────
            profile_rows = []
            for m in range(1, 13):
                if m == month:
                    profile_rows.append(updated_entry)
                elif m in profile_dict:
                    e = profile_dict[m]
                    profile_rows.append({
                        "month":               m,
                        "mean_bias_f":         e["mean_bias_f"],
                        "variance_multiplier": e["variance_multiplier"],
                        "sample_count":        e["sample_count"],
                        "last_updated":        e["last_updated"],
                    })
                else:
                    profile_rows.append({
                        "month":               m,
                        "mean_bias_f":         float("nan"),
                        "variance_multiplier": float("nan"),
                        "sample_count":        0,
                        "last_updated":        date.today(),
                    })

            new_profile_df = pd.DataFrame(profile_rows).sort_values("month").reset_index(drop=True)
            _write_parquet_atomic(new_profile_df, _profile_path(station_id, dir_))

            # ── 5. Log significant shifts ─────────────────────────────────────
            new_mean_bias = updated_entry["mean_bias_f"]
            if not math.isnan(prev_mean_bias) and not math.isnan(new_mean_bias):
                shift = abs(new_mean_bias - prev_mean_bias)
                if shift > _SIGNIFICANT_SHIFT_F:
                    logger.info(
                        "Bias update — %s month %d: mean_bias_f %.3f → %.3f "
                        "(shift %.3f°F, n=%d)",
                        station_id, month,
                        prev_mean_bias, new_mean_bias,
                        shift, updated_entry["sample_count"],
                    )
            else:
                logger.debug(
                    "Bias update — %s month %d: mean_bias_f=%.3f (n=%d, first estimate)",
                    station_id, month, new_mean_bias, updated_entry["sample_count"],
                )

        except Exception:
            logger.exception(
                "Unexpected error in update_profile_after_settlement "
                "for %s / %s — profile may be unchanged",
                station_id, settlement_date,
            )
        finally:
            if file_lock_path is not None:
                _release_file_lock(file_lock_path)
