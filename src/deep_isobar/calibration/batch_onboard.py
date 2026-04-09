#!/usr/bin/env python3
"""batch_onboard.py — Onboard multiple cities in parallel across CPU cores.

Each row in the input CSV maps to one ``onboard_city.py`` pipeline run executed
in its own child process.  Results are collected as futures complete, a live
progress table is printed to stdout, and a summary CSV is written to
``data/onboarding_report_{timestamp}.csv``.

Usage::

    python batch_onboard.py --cities-file cities_to_onboard.csv \\
                            --history-years 5 \\
                            --workers 8 \\
                            --auto-approve

Input CSV format::

    city,station,lat,lon,timezone
    Dallas,KDFW,32.897,-97.040,America/Chicago
    New York,KJFK,40.639,-73.778,America/New_York
    Miami,KMIA,25.796,-80.287,America/New_York

Notes
-----
- ``--workers`` controls the number of *city-level* child processes.
  Each child runs its own GFS ThreadPoolExecutor internally (``--inner-workers``
  threads, default 4).
- One bad city never stops the batch — exceptions (including ``SystemExit``)
  are caught and recorded as FAIL in the report.
- Cities that already exist in ``config/cities.yaml`` are skipped at write-time
  (the calibration still runs so you get updated metrics).
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrap — works when called as:
#   python src/deep_isobar/calibration/batch_onboard.py
#   python -m deep_isobar.calibration.batch_onboard
# ---------------------------------------------------------------------------

_FILE_PATH = Path(__file__).resolve()
_PROJECT_ROOT = _FILE_PATH.parents[3]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("batch_onboard")

# ---------------------------------------------------------------------------
# Report output directory
# ---------------------------------------------------------------------------

_REPORTS_DIR = _PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Worker — runs in a child process (must be a module-level function so
# ProcessPoolExecutor can pickle it on Windows / spawn start method)
# ---------------------------------------------------------------------------


def _onboard_city_worker(
    city: str,
    station: str,
    lat: float,
    lon: float,
    timezone: str,
    history_years: int,
    auto_approve: bool,
    inner_workers: int,
) -> dict[str, Any]:
    """Run all onboarding phases for one city.  Executes in a child process.

    Returns a result dict regardless of outcome so the parent can always
    collect metrics.  Failures are encoded as ``status="FAIL"`` with the
    exception message in ``error``.
    """
    # Re-bootstrap path (child process on Windows uses spawn — no inherited state).
    _proj = Path(__file__).resolve().parents[3]
    _src = _proj / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

    # Suppress INFO noise from deep_isobar loggers — the parent owns stdout.
    # basicConfig is a no-op here (handlers already installed by module-level
    # call), so we set the root logger level directly.
    logging.getLogger().setLevel(logging.WARNING)

    result: dict[str, Any] = {
        "city":                   city,
        "station":                station,
        "status":                 "FAIL",
        "error":                  "",
        "holdout_rmse":           float("nan"),
        "win_rate_pct":           float("nan"),
        "sharpe_raw":             float("nan"),
        "mean_bias_correction_f": float("nan"),
        "variance_multiplier":    float("nan"),
        "elapsed_s":              0.0,
    }
    t0 = time.monotonic()

    try:
        import shutil  # noqa: PLC0415
        from deep_isobar.calibration.onboard_city import (  # noqa: PLC0415
            _BIAS_PROFILES_DIR,
            _build_city_entry,
            _build_provisional_profile,
            _register_station_coords,
            _write_temp_city_yaml,
            phase1_validate_station,
            phase2_gfs_fetch,
            phase3_noaa_fetch,
            phase4_replay,
            phase5_holdout,
        )
        from deep_isobar.calibration.historical_replay import save_profiles  # noqa: PLC0415

        end_date   = date.today() - timedelta(days=1)
        start_date = date(end_date.year - history_years, end_date.month, 1)
        years      = list(range(start_date.year, end_date.year + 1))

        # Phase 1 — validate station (raises SystemExit on failure)
        phase1_validate_station(station)

        # Setup: register coordinates + build temp cities.yaml
        _register_station_coords(station, lat, lon)
        args = argparse.Namespace(
            city=city,
            station=station,
            lat=lat,
            lon=lon,
            timezone=timezone,
            history_years=history_years,
            dry_run=False,
            auto_approve=auto_approve,
            cache_dir=None,
            workers=inner_workers,
        )
        profile = _build_provisional_profile(args)
        tmp_dir = _write_temp_city_yaml(profile)
        cfg_dir = str(tmp_dir)

        try:
            # Phase 2 — pre-fetch GFS (warms GRIB2 cache for phase 4)
            phase2_gfs_fetch(
                profile=profile,
                years=years,
                config_dir=cfg_dir,
                cache_dir=None,
                workers=inner_workers,
            )

            # Phase 3 — NOAA settlement preflight (raises SystemExit if station bad)
            phase3_noaa_fetch(
                profile=profile,
                start_date=start_date,
                end_date=end_date,
                config_dir=cfg_dir,
            )

            # Phase 4 — historical replay → raw errors + monthly profile
            raw_errors, monthly = phase4_replay(
                profile=profile,
                start_date=start_date,
                end_date=end_date,
                config_dir=cfg_dir,
                cache_dir=None,
                workers=inner_workers,
            )

            # Phase 5 — holdout backtest metrics
            holdout = phase5_holdout(raw_errors)

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        # Save bias profiles (always, even if cities.yaml write is skipped)
        if not raw_errors.empty:
            save_profiles(
                station_id=station,
                raw_errors_df=raw_errors,
                monthly_profile_df=monthly,
                output_dir=_BIAS_PROFILES_DIR,
            )

        # Build the cities.yaml entry but do NOT write it here — concurrent
        # workers racing on open(file, "a") can interleave bytes and corrupt
        # the YAML.  main() collects entries and writes them serially.
        city_entry = _build_city_entry(args, holdout)

        result.update({
            "status":                 "PASS",
            "holdout_rmse":           holdout["holdout_rmse"],
            "win_rate_pct":           holdout["win_rate"] * 100,
            "sharpe_raw":             holdout["sharpe_raw"],
            "mean_bias_correction_f": holdout["mean_bias_correction_f"],
            "variance_multiplier":    holdout["variance_multiplier"],
            "yaml_entry":             city_entry,
        })

    except SystemExit as exc:
        result["error"] = str(exc).strip().lstrip("\n")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    result["elapsed_s"] = round(time.monotonic() - t0, 1)
    return result


# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------


def _load_cities_csv(path: Path) -> list[dict[str, Any]]:
    """Parse the input CSV and return a list of city dicts.

    Required columns: city, station, lat, lon, timezone.
    Raises ``SystemExit`` with a helpful message on malformed input.
    """
    required = {"city", "station", "lat", "lon", "timezone"}
    cities: list[dict[str, Any]] = []

    try:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                raise SystemExit(f"ERROR  {path} is empty.")
            missing = required - {f.strip() for f in reader.fieldnames}
            if missing:
                raise SystemExit(
                    f"ERROR  {path} is missing required columns: {sorted(missing)}\n"
                    f"       Expected: city,station,lat,lon,timezone"
                )
            for i, row in enumerate(reader, start=2):
                try:
                    cities.append({
                        "city":     row["city"].strip(),
                        "station":  row["station"].strip().upper(),
                        "lat":      float(row["lat"]),
                        "lon":      float(row["lon"]),
                        "timezone": row["timezone"].strip(),
                    })
                except (ValueError, KeyError) as exc:
                    raise SystemExit(
                        f"ERROR  Bad value on row {i} of {path}: {exc}"
                    ) from exc
    except FileNotFoundError:
        raise SystemExit(f"ERROR  Cities file not found: {path}")

    if not cities:
        raise SystemExit(f"ERROR  {path} contains no data rows.")

    return cities


# ---------------------------------------------------------------------------
# Progress table printer
# ---------------------------------------------------------------------------

_COL_CITY    = 18
_COL_STATION =  7
_COL_STATUS  =  6
_COL_RMSE    =  8
_COL_WIN     =  7
_COL_SHARPE  =  7
_COL_ELAPSED =  8

_HDR = (
    f"  {'#':>3}  "
    f"{'City':<{_COL_CITY}}  "
    f"{'Station':<{_COL_STATION}}  "
    f"{'Status':<{_COL_STATUS}}  "
    f"{'RMSE(°F)':>{_COL_RMSE}}  "
    f"{'Win%':>{_COL_WIN}}  "
    f"{'Sharpe':>{_COL_SHARPE}}  "
    f"{'Elapsed':>{_COL_ELAPSED}}"
)
_SEP = "  " + "─" * (len(_HDR) - 2)


def _print_header() -> None:
    print()
    print(_HDR)
    print(_SEP)


def _fmt_f(v: float, fmt: str) -> str:
    return f"{v:{fmt}}" if not math.isnan(v) else "—"


def _print_row(idx: int, r: dict[str, Any]) -> None:
    status_tag = "PASS" if r["status"] == "PASS" else "FAIL"
    rmse_s   = _fmt_f(r["holdout_rmse"],   ".2f")
    win_s    = _fmt_f(r["win_rate_pct"],   ".1f")
    sharpe_s = _fmt_f(r["sharpe_raw"],     ".3f")
    elapsed  = f"{r['elapsed_s']:.0f}s"
    city_s   = r["city"][:_COL_CITY]
    station_s = r["station"][:_COL_STATION]
    row = (
        f"  {idx:>3}  "
        f"{city_s:<{_COL_CITY}}  "
        f"{station_s:<{_COL_STATION}}  "
        f"{status_tag:<{_COL_STATUS}}  "
        f"{rmse_s:>{_COL_RMSE}}  "
        f"{win_s:>{_COL_WIN}}  "
        f"{sharpe_s:>{_COL_SHARPE}}  "
        f"{elapsed:>{_COL_ELAPSED}}"
    )
    if r["status"] == "FAIL" and r["error"]:
        # Truncate long error messages to keep the table readable.
        err_preview = r["error"].replace("\n", " ")[:72]
        row += f"\n        ERROR: {err_preview}"
    print(row)


def _print_footer(results: list[dict[str, Any]], elapsed_total: float) -> None:
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_fail = len(results) - n_pass
    print(_SEP)
    print(
        f"  {len(results)} cities processed  —  "
        f"{n_pass} PASS  /  {n_fail} FAIL  "
        f"(wall time {elapsed_total:.0f}s)"
    )
    print()


# ---------------------------------------------------------------------------
# Report CSV writer
# ---------------------------------------------------------------------------

_REPORT_FIELDS = [
    "city", "station", "status", "error",
    "holdout_rmse", "win_rate_pct", "sharpe_raw",
    "mean_bias_correction_f", "variance_multiplier",
    "elapsed_s",
]


def _write_report(results: list[dict[str, Any]]) -> Path:
    timestamp = date.today().strftime("%Y%m%d") + f"_{int(time.time()) % 100000:05d}"
    report_path = _REPORTS_DIR / f"onboarding_report_{timestamp}.csv"
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    def _fmt_val(v: Any) -> str:
        if isinstance(v, float) and math.isnan(v):
            return ""
        return str(v)

    with report_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_REPORT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow({k: _fmt_val(r.get(k, "")) for k in _REPORT_FIELDS})

    return report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Onboard multiple cities in parallel using onboard_city.py logic.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--cities-file", required=True, type=Path,
        metavar="CSV",
        help="CSV with columns: city,station,lat,lon,timezone",
    )
    p.add_argument(
        "--history-years", dest="history_years", type=int, default=5,
        help="Years of GFS + NOAA history to fetch and replay per city",
    )
    p.add_argument(
        "--workers", type=int, default=4,
        help="Number of city-level child processes (parallelism across cities)",
    )
    p.add_argument(
        "--inner-workers", dest="inner_workers", type=int, default=4,
        help="GFS year-fetch threads per city (within each child process)",
    )
    p.add_argument(
        "--auto-approve", dest="auto_approve", action="store_true",
        help="Write passing cities to cities.yaml without interactive prompts",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = _parse_args()

    cities = _load_cities_csv(args.cities_file)

    logger.info(
        "Batch onboard: %d cities  workers=%d  inner_workers=%d  "
        "history_years=%d  auto_approve=%s",
        len(cities), args.workers, args.inner_workers,
        args.history_years, args.auto_approve,
    )

    wall_t0 = time.monotonic()
    results: list[dict[str, Any]] = []
    # Map future → (original CSV index, city_name) so progress rows key to input order.
    future_meta: dict[Any, tuple[int, str]] = {}

    _print_header()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, city_row in enumerate(cities, start=1):
            fut = pool.submit(
                _onboard_city_worker,
                city_row["city"],
                city_row["station"],
                city_row["lat"],
                city_row["lon"],
                city_row["timezone"],
                args.history_years,
                args.auto_approve,
                args.inner_workers,
            )
            future_meta[fut] = (i, city_row["city"])

        for fut in as_completed(future_meta):
            idx, city_name = future_meta[fut]
            try:
                result = fut.result()
            except Exception as exc:
                # Unexpected error from the executor itself (not the worker).
                result = {
                    "city":                   city_name,
                    "station":                "?",
                    "status":                 "FAIL",
                    "error":                  f"Executor error: {type(exc).__name__}: {exc}",
                    "holdout_rmse":           float("nan"),
                    "win_rate_pct":           float("nan"),
                    "sharpe_raw":             float("nan"),
                    "mean_bias_correction_f": float("nan"),
                    "variance_multiplier":    float("nan"),
                    "elapsed_s":              0.0,
                }
            results.append(result)
            _print_row(idx, result)

    wall_elapsed = time.monotonic() - wall_t0
    _print_footer(results, wall_elapsed)

    # Write cities.yaml entries serially — workers built but did not write them
    # to avoid concurrent open(file, "a") corruption across processes.
    if args.auto_approve:
        from deep_isobar.calibration.onboard_city import (  # noqa: PLC0415
            _append_city_to_yaml,
            _city_exists,
        )
        for r in results:
            entry = r.get("yaml_entry")
            if entry and not _city_exists(entry["city"]):
                _append_city_to_yaml(entry)
                logger.info("Appended %s to cities.yaml  (active: false)", entry["city"])

    report_path = _write_report(results)
    print(f"  Report written → {report_path}")
    print()

    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
