"""GEFS ensemble-member backfill — historical T+1 member spread from AWS.

The free APIs only serve ~92 days of forecast history and NO historical
ensemble members (verified 2026-07-01), but the raw GEFS archive on AWS
(``noaa-gefs-pds``) holds all 31 members at 0.25° with ``.idx`` byte-range
indexes back to at least 2021.  This worker extracts, per (station, date,
member), the T+1-lead daily max/min — unlocking:

1. **Flow-dependent variance now** — historical pooled member spread fills
   the ``ens_spread_var`` column years deep, so EMOS's variance gate stops
   waiting for one-per-day live accumulation.
2. **The Forecast Lab's uncertainty data** — member-level vintages for
   regime experiments (including the trend-variance term that failed its
   2026-07-08 holdout for lack of exactly this history).

Design: an embarrassingly parallel **filesystem work queue** (shared NFS
dataset on TrueNAS) — no broker to operate.  Tasks are JSON files; workers
claim by atomic rename; failures land in ``failed/`` with the error
attached.  One task = one (station, metric, month).

Sampling note: GEFS pgrb2s is 3-hourly, so the "daily max" is max over
3-hourly TMP samples — a consistent slight underestimate of the true hourly
max.  EMOS mean coefficients absorb constant scale bias; document, don't
fudge.

CLI::

    # Seed the queue (coordinator, once):
    python -m deep_isobar.lab.gefs_backfill --seed --start 2021-01 --end 2026-06 \
        --queue /mnt/lake/backfill/queue --out /mnt/lake/backfill/out

    # Run a worker (each R520 VM, via systemd):
    python -m deep_isobar.lab.gefs_backfill --work \
        --queue /mnt/lake/backfill/queue --out /mnt/lake/backfill/out

    # Prototype one task locally:
    python -m deep_isobar.lab.gefs_backfill --once --station KMDW --month 2026-06
"""

from __future__ import annotations

import argparse
import calendar
import json
import logging
import os
import socket
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger(__name__)

_GEFS_BASE = "https://noaa-gefs-pds.s3.amazonaws.com"
# Control + 30 perturbed members (the full operational GEFS v12 set).
MEMBERS = ["gec00"] + [f"gep{i:02d}" for i in range(1, 31)]
# T+1 coverage from the 00z run of D-1: f024..f048 (3-hourly) spans all of
# local day D for every US timezone once filtered to local-day samples.
FHOURS = list(range(24, 49, 3))
CYCLE = "00"


def _member_urls(run_day: date, member: str, fhour: int) -> tuple[str, str]:
    base = (
        f"{_GEFS_BASE}/gefs.{run_day:%Y%m%d}/{CYCLE}/atmos/pgrb2sp25/"
        f"{member}.t{CYCLE}z.pgrb2s.0p25.f{fhour:03d}"
    )
    return base, base + ".idx"


def extract_member_daily(
    station_lat: float,
    station_lon_360: float,
    timezone_name: str,
    target_date: date,
    member: str,
    cache_dir: Path,
    agg: str = "max",
) -> float | None:
    """One member's T+1 daily max/min (°F) for *target_date*.

    Fetches the TMP@2m record for each 3-hourly step of the D−1 00z run via
    byte-range, keeps samples whose valid time falls on the station's local
    *target_date*, and aggregates.  Returns None when fewer than 4 samples
    decoded (partial day = garbage aggregate).
    """
    from deep_isobar.data.historical_forecast_ingest import (
        _download_snippet,
        _extract_fahrenheit,
    )

    run_day = target_date - timedelta(days=1)
    tz = ZoneInfo(timezone_name)
    values: list[float] = []

    for fhour in FHOURS:
        valid_utc = datetime(run_day.year, run_day.month, run_day.day,
                             tzinfo=ZoneInfo("UTC")) + timedelta(hours=fhour)
        if valid_utc.astimezone(tz).date() != target_date:
            continue
        grib_url, idx_url = _member_urls(run_day, member, fhour)
        dest = cache_dir / f"{member}_{run_day:%Y%m%d}_f{fhour:03d}.grib2"
        try:
            _download_snippet(grib_url, idx_url, dest)
            values.append(_extract_fahrenheit(dest, station_lat, station_lon_360))
        except Exception as exc:  # noqa: BLE001
            logger.debug("%s %s f%03d failed: %s", member, run_day, fhour, exc)
        finally:
            dest.unlink(missing_ok=True)  # 31 members x 9 steps: never cache

    if len(values) < 4:
        return None
    return min(values) if agg == "min" else max(values)


def run_task(task: dict, out_dir: Path) -> Path:
    """Process one (station, metric, month) task → members parquet.

    Output: ``{station}_{metric}_{YYYYMM}_members.parquet`` with columns
    ``date, member, value_f``.  Idempotent: skips if output exists.
    """
    station = task["station_id"]
    metric = task.get("metric", "high")
    year, month = map(int, task["month"].split("-"))
    agg = "min" if metric == "low" else "max"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{station}_{metric}_{year}{month:02d}_members.parquet"
    if out_path.exists():
        logger.info("task %s already done — skipping", out_path.name)
        return out_path

    rows: list[dict] = []
    n_days = calendar.monthrange(year, month)[1]
    with tempfile.TemporaryDirectory(prefix="gefs_") as tmp:
        cache = Path(tmp)
        for day in range(1, n_days + 1):
            target = date(year, month, day)
            if target >= date.today():
                break
            for member in MEMBERS:
                v = extract_member_daily(
                    task["lat"], task["lon_360"], task["timezone"],
                    target, member, cache, agg=agg,
                )
                if v is not None:
                    rows.append({"date": target, "member": member, "value_f": v})
            logger.info("[%s %s] %s: %d member values so far",
                        station, metric, target, len(rows))

    pd.DataFrame(rows).to_parquet(out_path, index=False)
    logger.info("task complete → %s (%d rows)", out_path, len(rows))
    return out_path


def summarize_members(members_parquet: Path) -> pd.DataFrame:
    """Reduce a members parquet to per-day (mean, spread var, count).

    ``ens_spread_var`` here matches the live pipeline's definition (within-
    ensemble variance, ddof=1), so these rows merge directly into
    ``{station}_t1_spread.parquet`` history.
    """
    df = pd.read_parquet(members_parquet)
    g = df.groupby("date")["value_f"]
    out = pd.DataFrame({
        "ens_mean_f": g.mean(),
        "ens_spread_var": g.var(ddof=1),
        "n_members": g.count(),
    }).reset_index()
    return out[out["n_members"] >= 10].reset_index(drop=True)


# ── Filesystem work queue ────────────────────────────────────────────────────


def seed_queue(queue_dir: Path, start: str, end: str,
               metrics: tuple[str, ...] = ("high", "low")) -> int:
    """Create one task file per (active station, metric, month)."""
    from deep_isobar.data.city_universe import get_city_universe

    queue_dir.mkdir(parents=True, exist_ok=True)
    y0, m0 = map(int, start.split("-"))
    y1, m1 = map(int, end.split("-"))
    months = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1

    n = 0
    for city in (c for c in get_city_universe() if c.active):
        for metric in metrics:
            for month in months:
                name = f"{city.station_id}_{metric}_{month}.json"
                path = queue_dir / name
                if path.exists():
                    continue
                path.write_text(json.dumps({
                    "station_id": city.station_id,
                    "lat": city.nws_lat,
                    "lon_360": city.nws_lon % 360.0,
                    "timezone": city.timezone,
                    "metric": metric,
                    "month": month,
                }), encoding="utf-8")
                n += 1
    logger.info("seeded %d task(s) into %s", n, queue_dir)
    return n


def claim_task(queue_dir: Path) -> tuple[Path, dict] | None:
    """Atomically claim the next task (rename into ``claimed/``)."""
    claimed_dir = queue_dir / "claimed"
    claimed_dir.mkdir(exist_ok=True)
    me = f"{socket.gethostname()}.{os.getpid()}"
    for f in sorted(queue_dir.glob("*.json")):
        target = claimed_dir / f"{f.stem}.{me}.json"
        try:
            f.rename(target)  # atomic on one filesystem — the whole queue
        except OSError:
            continue  # another worker got it
        return target, json.loads(target.read_text(encoding="utf-8"))
    return None


def worker_loop(queue_dir: Path, out_dir: Path) -> int:
    """Consume tasks until the queue is empty.  Systemd restarts us daily."""
    done = 0
    failed_dir = queue_dir / "failed"
    while True:
        claim = claim_task(queue_dir)
        if claim is None:
            logger.info("queue empty — %d task(s) completed this run", done)
            return done
        claimed_path, task = claim
        try:
            run_task(task, out_dir)
            claimed_path.unlink()
            done += 1
        except Exception as exc:  # noqa: BLE001
            logger.exception("task %s failed", claimed_path.name)
            failed_dir.mkdir(exist_ok=True)
            task["error"] = str(exc)[:500]
            (failed_dir / claimed_path.name).write_text(
                json.dumps(task), encoding="utf-8"
            )
            claimed_path.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GEFS member backfill")
    parser.add_argument("--queue", type=Path, default=Path("data/backfill/queue"))
    parser.add_argument("--out", type=Path, default=Path("data/backfill/out"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--seed", action="store_true")
    mode.add_argument("--work", action="store_true")
    mode.add_argument("--once", action="store_true", help="run one local task")
    parser.add_argument("--start", default="2021-01")
    parser.add_argument("--end", default="2026-06")
    parser.add_argument("--station")
    parser.add_argument("--month")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.seed:
        seed_queue(args.queue, args.start, args.end)
        return 0
    if args.work:
        worker_loop(args.queue, args.out)
        return 0
    # --once: prototype a single task without a queue
    from deep_isobar.data.city_universe import get_city_universe
    city = next(c for c in get_city_universe() if c.station_id == args.station)
    run_task({
        "station_id": city.station_id, "lat": city.nws_lat,
        "lon_360": city.nws_lon % 360.0, "timezone": city.timezone,
        "metric": "high", "month": args.month,
    }, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
