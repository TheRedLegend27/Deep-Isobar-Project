"""Daily paper trade session for Deep Isobar — multi-city orchestrator.

Loads all active cities from ``config/cities.yaml``, runs each city's
forecast → ensemble → Kalshi → spreading pipeline concurrently, and logs
signals to CSV.

Run once per day before the decision cutoff (~12:00 UTC / 7 am CDT)::

    python -m deep_isobar.research.paper_trade_session
    python -m deep_isobar.research.paper_trade_session --dry-run

Output files
------------
``data/paper_trades/paper_trades.csv``
    One row per trade logged (``abs(alpha) >= SIGNAL_THRESHOLD``).
    Status is initialised to ``OPEN``; settled by ``settle_paper_trades.py``.

``data/paper_trades/daily_log.csv``
    One row per contract evaluated regardless of signal strength.
    Useful for monitoring the pipeline on no-signal days.

CSV schema
----------
date, contract_ticker, direction, alpha, model_prob, market_prob,
ensemble_mean_f, entry_price, position_size, status, realized_pnl,
settled_temp, threshold_f, strike_type, floor_strike, cap_strike,
anomaly_flags, anomaly_penalty_f, anomaly_adjusted_signal,
anomaly_confidence, anomaly_reasoning, spread_rank,
spread_total_contracts, sizing_base_usd, sizing_final_usd,
sizing_reasoning, city

See ``_CSV_COLUMNS`` for the authoritative list.

GFS forecast note
-----------------
The script tries the 12z run (f030, 30 h lead) first, then falls back
to the 00z run (f042, 42 h lead).  Both require cfgrib / xarray / eccodes::

    pip install eccodes cfgrib xarray
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import threading
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")  # load KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PATH, etc.

from deep_isobar.calibration.emos import emos_predict, load_params as load_emos_params
from deep_isobar.core.types import CityProfile, ForecastPoint, TradeSignal
from deep_isobar.data.city_universe import get_city_universe
from deep_isobar.data.ensemble_ingest import (
    fetch_member_daily_maxes,
    pooled_member_variance,
    record_t1_spread,
)
from deep_isobar.data.historical_forecast_ingest import (
    _AWS_BASE,
    _LEAD_FHOUR,
    _STATION_COORDS,
    _cache_path,
    _check_cfgrib,
    _download_snippet,
    _extract_fahrenheit,
    _resolve_gfs_idx_url,
)
from deep_isobar.market.kalshi_client import (
    fetch_live_contracts,
    fetch_orderbook_for_contract,
)
from deep_isobar.market.market_scanner import evaluate_contract_opportunity
from deep_isobar.market.microstructure_scanner import compute_microstructure_score
from deep_isobar.models.probability_engine import probability_for_contract
from deep_isobar.models.temperature_ensemble import build_temperature_ensemble
from deep_isobar.trading.distribution_tail_alpha import is_tail_threshold
from deep_isobar.trading.preflight import run_preflight, training_history_bounds
from deep_isobar.market.kalshi_client import is_live_mode as kalshi_is_live_mode
from deep_isobar.config import get_setting
from deep_isobar.notifications.discord_notifier import (
    COLOR_AMBER,
    COLOR_BLUE,
    COLOR_GRAY,
    COLOR_GREEN,
    COLOR_RED,
    post_embed,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PAPER_TRADES_DIR = _PROJECT_ROOT / "data" / "paper_trades"
_PAPER_TRADES_CSV = _PAPER_TRADES_DIR / "paper_trades.csv"
_DAILY_LOG_CSV = _PAPER_TRADES_DIR / "daily_log.csv"
_GFS_CACHE_DIR = _PROJECT_ROOT / "data" / "historical" / "forecasts" / ".grib_cache"

SIGNAL_THRESHOLD: float = get_setting("risk.alpha_threshold", default=0.25)
logger.info(f"Signal threshold loaded from config: {SIGNAL_THRESHOLD}")
POSITION_SIZE = 10.0        # contracts per trade (paper)
METRIC = "high_temp_f"
# LEGACY-ONLY floor, applied when a station has no fitted EMOS params and
# the session falls back to the snapshot + bias-profile pipeline.  Stations
# with EMOS params use the calibrated sigma (floored at ~1.3 F inside
# emos_predict) — the 5.5 F floor was identified as the root cause of the
# chronic under-confidence in the 2026-06-11 research report.
_MIN_ENSEMBLE_STD_F = 5.5

# Thread lock for CSV writes — multiple cities run concurrently.
_CSV_LOCK = threading.Lock()

_CSV_COLUMNS = [
    "date",
    "contract_ticker",
    "direction",
    "alpha",
    "model_prob",
    "market_prob",
    "ensemble_mean_f",
    "entry_price",
    "position_size",
    "status",
    "realized_pnl",
    "settled_temp",
    "threshold_f",
    "strike_type",
    "floor_strike",
    "cap_strike",
    "anomaly_flags",
    "anomaly_penalty_f",
    "anomaly_adjusted_signal",
    "anomaly_confidence",
    "anomaly_reasoning",
    "spread_rank",
    "spread_total_contracts",
    "sizing_base_usd",
    "sizing_final_usd",
    "sizing_reasoning",
    "city",
    # 2026-07 ensemble upgrade — appended last so _ensure_csv's prefix
    # migration upgrades existing files in place.
    "nbm_max_f",
    "ens_spread_var",
]


# ---------------------------------------------------------------------------
# Live forecast fetch (GFS via GRIB2/AWS + ECMWF/ICON/GEM via Open-Meteo)
# ---------------------------------------------------------------------------

# Open-Meteo model identifiers per Deep Isobar model name.  GFS from
# Open-Meteo is only used when the GRIB2/AWS path fails — the GRIB path is
# what the bias profiles were calibrated on and stays the primary source.
_OPEN_METEO_MODELS: dict[str, str] = {
    "GFS":   "gfs_seamless",
    "ECMWF": "ecmwf_ifs025",
    "ICON":  "icon_seamless",
    "GEM":   "gem_seamless",
    "NBM":   "ncep_nbm_conus",
}
_EXTRA_MODELS = ("ECMWF", "ICON", "GEM")
# EMOS-mode member set — must cover emos_training.EMOS_MODELS.  NBM stays
# out of the legacy path: the old bias profiles were never calibrated on it.
_EMOS_SESSION_MODELS = ("GFS",) + _EXTRA_MODELS + ("NBM",)
# EMOS mu vs NBM MaxT divergence (deg F) that triggers a benchmark warning —
# NBM is NOAA's calibrated blend, so a big gap usually means we are wrong.
_NBM_DIVERGENCE_WARN_F = 4.0


def _fetch_live_forecasts_t24(
    city_profile: CityProfile,
    run_date: date,
    daily_max: bool = False,
) -> list[ForecastPoint]:
    """Fetch all model forecasts for *run_date*, targeting tomorrow's high.

    When *daily_max* is True (station has fitted EMOS params), every model
    comes from one Open-Meteo call requesting the native ``temperature_2m_max``
    daily variable over the local settlement day — the max-of-trace quantity
    the EMOS coefficients were trained on.

    Legacy mode (no EMOS params): GFS comes from the GRIB2/AWS 18z-snapshot
    path (the source the old bias profiles were calibrated on), ECMWF / ICON /
    GEM from the Open-Meteo 18z hourly value, with Open-Meteo GFS as fill-in.

    Returns all successfully retrieved points.  An empty list means no data
    was available from any path.
    """
    station_id = city_profile.station_id
    city_lat, city_lon_360 = _STATION_COORDS.get(
        station_id, (41.7868, 272.2478)  # KMDW fallback
    )
    # Prefer settlement-station coordinates from cities.yaml when present —
    # _STATION_COORDS may not know newly added stations (e.g. KNYC).
    if city_profile.nws_lat and city_profile.nws_lon:
        city_lat = city_profile.nws_lat
        city_lon_360 = city_profile.nws_lon % 360.0
    tomorrow = run_date + timedelta(days=1)

    if daily_max:
        points = _fetch_open_meteo_daily_max(
            city_profile, city_lat, city_lon_360, run_date, tomorrow,
            _EMOS_SESSION_MODELS,
        )
        if points:
            logger.info(
                "[%s] Forecast points (daily max-of-trace): %s",
                city_profile.city,
                {f"{p.model_name}({p.source_name})": round(p.forecast_value_f, 1) for p in points},
            )
        return points

    # ── GFS via GRIB2/AWS ────────────────────────────────────────────────────
    gfs_points = _fetch_grib_gfs_t24(city_profile, city_lat, city_lon_360, run_date, tomorrow)

    # ── ECMWF / ICON / GEM (plus GFS fallback) via Open-Meteo ───────────────
    om_models = _EXTRA_MODELS if gfs_points else ("GFS",) + _EXTRA_MODELS
    om_points = _fetch_open_meteo_models(
        city_profile, city_lat, city_lon_360, run_date, tomorrow, om_models
    )

    points = gfs_points + om_points
    if points:
        logger.info(
            "[%s] Forecast points: %s",
            city_profile.city,
            {f"{p.model_name}({p.source_name})": round(p.forecast_value_f, 1) for p in points},
        )
    return points


def _fetch_grib_gfs_t24(
    city_profile: CityProfile,
    city_lat: float,
    city_lon_360: float,
    run_date: date,
    tomorrow: date,
) -> list[ForecastPoint]:
    """Download GFS T+24 forecasts from the GRIB2/AWS path (12z, then 00z).

    Returns an empty list when cfgrib/ecCodes is unavailable or no run could
    be retrieved — the caller falls back to Open-Meteo GFS in that case.
    """
    station_id = city_profile.station_id
    try:
        _check_cfgrib()
    except (ImportError, RuntimeError, OSError) as exc:
        logger.info(
            "[%s] cfgrib/ecCodes unavailable (%s) — GFS will come from Open-Meteo",
            city_profile.city, exc,
        )
        return []

    points: list[ForecastPoint] = []
    for cycle in ("12", "00"):
        fhour = _LEAD_FHOUR[cycle][1]   # lead_day=1 → 18z UTC on target date
        fhour_str = f"{fhour:03d}"
        run_time_utc = datetime(
            run_date.year, run_date.month, run_date.day,
            int(cycle), 0, 0, tzinfo=timezone.utc,
        )
        date_str = run_date.strftime("%Y%m%d")
        dest = _cache_path(_GFS_CACHE_DIR, run_date, cycle, fhour)

        try:
            idx_url = _resolve_gfs_idx_url(_AWS_BASE, date_str, cycle, fhour_str)
            if idx_url is None:
                logger.warning(
                    "GFS %sz f%s not found on AWS (both layouts 404) — skipping",
                    cycle, fhour_str,
                )
                continue
            grib2_url = idx_url[:-4]  # strip ".idx"
            _download_snippet(grib2_url, idx_url, dest)
            t_f = _extract_fahrenheit(dest, city_lat, city_lon_360)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "GFS %sz f%03d unavailable (%s) — skipping", cycle, fhour, exc
            )
            continue

        logger.info(
            "GFS %sz f%03d: run=%s → %.1f°F  (target %s)",
            cycle, fhour, run_date, t_f, tomorrow,
        )
        points.append(ForecastPoint(
            city=city_profile.city,
            station_id=station_id,
            model_name="GFS",
            run_time_utc=run_time_utc,
            target_date=tomorrow,
            metric=METRIC,
            forecast_value_f=t_f,
            lead_hours=fhour,
            source_name="NOAA",
        ))

    if not points:
        logger.info(
            "[%s] GRIB2 path returned no GFS points — Open-Meteo GFS will fill in",
            city_profile.city,
        )
    return points


def _fetch_open_meteo_models(
    city_profile: CityProfile,
    city_lat: float,
    city_lon_360: float,
    run_date: date,
    tomorrow: date,
    models: tuple[str, ...] = _EXTRA_MODELS,
) -> list[ForecastPoint]:
    """Fetch 2m temperature for several models via one Open-Meteo call.

    Extracts the 18:00 UTC value on *tomorrow* for each requested model —
    the same afternoon-high proxy the GRIB2 path uses, so the station bias
    correction (which mostly absorbs the snapshot-vs-daily-high offset)
    applies consistently across models.  Returns one ForecastPoint per
    model that had data; models with a null/missing value are skipped.
    """
    station_id = city_profile.station_id
    city_lon = city_lon_360 - 360.0  # 360-lon → standard (-180 … 180)

    om_ids = ",".join(_OPEN_METEO_MODELS[m] for m in models)
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={city_lat}&longitude={city_lon}"
        "&hourly=temperature_2m"
        f"&models={om_ids}"
        "&temperature_unit=fahrenheit"
        "&forecast_days=3"
        "&timezone=UTC"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "deep-isobar/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] Open-Meteo request failed: %s", city_profile.city, exc)
        return []

    hourly: dict = data.get("hourly", {})
    hourly_times: list[str] = hourly.get("time", [])

    # Target: 18:00 UTC on tomorrow (afternoon-high proxy)
    target_str = tomorrow.strftime("%Y-%m-%d") + "T18:00"
    try:
        idx = hourly_times.index(target_str)
    except ValueError:
        logger.warning(
            "[%s] Open-Meteo: target time %s not found in response",
            city_profile.city, target_str,
        )
        return []

    # 12z run of run_date → 18z tomorrow = 30 h lead.
    run_time_utc = datetime(
        run_date.year, run_date.month, run_date.day, 12, 0, 0, tzinfo=timezone.utc,
    )
    lead_hours = 30

    points: list[ForecastPoint] = []
    for model in models:
        # Multi-model responses suffix the key with the model id; a
        # single-model request returns a plain "temperature_2m" key.
        key = f"temperature_2m_{_OPEN_METEO_MODELS[model]}"
        temps = hourly.get(key)
        if temps is None and len(models) == 1:
            temps = hourly.get("temperature_2m")
        t_f = temps[idx] if temps and idx < len(temps) else None
        if t_f is None:
            logger.warning(
                "[%s] Open-Meteo: no %s value at %s — skipping model",
                city_profile.city, model, target_str,
            )
            continue

        logger.info(
            "[%s] Open-Meteo %s 18z UTC: run=%s → %.1f°F  (target %s)",
            city_profile.city, model, run_date, t_f, tomorrow,
        )
        points.append(ForecastPoint(
            city=city_profile.city,
            station_id=station_id,
            model_name=model,
            run_time_utc=run_time_utc,
            target_date=tomorrow,
            metric=METRIC,
            forecast_value_f=float(t_f),
            lead_hours=lead_hours,
            source_name="Open-Meteo",
        ))

    return points


def _fetch_open_meteo_daily_max(
    city_profile: CityProfile,
    city_lat: float,
    city_lon_360: float,
    run_date: date,
    tomorrow: date,
    models: tuple[str, ...] = ("GFS",) + _EXTRA_MODELS,
) -> list[ForecastPoint]:
    """Fetch native daily-max temperature for several models in one call.

    Requests Open-Meteo's ``temperature_2m_max`` daily variable aggregated in
    the city's **local timezone** — i.e. the max over the hourly trace of the
    settlement day, matching how NWS computes the climate-report high and how
    the EMOS training data was built.  This replaces the 18z snapshot, which
    sampled hours before the afternoon peak (the Dallas cold-bias bug).
    """
    station_id = city_profile.station_id
    city_lon = city_lon_360 - 360.0 if city_lon_360 > 180.0 else city_lon_360

    om_ids = ",".join(_OPEN_METEO_MODELS[m] for m in models)
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={city_lat}&longitude={city_lon}"
        "&daily=temperature_2m_max"
        f"&models={om_ids}"
        "&temperature_unit=fahrenheit"
        "&forecast_days=3"
        f"&timezone={city_profile.timezone.replace('/', '%2F')}"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "deep-isobar/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] Open-Meteo daily-max request failed: %s", city_profile.city, exc)
        return []

    daily: dict = data.get("daily", {})
    days: list[str] = daily.get("time", [])
    target_str = tomorrow.strftime("%Y-%m-%d")
    try:
        idx = days.index(target_str)
    except ValueError:
        logger.warning(
            "[%s] Open-Meteo daily-max: target day %s not in response",
            city_profile.city, target_str,
        )
        return []

    run_time_utc = datetime(
        run_date.year, run_date.month, run_date.day, 12, 0, 0, tzinfo=timezone.utc,
    )

    points: list[ForecastPoint] = []
    for model in models:
        # Multi-model responses suffix the key with the model id; a
        # single-model request returns a plain "temperature_2m_max" key.
        key = f"temperature_2m_max_{_OPEN_METEO_MODELS[model]}"
        vals = daily.get(key)
        if vals is None and len(models) == 1:
            vals = daily.get("temperature_2m_max")
        t_f = vals[idx] if vals and idx < len(vals) else None
        if t_f is None:
            logger.warning(
                "[%s] Open-Meteo daily-max: no %s value for %s — skipping model",
                city_profile.city, model, target_str,
            )
            continue

        points.append(ForecastPoint(
            city=city_profile.city,
            station_id=station_id,
            model_name=model,
            run_time_utc=run_time_utc,
            target_date=tomorrow,
            metric=METRIC,
            forecast_value_f=float(t_f),
            lead_hours=30,
            source_name="Open-Meteo-MaxT",
        ))

    return points


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def _ensure_csv(path: Path) -> None:
    """Create the CSV file with a header row if it does not already exist.

    When the file exists and its header is a strict prefix of the current
    ``_CSV_COLUMNS`` (i.e. new columns were appended to the schema), the file
    is automatically migrated: the header is rewritten and existing rows
    receive empty values for the new columns.

    Raises RuntimeError if the file exists but its header is incompatible
    with ``_CSV_COLUMNS`` (column order or names differ, not just missing
    trailing columns).
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=_CSV_COLUMNS).writeheader()
        return
    with path.open(newline="", encoding="utf-8") as fh:
        existing = next(csv.reader(fh), [])
    if existing == _CSV_COLUMNS:
        return  # schema matches exactly — nothing to do
    # Auto-migrate if existing header is a prefix of the current schema.
    # This handles the case where new columns (e.g. city) were appended.
    if existing == _CSV_COLUMNS[: len(existing)]:
        _migrate_csv_add_columns(path)
        logger.info(
            "Auto-migrated %s: added columns %s",
            path.name,
            _CSV_COLUMNS[len(existing):],
        )
        return
    raise RuntimeError(
        f"Schema mismatch in {path.name}: header has {len(existing)} columns "
        f"but _CSV_COLUMNS defines {len(_CSV_COLUMNS)}. "
        "Migrate the CSV to the new schema before running a session."
    )


def _migrate_csv_add_columns(path: Path) -> None:
    """Rewrite *path* with the full ``_CSV_COLUMNS`` schema.

    Existing rows keep their current values; new columns are written as
    empty strings.  This is a forward-only migration — columns already
    present are never removed or reordered.
    """
    with path.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _append_csv_row(path: Path, row: dict) -> None:
    """Append one row dict to the CSV (header assumed to exist)."""
    with path.open("a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=_CSV_COLUMNS).writerow(row)


def _logged_trade_keys(path: Path) -> set[tuple[str, str]]:
    """Return ``(date, contract_ticker)`` pairs already logged to *path*.

    Used to guard against duplicate trades when the session is run more
    than once on the same day (e.g. a manual re-run after a crash).
    """
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as fh:
        return {
            (r.get("date", ""), r.get("contract_ticker", ""))
            for r in csv.DictReader(fh)
        }


# ---------------------------------------------------------------------------
# Signal printer (dry-run)
# ---------------------------------------------------------------------------


def _print_signal(row: dict, signal: Any, orderbook: Any) -> None:
    """Print a single evaluated contract to stdout (dry-run mode)."""
    is_trade = row["status"] == "OPEN"
    tag = ">>> TRADE SIGNAL (would be logged)" if is_trade else "    no trade"
    bid = orderbook.best_bid
    ask = orderbook.best_ask

    st = row.get("strike_type", "")
    cap = row.get("cap_strike", "")
    flr = row.get("floor_strike", "")
    if st == "less":
        condition = f"YES if actual < {cap}\u00b0F"
    elif st == "greater":
        condition = f"YES if actual > {flr}\u00b0F"
    else:
        condition = "unknown"

    print(
        f"\n  City        : {row['city']}\n"
        f"  Contract    : {row['contract_ticker']}\n"
        f"  Condition   : {condition}\n"
        f"  Direction   : {row['direction']}\n"
        f"  Alpha       : {float(row['alpha']):+.4f}\n"
        f"  Model prob  : {float(row['model_prob']):.4f}\n"
        f"  Market prob : {float(row['market_prob']):.4f}\n"
        f"  Bid / Ask   : {bid} / {ask}\n"
        f"  Entry price : {row['entry_price']}\n"
        f"  Conf. score : {signal.confidence_score:.3f}\n"
        f"  {tag}"
    )


# ---------------------------------------------------------------------------
# Per-city session
# ---------------------------------------------------------------------------


def run_city_session(city: CityProfile, dry_run: bool = False) -> dict:
    """Run one paper trade session for a single city.

    Fetches GFS T+24 forecast for tomorrow's high, pulls live Kalshi
    bid/ask for the city's contracts, runs the ensemble → probability →
    alpha pipeline, and logs signals to CSV.

    Args:
        city: City configuration profile.
        dry_run: When ``True``, print signals to stdout without writing to
            any CSV file.

    Returns:
        Dict with keys ``city`` (str), ``signals`` (int), ``error`` (str or None).
    """
    today = date.today()
    tomorrow = today + timedelta(days=1)
    city_label = city.city

    logger.info("=== [%s] Paper trade session  target_date=%s ===", city_label, tomorrow)

    result: dict = {"city": city_label, "signals": 0, "error": None}

    # ── EMOS calibrated distribution (research upgrade 2026-06) ────────────
    # When fitted params exist, the session fetches native daily-max
    # forecasts and uses the EMOS (mu, sigma) directly; otherwise it falls
    # back to the legacy snapshot + bias-profile + 5.5 F floor pipeline.
    emos_params = load_emos_params(city.station_id)

    # ── Model forecasts (GFS + ECMWF/ICON/GEM) ─────────────────────────────
    try:
        forecast_points = _fetch_live_forecasts_t24(
            city, today, daily_max=emos_params is not None
        )
    except (ImportError, RuntimeError, OSError) as exc:
        logger.error("[%s] cfgrib/xarray/eccodes not installed: %s", city_label, exc)
        result["error"] = f"cfgrib not installed: {exc}"
        return result

    if not forecast_points:
        msg = (
            f"No T+24 forecasts available for {city_label} run_date={today} "
            "from either the GRIB2/AWS path or Open-Meteo."
        )
        logger.error("[%s] %s", city_label, msg)
        result["error"] = msg
        return result

    # ── Temperature ensemble ───────────────────────────────────────────────
    ensemble = build_temperature_ensemble(
        city_profile=city,
        forecasts=forecast_points,
        target_date=tomorrow,
        metric=METRIC,
        ensemble_run_time_utc=forecast_points[0].run_time_utc,
    )

    nbm_max_f: float | None = None
    ens_spread_var: float | None = None

    if emos_params is not None:
        # The legacy bias profiles were calibrated on 18z snapshots and do
        # NOT apply to daily-max inputs — never mix the two pipelines.  If
        # EMOS cannot produce a distribution, skip the city.

        # Pooled GEFS/EPS member spread — today's flow-dependent uncertainty.
        # A failed fetch degrades to the climatological spread inside
        # emos_predict, never blocks the session.
        if city.nws_lat and city.nws_lon:
            try:
                members_by_model = fetch_member_daily_maxes(
                    city.nws_lat, city.nws_lon, city.timezone, tomorrow,
                )
                ens_spread_var = pooled_member_variance(members_by_model)
                logger.info(
                    "[%s] Ensemble members: %s → pooled spread var %s degF^2",
                    city_label,
                    {k: len(v) for k, v in members_by_model.items()},
                    f"{ens_spread_var:.2f}" if ens_spread_var is not None else "n/a",
                )
                if ens_spread_var is not None:
                    # Historical member spread is not retrievable, so every
                    # session banks today's observation for EMOS training.
                    try:
                        record_t1_spread(city.station_id, tomorrow, ens_spread_var)
                    except Exception:  # noqa: BLE001
                        logger.exception(
                            "[%s] could not record T+1 spread history", city_label
                        )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[%s] Ensemble member fetch failed (%s) — EMOS will use "
                    "climatological spread", city_label, exc,
                )

        try:
            effective_mean, effective_std = emos_predict(
                emos_params,
                {p.model_name: p.forecast_value_f for p in forecast_points},
                ensemble_spread_var=ens_spread_var,
            )
        except ValueError as exc:
            msg = f"EMOS predict failed ({exc}) — skipping city rather than trading uncalibrated."
            logger.error("[%s] %s", city_label, msg)
            result["error"] = msg
            return result
        dist_source = "EMOS"

        # NBM benchmark: NOAA's operationally calibrated blend.  Divergence
        # beyond _NBM_DIVERGENCE_WARN_F usually means our mean is the wrong
        # one (research report 2026-06, Q2).
        nbm_max_f = next(
            (p.forecast_value_f for p in forecast_points if p.model_name == "NBM"),
            None,
        )
        if nbm_max_f is not None and abs(effective_mean - nbm_max_f) > _NBM_DIVERGENCE_WARN_F:
            logger.warning(
                "[%s] EMOS mu %.1f°F diverges from NBM MaxT %.1f°F by %.1f°F "
                "(> %.1f) — treat today's edge with suspicion",
                city_label, effective_mean, nbm_max_f,
                abs(effective_mean - nbm_max_f), _NBM_DIVERGENCE_WARN_F,
            )
    else:
        effective_mean = ensemble.bias_corrected_mean_f
        effective_std = max(ensemble.adjusted_std_f, _MIN_ENSEMBLE_STD_F)
        dist_source = "legacy"

    logger.info(
        "[%s] Distribution (%s): raw_mean=%.1f°F  calibrated_mean=%.1f°F  "
        "sigma=%.2f°F",
        city_label,
        dist_source,
        ensemble.ensemble_mean_f,
        effective_mean,
        effective_std,
    )

    # ── Anomaly detection ─────────────────────────────────────────────────
    # Use city.nws_lat/nws_lon for NWS; city.station_id for METAR.
    # When anomaly.enabled is false, skip entirely: anomaly=None means the
    # position sizer applies the NONE multiplier (1.0 — no penalty) instead
    # of the LOW multiplier (0.5) that a failed API call would trigger.
    if not get_setting("anomaly.enabled", default=True):
        anomaly = None
        logger.info("[%s] Anomaly check disabled via config (anomaly.enabled=false)", city_label)
    else:
        try:
            from deep_isobar.anomaly import check_anomalies, parse_metar_fields
            from deep_isobar.anomaly.metar_fetcher import fetch_metar
            from deep_isobar.anomaly.nws_fetcher import fetch_nws_high_forecast_f

            raw_metar = fetch_metar(city.station_id)
            metar_fields = parse_metar_fields(raw_metar)

            nws_lat = city.nws_lat if city.nws_lat != 0.0 else None
            nws_lon = city.nws_lon if city.nws_lon != 0.0 else None

            nws_forecast_f = None
            if nws_lat is not None and nws_lon is not None:
                try:
                    nws_forecast_f = fetch_nws_high_forecast_f(lat=nws_lat, lon=nws_lon)
                except Exception:
                    logger.debug("[%s] NWS fetcher raised unexpectedly — defaulting to None", city_label)

            if nws_forecast_f is not None:
                logger.info("[%s] NWS forecast high: %.1f°F", city_label, nws_forecast_f)
            else:
                logger.debug("[%s] NWS forecast unavailable — NWS_MODEL_DIVERGENCE will not fire", city_label)

            anomaly = check_anomalies(
                metar=raw_metar,
                nws_forecast_f=nws_forecast_f,
                model_mean_f=ensemble.ensemble_mean_f,
                wind_dir=metar_fields.get("wind_dir", ""),
                sky_cover=metar_fields.get("sky_cover", ""),
            )
            logger.info(
                "[%s] Anomaly check: flags=%s  penalty=%.1f°F  signal=%s  confidence=%s",
                city_label,
                [f.code for f in anomaly.flags],
                anomaly.total_temp_penalty_f,
                anomaly.adjusted_signal,
                anomaly.confidence,
            )
        except Exception as e:
            anomaly = None
            logger.debug("[%s] Anomaly check skipped: %s", city_label, e)

    if not dry_run:
        post_embed(
            title=f"Deep Isobar \u2014 [{city_label}] Morning run started",
            color=COLOR_BLUE,
            fields=[
                {"name": "Target date",        "value": str(tomorrow)},
                {"name": "Distribution",       "value": dist_source},
                {"name": "Raw ensemble mean",  "value": f"{ensemble.ensemble_mean_f:.1f}\u00b0F"},
                {"name": "Calibrated mean",    "value": f"{effective_mean:.1f}\u00b0F"},
                {"name": "Sigma",              "value": f"{effective_std:.2f}\u00b0F"},
            ],
        )

    # ── Live Kalshi contracts for tomorrow ────────────────────────────────
    try:
        # Use city.kalshi_series to fetch only this city's contracts.
        series = city.kalshi_series or None
        all_contracts = fetch_live_contracts("Kalshi", series_ticker=series)
        tomorrow_contracts = [
            c for c in all_contracts
            if c.target_date == tomorrow
            and c.metric == METRIC
            and c.strike_type in ("less", "greater", "between")
        ]

        if not tomorrow_contracts:
            logger.warning(
                "[%s] No active Kalshi high_temp_f contracts found for %s — "
                "market may not be listed yet.",
                city_label, tomorrow,
            )
            if not dry_run:
                post_embed(
                    title=f"[{city_label}] No contracts found",
                    description=(
                        f"No active Kalshi high_temp_f contracts for {tomorrow}. "
                        "The market may not be listed yet."
                    ),
                    color=COLOR_GRAY,
                )
            result["signals"] = 0
            return result

        logger.info(
            "[%s] Found %d contracts for %s: thresholds=%s",
            city_label,
            len(tomorrow_contracts),
            tomorrow,
            sorted(c.threshold_f for c in tomorrow_contracts),
        )

        # ── Pre-trade circuit breakers ─────────────────────────────────────
        # Any failure refuses the whole city for the day — a wrong refusal
        # costs one day's edge; a wrong trade against broken inputs or stub
        # markets costs real money (see preflight module docstring).
        hist_lo, hist_hi = training_history_bounds(city.station_id)
        preflight = run_preflight(
            city_name=city_label,
            effective_mean=effective_mean,
            effective_std=effective_std,
            dist_source=dist_source,
            nbm_max_f=nbm_max_f,
            emos_params=emos_params,
            n_contracts=len(tomorrow_contracts),
            market_is_live=kalshi_is_live_mode(),
            hist_min_f=hist_lo,
            hist_max_f=hist_hi,
        )
        if not preflight.ok:
            msg = f"Preflight blocked trading: {preflight.summary()}"
            result["error"] = msg
            if not dry_run:
                post_embed(
                    title=f"🛑 [{city_label}] PREFLIGHT BLOCKED",
                    description=preflight.summary(),
                    color=COLOR_RED,
                )
            return result

        # ── Probability surface ────────────────────────────────────────────
        # Keyed by contract_id, NOT threshold_f: a "T98" tail and a 98–99
        # bracket share threshold 98, and threshold keys silently collide
        # (the wrong-sided 2026-07-04 NY trade: bracket prob overwrote the
        # tail prob and flipped the signal's side).
        probability_surface: dict[str, float] = {}
        for _c in tomorrow_contracts:
            probability_surface[_c.contract_id] = probability_for_contract(
                strike_type=_c.strike_type,
                floor_strike=_c.floor_strike,
                cap_strike=_c.cap_strike,
                mean_f=effective_mean,
                std_f=effective_std,
            )

        def _surface_label(c: Any) -> str:
            if c.strike_type == "less":
                return f"P(T<{c.cap_strike})"
            if c.strike_type == "greater":
                return f"P(T>{c.floor_strike})"
            if c.strike_type == "between":
                return f"P({c.floor_strike}≤T<{c.cap_strike})"
            return f"P(?{c.threshold_f})"

        logger.info(
            "[%s] Probability surface: %s",
            city_label,
            {_surface_label(_c): f"{probability_surface[_c.contract_id]:.3f}"
             for _c in sorted(tomorrow_contracts, key=lambda x: x.threshold_f)},
        )

        # ── Evaluate each contract ─────────────────────────────────────────
        all_rows: list[dict] = []
        dry_run_signals: list[tuple[dict, object, object]] = []
        signal_lookup: dict[str, TradeSignal] = {}

        for contract in sorted(tomorrow_contracts, key=lambda c: c.threshold_f):
            orderbook = fetch_orderbook_for_contract("Kalshi", contract.contract_id)
            now_utc = datetime.now(timezone.utc)

            micro_score = compute_microstructure_score(orderbook, now_utc)
            tail_flag = is_tail_threshold(
                ensemble_mean_f=effective_mean,
                threshold_f=contract.threshold_f,
                adjusted_std_f=effective_std,
            )

            signal = evaluate_contract_opportunity(
                contract=contract,
                probability_surface=probability_surface,
                orderbook=orderbook,
                signal_threshold=SIGNAL_THRESHOLD,
                timestamp_utc=now_utc,
                microstructure_score=micro_score,
                tail_opportunity_flag=tail_flag,
                tail_multiplier=city.tail_multiplier,
            )
            signal_lookup[contract.contract_id] = signal

            if signal.signal_side == "BUY":
                entry_price = orderbook.best_ask
            elif signal.signal_side == "SELL":
                entry_price = orderbook.best_bid
            else:
                if orderbook.best_bid is not None and orderbook.best_ask is not None:
                    entry_price = (orderbook.best_bid + orderbook.best_ask) / 2.0
                else:
                    entry_price = orderbook.best_bid if orderbook.best_bid is not None else orderbook.best_ask

            is_trade = (
                signal.signal_side != "HOLD"
                and abs(signal.alpha) >= SIGNAL_THRESHOLD
            )

            row: dict = {
                "date": str(tomorrow),
                "contract_ticker": contract.contract_id,
                "direction": signal.signal_side,
                "alpha": round(signal.alpha, 6),
                "model_prob": round(signal.model_probability, 6),
                "market_prob": round(signal.market_probability, 6),
                "ensemble_mean_f": round(ensemble.ensemble_mean_f, 4),
                "entry_price": round(entry_price, 6) if entry_price is not None else "",
                "position_size": POSITION_SIZE,
                "status": "OPEN" if is_trade else "NO_SIGNAL",
                "realized_pnl": "",
                "settled_temp": "",
                "threshold_f": contract.threshold_f,
                "strike_type": contract.strike_type,
                "floor_strike": contract.floor_strike if contract.floor_strike is not None else "",
                "cap_strike":   contract.cap_strike   if contract.cap_strike   is not None else "",
                "anomaly_flags":           ",".join(f.code for f in anomaly.flags) if anomaly else "",
                "anomaly_penalty_f":       round(anomaly.total_temp_penalty_f, 2) if anomaly else "",
                "anomaly_adjusted_signal": anomaly.adjusted_signal if anomaly else "",
                "anomaly_confidence":      anomaly.confidence if anomaly else "",
                "anomaly_reasoning":       anomaly.reasoning if anomaly else "",
                "spread_rank": 1,
                "spread_total_contracts": 1,
                "sizing_base_usd": "",
                "sizing_final_usd": "",
                "sizing_reasoning": "",
                "city": city.city,
                "nbm_max_f": round(nbm_max_f, 2) if nbm_max_f is not None else "",
                "ens_spread_var": round(ens_spread_var, 4) if ens_spread_var is not None else "",
            }

            all_rows.append(row)
            if dry_run:
                dry_run_signals.append((row, signal, orderbook))

        # ── Deduplicate trade signals ──────────────────────────────────────
        best_by_key: dict[tuple, dict] = {}
        for row in all_rows:
            if row["status"] != "OPEN":
                continue
            # Key on the full strike definition, not threshold alone: a T98
            # tail and a 98-99 bracket share threshold_f but are DIFFERENT
            # outcomes — a threshold-only key silently dropped one of them
            # as a "duplicate" (same family as the probability-surface
            # collision fixed 2026-07-04).
            key = (
                row["strike_type"], row["floor_strike"], row["cap_strike"],
                row["direction"],
            )
            existing = best_by_key.get(key)
            if existing is None or abs(row["alpha"]) > abs(existing["alpha"]):
                if existing is not None:
                    logger.warning(
                        "[%s] Dropping duplicate signal: %s (threshold=%.0f°F already covered"
                        " by %s with higher alpha)",
                        city_label,
                        existing["contract_ticker"],
                        existing["threshold_f"],
                        row["contract_ticker"],
                    )
                best_by_key[key] = row
            else:
                logger.warning(
                    "[%s] Dropping duplicate signal: %s (threshold=%.0f°F already covered"
                    " by %s with higher alpha)",
                    city_label,
                    row["contract_ticker"],
                    row["threshold_f"],
                    existing["contract_ticker"],
                )

        winning_tickers = {r["contract_ticker"] for r in best_by_key.values()}
        for row in all_rows:
            if row["status"] == "OPEN" and row["contract_ticker"] not in winning_tickers:
                row["status"] = "DEDUP_DROP"

        # ── Multi-bracket spreading ────────────────────────────────────────
        multi_bracket_cfg = get_setting("risk.multi_bracket", default={})
        spreading_enabled = multi_bracket_cfg.get("enabled", False)

        if spreading_enabled:
            from deep_isobar.trading.bracket_spreader import build_spread, log_spread_summary

            dynamic_cfg = multi_bracket_cfg.get("dynamic_sizing", {})
            if dynamic_cfg.get("enabled", False):
                from deep_isobar.trading.position_sizer import compute_exposure, log_sizing_decision
                sizing = compute_exposure(
                    anomaly_report=anomaly,
                    ensemble_std_f=ensemble.ensemble_std_f,
                    cfg=dynamic_cfg,
                )
                log_sizing_decision(sizing, logger)
                daily_exposure_cap_usd = sizing.final_exposure_usd
            else:
                sizing = None
                daily_exposure_cap_usd = multi_bracket_cfg.get("daily_exposure_cap_usd", 50.0)

            for row in all_rows:
                if sizing is not None:
                    row["sizing_base_usd"] = sizing.base_exposure_usd
                    row["sizing_final_usd"] = sizing.final_exposure_usd
                    row["sizing_reasoning"] = sizing.reasoning
                else:
                    row["sizing_base_usd"] = daily_exposure_cap_usd
                    row["sizing_final_usd"] = daily_exposure_cap_usd
                    row["sizing_reasoning"] = "dynamic sizing disabled"

            open_rows_pre_spread = [r for r in all_rows if r["status"] == "OPEN"]
            open_signals = [signal_lookup[r["contract_ticker"]] for r in open_rows_pre_spread]

            if open_signals:
                # Actual fill prices (BUY at ask / SELL at bid) — kelly sizes
                # on these, and they convert risk dollars to contract counts.
                entry_prices = {
                    r["contract_ticker"]: float(r["entry_price"])
                    for r in open_rows_pre_spread
                    if r["entry_price"] != ""
                }
                kelly_cfg = dict(multi_bracket_cfg.get("kelly", {}))
                # Haircut for same-airmass correlation: every active city
                # trades the same day, so N defaults to the city count.
                kelly_cfg.setdefault(
                    "n_correlated_bets",
                    sum(1 for c in get_city_universe() if c.active),
                )

                allocations = build_spread(
                    signals=open_signals,
                    daily_exposure_cap_usd=daily_exposure_cap_usd,
                    max_contracts=multi_bracket_cfg.get("max_contracts_per_session", 3),
                    min_alpha=multi_bracket_cfg.get("min_alpha_to_spread", SIGNAL_THRESHOLD),
                    allocation_method=multi_bracket_cfg.get("allocation_method", "proportional"),
                    entry_prices=entry_prices,
                    kelly_cfg=kelly_cfg,
                )
                log_spread_summary(allocations, logger)

                if allocations:
                    n_total = len(allocations)
                    alloc_by_id = {a.signal.contract_id: a for a in allocations}

                    for row in all_rows:
                        if row["status"] != "OPEN":
                            continue
                        ticker = row["contract_ticker"]
                        if ticker in alloc_by_id:
                            alloc = alloc_by_id[ticker]
                            # position_size is CONTRACTS (settlement P&L
                            # multiplies price moves by it); fall back to
                            # risk dollars only when no fill price existed.
                            row["position_size"] = (
                                alloc.contracts if alloc.contracts is not None
                                else alloc.allocated_usd
                            )
                            row["spread_rank"] = alloc.rank
                            row["spread_total_contracts"] = n_total
                        else:
                            row["status"] = "SPREAD_SKIP"
                            row["spread_total_contracts"] = n_total

        # ── Same-day duplicate guard (across session re-runs) ─────────────
        # The dedup above only covers rows produced in this run. If the
        # session is executed again on the same day, every signal would be
        # logged a second time — so check paper_trades.csv for trades
        # already logged for this (date, ticker) and skip them.
        with _CSV_LOCK:
            already_logged = _logged_trade_keys(_PAPER_TRADES_CSV)
        for row in all_rows:
            if (
                row["status"] == "OPEN"
                and (row["date"], row["contract_ticker"]) in already_logged
            ):
                row["status"] = "DUP_SKIP"
                logger.warning(
                    "[%s] Skipping duplicate trade: %s already logged for %s "
                    "in an earlier session run today",
                    city_label,
                    row["contract_ticker"],
                    row["date"],
                )

        # ── Dry-run output ─────────────────────────────────────────────────
        if dry_run:
            signals_logged = 0
            for row, signal, orderbook in dry_run_signals:
                _print_signal(row, signal, orderbook)
                if row["status"] == "OPEN":
                    signals_logged += 1
            result["signals"] = signals_logged
            return result

        # ── Discord notifications (per-city trade embeds) ──────────────────
        open_rows = [r for r in all_rows if r["status"] == "OPEN"]
        if open_rows:
            for row in open_rows:
                entry_str = f"{row['entry_price']}" if row["entry_price"] != "" else "—"
                post_embed(
                    title=f"[{city_label}] Signal: {row['contract_ticker']}",
                    color=COLOR_AMBER,
                    fields=[
                        {"name": "Direction",    "value": row["direction"]},
                        {"name": "Alpha",        "value": f"{row['alpha']:+.4f}"},
                        {"name": "Model prob",   "value": f"{row['model_prob']:.4f}"},
                        {"name": "Market prob",  "value": f"{row['market_prob']:.4f}"},
                        {"name": "Entry price",  "value": entry_str},
                        {"name": "Threshold",    "value": f"{row['threshold_f']:.0f}\u00b0F"},
                    ],
                )
        else:
            post_embed(
                title=f"[{city_label}] No signals today",
                description=f"No contracts met the |alpha| \u2265 {SIGNAL_THRESHOLD} threshold for {tomorrow}.",
                color=COLOR_GRAY,
            )

        # ── Write to CSV files (thread-safe) ──────────────────────────────
        signals_logged = 0
        with _CSV_LOCK:
            _ensure_csv(_PAPER_TRADES_CSV)
            _ensure_csv(_DAILY_LOG_CSV)

            for row in all_rows:
                _append_csv_row(_DAILY_LOG_CSV, row)

                if row["status"] == "OPEN":
                    _append_csv_row(_PAPER_TRADES_CSV, row)
                    signals_logged += 1
                    logger.info(
                        "[%s] TRADE LOGGED: %s %s  alpha=%+.3f  entry=%s  threshold=%.0f°F"
                        "  pos=$%.2f  spread=#%s/%s",
                        city_label,
                        row["direction"],
                        row["contract_ticker"],
                        row["alpha"],
                        row["entry_price"] or 0.0,
                        row["threshold_f"],
                        row["position_size"],
                        row["spread_rank"],
                        row["spread_total_contracts"],
                    )
                else:
                    logger.info(
                        "[%s] No trade: %-35s  alpha=%+.3f  side=%s  status=%s",
                        city_label,
                        row["contract_ticker"],
                        row["alpha"],
                        row["direction"],
                        row["status"],
                    )

        logger.info(
            "[%s] Session complete: %d trade(s) logged",
            city_label, signals_logged,
        )
        result["signals"] = signals_logged
        return result

    except Exception as exc:  # noqa: BLE001
        tb_lines = traceback.format_exc().splitlines()[-20:]
        while tb_lines and len("\n".join(tb_lines)) > 1016:
            tb_lines.pop(0)
        tb_truncated = "\n".join(tb_lines)
        exc_type = type(exc).__name__

        logger.exception("[%s] Unhandled exception in contract evaluation loop", city_label)

        try:
            post_embed(
                title=f"[{city_label}] Session crashed \u2014 unhandled exception",
                color=COLOR_RED,
                fields=[
                    {"name": "Exception type", "value": exc_type},
                    {"name": "Message",        "value": str(exc) or "(no message)"},
                    {"name": "Traceback (tail)", "value": f"```\n{tb_truncated}\n```"},
                ],
            )
        except Exception:  # noqa: BLE001
            logger.warning("[%s] Failed to send error embed to Discord", city_label, exc_info=True)

        result["error"] = f"{exc_type}: {exc}"
        return result


# ---------------------------------------------------------------------------
# Multi-city orchestrator
# ---------------------------------------------------------------------------


def main(dry_run: bool = False) -> None:
    """Load all active cities and run a paper trade session for each.

    Uses ``ThreadPoolExecutor(max_workers=4)`` to run cities concurrently.
    One city failure never blocks others.  Posts a single summary Discord
    embed at the end listing every city with its result.

    Args:
        dry_run: When ``True``, print signals to stdout without writing to
            any CSV file.
    """
    today = date.today()
    tomorrow = today + timedelta(days=1)

    all_profiles = get_city_universe()
    active_cities = [p for p in all_profiles if p.active]

    if not active_cities:
        logger.warning("No active cities found in config/cities.yaml — nothing to do.")
        return

    city_names = [c.city for c in active_cities]
    logger.info(
        "=== Multi-city session  target_date=%s  cities=%s ===",
        tomorrow, city_names,
    )

    results: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_city = {
            executor.submit(run_city_session, city, dry_run): city
            for city in active_cities
        }
        for future in as_completed(future_to_city):
            city = future_to_city[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                logger.exception("[%s] Future raised unexpectedly: %s", city.city, exc)
                result = {"city": city.city, "signals": 0, "error": str(exc)}
            results[city.city] = result

    # ── Session log ────────────────────────────────────────────────────────
    session_log_path = _PAPER_TRADES_DIR / "session_log.txt"
    session_log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with session_log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n=== Session {ts}  target={tomorrow} ===\n")
        for city_name, r in results.items():
            if r["error"]:
                fh.write(f"  {city_name}: ERROR — {r['error']}\n")
            else:
                fh.write(f"  {city_name}: {r['signals']} signal(s)\n")

    # ── Summary Discord embed ──────────────────────────────────────────────
    if not dry_run:
        summary_fields = []
        for city_name, r in results.items():
            if r["error"]:
                value = f"\u274c Error: {r['error'][:80]}"
            elif r["signals"] == 0:
                value = "\u23f9 No signals"
            else:
                value = f"\u2705 {r['signals']} signal(s) logged"
            summary_fields.append({"name": city_name, "value": value})

        post_embed(
            title=f"Daily Session Summary \u2014 {tomorrow}",
            color=COLOR_GREEN if all(not r["error"] for r in results.values()) else COLOR_AMBER,
            fields=summary_fields,
        )

    total_signals = sum(r["signals"] for r in results.values())
    errors = [r for r in results.values() if r["error"]]
    logger.info(
        "=== All cities complete: %d signal(s) total, %d error(s) ===",
        total_signals, len(errors),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Daily paper trade session for all active cities.\n\n"
            "Loads active cities from config/cities.yaml, fetches GFS T+24\n"
            "forecasts, evaluates live Kalshi contracts, and logs any trade\n"
            f"signals (|alpha| >= {SIGNAL_THRESHOLD}) to data/paper_trades/."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print signals to stdout without writing to any CSV file.",
    )
    args = parser.parse_args()

    today = date.today()
    tomorrow = today + timedelta(days=1)

    if args.dry_run:
        print(f"\n[DRY RUN] Paper trade session — {today}")
        print(f"  Target date      : {tomorrow}")
        print(f"  Signal threshold : |alpha| >= {SIGNAL_THRESHOLD}")
        print(f"  Position size    : {POSITION_SIZE} contracts")
        print("-" * 55)

    main(dry_run=args.dry_run)
