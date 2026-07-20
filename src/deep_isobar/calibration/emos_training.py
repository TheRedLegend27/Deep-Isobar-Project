"""EMOS training-data builder and station fitter for Deep Isobar.

Builds paired (member daily-max forecasts, observed settlement high)
training rows for each station and fits per-station EMOS parameters.

Forecast side — Open-Meteo **Previous Runs API**: ``temperature_2m_previous_day1``
is the hourly 2 m temperature trace as forecast ~one day earlier, i.e. the
same T+1 lead the live session trades.  The daily max is taken over the
**local settlement day** (max-of-trace), matching how NWS computes the
climate-report high — NOT the legacy 18z snapshot, which sampled hours
before the afternoon peak and caused the Dallas cold bias.

Observation side — ACIS settlement highs via
:func:`deep_isobar.data.historical_noaa_ingest.fetch_settlement_observations`
(the authoritative settlement source for both Kalshi and Polymarket).

Training rows accumulate in ``data/emos_training/{station_id}_t1_training.parquet``
(appended and deduplicated by date), so the rolling window can eventually
grow past the API's ~92-day history limit via daily runs.

CLI::

    python -m deep_isobar.calibration.emos_training            # all active cities
    python -m deep_isobar.calibration.emos_training --city Chicago
    python -m deep_isobar.calibration.emos_training --window-days 60
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from deep_isobar.calibration.emos import EMOSParams, fit_emos, save_params
from deep_isobar.core.lst import lst_timezone
from deep_isobar.core.types import CityProfile
from deep_isobar.data.city_universe import get_city_universe
from deep_isobar.data.ensemble_ingest import (
    fetch_member_daily_maxes,
    load_t1_spread_history,
    pooled_member_variance,
    record_t1_spread,
)
from deep_isobar.data.historical_noaa_ingest import fetch_settlement_observations

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_TRAINING_DIR = _PROJECT_ROOT / "data" / "emos_training"

_PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
_HTTP_TIMEOUT_S = 60

# Member set: must match the live session's Open-Meteo model codes so the
# fitted coefficients apply to live forecasts of the same quantity.  NBM is
# NOAA's operationally calibrated blend (research report 2026-06, Q2) — it
# enters the mean regression like any other member; its ensemble means are
# deliberately NOT added as further predictors (collinear with GFS/ECMWF).
EMOS_MODELS: dict[str, str] = {
    "GFS":   "gfs_seamless",
    "ECMWF": "ecmwf_ifs025",
    "ICON":  "icon_seamless",
    "GEM":   "gem_seamless",
    "NBM":   "ncep_nbm_conus",
}

# Open-Meteo Previous Runs API serves at most ~92 days of history.
MAX_PAST_DAYS = 92
DEFAULT_WINDOW_DAYS = 90

# Fraction of window days that must have recorded ensemble spread before the
# variance model trains on it (spread history accumulates one day per run).
MIN_SPREAD_COVERAGE = 0.6


def fetch_t1_member_maxes(
    lat: float,
    lon: float,
    timezone_name: str,
    past_days: int = MAX_PAST_DAYS,
    agg: str = "max",
) -> pd.DataFrame:
    """Fetch T+1-lead daily-max forecasts per member from Open-Meteo.

    Requests the hourly ``temperature_2m_previous_day1`` trace for every
    model in :data:`EMOS_MODELS` in the station's local timezone, then takes
    the max over each local calendar day (max-of-trace).

    Args:
        lat / lon: Settlement-station coordinates.
        timezone_name: IANA timezone of the settlement day.
        past_days: How many past days to request (max ~92).

    Returns:
        DataFrame indexed by ``date`` (datetime.date) with one column per
        model name (``GFS``, ``ECMWF``, ...), values in deg F (NaN when a
        model had no data for that day).
    """
    om_ids = ",".join(EMOS_MODELS.values())
    # Fixed standard-time zone, NOT the DST-aware local zone: the CLI
    # settlement day is midnight-midnight LST year-round, so the grouping
    # below must see LST timestamps (see deep_isobar.core.lst).
    url = (
        f"{_PREVIOUS_RUNS_URL}?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m_previous_day1,temperature_2m_previous_day2"
        f"&models={om_ids}"
        "&temperature_unit=fahrenheit"
        f"&timezone={quote(lst_timezone(timezone_name), safe='')}"
        f"&past_days={min(past_days, MAX_PAST_DAYS)}"
        "&forecast_days=1"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "deep-isobar/1.0"})
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
        data = json.loads(resp.read())
    if data.get("error"):
        raise ValueError(f"Open-Meteo error: {data.get('reason', 'unknown')}")

    hourly: dict = data.get("hourly", {})
    times = pd.to_datetime(hourly.get("time", []))
    if len(times) == 0:
        raise ValueError("Open-Meteo previous-runs response had no hourly times")

    frame = pd.DataFrame({"local_day": times.date})
    frame2 = pd.DataFrame({"local_day": times.date})  # previous_day2 view
    for model, om_id in EMOS_MODELS.items():
        vals = hourly.get(f"temperature_2m_previous_day1_{om_id}")
        if vals is None and len(EMOS_MODELS) == 1:
            vals = hourly.get("temperature_2m_previous_day1")
        frame[model] = pd.to_numeric(pd.Series(vals if vals is not None else [np.nan] * len(times)), errors="coerce")
        vals2 = hourly.get(f"temperature_2m_previous_day2_{om_id}")
        if vals2 is None and len(EMOS_MODELS) == 1:
            vals2 = hourly.get("temperature_2m_previous_day2")
        frame2[model] = pd.to_numeric(pd.Series(vals2 if vals2 is not None else [np.nan] * len(times)), errors="coerce")

    def _daily(fr):
        return (fr.groupby("local_day").min(numeric_only=True) if agg == "min"
                else fr.groupby("local_day").max(numeric_only=True))

    daily = _daily(frame)
    daily2 = _daily(frame2)
    # Run-to-run forecast trend: how much the multi-model mean view of each
    # target day moved between the D-2 and D-1 runs.  Regime-change days
    # (fronts) show big values — the variance model's e_trend widens sigma
    # there (2026-07-05 bust mode: NY forecast fell ~14F day-over-day).
    daily["trend_f"] = daily[list(EMOS_MODELS)].mean(axis=1) - daily2[list(EMOS_MODELS)].mean(axis=1)
    daily.index.name = "date"
    for model in EMOS_MODELS:
        if model in daily.columns and daily[model].isna().all():
            logger.warning(
                "previous-runs: model %s returned no data — its column is "
                "all-NaN and those training rows will be dropped at fit time",
                model,
            )
    # Drop today's partial day — the trace is incomplete until local midnight.
    return daily[daily.index < date.today()]


def _runview_path(station_id: str, training_dir: Path | None = None) -> Path:
    d = training_dir or _TRAINING_DIR
    return d / f"{station_id}_runviews.parquet"


def _record_run_views(
    station_id: str, views: dict[date, float], training_dir: Path | None = None
) -> None:
    """Bank today's multi-model mean view of each future target day.

    The Previous Runs API serves ``previous_dayN`` only for PAST timestamps
    (for future hours it mirrors the current run — verified live 2026-07-19),
    so yesterday's view of tomorrow is not fetchable.  Like the T+1 spread
    history, run views therefore accumulate forward: each session records
    today's views, and tomorrow's session diffs against them.
    """
    path = _runview_path(station_id, training_dir)
    rows = pd.DataFrame(
        {
            "recorded_date": [date.today()] * len(views),
            "target_date": list(views.keys()),
            "mean_view_f": list(views.values()),
        }
    )
    if path.exists():
        rows = pd.concat([pd.read_parquet(path), rows], ignore_index=True)
    rows = rows.drop_duplicates(subset=["recorded_date", "target_date"], keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(path, index=False)


def fetch_live_trend_f(
    lat: float,
    lon: float,
    timezone_name: str,
    target_date: date,
    station_id: str,
    agg: str = "max",
) -> float | None:
    """Run-to-run trend of *target_date*'s multi-model mean view, at session time.

    Matches the training column exactly: the training row for target day T
    holds ``pd1(T) − pd2(T)`` — the view of T issued on T−1 minus the view
    issued on T−2.  At session time (day T−1) the first term is today's
    run, fetched fresh; the second is yesterday's session's recorded view
    of T (see :func:`_record_run_views`).  Feed the result to
    :func:`emos_predict` as ``trend_f`` so the fitted ``e_trend`` variance
    term applies live.

    Always records today's views of T+1/T+2 as a side effect, so the diff
    becomes available from the second session onward.  Returns None when
    yesterday's view of *target_date* was never recorded (bootstrap day,
    missed session) or the fetch has no usable rows.
    """
    om_ids = ",".join(EMOS_MODELS.values())
    url = (
        f"{_PREVIOUS_RUNS_URL}?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m"
        f"&models={om_ids}"
        "&temperature_unit=fahrenheit"
        f"&timezone={quote(lst_timezone(timezone_name), safe='')}"
        "&forecast_days=3"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "deep-isobar/1.0"})
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
        data = json.loads(resp.read())
    if data.get("error"):
        raise ValueError(f"Open-Meteo error: {data.get('reason', 'unknown')}")

    hourly: dict = data.get("hourly", {})
    times = pd.to_datetime(hourly.get("time", []))
    if len(times) == 0:
        return None

    frame = pd.DataFrame({"local_day": times.date})
    for model, om_id in EMOS_MODELS.items():
        vals = hourly.get(f"temperature_2m_{om_id}")
        if vals is None and len(EMOS_MODELS) == 1:
            vals = hourly.get("temperature_2m")
        frame[model] = pd.to_numeric(
            pd.Series(vals if vals is not None else [np.nan] * len(times)), errors="coerce"
        )
    daily = (frame.groupby("local_day").min(numeric_only=True) if agg == "min"
             else frame.groupby("local_day").max(numeric_only=True))
    # Same skipna mean-across-members as the training column.
    today_views = {
        d: float(daily.loc[d].mean())
        for d in daily.index
        if d > date.today() and math.isfinite(daily.loc[d].mean())
    }
    if today_views:
        _record_run_views(station_id, today_views)

    current = today_views.get(target_date)
    if current is None:
        return None
    path = _runview_path(station_id)
    if not path.exists():
        return None
    history = pd.read_parquet(path)
    prior = history[
        (history["target_date"] == target_date)
        & (history["recorded_date"] < date.today())
    ].sort_values("recorded_date")
    if prior.empty:
        return None
    trend = current - float(prior.iloc[-1]["mean_view_f"])
    return trend if math.isfinite(trend) else None


def _fit_option(station_key: str, name: str, default):
    """Fit option for one station key, with per-station config override.

    ``emos.station_overrides.<key>.<name>`` beats the global ``emos.<name>``.
    The key is the fit key — ``KDFW`` for highs, ``KDFW_low`` for lows — so
    the two models of one station can be configured independently.  Model
    changes still ship only through the holdout race
    (research/validate_emos_variants.py).
    """
    from deep_isobar.config import get_setting

    overrides = get_setting("emos.station_overrides", default={}) or {}
    station_cfg = overrides.get(station_key) or {}
    if name in station_cfg:
        return station_cfg[name]
    return get_setting(f"emos.{name}", default=default)


def _training_path(station_id: str, training_dir: Path | None = None) -> Path:
    d = training_dir or _TRAINING_DIR
    return d / f"{station_id}_t1_training.parquet"


def _station_key(city: CityProfile, metric: str) -> str:
    """Filesystem key for params/training/spread: highs keep the bare
    station id (backward compatible); lows append ``_low``."""
    return city.station_id if metric == "high" else f"{city.station_id}_low"


def build_training_data(
    city: CityProfile,
    past_days: int = MAX_PAST_DAYS,
    training_dir: Path | None = None,
    metric: str = "high",
) -> pd.DataFrame:
    """Build/refresh the paired training parquet for one city.

    Fetches member daily maxes and ACIS settlement highs, joins on date,
    and merges into the existing training parquet (new dates win).

    Returns:
        The full merged training DataFrame with columns ``date``,
        one column per model, and ``actual_f``.
    """
    if not (city.nws_lat and city.nws_lon):
        raise ValueError(f"{city.city}: nws_lat/nws_lon required for EMOS training")

    key = _station_key(city, metric)
    agg = "min" if metric == "low" else "max"
    obs_col = "low_temp_f" if metric == "low" else "high_temp_f"
    forecasts = fetch_t1_member_maxes(
        city.nws_lat, city.nws_lon, city.timezone, past_days, agg=agg
    )
    logger.info(
        "[%s] previous-runs forecasts: %d days (%s .. %s)",
        city.city, len(forecasts), forecasts.index.min(), forecasts.index.max(),
    )

    # Record tomorrow's live T+1 pooled GEFS/EPS member spread — historical
    # member spread is not retrievable (see ensemble_ingest docstring), so
    # the history accumulates one day per run.  Best-effort: a failed fetch
    # never blocks the training build.
    try:
        members_by_model = fetch_member_daily_maxes(
            city.nws_lat, city.nws_lon, city.timezone,
            date.today() + timedelta(days=1),
            daily_var="temperature_2m_min" if metric == "low" else "temperature_2m_max",
        )
        live_var = pooled_member_variance(members_by_model)
        if live_var is not None:
            record_t1_spread(key, date.today() + timedelta(days=1), live_var)
    except Exception:  # noqa: BLE001
        logger.exception("[%s] live ensemble spread fetch failed — not recorded", city.city)

    obs = fetch_settlement_observations(
        city.city,
        start_date=forecasts.index.min(),
        end_date=forecasts.index.max(),
    )
    actuals = (
        obs[["target_date", obs_col]]
        .rename(columns={"target_date": "date", obs_col: "actual_f"})
        .set_index("date")
    )

    merged = forecasts.join(actuals, how="inner").reset_index()
    merged["date"] = pd.to_datetime(merged["date"]).dt.date
    merged = merged.dropna(subset=["actual_f"])

    path = _training_path(key, training_dir)
    if path.exists():
        try:
            existing = pd.read_parquet(path)
            existing["date"] = pd.to_datetime(existing["date"]).dt.date
            existing = existing[~existing["date"].isin(set(merged["date"]))]
            merged = pd.concat([existing, merged], ignore_index=True)
        except Exception:
            logger.exception(
                "[%s] could not read existing training parquet — rebuilding fresh",
                city.city,
            )
    # The spread-history parquet is the single source of truth for
    # ens_spread_var — re-join it fully each run rather than merging stale
    # column values through the dedup above.
    spread_hist = load_t1_spread_history(key)
    merged = merged.drop(columns=["ens_spread_var"], errors="ignore")
    if not spread_hist.empty:
        merged = merged.merge(spread_hist, on="date", how="left")
    else:
        merged["ens_spread_var"] = np.nan
    logger.info(
        "[%s] ensemble T+1 spread coverage: %d/%d training days",
        city.city, int(merged["ens_spread_var"].notna().sum()), len(merged),
    )

    merged = merged.sort_values("date").reset_index(drop=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False)
    logger.info(
        "[%s] training parquet: %d paired days → %s", city.city, len(merged), path
    )
    return merged


def fit_station(
    city: CityProfile,
    window_days: int = DEFAULT_WINDOW_DAYS,
    training_dir: Path | None = None,
    params_dir: Path | None = None,
    metric: str = "high",
) -> EMOSParams:
    """Fit EMOS for one station from its training parquet and save params.

    Uses the most recent *window_days* of paired rows (rolling window — the
    research report's recommended alternative to per-month fits while the
    history is still short).
    """
    key = _station_key(city, metric)
    path = _training_path(key, training_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No training parquet for {key} — run build_training_data first"
        )
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    cutoff = date.today() - timedelta(days=window_days)
    window = df[df["date"] >= cutoff].sort_values("date")

    # Tolerate parquets built before a model was added (e.g. pre-NBM): fit
    # on the columns that exist — params.model_names records what was used.
    model_names = [m for m in EMOS_MODELS if m in window.columns]
    missing = [m for m in EMOS_MODELS if m not in window.columns]
    if missing:
        logger.warning(
            "[%s] training parquet lacks %s — fitting on %s only; re-run the "
            "fetch to add the missing member(s)",
            city.city, missing, model_names,
        )
    members = window[model_names].to_numpy(dtype=float)
    actuals = window["actual_f"].to_numpy(dtype=float)

    # Train the variance on ensemble spread only once coverage clears
    # MIN_SPREAD_COVERAGE — a `d` fit mostly on the member-variance fallback
    # would be scaled to the wrong predictor.  Below the gate the params
    # stay "members"-source and live prediction matches automatically.
    ens_spread_var = None
    if "ens_spread_var" in window.columns and len(window):
        coverage = float(window["ens_spread_var"].notna().mean())
        if coverage >= MIN_SPREAD_COVERAGE:
            ens_spread_var = window["ens_spread_var"].to_numpy(dtype=float)
        else:
            logger.info(
                "[%s] ensemble spread coverage %.0f%% < %.0f%% — fitting on "
                "member variance until history accumulates",
                city.city, coverage * 100, MIN_SPREAD_COVERAGE * 100,
            )

    use_trend = bool(_fit_option(key, "trend_variance", default=False))
    params = fit_emos(
        members,
        actuals,
        model_names,
        station_id=key,
        train_dates=(window["date"].min(), window["date"].max()),
        ensemble_spread_var=ens_spread_var,
        trend=(window["trend_f"].to_numpy(dtype=float)
               if use_trend and "trend_f" in window.columns else None),
        nonneg_weights=bool(_fit_option(key, "nonneg_weights", default=False)),
        ridge_lambda=float(_fit_option(key, "ridge_lambda", default=0.0)),
    )
    params.metric = "low_temp_f" if metric == "low" else "high_temp_f"
    save_params(params, params_dir=params_dir)
    return params


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build EMOS training data and fit per-station params")
    parser.add_argument("--city", help="Single city name (default: all active cities)")
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    parser.add_argument("--skip-fetch", action="store_true", help="Fit from existing parquet only")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cities = [c for c in get_city_universe() if c.active]
    if args.city:
        cities = [c for c in cities if c.city.lower() == args.city.lower()]
        if not cities:
            print(f"City {args.city!r} not found among active cities", file=sys.stderr)
            return 2

    failures = 0
    for city in cities:
        metrics = ["high"] + (["low"] if city.kalshi_low_series else [])
        for metric in metrics:
            try:
                if not args.skip_fetch:
                    build_training_data(city, metric=metric)
                params = fit_station(
                    city, window_days=args.window_days, metric=metric
                )
                print(
                    f"{city.city:>14} ({params.station_id}): n={params.n_train}  "
                    f"a0={params.a0:+.2f}  a={[round(v, 3) for v in params.a]}  "
                    f"c={params.c:.3f}  d={params.d:.3f}  "
                    f"train CRPS={params.train_crps:.3f}degF"
                )
            except Exception as exc:  # noqa: BLE001
                failures += 1
                logger.exception("[%s/%s] EMOS build/fit failed", city.city, metric)
                print(f"{city.city:>14} ({metric}): FAILED — {exc}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
