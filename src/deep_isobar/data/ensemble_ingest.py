"""GEFS / ECMWF-EPS ensemble-member ingestion via the Open-Meteo Ensemble API.

The 2026-06 research report (Q2) found that the EMOS variance predictor was
being fed the spread of 4 deterministic models — a much weaker flow-dependent
uncertainty signal than true ensemble-member spread.  This module supplies
that signal from the two big global ensembles:

    GEFS  — NCEP GEFS 0.25 deg, control + 30 perturbed members
    EPS   — ECMWF IFS 0.25 deg ensemble, control + 50 perturbed members

Both are served by Open-Meteo with a native daily ``temperature_2m_max``
aggregated in the station's local timezone (max-of-trace, the same quantity
EMOS is trained on).

Live side  — :func:`fetch_member_daily_maxes` returns every member's daily
max for a target date from the Ensemble API.

Training side — historical T+1 member spread is NOT retrievable: the
Previous Runs API returns all-null ``_previous_day1`` values for ensemble
models, and the Ensemble API's ``past_days`` archive holds short-lead
(near-analysis) runs whose spread is ~3x too tight (verified 2026-07-01:
archived GEFS daily-max std ~0.5-0.9 F vs 1.83 F for the live T+1 run).
So the spread history is accumulated forward instead: each morning run
records the live T+1 pooled variance via :func:`record_t1_spread`, and
EMOS training joins :func:`load_t1_spread_history` by target date.  Until
coverage builds up, fitting falls back to deterministic-member variance.

Spread pooling: the per-day variance is computed within each ensemble
(ddof=1, control included) and the two variances are averaged.  Averaging
within-model variances keeps inter-model mean offsets (model bias) out of
the spread signal — that bias belongs to the EMOS mean coefficients, not
the variance.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_SPREAD_DIR = _PROJECT_ROOT / "data" / "emos_training"

_ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
_HTTP_TIMEOUT_S = 120

# Open-Meteo model ids per Deep Isobar ensemble name.
ENSEMBLE_MODELS: dict[str, str] = {
    "GEFS": "ncep_gefs025",
    "EPS":  "ecmwf_ifs025_ensemble",
}


def _http_get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "deep-isobar/1.0"})
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
        data = json.loads(resp.read())
    if isinstance(data, dict) and data.get("error"):
        raise ValueError(f"Open-Meteo error: {data.get('reason', 'unknown')}")
    return data


def fetch_member_daily_maxes(
    lat: float,
    lon: float,
    timezone_name: str,
    target_date: date,
    models: dict[str, str] | None = None,
    daily_var: str = "temperature_2m_max",
) -> dict[str, list[float]]:
    """Fetch per-member daily-max forecasts for *target_date* (live path).

    One Ensemble API request covering all *models*; the daily
    ``temperature_2m_max`` is aggregated by Open-Meteo over the station's
    local calendar day.

    Returns:
        ``{ensemble_name: [member maxes in deg F]}`` — control member
        included.  Ensembles with no data for the target date are omitted.

    Raises:
        ValueError: On API error or when *target_date* is outside the
            returned daily range.
    """
    models = models or ENSEMBLE_MODELS
    om_ids = ",".join(models.values())
    url = (
        f"{_ENSEMBLE_URL}?latitude={lat}&longitude={lon}"
        f"&daily={daily_var}"
        f"&models={om_ids}"
        "&temperature_unit=fahrenheit"
        f"&timezone={timezone_name.replace('/', '%2F')}"
        "&forecast_days=3"
    )
    data = _http_get_json(url)

    daily: dict = data.get("daily", {})
    days: list[str] = daily.get("time", [])
    target_str = target_date.strftime("%Y-%m-%d")
    try:
        idx = days.index(target_str)
    except ValueError:
        raise ValueError(
            f"Ensemble API: target day {target_str} not in response range {days[:1]}..{days[-1:]}"
        ) from None

    out: dict[str, list[float]] = {}
    for name, om_id in models.items():
        # Keys: {daily_var}_{om_id} (control) and
        # {daily_var}_memberNN_{om_id} (perturbed members).
        pattern = re.compile(rf"^{re.escape(daily_var)}(_member\d+)?_{re.escape(om_id)}$")
        values: list[float] = []
        for key, series in daily.items():
            if not pattern.match(key):
                continue
            v = series[idx] if series and idx < len(series) else None
            if v is not None:
                values.append(float(v))
        if values:
            out[name] = values
        else:
            logger.warning(
                "Ensemble API: no %s member values for %s at (%.3f, %.3f)",
                name, target_str, lat, lon,
            )
    return out


def pooled_member_variance(members_by_model: dict[str, list[float]]) -> float | None:
    """Average of within-ensemble member variances (ddof=1).

    Returns ``None`` when no ensemble has at least 2 members — callers fall
    back to the climatological spread stored in the EMOS params.
    """
    variances = [
        float(np.var(vals, ddof=1))
        for vals in members_by_model.values()
        if len(vals) >= 2
    ]
    if not variances:
        return None
    return float(np.mean(variances))


def _spread_history_path(station_id: str, spread_dir: Path | None = None) -> Path:
    d = spread_dir or _DEFAULT_SPREAD_DIR
    return d / f"{station_id}_t1_spread.parquet"


def record_t1_spread(
    station_id: str,
    target_date: date,
    spread_var: float,
    spread_dir: Path | None = None,
) -> Path:
    """Append one live T+1 pooled member variance to the station's history.

    Idempotent per (station, target_date): re-recording the same date
    overwrites the previous value (later runs see fresher model cycles).
    Written atomically (tmp + replace) — the morning session threads and the
    training cron may both record around the same time.
    """
    path = _spread_history_path(station_id, spread_dir)
    row = pd.DataFrame({"date": [target_date], "ens_spread_var": [float(spread_var)]})
    if path.exists():
        try:
            existing = pd.read_parquet(path)
            existing["date"] = pd.to_datetime(existing["date"]).dt.date
            existing = existing[existing["date"] != target_date]
            row = pd.concat([existing, row], ignore_index=True)
        except Exception:
            logger.exception("Could not read %s — rewriting with one row", path)
    row = row.sort_values("date").reset_index(drop=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    row.to_parquet(tmp, index=False)
    tmp.replace(path)
    logger.info(
        "Recorded T+1 spread for %s %s: %.2f degF^2 (%d days of history)",
        station_id, target_date, spread_var, len(row),
    )
    return path


def load_t1_spread_history(
    station_id: str,
    spread_dir: Path | None = None,
) -> pd.DataFrame:
    """Load the accumulated T+1 spread history for *station_id*.

    Returns:
        DataFrame with columns ``date`` (datetime.date) and
        ``ens_spread_var``; empty when no history exists yet.
    """
    path = _spread_history_path(station_id, spread_dir)
    if not path.exists():
        return pd.DataFrame(columns=["date", "ens_spread_var"])
    try:
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df
    except Exception:
        logger.exception("Could not read spread history %s — treating as empty", path)
        return pd.DataFrame(columns=["date", "ens_spread_var"])
