"""Chicago (CHI) backtest driver for Deep Isobar — real historical data.

Replaces the synthetic simulation with three real archived data sources:

  - **NOAA/ACIS settlement observations**
    ``data/historical/settlement/chicago_2023.parquet``
    → daily realized high_temp_f (the truth)

  - **GFS archived forecast runs** (AWS Open Data)
    ``data/historical/forecasts/gfs_chicago_2023.parquet``
    → what the model was actually predicting at 1–5 day lead times

  - **Kalshi order-book snapshots**
    ``data/historical/markets/kalshi_kxhighchi_2023.parquet``
    → actual bid/ask prices for KXHIGHCHI contracts

  - **Kalshi contract metadata**
    ``data/historical/markets/kalshi_kxhighchi_2023_contracts.parquet``
    → threshold_f and target_date per contract_id

Generate the parquet files before running::

    python -m deep_isobar.data.historical_noaa_ingest \\
        --city Chicago --start 2023-05-01 --end 2023-09-30 \\
        --out data/historical/settlement/chicago_2023.parquet

    python -m deep_isobar.data.historical_forecast_ingest \\
        --city Chicago --year 2023 --month 8 \\
        --out data/historical/forecasts/gfs_chicago_2023.parquet

    python -m deep_isobar.market.historical_kalshi_ingest \\
        --series KXHIGHCHI --start 2023-05-01 --end 2023-09-30 \\
        --out data/historical/markets/kalshi_kxhighchi_2023.parquet

Train / test split
------------------
Calibration window : May–July 2023   (derive bias_correction & variance_multiplier here)
Holdout window     : Aug–Sep  2023   (evaluate out-of-sample performance here)

The calibrated parameters are derived exclusively from the calibration window.
Holdout data never touches the parameter estimation step.

Pipeline per (target_date, threshold_f)
-----------------------------------------
1. Realized high_temp_f from settlement parquet  (truth)
2. Best GFS forecast for target_date (shortest lead ≤ 48 h, else ≤ 72 h)
3. Build EnsembleSummary via ``temperature_ensemble``
4. Compute P(high ≥ threshold) via ``probability_engine``
5. Last Kalshi snapshot before decision cutoff (noon UTC on target_date)
6. Evaluate contract opportunity → TradeSignal
7. Settle with actual high_temp_f from NOAA

Usage::

    python -m deep_isobar.research.run_chicago_backtest

or::

    python src/deep_isobar/research/run_chicago_backtest.py
"""

from __future__ import annotations

import dataclasses
import logging
import math
import statistics
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from deep_isobar.core.types import (
    CityProfile,
    ForecastPoint,
    MarketContract,
    OrderBookSnapshot,
)
from deep_isobar.data.city_universe import get_city_profile
from deep_isobar.market.market_scanner import evaluate_contract_opportunity
from deep_isobar.market.microstructure_scanner import compute_microstructure_score
from deep_isobar.models.temperature_ensemble import build_temperature_ensemble
from deep_isobar.models.probability_engine import probability_ge_normal
from deep_isobar.research.backtest_engine import (
    simulate_backtest_trades,
    summarize_backtest_results,
)
from deep_isobar.trading.distribution_tail_alpha import is_tail_threshold

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_chicago_backtest")

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_SETTLEMENT_PATH = _PROJECT_ROOT / "data/historical/settlement/chicago_2023.parquet"
_FORECAST_PATH   = _PROJECT_ROOT / "data/historical/forecasts/gfs_chicago_2023.parquet"
_SNAPSHOTS_PATH  = _PROJECT_ROOT / "data/historical/markets/kalshi_kxhighchi_2023.parquet"
_CONTRACTS_PATH  = _PROJECT_ROOT / "data/historical/markets/kalshi_kxhighchi_2023_contracts.parquet"

# 2024 out-of-time data  (generate with the same ingest scripts, year=2024)
_SETTLEMENT_PATH_2024 = _PROJECT_ROOT / "data/historical/settlement/chicago_2024.parquet"
_FORECAST_PATH_2024   = _PROJECT_ROOT / "data/historical/forecasts/gfs_chicago_2024.parquet"
_SNAPSHOTS_PATH_2024  = _PROJECT_ROOT / "data/historical/markets/kalshi_kxhighchi_2024.parquet"
_CONTRACTS_PATH_2024  = _PROJECT_ROOT / "data/historical/markets/kalshi_kxhighchi_2024_contracts.parquet"

QUANTITY         = 10.0    # contracts per trade
KALSHI_FEE_RATE  = 0.07    # 7 % fee deducted from gross profit on winning trades
SIGNAL_THRESHOLD = 0.25    # minimum |alpha| to generate BUY/SELL
COMPARISON_OPERATOR = "ge"
METRIC           = "high_temp_f"
THRESHOLDS       = [80, 90]
# Minimum ensemble std when only one GFS run covers a date (single-point → σ=0).
# 5.5 °F represents a conservative short-range uncertainty floor for Chicago summer.
_MIN_ENSEMBLE_STD_F = 5.5

# Use the shortest lead available up to this many hours for the GFS forecast.
# 48 h = day+1 (00z f042 / 12z f030). Falls back to 72 h if day+1 absent.
_PRIMARY_MAX_LEAD   = 48
_FALLBACK_MAX_LEAD  = 72

# Decision cutoff: the last Kalshi snapshot before this UTC hour on target_date
# is used as the entry price. 12:00 UTC = ~7 am CDT, well before the daily
# high temperature is observed (typically 14:00–19:00 CDT).
_DECISION_CUTOFF_HOUR_UTC = 12

# September heat-bias adjustment applied when the month is anomalously warm.
# Value derived from the 2024 OOT analysis: Sep 2024 mean daily high was
# +7.57°F above the 1991-2020 NOAA normal, producing a +2.187°F DRIFT flag.
# A -2.0°F correction reduces that to ~+0.19°F (within the ±2.0°F threshold).
_SEP_HEAT_BIAS_ADJUSTMENT_F = -2.0

# ---------------------------------------------------------------------------
# Train / test date windows  — never mix these
# ---------------------------------------------------------------------------

# Calibration window: bias_correction and variance_multiplier are derived here.
_CALIBRATION_START = date(2023, 5, 1)
_CALIBRATION_END   = date(2023, 7, 31)   # inclusive

# Holdout window: out-of-sample, same year — no parameter fitting.
_HOLDOUT_START     = date(2023, 8, 1)
_HOLDOUT_END       = date(2023, 9, 30)   # inclusive

# Out-of-time window: completely independent year — 2023 params applied verbatim.
_OOT_START         = date(2024, 5, 1)
_OOT_END           = date(2024, 9, 30)   # inclusive


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _require(path: Path, script_hint: str) -> None:
    """Raise FileNotFoundError with a helpful message if *path* is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing data file: {path}\n"
            f"Generate it with:\n    {script_hint}"
        )


def _load_settlement(path: Path | None = None) -> pd.DataFrame:
    """Load and validate a NOAA settlement parquet.

    *path* defaults to ``_SETTLEMENT_PATH`` (2023 data).  Pass
    ``_SETTLEMENT_PATH_2024`` (or any other path) for a different year.
    """
    path = path or _SETTLEMENT_PATH
    year = path.stem.split("_")[-1]   # e.g. "chicago_2023" → "2023"
    _require(
        path,
        f"python -m deep_isobar.data.historical_noaa_ingest "
        f"--city Chicago --start {year}-05-01 --end {year}-09-30 "
        f"--out {path}",
    )
    df = pd.read_parquet(path)
    # Normalise target_date to datetime.date
    if pd.api.types.is_datetime64_any_dtype(df["target_date"]):
        df["target_date"] = df["target_date"].dt.date
    df = df[df["quality_flag"] != "missing"].copy()
    logger.info("Settlement (%s): %d usable rows loaded", year, len(df))
    return df


def _load_forecasts(path: Path | None = None) -> pd.DataFrame:
    """Load and validate a GFS forecast parquet.

    *path* defaults to ``_FORECAST_PATH`` (2023 data).
    """
    path = path or _FORECAST_PATH
    year = path.stem.split("_")[-1]
    _require(
        path,
        f"python -m deep_isobar.data.historical_forecast_ingest "
        f"--city Chicago --year {year} --month 8 "
        f"--out {path}",
    )
    df = pd.read_parquet(path)
    if pd.api.types.is_datetime64_any_dtype(df["target_date"]):
        df["target_date"] = df["target_date"].dt.date
    # Ensure run_time_utc is timezone-aware
    if df["run_time_utc"].dt.tz is None:
        df["run_time_utc"] = df["run_time_utc"].dt.tz_localize("UTC")
    logger.info("Forecasts (%s): %d rows loaded", year, len(df))
    return df


def _load_market_data(
    snapshots_path: Path | None = None,
    contracts_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Kalshi snapshots and contract metadata.

    Both paths default to the 2023 files.  Pass the 2024 constants for
    the out-of-time validation.

    Returns ``(snapshots_df, contracts_df)``.
    """
    snapshots_path = snapshots_path or _SNAPSHOTS_PATH
    contracts_path = contracts_path or _CONTRACTS_PATH
    year = snapshots_path.stem.split("_")[-1]
    _require(
        snapshots_path,
        f"python -m deep_isobar.market.historical_kalshi_ingest "
        f"--series KXHIGHCHI --start {year}-05-01 --end {year}-09-30 "
        f"--out {snapshots_path}",
    )
    snapshots = pd.read_parquet(snapshots_path)
    if snapshots["timestamp_utc"].dt.tz is None:
        snapshots["timestamp_utc"] = snapshots["timestamp_utc"].dt.tz_localize("UTC")

    if contracts_path.exists():
        contracts = pd.read_parquet(contracts_path)
        if pd.api.types.is_datetime64_any_dtype(contracts["target_date"]):
            contracts["target_date"] = contracts["target_date"].dt.date
        contracts["threshold_f"] = pd.to_numeric(
            contracts["threshold_f"], errors="coerce"
        )
    else:
        logger.warning(
            "Contracts parquet not found at %s — deriving contract metadata "
            "from T-format contract_ids in snapshots.",
            contracts_path,
        )
        # Derive contracts table from T-format snapshot contract_ids.
        # Format: HIGHCHI-24AUG08-T80 or KXHIGHCHI-24AUG08-T80
        t_snaps = snapshots[
            snapshots["contract_id"].str.contains(r"-T\d", regex=True, na=False)
        ][["contract_id"]].drop_duplicates().copy()
        t_snaps["threshold_f"] = (
            t_snaps["contract_id"].str.extract(r"-T(\d+)")[0].astype(float)
        )
        # Parse target_date from the date component (e.g. 24AUG08 -> 2024-08-08)
        t_snaps["target_date"] = pd.to_datetime(
            t_snaps["contract_id"].str.extract(r"-(\d{2}[A-Z]{3}\d{2})-")[0],
            format="%y%b%d",
        ).dt.date
        contracts = t_snaps.dropna(subset=["threshold_f", "target_date"]).copy()

    # Keep only T-format contracts
    if "contract_id" in contracts.columns and not contracts.empty:
        contracts = contracts[
            contracts["contract_id"].str.contains(r"-T\d", regex=True, na=False)
        ].copy()

    logger.info(
        "Market (%s): %d snapshot rows, %d T-format contract rows loaded",
        year, len(snapshots), len(contracts),
    )
    return snapshots, contracts


# ---------------------------------------------------------------------------
# Per-date helpers
# ---------------------------------------------------------------------------


def _build_contract_index(
    contracts_df: pd.DataFrame,
) -> tuple[dict[tuple, str], dict[date, list[int]]]:
    """Build lookup structures from a contracts DataFrame.

    Returns
    -------
    idx_map : dict[(target_date, threshold_int) -> contract_id]
    date_thresholds : dict[target_date -> sorted list[int]]
    """
    idx_map: dict[tuple, str] = {}
    date_thresholds: dict[date, list[int]] = defaultdict(list)
    for _, row in contracts_df.iterrows():
        td  = row["target_date"]
        thr = row["threshold_f"]
        cid = row["contract_id"]
        if pd.notna(thr) and pd.notna(td):
            key = (td, int(thr))
            idx_map[key] = cid
            date_thresholds[td].append(int(thr))
    date_thresholds = {td: sorted(thrs) for td, thrs in date_thresholds.items()}
    return idx_map, date_thresholds


def _select_forecast(
    forecasts_df: pd.DataFrame,
    target_date: date,
) -> list[ForecastPoint]:
    """Return the best available GFS ForecastPoints for *target_date*.

    Prefers the shortest lead ≤ 48 h (day+1).  Falls back to ≤ 72 h.
    Returns an empty list when no qualifying forecast exists.
    When multiple cycles (00z / 12z) are available at the same lead tier,
    all qualifying rows are returned so the ensemble can weight them.
    """
    day_rows = forecasts_df[forecasts_df["target_date"] == target_date]
    if day_rows.empty:
        return []

    for max_lead in (_PRIMARY_MAX_LEAD, _FALLBACK_MAX_LEAD):
        candidates = day_rows[day_rows["lead_hours"] <= max_lead]
        if not candidates.empty:
            # Keep only the rows at the minimum lead hour available
            min_lead = candidates["lead_hours"].min()
            best = candidates[candidates["lead_hours"] == min_lead]
            return [_row_to_forecast_point(row) for _, row in best.iterrows()]

    return []


def _row_to_forecast_point(row: pd.Series) -> ForecastPoint:
    """Convert a parquet row to a ForecastPoint dataclass."""
    return ForecastPoint(
        city=str(row["city"]),
        station_id=str(row["station_id"]),
        model_name=str(row["model_name"]),
        run_time_utc=row["run_time_utc"],
        target_date=row["target_date"],
        metric=str(row["metric"]),
        forecast_value_f=float(row["forecast_value_f"]),
        lead_hours=int(row["lead_hours"]),
        source_name=str(row["source_name"]),
    )


def _select_market_snapshot(
    snapshots_df: pd.DataFrame,
    contract_id: str,
    decision_cutoff: datetime,
) -> tuple[float, float] | None:
    """Return (best_bid, best_ask) from the last snapshot before *decision_cutoff*.

    Returns ``None`` if no qualifying snapshot exists.
    """
    contract_snaps = snapshots_df[snapshots_df["contract_id"] == contract_id]
    before = contract_snaps[contract_snaps["timestamp_utc"] < decision_cutoff]
    if before.empty:
        return None
    last = before.sort_values("timestamp_utc").iloc[-1]
    return float(last["best_bid"]), float(last["best_ask"])


def _build_contracts_index(
    contracts_df: pd.DataFrame,
    thresholds: list[int],
) -> pd.DataFrame:
    """Return a DataFrame with columns [contract_id, target_date, threshold_f].

    Filters the contract metadata to only the requested thresholds.
    Returns an empty DataFrame when *contracts_df* is empty or contains
    no rows matching *thresholds*.
    """
    if contracts_df.empty:
        return pd.DataFrame(columns=["contract_id", "target_date", "threshold_f"])
    return contracts_df[
        contracts_df["threshold_f"].isin(thresholds)
    ][["contract_id", "target_date", "threshold_f"]].copy()


# ---------------------------------------------------------------------------
# Window runner
# ---------------------------------------------------------------------------


def _run_window(
    window_dates: list[date],
    settlement_by_date: dict[date, float],
    forecasts_df: pd.DataFrame,
    snapshots_df: pd.DataFrame,
    idx_map: dict[tuple, str],
    date_thresholds: dict[date, list[int]],
    city_profile: CityProfile,
) -> tuple[pd.DataFrame, list[dict]]:
    """Evaluate opportunities over *window_dates* using *city_profile*.

    Returns
    -------
    opportunities_df : pd.DataFrame
        One row per BUY/SELL signal (empty DataFrame when none generated).
    error_records : list[dict]
        One entry per date where a forecast existed, with keys:
        ``target_date``, ``ensemble_mean_f``, ``ensemble_std_f``,
        ``true_high_f``.  Used by the calibration phase to derive
        bias_correction and variance_multiplier.
    """
    opportunity_rows: list[dict] = []
    error_records: list[dict] = []
    skipped_no_forecast = 0
    skipped_no_snapshot = 0
    evaluated_count = 0

    for target_date in window_dates:
        true_high_f = settlement_by_date[target_date]

        # ── GFS forecast → ensemble ────────────────────────────────────────
        forecast_points = _select_forecast(forecasts_df, target_date)
        if not forecast_points:
            skipped_no_forecast += len(date_thresholds.get(target_date, [1]))
            logger.debug("No GFS forecast for %s — skipping", target_date)
            continue

        ensemble_run_time = forecast_points[0].run_time_utc
        ensemble = build_temperature_ensemble(
            city_profile=city_profile,
            forecasts=forecast_points,
            target_date=target_date,
            metric=METRIC,
            ensemble_run_time_utc=ensemble_run_time,
        )

        # Always record raw forecast error for calibration diagnostics.
        error_records.append({
            "target_date":      target_date,
            "ensemble_mean_f":  ensemble.ensemble_mean_f,
            "ensemble_std_f":   ensemble.ensemble_std_f,
            "true_high_f":      true_high_f,
        })

        # ── Probability surface ────────────────────────────────────────────
        # Guard: a single GFS run produces std=0; apply a minimum floor so
        # the probability engine can still compute a useful estimate.
        effective_std = max(ensemble.adjusted_std_f, _MIN_ENSEMBLE_STD_F)

        # Derive thresholds for this date from contracts (dynamic per Kalshi listing)
        day_thresholds = date_thresholds.get(target_date, [])
        if not day_thresholds:
            logger.debug("No T-format contracts for %s — skipping", target_date)
            continue

        probability_surface: dict[int, float] = {
            thr: probability_ge_normal(
                mean_f=ensemble.bias_corrected_mean_f,
                std_f=effective_std,
                threshold_f=float(thr),
            )
            for thr in day_thresholds
        }

        # Decision cutoff: noon UTC on target_date
        decision_cutoff = datetime(
            target_date.year, target_date.month, target_date.day,
            _DECISION_CUTOFF_HOUR_UTC, 0, 0, tzinfo=timezone.utc,
        )

        # ── Evaluate each threshold ────────────────────────────────────────
        for threshold_f in day_thresholds:
            contract_id = idx_map.get((target_date, threshold_f))
            if contract_id is None:
                skipped_no_snapshot += 1
                continue

            price = _select_market_snapshot(snapshots_df, contract_id, decision_cutoff)
            if price is None:
                skipped_no_snapshot += 1
                logger.debug(
                    "No snapshot for %s before %s — skipping",
                    contract_id,
                    decision_cutoff.isoformat(),
                )
                continue

            best_bid, best_ask = price
            evaluated_count += 1

            contract = MarketContract(
                contract_id=contract_id,
                market_source="Kalshi",
                city="Chicago",
                metric=METRIC,
                comparison_operator=COMPARISON_OPERATOR,
                threshold_f=threshold_f,
                target_date=target_date,
                settlement_source="NWS",
            )

            snapshot = OrderBookSnapshot(
                timestamp_utc=decision_cutoff,
                contract_id=contract_id,
                market_source="Kalshi",
                best_bid=best_bid,
                best_ask=best_ask,
            )

            micro_score = compute_microstructure_score(snapshot, decision_cutoff)
            tail_flag = is_tail_threshold(
                ensemble_mean_f=ensemble.bias_corrected_mean_f,
                threshold_f=threshold_f,
                adjusted_std_f=effective_std,
            )

            signal = evaluate_contract_opportunity(
                contract=contract,
                probability_surface=probability_surface,
                orderbook=snapshot,
                signal_threshold=SIGNAL_THRESHOLD,
                timestamp_utc=decision_cutoff,
                microstructure_score=micro_score,
                tail_opportunity_flag=tail_flag,
            )

            if signal.signal_side == "HOLD":
                continue

            # Fill at the side you actually cross: BUY lifts the ask,
            # SELL hits the bid.  Mid-price fills overstate PnL by ~½ spread.
            if signal.signal_side == "BUY":
                simulated_price = best_ask
            else:
                simulated_price = best_bid
            realized_outcome = int(true_high_f >= threshold_f)

            opportunity_rows.append({
                "contract_id":        contract_id,
                "target_date":        target_date,
                "threshold_f":        threshold_f,
                "side":               signal.signal_side,
                "model_probability":  signal.model_probability,
                "market_probability": signal.market_probability,
                "alpha":              signal.alpha,
                "absolute_alpha":     signal.absolute_alpha,
                "rank_score":         signal.rank_score,
                "simulated_price":    simulated_price,
                "realized_outcome":   realized_outcome,
            })

    label = (
        f"{window_dates[0]} → {window_dates[-1]}"
        if window_dates else "empty"
    )
    logger.info(
        "Window [%s]: evaluated=%d  skipped_no_forecast=%d  skipped_no_snapshot=%d",
        label, evaluated_count, skipped_no_forecast, skipped_no_snapshot,
    )

    if not opportunity_rows:
        return pd.DataFrame(), error_records
    return pd.DataFrame(opportunity_rows), error_records


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------


def _derive_calibration_params(
    error_records: list[dict],
) -> tuple[float, float]:
    """Derive bias_correction_f and variance_multiplier from calibration errors.

    Both parameters are derived from *calibration-window data only*.

    ``mean_bias_correction_f``
        Negative of the mean forecast error (forecast_mean − actual).
        Corrects the systematic warm/cold bias in the raw ensemble mean.

    ``variance_multiplier``
        Ratio of the RMSE (after bias removal) to the mean effective ensemble
        spread.  The "effective" spread uses the same ``_MIN_ENSEMBLE_STD_F``
        floor applied during trading to keep the estimator consistent.

    Returns
    -------
    (mean_bias_correction_f, variance_multiplier)
    """
    if not error_records:
        logger.warning("_derive_calibration_params: no error records — returning neutral (0, 1)")
        return 0.0, 1.0

    raw_errors    = [r["ensemble_mean_f"] - r["true_high_f"] for r in error_records]
    raw_stds      = [r["ensemble_std_f"] for r in error_records]

    mean_error        = statistics.mean(raw_errors)
    bias_correction   = -mean_error

    # RMSE after bias removal — this is the residual spread we want the
    # calibrated std to match.
    bias_removed = [e - mean_error for e in raw_errors]
    rmse_after_bias = math.sqrt(statistics.mean(x * x for x in bias_removed))

    # Use the same floor as the trading loop so the ratio is dimensionally consistent.
    effective_stds = [max(s, _MIN_ENSEMBLE_STD_F) for s in raw_stds]
    mean_effective_std = statistics.mean(effective_stds)

    variance_multiplier = (
        rmse_after_bias / mean_effective_std if mean_effective_std > 0 else 1.0
    )

    logger.info("  n_dates            : %d", len(error_records))
    logger.info("  mean_error         : %+.3f °F  (raw forecast − actual)", mean_error)
    logger.info("  rmse_after_bias    : %.3f °F", rmse_after_bias)
    logger.info("  mean_effective_std : %.3f °F  (with %.1f °F floor)", mean_effective_std, _MIN_ENSEMBLE_STD_F)
    logger.info("  → bias_correction  : %+.4f °F", bias_correction)
    logger.info("  → variance_mult    : %.4f", variance_multiplier)

    return bias_correction, variance_multiplier, rmse_after_bias


def _compute_sep_heat_bias_adjustment(
    city_profile: CityProfile,
    sep_mean_actual_f: float,
) -> float:
    """Return the September heat-bias adjustment when an anomaly is detected.

    The adjustment is conditional on two criteria both being true:

    1. *city_profile* has ``sep_climate_normal_f`` and ``sep_anomaly_trigger_f``
       configured (not None).
    2. *sep_mean_actual_f* exceeds ``sep_climate_normal_f`` by more than
       ``sep_anomaly_trigger_f`` degrees.

    When triggered, returns ``_SEP_HEAT_BIAS_ADJUSTMENT_F`` (-2.0°F) and logs
    a diagnostic line.  Returns ``0.0`` for normal Septembers or when the
    thresholds are not configured for this city, leaving the main
    ``mean_bias_correction_f`` untouched.

    The threshold is intentionally strict (>5°F above normal) so that
    near-normal Septembers are never penalised by an over-eager correction.
    """
    if (
        city_profile.sep_climate_normal_f is None
        or city_profile.sep_anomaly_trigger_f is None
    ):
        return 0.0

    anomaly = sep_mean_actual_f - city_profile.sep_climate_normal_f
    if anomaly > city_profile.sep_anomaly_trigger_f:
        logger.info(
            "Sep anomaly detected (+%.1fF above %.1fF normal) — "
            "applying heat_bias_adjustment_f: %.1fF",
            anomaly,
            city_profile.sep_climate_normal_f,
            _SEP_HEAT_BIAS_ADJUSTMENT_F,
        )
        return _SEP_HEAT_BIAS_ADJUSTMENT_F

    return 0.0


# ---------------------------------------------------------------------------
# Reporting helper
# ---------------------------------------------------------------------------


def _apply_kalshi_fee(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Deduct Kalshi's KALSHI_FEE_RATE fee from gross profit on winning trades.

    The fee applies only when realized_pnl > 0 (Kalshi charges on profits,
    not on losses).  Both ``pnl_per_unit`` and ``realized_pnl`` are adjusted
    so downstream summaries stay consistent.
    """
    df = trades_df.copy()
    win_mask = df["realized_pnl"] > 0
    df.loc[win_mask, "pnl_per_unit"] *= (1.0 - KALSHI_FEE_RATE)
    df.loc[win_mask, "realized_pnl"] *= (1.0 - KALSHI_FEE_RATE)
    return df


def _compute_sharpe(trades_df: pd.DataFrame) -> tuple[float, float]:
    """Compute raw and annualized Sharpe from daily PnL, with diagnostics.

    Groups ``realized_pnl`` by ``target_date`` to form daily returns, then:

    - **raw Sharpe**        : ``mean / std``  (no annualization)
    - **annualized Sharpe** : ``mean / std * sqrt(252)``

    The annualized figure assumes the strategy runs every calendar trading
    day.  For a seasonal strategy (e.g. summer only) it is an extrapolation;
    the raw Sharpe is the more honest within-period measure.

    Returns ``(raw_sharpe, annualized_sharpe)``, both 0.0 on degenerate input.
    """
    if trades_df.empty or "target_date" not in trades_df.columns:
        return 0.0, 0.0
    daily = trades_df.groupby("target_date")["realized_pnl"].sum()
    n_days = len(daily)
    if n_days < 2:
        return 0.0, 0.0
    mean_pnl = daily.mean()
    std_pnl  = daily.std(ddof=1)
    if std_pnl == 0.0:
        return 0.0, 0.0

    raw_sharpe  = float(mean_pnl / std_pnl)
    ann_sharpe  = float(raw_sharpe * math.sqrt(252))

    # ── Diagnostics ──────────────────────────────────────────────────────
    sample = daily.head(10).round(4).to_dict()
    logger.info(
        "  [Sharpe diag] n_trading_days=%d  mean_daily_pnl=%.4f  "
        "std_daily_pnl=%.4f",
        n_days, mean_pnl, std_pnl,
    )
    logger.info(
        "  [Sharpe diag] raw=%.3f  annualized(×√252)=%.3f  "
        "NOTE: annualized extrapolates %d summer days to a 252-day year",
        raw_sharpe, ann_sharpe, n_days,
    )
    logger.info("  [Sharpe diag] first 10 daily PnL values: %s", sample)

    return raw_sharpe, ann_sharpe


def _report_window(opps_df: pd.DataFrame, label: str) -> dict:
    """Simulate trades from *opps_df*, log all metrics, return summary dict.

    Adds a ``sharpe`` key to the returned dictionary.
    """
    if opps_df.empty:
        logger.warning(
            "%s: No BUY/SELL signals generated — all HOLD or no joinable data. "
            "Try lowering SIGNAL_THRESHOLD (currently %.2f).",
            label, SIGNAL_THRESHOLD,
        )
        return {
            "total_trades": 0, "win_rate": 0.0,
            "gross_pnl": 0.0, "max_drawdown": 0.0,
            "sharpe": 0.0, "sharpe_raw": 0.0, "sharpe_ann": 0.0,
        }

    trades_df = simulate_backtest_trades(opps_df, quantity=QUANTITY)
    trades_df = _apply_kalshi_fee(trades_df)           # deduct 7% fee on wins
    summary   = summarize_backtest_results(trades_df)
    raw_sharpe, ann_sharpe = _compute_sharpe(trades_df)

    logger.info("  fill_model   : ask for BUY, bid for SELL  (spread crossed)")
    logger.info("  kalshi_fee   : %.0f%% deducted from winning trades", KALSHI_FEE_RATE * 100)
    logger.info("  total_trades : %d", summary["total_trades"])
    logger.info("  win_rate     : %.1f%%", summary["win_rate"] * 100)
    logger.info(
        "  gross_pnl    : %.4f  (probability-point × contracts, after fee)",
        summary["gross_pnl"],
    )
    logger.info("  max_drawdown : %.4f", summary["max_drawdown"])
    logger.info("  sharpe_raw   : %.3f  (per-period, no annualization)", raw_sharpe)
    logger.info("  sharpe_ann   : %.3f  (× √252 — seasonal extrapolation)", ann_sharpe)

    # Per-threshold breakdown
    for thr in sorted(opps_df["threshold_f"].unique()):
        sub = trades_df[trades_df["contract_id"].str.contains(
            f"-T{int(thr)}$", regex=True, na=False
        )]
        if sub.empty:
            continue
        ss = summarize_backtest_results(sub)
        logger.info(
            "  >= %d°F  trades=%d  win_rate=%.1f%%  gross_pnl=%.4f",
            int(thr), ss["total_trades"], ss["win_rate"] * 100, ss["gross_pnl"],
        )

    summary["sharpe_raw"] = raw_sharpe
    summary["sharpe_ann"] = ann_sharpe
    summary["sharpe"]     = raw_sharpe   # primary key = raw (honest)
    return summary


# ---------------------------------------------------------------------------
# Forecast-error OOT helpers  (no Kalshi data required)
# ---------------------------------------------------------------------------


def _run_forecast_error_oot(
    oot_dates: list[date],
    settlement_by_date: dict[date, float],
    forecasts_df: pd.DataFrame,
    city_profile: CityProfile,
) -> list[dict]:
    """Collect raw ensemble forecast errors for *oot_dates*.

    Mirrors the error-record half of ``_run_window`` but requires no Kalshi
    market data — pure meteorological calibration transfer test.

    Returns a list of dicts with keys:
        ``target_date``, ``ensemble_mean_f``, ``ensemble_std_f``, ``true_high_f``

    ``ensemble_mean_f`` is the **raw** (pre-bias-correction) ensemble mean so
    the same ``_derive_calibration_params`` arithmetic can be applied in post.
    """
    error_records: list[dict] = []
    skipped = 0
    for target_date in oot_dates:
        true_high_f = settlement_by_date.get(target_date)
        if true_high_f is None:
            skipped += 1
            continue
        forecast_points = _select_forecast(forecasts_df, target_date)
        if not forecast_points:
            skipped += 1
            logger.debug("No GFS forecast for %s — skipping", target_date)
            continue
        ensemble_run_time = forecast_points[0].run_time_utc
        ensemble = build_temperature_ensemble(
            city_profile=city_profile,
            forecasts=forecast_points,
            target_date=target_date,
            metric=METRIC,
            ensemble_run_time_utc=ensemble_run_time,
        )
        error_records.append({
            "target_date":     target_date,
            "ensemble_mean_f": ensemble.ensemble_mean_f,
            "ensemble_std_f":  ensemble.ensemble_std_f,
            "true_high_f":     true_high_f,
        })

    logger.info(
        "OOT forecast error collection: %d dates evaluated, %d skipped",
        len(error_records), skipped,
    )
    return error_records


def _report_forecast_accuracy_oot(
    oot_error_records: list[dict],
    cal_error_records: list[dict],
    cal_bias: float,
    cal_rmse_after_bias: float,
    city_profile: CityProfile,
) -> dict:
    """Log and return forecast accuracy metrics for the OOT window.

    Parameters
    ----------
    oot_error_records   : error dicts from ``_run_forecast_error_oot``
    cal_error_records   : error dicts from the calibration Phase 1 pass
    cal_bias            : mean_bias_correction_f derived from calibration
    cal_rmse_after_bias : RMSE of calibration errors after bias removal

    Metrics computed
    ----------------
    - **OOT mean bias** : mean corrected residual (should be ~0 if correction transfers)
    - **OOT RMSE**      : root-mean-squared corrected residual
    - **OOT std**       : std-dev of corrected residuals (spread around mean bias)
    - **Per-month bias** : flag seasonal drift (e.g. model runs hot in Aug only)
    - **calibration_transfer_score** : 1 - (OOT_RMSE / CAL_RMSE_after_bias)
      1.0 = perfect transfer; 0.0 = OOT RMSE equals uncalibrated CAL RMSE;
      negative = calibration hurt OOT generalization.

    Returns a dict with the scalar metrics.
    """
    if not oot_error_records:
        logger.warning("FORECAST ACCURACY OOT: no error records — skipping metrics")
        return {
            "oot_n": 0, "oot_mean_bias": None, "oot_rmse": None,
            "oot_std": None, "calibration_transfer_score": None,
        }

    # Cal raw errors (neutral profile — used to derive cal_bias = -mean_error)
    cal_raw_errors = [r["ensemble_mean_f"] - r["true_high_f"] for r in cal_error_records]
    cal_mean_error = statistics.mean(cal_raw_errors)   # = -cal_bias
    cal_corrected  = [e - cal_mean_error for e in cal_raw_errors]
    cal_bias_metric = statistics.mean(cal_corrected)   # should be ~0 by construction

    # OOT raw errors, then corrected using 2023-derived cal_bias
    # bias_corrected_error = raw_error + cal_bias = raw_error - cal_mean_error
    oot_raw_errors = [r["ensemble_mean_f"] - r["true_high_f"] for r in oot_error_records]
    oot_corrected  = [e - cal_mean_error for e in oot_raw_errors]

    # ── September heat-bias anomaly adjustment ────────────────────────────
    # Applied only when the observed Sep mean daily high exceeds the city's
    # configured climate normal by more than sep_anomaly_trigger_f.  Normal
    # Septembers (within ±5°F of 72.7°F for Chicago) receive 0.0 adjustment
    # so the main mean_bias_correction_f handles them unchanged.
    sep_actuals = [
        r["true_high_f"] for r in oot_error_records
        if r["target_date"].month == 9
    ]
    sep_mean_actual_f = statistics.mean(sep_actuals) if sep_actuals else 0.0
    sep_heat_adj_f = _compute_sep_heat_bias_adjustment(city_profile, sep_mean_actual_f)

    record_adjs = [
        sep_heat_adj_f if r["target_date"].month == 9 else 0.0
        for r in oot_error_records
    ]
    oot_corrected_adj = [c + a for c, a in zip(oot_corrected, record_adjs)]

    oot_mean_bias = statistics.mean(oot_corrected_adj)
    oot_rmse      = math.sqrt(statistics.mean(x * x for x in oot_corrected_adj))
    oot_std       = statistics.stdev(oot_corrected_adj) if len(oot_corrected_adj) > 1 else 0.0

    transfer_score = (
        1.0 - (oot_rmse / cal_rmse_after_bias) if cal_rmse_after_bias > 0 else float("nan")
    )

    # Per-month breakdown (uses Sep-adjusted errors)
    month_errors: dict[int, list[float]] = defaultdict(list)
    for rec, cerr in zip(oot_error_records, oot_corrected_adj):
        month_errors[rec["target_date"].month].append(cerr)

    month_names = {5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep"}

    logger.info("")
    logger.info("=" * 60)
    logger.info("FORECAST ACCURACY OOT (2024)  — no trade signals, no PnL")
    logger.info("-" * 60)
    logger.info("  Calibration params applied verbatim (no refitting):")
    logger.info("    mean_bias_correction_f    : %+.4f F", cal_bias)
    logger.info("    cal_rmse_after_bias       : %.4f F  (2023 calibration window)", cal_rmse_after_bias)
    if sep_heat_adj_f != 0.0:
        logger.info(
            "    sep_heat_bias_adjustment_f: %.1fF  "
            "(Sep mean actual %.2fF > %.1fF normal + %.1fF trigger)",
            sep_heat_adj_f,
            sep_mean_actual_f,
            city_profile.sep_climate_normal_f,
            city_profile.sep_anomaly_trigger_f,
        )
    else:
        logger.info(
            "    sep_heat_bias_adjustment_f: 0.0  "
            "(Sep mean actual %.2fF within normal range)",
            sep_mean_actual_f,
        )
    logger.info("-" * 60)
    logger.info("  OOT n_dates    : %d", len(oot_error_records))
    logger.info(
        "  OOT mean bias  : %+.4f F  (target ~0; 2023 cal was %+.4f F)",
        oot_mean_bias, cal_bias_metric,
    )
    logger.info("  OOT RMSE       : %.4f F  (2023 cal = %.4f F)", oot_rmse, cal_rmse_after_bias)
    logger.info("  OOT std        : %.4f F", oot_std)
    logger.info(
        "  transfer_score : %.4f  (1.0=perfect; 0.0=no better than uncal; <0=degraded)",
        transfer_score,
    )
    logger.info("-" * 60)
    logger.info("  Per-month corrected bias (drift check):")
    for month in sorted(month_errors):
        errs = month_errors[month]
        mean_bias = statistics.mean(errs)
        flag = "  <-- DRIFT" if abs(mean_bias) > 2.0 else ""
        adj_note = (
            f"  [sep_heat_adj {sep_heat_adj_f:+.1f}F applied]"
            if month == 9 and sep_heat_adj_f != 0.0
            else ""
        )
        logger.info(
            "    %s : n=%d  mean_bias=%+.3f F  rmse=%.3f F%s%s",
            month_names.get(month, str(month)),
            len(errs),
            mean_bias,
            math.sqrt(statistics.mean(x * x for x in errs)),
            flag,
            adj_note,
        )
    logger.info("=" * 60)

    return {
        "oot_n":                      len(oot_error_records),
        "oot_mean_bias":              oot_mean_bias,
        "oot_rmse":                   oot_rmse,
        "oot_std":                    oot_std,
        "calibration_transfer_score": transfer_score,
        "sep_heat_bias_adjustment_f": sep_heat_adj_f,
    }


# ---------------------------------------------------------------------------
# Main backtest  (calibration → holdout)
# ---------------------------------------------------------------------------


def run_backtest() -> dict:
    """Execute the Chicago backtest with a strict calibration / holdout split.

    Phases
    ------
    1. Run calibration window (May–Jul 2023) with a **neutral profile**
       (bias_correction=0, variance_multiplier=1) to collect raw forecast
       errors.  Holdout dates are never touched here.
    2. Derive ``mean_bias_correction_f`` and ``variance_multiplier``
       exclusively from the calibration errors.
    3. Re-run the calibration window with the derived profile to produce
       in-sample performance metrics.
    4. Run the holdout window (Aug–Sep 2023) with the **same derived profile**
       — no re-fitting.  This is the out-of-sample evaluation.

    Returns a dict with keys ``calibration`` and ``holdout`` (each a metrics
    sub-dict) plus ``calibrated_bias_correction_f`` and
    ``calibrated_variance_multiplier``.
    """
    # ── Load 2023 data ─────────────────────────────────────────────────────
    settlement_df  = _load_settlement()
    forecasts_df   = _load_forecasts()
    snapshots_df, contracts_df = _load_market_data()

    # ── City profile (weights / kde_bandwidth taken from YAML) ────────────
    base_profile = get_city_profile("Chicago")

    # ── Build 2023 contract index ──────────────────────────────────────────
    idx_map, date_thresholds = _build_contract_index(contracts_df)

    # Index settlement by date for O(1) lookup.
    settlement_by_date: dict[date, float] = {
        row["target_date"]: float(row["high_temp_f"])
        for _, row in settlement_df.iterrows()
    }

    # ── Identify 2023 active dates and split into windows ─────────────────
    settlement_dates = set(settlement_df["target_date"].unique())
    forecast_dates   = set(forecasts_df["target_date"].unique())
    active_dates     = sorted(settlement_dates & forecast_dates)

    if not active_dates:
        raise RuntimeError(
            "No target dates overlap between settlement and forecast parquets. "
            "Check that both cover the same period."
        )

    cal_dates  = [d for d in active_dates
                  if _CALIBRATION_START <= d <= _CALIBRATION_END]
    hold_dates = [d for d in active_dates
                  if _HOLDOUT_START <= d <= _HOLDOUT_END]

    logger.info("=" * 60)
    logger.info("Deep Isobar — Chicago (CHI) Backtest  [TRAIN / TEST SPLIT + OOT]")
    logger.info("Calibration : %s → %s  (%d dates available)",
                _CALIBRATION_START, _CALIBRATION_END, len(cal_dates))
    logger.info("Holdout     : %s → %s  (%d dates available)",
                _HOLDOUT_START, _HOLDOUT_END, len(hold_dates))
    logger.info("OOT         : %s → %s  (2024 data — loaded separately)",
                _OOT_START, _OOT_END)
    logger.info("Signal threshold : %.2f  |  Quantity : %.0f contracts",
                SIGNAL_THRESHOLD, QUANTITY)
    logger.info("=" * 60)

    if not cal_dates:
        raise RuntimeError(
            "No calibration dates found in May–Jul 2023. "
            "Ensure settlement and forecast parquets cover that range."
        )
    if not hold_dates:
        raise RuntimeError(
            "No holdout dates found in Aug–Sep 2023. "
            "Ensure settlement and forecast parquets cover that range."
        )

    # ── PHASE 1: Calibration error collection (neutral profile) ───────────
    # Zero out all bias/seasonal adjustments so we measure the raw model error.
    # Holdout dates are untouched in this phase.
    logger.info("")
    logger.info("── PHASE 1: Calibration error estimation  [neutral profile] ──")
    neutral_profile = dataclasses.replace(
        base_profile,
        mean_bias_correction_f=0.0,
        variance_multiplier=1.0,
        heat_bias_adjustment_f=0.0,
        cold_bias_adjustment_f=0.0,
    )
    _, cal_error_records = _run_window(
        cal_dates, settlement_by_date, forecasts_df,
        snapshots_df, idx_map, date_thresholds, neutral_profile,
    )

    # ── PHASE 2: Derive calibration parameters ────────────────────────────
    # Parameters are computed exclusively from calibration errors.
    # No holdout data is used here.
    logger.info("")
    logger.info("── PHASE 2: Deriving calibration parameters ──")
    cal_bias, cal_var_mult, cal_rmse_after_bias = _derive_calibration_params(cal_error_records)

    calibrated_profile = dataclasses.replace(
        base_profile,
        mean_bias_correction_f=cal_bias,
        variance_multiplier=cal_var_mult,
        # Zero out seasonal adjustments — the derived bias already absorbs them.
        heat_bias_adjustment_f=0.0,
        cold_bias_adjustment_f=0.0,
    )

    # ── PHASE 3: Calibration window performance ───────────────────────────
    logger.info("")
    logger.info("── PHASE 3: Calibration window performance  [in-sample] ──")
    cal_opps_df, _ = _run_window(
        cal_dates, settlement_by_date, forecasts_df,
        snapshots_df, idx_map, date_thresholds, calibrated_profile,
    )

    # ── PHASE 4: Holdout window performance ──────────────────────────────
    # calibrated_profile is unchanged — holdout data has zero influence
    # on the parameters.
    logger.info("")
    logger.info("── PHASE 4: Holdout window performance  [out-of-sample] ──")
    hold_opps_df, _ = _run_window(
        hold_dates, settlement_by_date, forecasts_df,
        snapshots_df, idx_map, date_thresholds, calibrated_profile,
    )

    # ── Results ───────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("CALIBRATION WINDOW  (May–Jul 2023)  — IN-SAMPLE")
    logger.info("-" * 60)
    cal_summary = _report_window(cal_opps_df, "CALIBRATION")

    logger.info("")
    logger.info("=" * 60)
    logger.info("HOLDOUT WINDOW  (Aug–Sep 2023)  — OUT-OF-SAMPLE")
    logger.info("-" * 60)
    hold_summary = _report_window(hold_opps_df, "HOLDOUT")

    # ── PHASE 5: Forecast-error OOT validation on 2024 data ──────────────
    # No Kalshi data required — pure meteorological calibration transfer test.
    # Uses 2023-derived params verbatim; zero refitting on 2024 data.
    logger.info("")
    logger.info("── PHASE 5: Forecast accuracy OOT  [2024, no refitting, no Kalshi] ──")
    oot_accuracy: dict = {"skipped": True}

    _oot_missing = [
        p for p in (_SETTLEMENT_PATH_2024, _FORECAST_PATH_2024) if not p.exists()
    ]
    if _oot_missing:
        logger.warning(
            "PHASE 5 SKIPPED — 2024 data not found: %s\n"
            "  Generate with (run from project root):\n"
            "\n"
            "  # 1. Settlement observations\n"
            "    python -m deep_isobar.data.historical_noaa_ingest"
            " --city Chicago --start 2024-05-01 --end 2024-09-30"
            " --out %s\n"
            "\n"
            "  # 2. GFS forecasts (--month one at a time; same --out merges):\n"
            "    python -m deep_isobar.data.historical_forecast_ingest"
            " --city Chicago --year 2024 --month 5 --out %s\n"
            "    python -m deep_isobar.data.historical_forecast_ingest"
            " --city Chicago --year 2024 --month 6 --out %s\n"
            "    python -m deep_isobar.data.historical_forecast_ingest"
            " --city Chicago --year 2024 --month 7 --out %s\n"
            "    python -m deep_isobar.data.historical_forecast_ingest"
            " --city Chicago --year 2024 --month 8 --out %s\n"
            "    python -m deep_isobar.data.historical_forecast_ingest"
            " --city Chicago --year 2024 --month 9 --out %s",
            [str(p) for p in _oot_missing],
            _SETTLEMENT_PATH_2024,
            _FORECAST_PATH_2024,
            _FORECAST_PATH_2024,
            _FORECAST_PATH_2024,
            _FORECAST_PATH_2024,
            _FORECAST_PATH_2024,
        )
    else:
        settlement_df_2024 = _load_settlement(_SETTLEMENT_PATH_2024)
        forecasts_df_2024  = _load_forecasts(_FORECAST_PATH_2024)

        settlement_by_date_2024: dict[date, float] = {
            row["target_date"]: float(row["high_temp_f"])
            for _, row in settlement_df_2024.iterrows()
        }

        oot_active = sorted(
            set(settlement_df_2024["target_date"].unique())
            & set(forecasts_df_2024["target_date"].unique())
        )
        oot_dates = [d for d in oot_active if _OOT_START <= d <= _OOT_END]
        logger.info(
            "OOT dates available: %d  (%s to %s)",
            len(oot_dates),
            oot_dates[0] if oot_dates else "-",
            oot_dates[-1] if oot_dates else "-",
        )

        # neutral_profile: raw GFS errors (bias_correction=0) so we can apply
        # the 2023-derived correction in _report_forecast_accuracy_oot the same
        # way we did during calibration Phase 1.
        oot_error_records = _run_forecast_error_oot(
            oot_dates, settlement_by_date_2024, forecasts_df_2024, neutral_profile,
        )

        oot_accuracy = _report_forecast_accuracy_oot(
            oot_error_records=oot_error_records,
            cal_error_records=cal_error_records,
            cal_bias=cal_bias,
            cal_rmse_after_bias=cal_rmse_after_bias,
            city_profile=base_profile,
        )
        oot_accuracy["skipped"] = False

    logger.info("")
    logger.info("=" * 60)
    logger.info("Derived calibration params  (safe to write to cities.yaml):")
    logger.info("  mean_bias_correction_f : %+.4f", cal_bias)
    logger.info("  variance_multiplier    : %.4f", cal_var_mult)
    logger.info("  (heat/cold_bias_adjustment_f set to 0.0 — absorbed into bias)")
    logger.info("=" * 60)

    return {
        "calibration":   cal_summary,
        "holdout":       hold_summary,
        "oot_accuracy":  oot_accuracy,
        "calibrated_bias_correction_f":   cal_bias,
        "calibrated_variance_multiplier": cal_var_mult,
        "calibrated_rmse_after_bias":     cal_rmse_after_bias,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    result = run_backtest()

    def _fmt_window(w: dict) -> None:
        print(f"  total_trades : {w['total_trades']}")
        print(f"  win_rate     : {w['win_rate'] * 100:.1f}%")
        print(f"  gross_pnl    : {w['gross_pnl']:.4f}  (after spread + 7% fee)")
        print(f"  max_drawdown : {w['max_drawdown']:.4f}")
        print(f"  sharpe_raw   : {w.get('sharpe_raw', w['sharpe']):.3f}  (per-period)")
        print(f"  sharpe_ann   : {w.get('sharpe_ann', 0.0):.3f}  (× √252, seasonal extrap.)")

    print("\n" + "=" * 60)
    print("CALIBRATION  (May–Jul 2023)  — IN-SAMPLE")
    print("-" * 60)
    _fmt_window(result["calibration"])

    print("\n" + "=" * 60)
    print("HOLDOUT  (Aug–Sep 2023)  — OUT-OF-SAMPLE")
    print("-" * 60)
    _fmt_window(result["holdout"])

    print("\n" + "=" * 60)
    print("FORECAST ACCURACY OOT (2024)  — no trade signals, no PnL")
    print("[2023-calibrated params applied verbatim; no refitting]")
    print("-" * 60)
    oot = result["oot_accuracy"]
    if oot.get("skipped"):
        print("  SKIPPED — 2024 settlement/forecast files not found (see log)")
    else:
        print(f"  n_dates          : {oot['oot_n']}")
        bias = oot["oot_mean_bias"]
        rmse = oot["oot_rmse"]
        std  = oot["oot_std"]
        ts   = oot["calibration_transfer_score"]
        cal_rmse = result["calibrated_rmse_after_bias"]
        print(f"  mean_bias        : {bias:+.4f} F  (target ~0; cal = ~0 by construction)")
        print(f"  RMSE             : {rmse:.4f} F  (2023 cal RMSE = {cal_rmse:.4f} F)")
        print(f"  std_residuals    : {std:.4f} F")
        print(f"  transfer_score   : {ts:.4f}  (1.0=perfect; <0=worse than uncal)")

    print("\n" + "=" * 60)
    print("Calibrated parameters (derived from 2023 calibration window only):")
    print(f"  mean_bias_correction_f : {result['calibrated_bias_correction_f']:+.4f}")
    print(f"  variance_multiplier    : {result['calibrated_variance_multiplier']:.4f}")
    print("=" * 60)
