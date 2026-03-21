"""Dallas (DFW) backtest driver for Deep Isobar — real historical data.

Mirrors the Chicago backtest framework (run_chicago_backtest.py) but targets
Dallas/Fort Worth using KDFW observations and KXHIGHDFW Kalshi contracts.

Data sources required (same pipeline as Chicago):

  - **NOAA/ACIS settlement observations**
    ``data/historical/settlement/dallas_2023.parquet``
    → daily realized high_temp_f for KDFW

  - **GFS archived forecast runs** (AWS Open Data)
    ``data/historical/forecasts/gfs_dallas_2023.parquet``
    → GFS predictions at 1–5 day lead times for KDFW grid point

  - **Kalshi order-book snapshots**
    ``data/historical/markets/kalshi_kxhightdal_2023.parquet``
    → bid/ask prices for KXHIGHTDAL contracts

  - **Kalshi contract metadata**
    ``data/historical/markets/kalshi_kxhightdal_2023_contracts.parquet``
    → threshold_f and target_date per contract_id

Generate the parquet files before running::

    python -m deep_isobar.data.historical_noaa_ingest \\
        --city Dallas --start 2023-06-01 --end 2023-09-30 \\
        --out data/historical/settlement/dallas_2023.parquet

    python -m deep_isobar.data.historical_forecast_ingest \\
        --city Dallas --year 2023 --month 8 \\
        --out data/historical/forecasts/gfs_dallas_2023.parquet

    python -m deep_isobar.market.historical_kalshi_ingest \\
        --series KXHIGHTDAL --start 2023-06-01 --end 2023-09-30 \\
        --out data/historical/markets/kalshi_kxhightdal_2023.parquet

Dallas-specific design choices vs. Chicago:
  - Thresholds are [100, 105] °F — 80/90°F are almost always exceeded in
    DFW August, so those contracts carry negligible signal.
  - _MIN_ENSEMBLE_STD_F = 5.5°F (same conservative floor as Chicago).
  - Decision cutoff is noon UTC = 7 am CDT, before the afternoon peak.

Usage::

    python -m deep_isobar.research.run_dallas_backtest

or::

    python src/deep_isobar/research/run_dallas_backtest.py
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from deep_isobar.core.types import (
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
logger = logging.getLogger("run_dallas_backtest")

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_SETTLEMENT_PATH = _PROJECT_ROOT / "data/historical/settlement/dallas_2023.parquet"
_FORECAST_PATH   = _PROJECT_ROOT / "data/historical/forecasts/gfs_dallas_2023.parquet"
_SNAPSHOTS_PATH  = _PROJECT_ROOT / "data/historical/markets/kalshi_kxhightdal_2023.parquet"
_CONTRACTS_PATH  = _PROJECT_ROOT / "data/historical/markets/kalshi_kxhightdal_2023_contracts.parquet"

QUANTITY            = 10.0    # contracts per trade
SIGNAL_THRESHOLD    = 0.08    # minimum |alpha| to generate BUY/SELL
COMPARISON_OPERATOR = "ge"
METRIC              = "high_temp_f"
# 100°F and 105°F — the relevant Kalshi thresholds for DFW summer.
# 80/90°F contracts are almost always ITM in August and carry little edge.
THRESHOLDS          = [100, 105]

# Minimum ensemble std when only one GFS run covers a date (single-point → σ=0).
# 5.5 °F is a conservative short-range uncertainty floor for DFW summer.
_MIN_ENSEMBLE_STD_F = 5.5

# Use the shortest lead available up to this many hours for the GFS forecast.
_PRIMARY_MAX_LEAD   = 48
_FALLBACK_MAX_LEAD  = 72

# Decision cutoff: last Kalshi snapshot before noon UTC on target_date.
# Noon UTC = 7 am CDT — well before the afternoon high is observed.
_DECISION_CUTOFF_HOUR_UTC = 12


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


def _load_settlement() -> pd.DataFrame:
    """Load and validate the NOAA settlement parquet for Dallas."""
    _require(
        _SETTLEMENT_PATH,
        "python -m deep_isobar.data.historical_noaa_ingest "
        "--city Dallas --start 2023-06-01 --end 2023-09-30 "
        f"--out {_SETTLEMENT_PATH}",
    )
    df = pd.read_parquet(_SETTLEMENT_PATH)
    if pd.api.types.is_datetime64_any_dtype(df["target_date"]):
        df["target_date"] = df["target_date"].dt.date
    df = df[df["quality_flag"] != "missing"].copy()
    logger.info("Settlement: %d usable rows loaded", len(df))
    return df


def _load_forecasts() -> pd.DataFrame:
    """Load and validate the GFS forecast parquet for Dallas."""
    _require(
        _FORECAST_PATH,
        "python -m deep_isobar.data.historical_forecast_ingest "
        "--city Dallas --year 2023 --month 8 "
        f"--out {_FORECAST_PATH}",
    )
    df = pd.read_parquet(_FORECAST_PATH)
    if pd.api.types.is_datetime64_any_dtype(df["target_date"]):
        df["target_date"] = df["target_date"].dt.date
    if df["run_time_utc"].dt.tz is None:
        df["run_time_utc"] = df["run_time_utc"].dt.tz_localize("UTC")
    logger.info("Forecasts: %d rows loaded", len(df))
    return df


def _load_market_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Kalshi snapshots and contract metadata for KXHIGHDFW.

    Returns ``(snapshots_df, contracts_df)``.
    """
    _require(
        _SNAPSHOTS_PATH,
        "python -m deep_isobar.market.historical_kalshi_ingest "
        "--series KXHIGHTDAL --start 2023-06-01 --end 2023-09-30 "
        f"--out {_SNAPSHOTS_PATH}",
    )
    snapshots = pd.read_parquet(_SNAPSHOTS_PATH)
    if snapshots["timestamp_utc"].dt.tz is None:
        snapshots["timestamp_utc"] = snapshots["timestamp_utc"].dt.tz_localize("UTC")

    if _CONTRACTS_PATH.exists():
        contracts = pd.read_parquet(_CONTRACTS_PATH)
        if pd.api.types.is_datetime64_any_dtype(contracts["target_date"]):
            contracts["target_date"] = contracts["target_date"].dt.date
        contracts["threshold_f"] = pd.to_numeric(
            contracts["threshold_f"], errors="coerce"
        )
    else:
        logger.warning(
            "Contracts metadata not found at %s — threshold_f will be parsed "
            "from contract_id strings. Results may be incomplete.",
            _CONTRACTS_PATH,
        )
        contracts = pd.DataFrame()

    logger.info(
        "Market: %d snapshot rows, %d contract rows loaded",
        len(snapshots),
        len(contracts),
    )
    return snapshots, contracts


# ---------------------------------------------------------------------------
# Per-date helpers
# ---------------------------------------------------------------------------


def _select_forecast(
    forecasts_df: pd.DataFrame,
    target_date: date,
) -> list[ForecastPoint]:
    """Return the best available GFS ForecastPoints for *target_date*.

    Prefers the shortest lead ≤ 48 h.  Falls back to ≤ 72 h.
    Returns an empty list when no qualifying forecast exists.
    """
    day_rows = forecasts_df[forecasts_df["target_date"] == target_date]
    if day_rows.empty:
        return []

    for max_lead in (_PRIMARY_MAX_LEAD, _FALLBACK_MAX_LEAD):
        candidates = day_rows[day_rows["lead_hours"] <= max_lead]
        if not candidates.empty:
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
    """Return a DataFrame with columns [contract_id, target_date, threshold_f]."""
    if contracts_df.empty:
        return pd.DataFrame(columns=["contract_id", "target_date", "threshold_f"])
    return contracts_df[
        contracts_df["threshold_f"].isin(thresholds)
    ][["contract_id", "target_date", "threshold_f"]].copy()


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------


def run_backtest() -> dict:
    """Execute the full Dallas backtest on real historical data."""
    # ── Load all data ──────────────────────────────────────────────────────
    settlement_df  = _load_settlement()
    forecasts_df   = _load_forecasts()
    snapshots_df, contracts_df = _load_market_data()

    # ── City profile (calibrated values from config/cities.yaml) ──────────
    city_profile = get_city_profile("Dallas")

    # ── Build contract index ───────────────────────────────────────────────
    contract_index = _build_contracts_index(contracts_df, THRESHOLDS)
    idx_map: dict[tuple, str] = {
        (row["target_date"], int(row["threshold_f"])): row["contract_id"]
        for _, row in contract_index.iterrows()
    }

    # ── Identify overlapping target dates ─────────────────────────────────
    settlement_dates = set(settlement_df["target_date"].unique())
    forecast_dates   = set(forecasts_df["target_date"].unique())
    active_dates     = sorted(settlement_dates & forecast_dates)

    if not active_dates:
        raise RuntimeError(
            "No target dates overlap between settlement and forecast parquets. "
            "Check that both cover the same period."
        )

    logger.info("=" * 60)
    logger.info("Deep Isobar — Dallas (DFW) Backtest  [REAL DATA]")
    logger.info("Settlement dates : %d", len(settlement_dates))
    logger.info("Forecast dates   : %d", len(forecast_dates))
    logger.info("Overlapping      : %d", len(active_dates))
    logger.info("Thresholds       : %s °F", THRESHOLDS)
    logger.info("Signal threshold : %.2f  |  Quantity : %.0f contracts",
                SIGNAL_THRESHOLD, QUANTITY)
    logger.info("=" * 60)

    opportunity_rows: list[dict] = []
    skipped_no_forecast  = 0
    skipped_no_snapshot  = 0
    evaluated_count      = 0

    settlement_by_date: dict[date, float] = {
        row["target_date"]: float(row["high_temp_f"])
        for _, row in settlement_df.iterrows()
    }

    for target_date in active_dates:
        true_high_f = settlement_by_date[target_date]

        # ── GFS forecast → ensemble ────────────────────────────────────────
        forecast_points = _select_forecast(forecasts_df, target_date)
        if not forecast_points:
            skipped_no_forecast += len(THRESHOLDS)
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

        # ── Probability surface ────────────────────────────────────────────
        effective_std = max(ensemble.adjusted_std_f, _MIN_ENSEMBLE_STD_F)
        probability_surface: dict[int, float] = {
            thr: probability_ge_normal(
                mean_f=ensemble.bias_corrected_mean_f,
                std_f=effective_std,
                threshold_f=float(thr),
            )
            for thr in THRESHOLDS
        }

        decision_cutoff = datetime(
            target_date.year, target_date.month, target_date.day,
            _DECISION_CUTOFF_HOUR_UTC, 0, 0, tzinfo=timezone.utc,
        )

        # ── Evaluate each threshold ────────────────────────────────────────
        for threshold_f in THRESHOLDS:
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
                city="Dallas",
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

            simulated_price = (best_bid + best_ask) / 2.0
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

    # ── Coverage report ────────────────────────────────────────────────────
    logger.info(
        "Coverage: evaluated=%d  skipped_no_forecast=%d  skipped_no_snapshot=%d",
        evaluated_count,
        skipped_no_forecast,
        skipped_no_snapshot,
    )

    # ── Build opportunities DataFrame ──────────────────────────────────────
    if not opportunity_rows:
        logger.warning(
            "No BUY/SELL signals generated — all HOLD or no joinable data. "
            "Check that all three parquet files cover the same date range and "
            "that contract_id / threshold_f can be joined. "
            "Try lowering SIGNAL_THRESHOLD (currently %.2f).",
            SIGNAL_THRESHOLD,
        )
        return {"total_trades": 0, "win_rate": 0.0, "gross_pnl": 0.0, "max_drawdown": 0.0}

    opportunities_df = pd.DataFrame(opportunity_rows)

    logger.info(
        "Signal breakdown:  BUY=%d  SELL=%d  (from %d evaluated contract-days)",
        (opportunities_df["side"] == "BUY").sum(),
        (opportunities_df["side"] == "SELL").sum(),
        evaluated_count,
    )

    # ── Run backtest engine ────────────────────────────────────────────────
    trades_df = simulate_backtest_trades(opportunities_df, quantity=QUANTITY)
    summary   = summarize_backtest_results(trades_df)

    # ── Print summary ──────────────────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("BACKTEST SUMMARY  (real historical data — Dallas DFW)")
    logger.info("  total_trades     : %d", summary["total_trades"])
    logger.info("  win_rate         : %.1f%%", summary["win_rate"] * 100)
    logger.info(
        "  gross_pnl        : %.4f  (probability-point × contracts)",
        summary["gross_pnl"],
    )
    logger.info("  max_drawdown     : %.4f", summary["max_drawdown"])
    logger.info("-" * 60)

    for thr in THRESHOLDS:
        sub = trades_df[trades_df["contract_id"].str.contains(f"T{thr}|_{thr}_", regex=True)]
        if sub.empty:
            continue
        sub_summary = summarize_backtest_results(sub)
        logger.info(
            "  >= %d°F  trades=%d  win_rate=%.1f%%  gross_pnl=%.4f",
            thr,
            sub_summary["total_trades"],
            sub_summary["win_rate"] * 100,
            sub_summary["gross_pnl"],
        )

    logger.info("=" * 60)

    preview = trades_df[
        ["contract_id", "side", "simulated_price", "realized_outcome",
         "pnl_per_unit", "realized_pnl"]
    ].head(5)
    logger.info("First 5 trades:\n%s", preview.to_string(index=False))
    logger.info("=" * 60)

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    summary = run_backtest()
    print("\n--- backtest_summary (dict) ---")
    for key, value in summary.items():
        print(f"  {key:<20}: {value}")
