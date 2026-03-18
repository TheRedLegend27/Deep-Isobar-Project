"""Chicago (CHI) backtest driver for Deep Isobar.

Simulates the full Deep Isobar alpha-generation and settlement pipeline
for Chicago over a synthetic historical dataset, then runs the backtest
engine to compute theoretical PnL, win rate, and max drawdown.

Synthetic data design
---------------------
- 90 trading days (roughly one season)
- Two contract shapes per day: high_temp_f >= 80 and high_temp_f >= 90
- GFS and ECMWF forecasts with realistic Chicago summer bias characteristics
- Market prices set to a mix of correctly-priced and mispriced contracts,
  deliberately introducing alpha to demonstrate the engine's edge
- Realized outcomes drawn from the forecast-implied distribution with added
  noise, so the model has genuine (not perfect) predictive power

Pipeline steps
--------------
1.  Build synthetic ``daily_settlement_observation`` records
2.  Build synthetic ``forecast_temperature_point`` records (GFS + ECMWF)
3.  Build ``EnsembleSummary`` for each day using ``temperature_ensemble``
4.  Generate ``probability_surface`` for each day via ``probability_engine``
5.  Construct synthetic ``OrderBookSnapshot`` records (simulated market prices)
6.  Evaluate each contract with ``market_scanner.evaluate_contract_opportunity``
    to produce ``TradeSignal`` / ``alpha_opportunity`` rows
7.  Convert signals + realized outcomes into ``opportunities_df`` for the
    backtest engine
8.  Run ``simulate_backtest_trades`` and ``summarize_backtest_results``
9.  Print the ``backtest_summary`` to stdout

Usage::

    python -m deep_isobar.research.run_chicago_backtest

or::

    python src/deep_isobar/research/run_chicago_backtest.py
"""

from __future__ import annotations

import logging
import random
from datetime import date, datetime, timedelta, timezone
from typing import List

import pandas as pd

from deep_isobar.core.types import (
    CityProfile,
    ForecastPoint,
    MarketContract,
    OrderBookSnapshot,
)
from deep_isobar.market.market_scanner import evaluate_contract_opportunity
from deep_isobar.models.temperature_ensemble import build_temperature_ensemble
from deep_isobar.models.probability_engine import probability_ge_normal
from deep_isobar.research.backtest_engine import (
    simulate_backtest_trades,
    summarize_backtest_results,
)

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
# Simulation parameters
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
N_DAYS = 90                      # backtest window
START_DATE = date(2026, 6, 1)    # Chicago summer season (high-temp contracts)
QUANTITY = 10.0                  # contracts per trade
SIGNAL_THRESHOLD = 0.08          # minimum |alpha| to generate BUY/SELL

# Contract shapes: high_temp_f >= 80 and >= 90 (Fahrenheit)
THRESHOLDS = [80, 90]
COMPARISON_OPERATOR = "ge"
METRIC = "high_temp_f"

# Chicago city profile (synthetic calibration values)
CHICAGO_PROFILE = CityProfile(
    city="Chicago",
    city_code="CHI",
    station_id="KORD",
    timezone="America/Chicago",
    settlement_source="NWS",
    variance_multiplier=1.15,
    mean_bias_correction_f=-0.5,   # GFS runs slightly warm for Chicago summer
    kde_bandwidth=1.5,
    tail_multiplier=1.3,
    model_weight_gfs=0.55,
    model_weight_ecmwf=0.45,
    heat_bias_adjustment_f=-0.5,
)

# ---------------------------------------------------------------------------
# Synthetic data generation helpers
# ---------------------------------------------------------------------------


def _make_forecast_points(
    target_date: date,
    true_high_f: float,
    rng: random.Random,
) -> List[ForecastPoint]:
    """Return GFS and ECMWF ForecastPoints with realistic error distributions.

    GFS: +1 °F warm bias, ±3 °F noise (Chicago summer warm bias pattern)
    ECMWF: +0.3 °F warm bias, ±2 °F noise (slightly more accurate)
    """
    run_time = datetime(
        target_date.year, target_date.month, target_date.day,
        0, 0, 0, tzinfo=timezone.utc
    ) - timedelta(hours=24)

    gfs_value = true_high_f + 1.0 + rng.gauss(0, 3.0)
    ecmwf_value = true_high_f + 0.3 + rng.gauss(0, 2.0)

    return [
        ForecastPoint(
            city="Chicago",
            station_id="KORD",
            model_name="GFS",
            run_time_utc=run_time,
            target_date=target_date,
            metric=METRIC,
            forecast_value_f=round(gfs_value, 1),
            lead_hours=24,
            source_name="NOAA",
        ),
        ForecastPoint(
            city="Chicago",
            station_id="KORD",
            model_name="ECMWF",
            run_time_utc=run_time,
            target_date=target_date,
            metric=METRIC,
            forecast_value_f=round(ecmwf_value, 1),
            lead_hours=24,
            source_name="ECMWF",
        ),
    ]


def _make_market_price(
    model_prob: float,
    rng: random.Random,
    misprice_prob: float = 0.35,
) -> tuple[float, float]:
    """Synthesize a market-implied probability.

    With ``misprice_prob`` probability the market is significantly wrong
    (±10–20 pp), otherwise it is close to the model price (±3 pp).
    Clamped to [0.05, 0.95] to avoid degenerate spreads.
    """
    if rng.random() < misprice_prob:
        direction = rng.choice([-1, 1])
        error = rng.uniform(0.10, 0.20) * direction
    else:
        error = rng.gauss(0, 0.03)

    market_prob = max(0.05, min(0.95, model_prob + error))
    # Simulate a realistic bid/ask spread (±2%)
    half_spread = rng.uniform(0.015, 0.025)
    best_bid = max(0.01, market_prob - half_spread)
    best_ask = min(0.99, market_prob + half_spread)
    return best_bid, best_ask


def _realized_outcome(true_high_f: float, threshold_f: int) -> int:
    """Return 1 if the true high meets/exceeds the threshold, else 0."""
    return int(true_high_f >= threshold_f)


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------


def run_backtest() -> dict:
    """Execute the full Chicago backtest and return the summary dict."""
    rng = random.Random(RANDOM_SEED)

    logger.info("=" * 60)
    logger.info("Deep Isobar — Chicago (CHI) Backtest")
    logger.info("Period : %s → %s  (%d days)", START_DATE,
                START_DATE + timedelta(days=N_DAYS - 1), N_DAYS)
    logger.info("Thresholds : %s °F (high_temp_f >= X)", THRESHOLDS)
    logger.info("Signal threshold : %.2f  |  Quantity : %.0f contracts",
                SIGNAL_THRESHOLD, QUANTITY)
    logger.info("=" * 60)

    # True Chicago July-August high temperatures: mean ≈ 84 °F, std ≈ 7 °F
    CHICAGO_MEAN_HIGH_F = 84.0
    CHICAGO_STD_HIGH_F = 7.0

    opportunity_rows: list[dict] = []
    eval_timestamp = datetime(2026, 6, 1, 18, 0, 0, tzinfo=timezone.utc)

    for day_idx in range(N_DAYS):
        target_date = START_DATE + timedelta(days=day_idx)

        # ── Realized weather ───────────────────────────────────────────────
        true_high_f = rng.gauss(CHICAGO_MEAN_HIGH_F, CHICAGO_STD_HIGH_F)
        true_high_f = max(55.0, min(110.0, true_high_f))  # physical bounds

        # ── Forecasts → ensemble ───────────────────────────────────────────
        forecast_points = _make_forecast_points(target_date, true_high_f, rng)
        ensemble_run_time = datetime(
            target_date.year, target_date.month, target_date.day,
            6, 0, 0, tzinfo=timezone.utc
        )
        ensemble = build_temperature_ensemble(
            city_profile=CHICAGO_PROFILE,
            forecasts=forecast_points,
            target_date=target_date,
            metric=METRIC,
            ensemble_run_time_utc=ensemble_run_time,
        )

        # ── Probability surface ────────────────────────────────────────────
        probability_surface: dict[int, float] = {}
        for threshold_f in THRESHOLDS:
            probability_surface[threshold_f] = probability_ge_normal(
                mean_f=ensemble.bias_corrected_mean_f,
                std_f=ensemble.adjusted_std_f,
                threshold_f=float(threshold_f),
            )

        # ── Evaluate each contract ─────────────────────────────────────────
        for threshold_f in THRESHOLDS:
            contract_id = (
                f"CHI_{METRIC.upper()}_{COMPARISON_OPERATOR.upper()}"
                f"_{threshold_f}_{target_date.strftime('%Y%m%d')}"
            )
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

            model_prob = probability_surface[threshold_f]
            best_bid, best_ask = _make_market_price(model_prob, rng)

            snapshot = OrderBookSnapshot(
                timestamp_utc=eval_timestamp + timedelta(days=day_idx),
                contract_id=contract_id,
                market_source="Kalshi",
                best_bid=best_bid,
                best_ask=best_ask,
                bid_size=rng.uniform(50, 200),
                ask_size=rng.uniform(50, 200),
                volume_24h=rng.uniform(1000, 8000),
            )

            signal = evaluate_contract_opportunity(
                contract=contract,
                probability_surface=probability_surface,
                orderbook=snapshot,
                signal_threshold=SIGNAL_THRESHOLD,
                timestamp_utc=eval_timestamp + timedelta(days=day_idx),
            )

            if signal.signal_side == "HOLD":
                continue  # no edge detected; skip this contract

            # Market mid-price is used as the simulated execution price
            simulated_price = (best_bid + best_ask) / 2.0
            outcome = _realized_outcome(true_high_f, threshold_f)

            opportunity_rows.append({
                "contract_id": contract_id,
                "target_date": target_date,
                "threshold_f": threshold_f,
                "side": signal.signal_side,
                "model_probability": signal.model_probability,
                "market_probability": signal.market_probability,
                "alpha": signal.alpha,
                "absolute_alpha": signal.absolute_alpha,
                "rank_score": signal.rank_score,
                "simulated_price": simulated_price,
                "realized_outcome": outcome,
            })

    # ── Build opportunities DataFrame ──────────────────────────────────────
    if not opportunity_rows:
        logger.warning("No BUY/SELL signals generated — all HOLD.  "
                       "Try lowering SIGNAL_THRESHOLD.")
        return {"total_trades": 0, "win_rate": 0.0, "gross_pnl": 0.0, "max_drawdown": 0.0}

    opportunities_df = pd.DataFrame(opportunity_rows)

    logger.info(
        "Signal breakdown:  BUY=%d  SELL=%d  (from %d contract-days evaluated)",
        (opportunities_df["side"] == "BUY").sum(),
        (opportunities_df["side"] == "SELL").sum(),
        N_DAYS * len(THRESHOLDS),
    )

    # ── Run backtest engine ────────────────────────────────────────────────
    trades_df = simulate_backtest_trades(opportunities_df, quantity=QUANTITY)
    summary = summarize_backtest_results(trades_df)

    # ── Print backtest_summary schema ─────────────────────────────────────
    logger.info("-" * 60)
    logger.info("BACKTEST SUMMARY  (backtest_summary schema)")
    logger.info("  total_trades     : %d", summary["total_trades"])
    logger.info("  win_rate         : %.1f%%", summary["win_rate"] * 100)
    logger.info("  gross_pnl        : %.4f  (units: probability-point × contracts)",
                summary["gross_pnl"])
    logger.info("  max_drawdown     : %.4f", summary["max_drawdown"])
    logger.info("-" * 60)

    # Per-threshold breakdown
    for thr in THRESHOLDS:
        sub = trades_df[trades_df["contract_id"].str.contains(f"_{thr}_")]
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

    # Raw trades preview (first 5 rows)
    preview = trades_df[["contract_id", "side", "simulated_price",
                          "realized_outcome", "pnl_per_unit", "realized_pnl"]].head(5)
    logger.info("First 5 simulated trades:\n%s", preview.to_string(index=False))
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
