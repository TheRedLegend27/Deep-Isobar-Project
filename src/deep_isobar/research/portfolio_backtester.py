"""Portfolio Backtester — multi-city, Phase 9 enhanced backtest for Deep Isobar.

Extends the single-city backtests (run_chicago_backtest.py, run_nyc_backtest.py)
with three key enhancements:

1. **Multi-city**: Iterates over every active city in config/cities.yaml, loads
   the city's data files, and simulates trading across a shared timeline.

2. **Phase 9 forecast-shift + market-lag wiring**: For every (date, threshold)
   pair, the backtester compares the current GFS run (t) to the most recent
   prior GFS run (t-1) via ``compute_forecast_shift``.  The resulting
   ``ForecastShiftEvent`` is used to call ``detect_market_lag`` with the
   market probability observed before and after the shift.  Both flags are
   forwarded into ``evaluate_contract_opportunity``.

3. **Confidence-based sizing**: Position size scales with the signal's
   ``confidence_score`` rather than a fixed quantity.  Higher-conviction
   signals trade up to 50 contracts; marginal-edge signals stay at 10.

Execution model (realistic spread):
    BUY  → filled at ``best_ask``  (pay the offer)
    SELL → filled at ``best_bid``  (hit the bid)

Output:
    Per-city PnL tables printed to the log + aggregate portfolio totals
    returned as a dict.

Usage::

    python -m deep_isobar.research.portfolio_backtester

or::

    python src/deep_isobar/research/portfolio_backtester.py

Data files must be present for a city to be included.  Cities without
matching parquet files are skipped with a WARNING log line and excluded
from the portfolio totals.  The module never raises due to missing data
for a single city — it logs and continues.
"""

from __future__ import annotations

import logging
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
from deep_isobar.data.city_universe import get_city_profile, list_active_cities
from deep_isobar.market.market_lag_detection import detect_market_lag
from deep_isobar.market.market_scanner import evaluate_contract_opportunity
from deep_isobar.market.microstructure_scanner import compute_microstructure_score
from deep_isobar.models.forecast_shift import compute_forecast_shift
from deep_isobar.models.probability_engine import probability_ge_normal
from deep_isobar.models.temperature_ensemble import build_temperature_ensemble
from deep_isobar.research.backtest_engine import summarize_backtest_results
from deep_isobar.trading.distribution_tail_alpha import is_tail_threshold

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("portfolio_backtester")

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SETTLEMENT_DIR = _PROJECT_ROOT / "data/historical/settlement"
_FORECAST_DIR   = _PROJECT_ROOT / "data/historical/forecasts"
_MARKET_DIR     = _PROJECT_ROOT / "data/historical/markets"

# Minimum |alpha| to generate a BUY or SELL (same threshold as single-city drivers).
_SIGNAL_THRESHOLD: float = 0.25

# Prediction-market contract type
_METRIC: str = "high_temp_f"
_COMPARISON_OPERATOR: str = "ge"

# Decision cutoff: last Kalshi snapshot before this UTC hour on target_date.
# 12:00 UTC ≈ 7 am CDT / 8 am EDT — well before afternoon high temperature.
_DECISION_CUTOFF_HOUR_UTC: int = 12

# GFS lead tiers used to select the "current" and "previous" forecast runs.
# Primary tier: shortest lead ≤ 48 h (day+1 from 00z/12z cycle).
# Fallback tier: ≤ 72 h when no day+1 run covers the target date.
_PRIMARY_MAX_LEAD: int  = 48
_FALLBACK_MAX_LEAD: int = 72

# Floor for ensemble std when only a single GFS run is available (std=0).
# Expressed as a multiplier on the city's kde_bandwidth; the minimum cap
# ensures we never use a std below 2.5 °F regardless of KDE configuration.
_MIN_ENSEMBLE_STD_FLOOR: float = 2.5

# Confidence-based sizing thresholds and corresponding contract counts.
# confidence_score = abs(alpha) + enhancement boosts (shift, tail).
_SIZING_TIERS: list[tuple[float, float]] = [
    (0.80, 50.0),   # very high conviction
    (0.65, 40.0),
    (0.50, 30.0),
    (0.35, 20.0),
    (0.00, 10.0),   # base size (floor — covers all non-negative confidence values)
]

# ---------------------------------------------------------------------------
# City data path registry
# ---------------------------------------------------------------------------
# Maps each city name (must match config/cities.yaml) to the three stems
# needed to build the parquet file paths:
#   settlement_stem : data/historical/settlement/{stem}.parquet
#   forecast_stem   : data/historical/forecasts/gfs_{stem}.parquet
#   market_series   : data/historical/markets/kalshi_{series}.parquet
#                     data/historical/markets/kalshi_{series}_contracts.parquet
# ---------------------------------------------------------------------------

_CITY_FILE_STEMS: dict[str, tuple[str, str, str]] = {
    "Chicago":   ("chicago_2023", "chicago_2023", "kxhighchi_2023"),
    "New York":  ("nyc_2023",     "nyc_2023",     "kxhighny_2023"),
    "Phoenix":   ("phx_2023",     "phx_2023",     "kxhighphx_2023"),
    "Denver":    ("den_2023",     "den_2023",      "kxhighden_2023"),
    "Dallas":    ("dal_2023",     "dal_2023",      "kxhighdal_2023"),
}


def _build_city_paths(city_name: str) -> dict[str, Path | None]:
    """Return canonical parquet paths for *city_name*.

    Returns a dict with keys: ``settlement``, ``forecasts``, ``snapshots``,
    ``contracts``.  ``contracts`` may not exist on disk (data can be derived
    from snapshots); the key is always present but callers must guard with
    ``path.exists()`` before reading.

    Returns ``None`` for all values when the city has no registered file stems.
    """
    stems = _CITY_FILE_STEMS.get(city_name)
    if stems is None:
        return {"settlement": None, "forecasts": None, "snapshots": None, "contracts": None}

    settlement_stem, forecast_stem, market_series = stems
    return {
        "settlement": _SETTLEMENT_DIR / f"{settlement_stem}.parquet",
        "forecasts":  _FORECAST_DIR   / f"gfs_{forecast_stem}.parquet",
        "snapshots":  _MARKET_DIR     / f"kalshi_{market_series}.parquet",
        "contracts":  _MARKET_DIR     / f"kalshi_{market_series}_contracts.parquet",
    }


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _load_settlement(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if pd.api.types.is_datetime64_any_dtype(df["target_date"]):
        df["target_date"] = df["target_date"].dt.date
    if "quality_flag" in df.columns:
        df = df[df["quality_flag"] != "missing"].copy()
    logger.info("  settlement: %d rows from %s", len(df), path.name)
    return df


def _load_forecasts(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if pd.api.types.is_datetime64_any_dtype(df["target_date"]):
        df["target_date"] = df["target_date"].dt.date
    if df["run_time_utc"].dt.tz is None:
        df["run_time_utc"] = df["run_time_utc"].dt.tz_localize("UTC")
    logger.info("  forecasts:  %d rows from %s", len(df), path.name)
    return df


def _load_market_data(
    snapshots_path: Path,
    contracts_path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(snapshots_df, contracts_df)``.

    When *contracts_path* is absent or does not exist, derives contract
    metadata from T-format contract_ids in the snapshots parquet using the
    same pattern as the single-city drivers.
    """
    snapshots = pd.read_parquet(snapshots_path)
    if snapshots["timestamp_utc"].dt.tz is None:
        snapshots["timestamp_utc"] = snapshots["timestamp_utc"].dt.tz_localize("UTC")

    if contracts_path is not None and contracts_path.exists():
        contracts = pd.read_parquet(contracts_path)
        if pd.api.types.is_datetime64_any_dtype(contracts["target_date"]):
            contracts["target_date"] = contracts["target_date"].dt.date
        contracts["threshold_f"] = pd.to_numeric(contracts["threshold_f"], errors="coerce")
    else:
        logger.warning(
            "  contracts parquet absent — deriving metadata from T-format contract_ids"
        )
        t_snaps = snapshots[
            snapshots["contract_id"].str.contains(r"-T\d", regex=True, na=False)
        ][["contract_id"]].drop_duplicates().copy()
        t_snaps["threshold_f"] = (
            t_snaps["contract_id"].str.extract(r"-T(\d+)")[0].astype(float)
        )
        t_snaps["target_date"] = pd.to_datetime(
            t_snaps["contract_id"].str.extract(r"-(\d{2}[A-Z]{3}\d{2})-")[0],
            format="%y%b%d",
        ).dt.date
        contracts = t_snaps.dropna(subset=["threshold_f", "target_date"]).copy()

    # Keep only T-format binary threshold contracts
    if "contract_id" in contracts.columns and not contracts.empty:
        contracts = contracts[
            contracts["contract_id"].str.contains(r"-T\d", regex=True, na=False)
        ].copy()

    logger.info(
        "  snapshots:  %d rows, %d T-format contracts",
        len(snapshots),
        len(contracts),
    )
    return snapshots, contracts


# ---------------------------------------------------------------------------
# Forecast selection helpers (Phase 9: t and t-1 runs)
# ---------------------------------------------------------------------------


def _row_to_forecast_point(row: pd.Series) -> ForecastPoint:
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


def _select_forecast_pair(
    forecasts_df: pd.DataFrame,
    target_date: date,
) -> tuple[ForecastPoint | None, ForecastPoint | None]:
    """Return ``(current_run, previous_run)`` for *target_date*.

    *current_run* is the freshest GFS run within the primary lead budget
    (≤48 h), or the fallback budget (≤72 h) when no day+1 run exists.

    *previous_run* is the most recent run with an **earlier** ``run_time_utc``
    than *current_run* — i.e., the t-1 run used for shift detection.  Returns
    ``None`` when only one run tier exists for *target_date*.
    """
    day_rows = forecasts_df[forecasts_df["target_date"] == target_date]
    if day_rows.empty:
        return None, None

    # Select current run: minimum lead_hours within budget, most recent run_time on tie
    current_row: pd.Series | None = None
    for max_lead in (_PRIMARY_MAX_LEAD, _FALLBACK_MAX_LEAD):
        candidates = day_rows[day_rows["lead_hours"] <= max_lead]
        if not candidates.empty:
            min_lead = candidates["lead_hours"].min()
            min_lead_rows = candidates[candidates["lead_hours"] == min_lead]
            current_row = min_lead_rows.sort_values("run_time_utc").iloc[-1]
            break

    if current_row is None:
        return None, None

    current_fp = _row_to_forecast_point(current_row)

    # Select previous run: most recent run with an earlier run_time_utc
    earlier_rows = day_rows[day_rows["run_time_utc"] < current_fp.run_time_utc]
    if earlier_rows.empty:
        return current_fp, None

    prev_run_time = earlier_rows["run_time_utc"].max()
    prev_run_rows = earlier_rows[earlier_rows["run_time_utc"] == prev_run_time]
    # When multiple rows share the same run_time (e.g., two cycles at same hour),
    # pick the shortest lead so forecast_value_f is the closest-range estimate.
    prev_row = prev_run_rows.sort_values("lead_hours").iloc[0]
    previous_fp = _row_to_forecast_point(prev_row)

    return current_fp, previous_fp


# ---------------------------------------------------------------------------
# Market snapshot helpers
# ---------------------------------------------------------------------------


def _select_market_snapshot(
    snapshots_df: pd.DataFrame,
    contract_id: str,
    cutoff_utc: datetime,
) -> tuple[float, float] | None:
    """Return ``(best_bid, best_ask)`` from the last snapshot before *cutoff_utc*.

    Returns ``None`` if no qualifying snapshot exists or if bid/ask are NaN.
    """
    contract_snaps = snapshots_df[snapshots_df["contract_id"] == contract_id]
    before = contract_snaps[contract_snaps["timestamp_utc"] < cutoff_utc]
    if before.empty:
        return None
    last = before.sort_values("timestamp_utc").iloc[-1]
    bid = last["best_bid"]
    ask = last["best_ask"]
    if pd.isna(bid) or pd.isna(ask):
        return None
    return float(bid), float(ask)


def _get_market_probability(
    snapshots_df: pd.DataFrame,
    contract_id: str,
    cutoff_utc: datetime,
) -> float | None:
    """Return the mid-price from the last snapshot before *cutoff_utc*.

    The mid-price ``(best_bid + best_ask) / 2`` is used as the market-implied
    probability for lag-detection inputs.  Returns ``None`` when no snapshot
    exists before the cutoff or when bid/ask are NaN or out of [0, 1].
    """
    price = _select_market_snapshot(snapshots_df, contract_id, cutoff_utc)
    if price is None:
        return None
    bid, ask = price
    mid = (bid + ask) / 2.0
    if not (0.0 <= mid <= 1.0):
        return None
    return mid


# ---------------------------------------------------------------------------
# Confidence-based position sizing
# ---------------------------------------------------------------------------


def _confidence_to_quantity(confidence: float) -> float:
    """Map a confidence score to a position size (10–50 contracts).

    Uses a tiered schedule so position sizes snap to clean lot boundaries:

    +---------------------+----------+
    | confidence_score    | quantity |
    +=====================+==========+
    | >= 0.80             | 50       |
    | >= 0.65             | 40       |
    | >= 0.50             | 30       |
    | >= 0.35             | 20       |
    | any (>= threshold)  | 10       |
    +---------------------+----------+
    """
    for threshold, qty in _SIZING_TIERS:
        if confidence >= threshold:
            return qty
    return 10.0  # unreachable but satisfies type checker


# ---------------------------------------------------------------------------
# Per-city backtest runner
# ---------------------------------------------------------------------------


def run_city_backtest(
    city_name: str,
    city_profile: CityProfile,
    settlement_df: pd.DataFrame,
    forecasts_df: pd.DataFrame,
    snapshots_df: pd.DataFrame,
    contracts_df: pd.DataFrame,
) -> list[dict]:
    """Simulate trades for a single city; return a list of trade-row dicts.

    Each dict contains all columns needed to assemble a trades DataFrame
    and is compatible with ``summarize_backtest_results``.

    Args:
        city_name:     Canonical city name (e.g. ``"Chicago"``).
        city_profile:  ``CityProfile`` loaded from ``cities.yaml``.
        settlement_df: NOAA settlement observations for this city.
        forecasts_df:  GFS archived forecast runs for this city.
        snapshots_df:  Kalshi order-book snapshots for this city.
        contracts_df:  Contract metadata (threshold_f, target_date) for
                       this city's T-format contracts.

    Returns:
        List of trade-row dicts.  Empty when no signals are generated.
    """
    # Ensemble std floor: max of city KDE bandwidth and the global minimum.
    min_ensemble_std = max(city_profile.kde_bandwidth, _MIN_ENSEMBLE_STD_FLOOR)

    # ── Build (target_date, threshold_f) → contract_id index ──────────────
    idx_map: dict[tuple[date, int], str] = {}
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

    # ── Identify overlapping dates ─────────────────────────────────────────
    settlement_dates = set(settlement_df["target_date"].unique())
    forecast_dates   = set(forecasts_df["target_date"].unique())
    active_dates     = sorted(settlement_dates & forecast_dates)

    if not active_dates:
        logger.warning(
            "%s: no overlapping dates between settlement (%d dates) and "
            "forecasts (%d dates) — skipping city",
            city_name, len(settlement_dates), len(forecast_dates),
        )
        return []

    settlement_by_date: dict[date, float] = {
        row["target_date"]: float(row["high_temp_f"])
        for _, row in settlement_df.iterrows()
    }

    trade_rows: list[dict] = []
    skipped_no_forecast  = 0
    skipped_no_snapshot  = 0
    evaluated            = 0
    phase9_shift_active  = 0   # dates where a significant shift was detected
    phase9_lag_active    = 0   # contract-days where stale_market_flag was True

    for target_date in active_dates:
        true_high_f = settlement_by_date[target_date]

        # ── Phase 9: select current and previous GFS runs ──────────────────
        current_fp, previous_fp = _select_forecast_pair(forecasts_df, target_date)
        if current_fp is None:
            skipped_no_forecast += len(date_thresholds.get(target_date, [1]))
            logger.debug("%s %s: no GFS forecast — skipping", city_name, target_date)
            continue

        # ── Build ensemble from current run ───────────────────────────────
        ensemble = build_temperature_ensemble(
            city_profile=city_profile,
            forecasts=[current_fp],
            target_date=target_date,
            metric=_METRIC,
            ensemble_run_time_utc=current_fp.run_time_utc,
        )
        effective_std = max(ensemble.adjusted_std_f, min_ensemble_std)

        # ── Probability surface across all Kalshi thresholds for this date ─
        day_thresholds = date_thresholds.get(target_date, [])
        if not day_thresholds:
            logger.debug(
                "%s %s: no T-format contracts — skipping", city_name, target_date
            )
            continue

        probability_surface: dict[int, float] = {
            thr: probability_ge_normal(
                mean_f=ensemble.bias_corrected_mean_f,
                std_f=effective_std,
                threshold_f=float(thr),
            )
            for thr in day_thresholds
        }

        # ── Phase 9: compute forecast shift (t vs t-1) ────────────────────
        forecast_shift_flag = False
        shift_event = None
        if previous_fp is not None:
            try:
                shift_event = compute_forecast_shift(previous_fp, current_fp)
                forecast_shift_flag = shift_event.significant_shift_flag
                if forecast_shift_flag:
                    phase9_shift_active += 1
            except ValueError as exc:
                logger.debug(
                    "%s %s: forecast shift skipped — %s", city_name, target_date, exc
                )

        # Decision cutoff: noon UTC on target_date
        decision_cutoff = datetime(
            target_date.year, target_date.month, target_date.day,
            _DECISION_CUTOFF_HOUR_UTC, 0, 0, tzinfo=timezone.utc,
        )

        # ── Per-threshold evaluation ───────────────────────────────────────
        for threshold_f in day_thresholds:
            contract_id = idx_map.get((target_date, threshold_f))
            if contract_id is None:
                skipped_no_snapshot += 1
                continue

            # Last market snapshot before the decision cutoff
            price = _select_market_snapshot(snapshots_df, contract_id, decision_cutoff)
            if price is None:
                skipped_no_snapshot += 1
                logger.debug(
                    "%s: no snapshot for %s before %s",
                    contract_id, target_date, decision_cutoff.isoformat(),
                )
                continue

            best_bid, best_ask = price
            evaluated += 1

            # ── Phase 9: market lag detection ─────────────────────────────
            # Compare market probability before the current GFS run was
            # published vs. after (at decision cutoff).  A stale market
            # fails to react to a meaningful temperature shift.
            stale_market_flag = False
            if shift_event is not None:
                prob_before = _get_market_probability(
                    snapshots_df, contract_id, current_fp.run_time_utc
                )
                # prob_after: reuse bid/ask already fetched at decision_cutoff
                # rather than scanning the snapshot DataFrame a second time.
                prob_after = (best_bid + best_ask) / 2.0
                if prob_before is not None:
                    try:
                        stale_market_flag = detect_market_lag(
                            shift_event=shift_event,
                            market_probability_before=prob_before,
                            market_probability_after=prob_after,
                        )
                    except ValueError as exc:
                        logger.debug(
                            "%s %s T%d: lag detection skipped — %s",
                            city_name, target_date, threshold_f, exc,
                        )
                    if stale_market_flag:
                        phase9_lag_active += 1

            # ── Market structure quality ───────────────────────────────────
            snapshot = OrderBookSnapshot(
                timestamp_utc=decision_cutoff,
                contract_id=contract_id,
                market_source="Kalshi",
                best_bid=best_bid,
                best_ask=best_ask,
            )
            micro_score = compute_microstructure_score(snapshot, decision_cutoff)
            tail_flag   = is_tail_threshold(
                ensemble_mean_f=ensemble.bias_corrected_mean_f,
                threshold_f=threshold_f,
                adjusted_std_f=effective_std,
            )

            contract = MarketContract(
                contract_id=contract_id,
                market_source="Kalshi",
                city=city_name,
                metric=_METRIC,
                comparison_operator=_COMPARISON_OPERATOR,
                threshold_f=threshold_f,
                target_date=target_date,
                settlement_source=city_profile.settlement_source,
            )

            signal = evaluate_contract_opportunity(
                contract=contract,
                probability_surface=probability_surface,
                orderbook=snapshot,
                signal_threshold=_SIGNAL_THRESHOLD,
                timestamp_utc=decision_cutoff,
                forecast_shift_flag=forecast_shift_flag,
                stale_market_flag=stale_market_flag,
                microstructure_score=micro_score,
                tail_opportunity_flag=tail_flag,
                tail_multiplier=city_profile.tail_multiplier,
            )

            if signal.signal_side == "HOLD":
                continue

            # ── Spread-realistic fill price ────────────────────────────────
            # BUY: we pay the offer (best_ask).
            # SELL: we hit the bid (best_bid).
            fill_price: float = best_ask if signal.signal_side == "BUY" else best_bid

            # ── Confidence-based position sizing ──────────────────────────
            quantity = _confidence_to_quantity(signal.confidence_score)

            # ── Settlement and PnL ────────────────────────────────────────
            realized_outcome = int(true_high_f >= threshold_f)

            if signal.signal_side == "BUY":
                pnl_per_unit = realized_outcome - fill_price
            else:
                pnl_per_unit = fill_price - realized_outcome

            realized_pnl = pnl_per_unit * quantity

            trade_rows.append({
                "city":                city_name,
                "contract_id":         contract_id,
                "target_date":         target_date,
                "threshold_f":         threshold_f,
                "side":                signal.signal_side,
                "fill_price":          fill_price,
                "best_bid":            best_bid,
                "best_ask":            best_ask,
                "quantity":            quantity,
                "model_probability":   signal.model_probability,
                "market_probability":  signal.market_probability,
                "alpha":               signal.alpha,
                "confidence_score":    signal.confidence_score,
                "rank_score":          signal.rank_score,
                "forecast_shift_flag": signal.forecast_shift_flag,
                "stale_market_flag":   signal.stale_market_flag,
                "tail_flag":           signal.tail_opportunity_flag,
                "microstructure_score": signal.microstructure_score,
                # Columns used by summarize_backtest_results:
                "realized_outcome":    realized_outcome,
                "pnl_per_unit":        pnl_per_unit,
                "realized_pnl":        realized_pnl,
            })

    logger.info(
        "%s: evaluated=%d  signals=%d  "
        "phase9_shifts=%d  phase9_lags=%d  "
        "skipped_no_forecast=%d  skipped_no_snapshot=%d",
        city_name,
        evaluated,
        len(trade_rows),
        phase9_shift_active,
        phase9_lag_active,
        skipped_no_forecast,
        skipped_no_snapshot,
    )
    return trade_rows


# ---------------------------------------------------------------------------
# Portfolio runner (outer multi-city loop)
# ---------------------------------------------------------------------------


def run_portfolio_backtest() -> dict:
    """Execute the full multi-city portfolio backtest.

    Iterates over every active city in ``config/cities.yaml``.  For each
    city, discovers the data files by convention, loads them, and delegates
    to :func:`run_city_backtest`.  Cities with missing data files are skipped
    gracefully.

    Prints a per-city summary table and a combined portfolio totals block to
    the log.

    Returns:
        Aggregate portfolio metrics dict with keys:
        ``total_trades``, ``win_rate``, ``gross_pnl``, ``max_drawdown``,
        ``cities_run``, ``cities_skipped``.
    """
    active_cities = list_active_cities()

    logger.info("=" * 70)
    logger.info("Deep Isobar — Multi-City Portfolio Backtest  (Phase 9)")
    logger.info("Active cities in config: %s", active_cities)
    logger.info("Signal threshold: %.2f", _SIGNAL_THRESHOLD)
    logger.info("Execution: BUY@ask / SELL@bid  |  Sizing: confidence-tiered 10–50")
    logger.info("=" * 70)

    all_trade_rows: list[dict]          = []
    city_summaries: dict[str, dict]     = {}
    cities_run     = 0
    cities_skipped = 0

    for city_name in active_cities:
        logger.info("-" * 50)
        logger.info("Loading data for: %s", city_name)

        paths = _build_city_paths(city_name)

        # A city is skipped if we have no registered stems OR any of the three
        # required parquets (settlement, forecasts, snapshots) is absent.
        required = ["settlement", "forecasts", "snapshots"]
        if any(paths.get(k) is None or not paths[k].exists() for k in required):
            logger.warning(
                "%s: one or more required data files missing — skipping.  "
                "Expected:\n"
                "  settlement: %s\n"
                "  forecasts:  %s\n"
                "  snapshots:  %s",
                city_name,
                paths.get("settlement"),
                paths.get("forecasts"),
                paths.get("snapshots"),
            )
            cities_skipped += 1
            continue

        try:
            city_profile  = get_city_profile(city_name)
            settlement_df = _load_settlement(paths["settlement"])
            forecasts_df  = _load_forecasts(paths["forecasts"])
            snapshots_df, contracts_df = _load_market_data(
                paths["snapshots"],
                paths.get("contracts"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "%s: unexpected error loading data — skipping city.  Error: %s",
                city_name, exc,
            )
            cities_skipped += 1
            continue

        try:
            trade_rows = run_city_backtest(
                city_name=city_name,
                city_profile=city_profile,
                settlement_df=settlement_df,
                forecasts_df=forecasts_df,
                snapshots_df=snapshots_df,
                contracts_df=contracts_df,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "%s: backtest error — skipping city.  Error: %s",
                city_name, exc,
            )
            cities_skipped += 1
            continue

        all_trade_rows.extend(trade_rows)
        cities_run += 1

        if trade_rows:
            city_df      = pd.DataFrame(trade_rows)
            city_summary = summarize_backtest_results(city_df)
        else:
            city_summary = {
                "total_trades": 0, "win_rate": 0.0,
                "gross_pnl": 0.0,  "max_drawdown": 0.0,
            }

        city_summaries[city_name] = city_summary

        logger.info(
            "%s summary:  trades=%d  win_rate=%.1f%%  pnl=%.4f  drawdown=%.4f",
            city_name,
            city_summary["total_trades"],
            city_summary["win_rate"] * 100,
            city_summary["gross_pnl"],
            city_summary["max_drawdown"],
        )

    # ── Aggregate portfolio summary ────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("PORTFOLIO SUMMARY  (all cities combined)")
    logger.info("  cities run     : %d", cities_run)
    logger.info("  cities skipped : %d", cities_skipped)

    if not all_trade_rows:
        logger.warning("No trades generated across any city.")
        return {
            "total_trades": 0, "win_rate": 0.0,
            "gross_pnl": 0.0, "max_drawdown": 0.0,
            "cities_run": cities_run, "cities_skipped": cities_skipped,
        }

    portfolio_df      = pd.DataFrame(all_trade_rows)
    portfolio_summary = summarize_backtest_results(portfolio_df)

    logger.info("  total_trades   : %d", portfolio_summary["total_trades"])
    logger.info("  win_rate       : %.1f%%", portfolio_summary["win_rate"] * 100)
    logger.info(
        "  gross_pnl      : %.4f  (probability-points × contracts)",
        portfolio_summary["gross_pnl"],
    )
    logger.info("  max_drawdown   : %.4f", portfolio_summary["max_drawdown"])

    # ── Per-city breakdown ─────────────────────────────────────────────────
    logger.info("-" * 70)
    logger.info("  %-14s %8s %9s %10s %11s", "city", "trades", "win_rate", "gross_pnl", "max_drawdown")
    logger.info("  %-14s %8s %9s %10s %11s", "-" * 14, "-" * 8, "-" * 9, "-" * 10, "-" * 11)
    for city_name in active_cities:
        if city_name in city_summaries:
            s = city_summaries[city_name]
            logger.info(
                "  %-14s %8d %8.1f%% %10.4f %11.4f",
                city_name,
                s["total_trades"],
                s["win_rate"] * 100,
                s["gross_pnl"],
                s["max_drawdown"],
            )

    # ── Phase 9 enhancement summary ───────────────────────────────────────
    shift_trades = portfolio_df["forecast_shift_flag"].sum()
    lag_trades   = portfolio_df["stale_market_flag"].sum()
    tail_trades  = portfolio_df["tail_flag"].sum()
    logger.info("-" * 70)
    logger.info(
        "  Phase 9 flags on executed trades:  "
        "shift=%d (%.1f%%)  lag=%d (%.1f%%)  tail=%d (%.1f%%)",
        shift_trades, shift_trades / len(portfolio_df) * 100,
        lag_trades,   lag_trades   / len(portfolio_df) * 100,
        tail_trades,  tail_trades  / len(portfolio_df) * 100,
    )

    # ── Confidence/sizing distribution ────────────────────────────────────
    qty_counts = portfolio_df["quantity"].value_counts().sort_index()
    logger.info(
        "  Quantity distribution:  %s",
        "  ".join(f"{int(q)}ct={n}" for q, n in qty_counts.items()),
    )

    logger.info("=" * 70)

    result = dict(portfolio_summary)
    result["cities_run"]     = cities_run
    result["cities_skipped"] = cities_skipped
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    summary = run_portfolio_backtest()
    print("\n--- portfolio_backtest_summary ---")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key:<22}: {value:.4f}")
        else:
            print(f"  {key:<22}: {value}")
