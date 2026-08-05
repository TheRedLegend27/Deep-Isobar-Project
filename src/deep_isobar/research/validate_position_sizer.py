"""Historical replay of the smart position sizer against settled paper trades.

Go-live safety build, Part 2 — this is the point of the exercise, not an
afterthought: a sizer that only looks good because it happened to size up
the one lottery hit is a failed sizer, and the only way to know is to run
it back over what actually happened.

For every settled (WIN/LOSS) row in ``data/paper_trades/paper_trades.csv``,
grouped by (date, city) exactly as the live session groups them, this
script re-derives the Kelly allocation from the recorded model probability
and entry price (``build_spread(..., allocation_method="kelly", ...)``,
unchanged), then runs it through the new sizer's evidence multipliers and
hard caps (``trading/position_sizer.adjust_allocations``). Per-trade P&L is
rescaled from the actual recorded ``realized_pnl`` by the ratio of new to
old contract count — the entry price and settlement outcome are historical
fact and are not re-derived.

Known limitation, stated plainly: the calibration-quality multiplier is
**not** exercised by this replay. Reconstructing genuine historical
rolling CRPS/PIT per station as of each trade's date would require
re-fitting EMOS over a rolling window for 25+ days x 5 stations, which is
a real data-engineering project on its own and out of scope here. The
replay passes ``station_calibration=None`` (neutral multiplier,
unchanged output) throughout — in production it is live via
``daily_scorecard.station_calibration()``. Everything else (entry-price
tail haircut, per-station track record with no lookahead, anomaly/spread
multiplier, both hard caps) runs for real.

CLI::

    python -m deep_isobar.research.validate_position_sizer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from deep_isobar.config import get_setting
from deep_isobar.core.types import TradeSignal
from deep_isobar.trading.bracket_spreader import build_spread
from deep_isobar.trading.kelly import risk_per_contract
from deep_isobar.trading.position_sizer import (
    StationTrackRecord,
    adjust_allocations,
    build_station_track_record,
    compute_city_daily_cap,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PAPER_TRADES_CSV = _PROJECT_ROOT / "data" / "paper_trades" / "paper_trades.csv"


# ---------------------------------------------------------------------------
# Minimal stand-ins for the anomaly report the live session passes in —
# the CSV only has the reduced (confidence, flag codes) form, not the
# original object.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FlagStub:
    code: str


@dataclass(frozen=True)
class _AnomalyStub:
    confidence: str | None
    flags: tuple[_FlagStub, ...]


def _anomaly_from_row(row: pd.Series) -> _AnomalyStub | None:
    confidence = row.get("anomaly_confidence")
    if not isinstance(confidence, str) or not confidence.strip():
        return None
    flags_raw = row.get("anomaly_flags")
    codes = (
        [c.strip() for c in str(flags_raw).split(",") if c.strip()]
        if isinstance(flags_raw, str) and flags_raw.strip()
        else []
    )
    return _AnomalyStub(confidence=confidence.strip(), flags=tuple(_FlagStub(c) for c in codes))


def _ensemble_std_from_row(row: pd.Series) -> float | None:
    var = row.get("ens_spread_var")
    if var is None or (isinstance(var, float) and pd.isna(var)) or var < 0:
        return None
    return float(var) ** 0.5


# ---------------------------------------------------------------------------
# Signal reconstruction
# ---------------------------------------------------------------------------

_OP_BY_STRIKE = {"less": "lt", "greater": "gt", "between": "between"}


def _signal_from_row(row: pd.Series) -> TradeSignal:
    return TradeSignal(
        timestamp_utc=datetime.now(timezone.utc),
        contract_id=str(row["contract_ticker"]),
        city=str(row["city"]),
        target_date=row["date"],
        metric="high_temp_f",
        threshold_f=int(row["threshold_f"]) if pd.notna(row["threshold_f"]) else 0,
        comparison_operator=_OP_BY_STRIKE.get(str(row.get("strike_type")), "ge"),
        market_probability=float(row["market_prob"]),
        model_probability=float(row["model_prob"]),
        alpha=float(row["alpha"]),
        absolute_alpha=abs(float(row["alpha"])),
        signal_side=str(row["direction"]),
        confidence_score=abs(float(row["alpha"])),
    )


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


@dataclass
class ReplayRow:
    trade_date: date
    city: str
    contract_ticker: str
    entry_price: float
    old_contracts: float
    old_stake_usd: float
    old_realized_pnl: float
    new_stake_usd: float
    new_contracts: float
    new_realized_pnl: float
    capped: bool
    reasoning: str


def replay(csv_path: Path = _PAPER_TRADES_CSV) -> list[ReplayRow]:
    """Re-run the smart sizer over every settled trade; no lookahead.

    Groups by (date, city) — the same unit the live session sizes — and,
    within each group, reconstructs the Kelly allocation and then applies
    the new evidence multipliers and hard caps. Track record for each
    group only ever sees strictly earlier dates (see
    :func:`~deep_isobar.trading.position_sizer.build_station_track_record`).
    """
    df = pd.read_csv(csv_path, dtype={"threshold_f": float, "position_size": float})
    df = df[df["status"].isin(["WIN", "LOSS"])].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for col in ("alpha", "model_prob", "market_prob", "entry_price", "position_size", "realized_pnl"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["entry_price", "position_size", "realized_pnl"])
    df = df[df["position_size"] > 0]

    sizing_cfg = get_setting("risk.position_sizing", default={})
    bankroll_usd = float(sizing_cfg.get("bankroll_usd", 500.0))
    n_cities = df["city"].nunique()

    kelly_base_cfg = get_setting("risk.multi_bracket.kelly", default={})
    max_contracts = get_setting("risk.multi_bracket.max_contracts_per_session", default=3)
    min_alpha = get_setting("risk.multi_bracket.min_alpha_to_spread", default=0.10)

    city_cap = compute_city_daily_cap(bankroll_usd, n_cities, sizing_cfg)

    results: list[ReplayRow] = []

    for (trade_date, city), group in df.sort_values("date").groupby(["date", "city"]):
        signals = [_signal_from_row(row) for _, row in group.iterrows()]
        entry_prices = {s.contract_id: float(p) for s, p in zip(signals, group["entry_price"])}

        kelly_cfg = dict(kelly_base_cfg)
        kelly_cfg["bankroll_usd"] = bankroll_usd
        kelly_cfg.setdefault("n_correlated_bets", n_cities)

        allocations = build_spread(
            signals=signals,
            daily_exposure_cap_usd=city_cap,
            max_contracts=max_contracts,
            min_alpha=min_alpha,
            allocation_method="kelly",
            entry_prices=entry_prices,
            kelly_cfg=kelly_cfg,
        )
        if not allocations:
            continue

        track_record: StationTrackRecord | None = build_station_track_record(
            df, city, asof=trade_date
        )

        # Anomaly/spread evidence: identical for every row in this (date,
        # city) group in production too — take it from the first row.
        first_row = group.iloc[0]
        decisions = adjust_allocations(
            allocations,
            bankroll_usd=bankroll_usd,
            entry_prices=entry_prices,
            anomaly_report=_anomaly_from_row(first_row),
            ensemble_std_f=_ensemble_std_from_row(first_row),
            station_calibration=None,  # see module docstring — known limitation
            station_track_record=track_record,
            cfg=sizing_cfg,
        )

        decision_by_id = {d.contract_id: d for d in decisions}
        row_by_id = {row["contract_ticker"]: row for _, row in group.iterrows()}

        for contract_id, decision in decision_by_id.items():
            row = row_by_id[contract_id]
            old_contracts = float(row["position_size"])
            old_pnl = float(row["realized_pnl"])
            pnl_per_contract = old_pnl / old_contracts if old_contracts else 0.0

            price = entry_prices[contract_id]
            rpc = risk_per_contract(price, str(row["direction"]))
            new_contracts = decision.final_stake_usd / rpc if rpc > 0 else 0.0
            new_pnl = pnl_per_contract * new_contracts

            results.append(ReplayRow(
                trade_date=trade_date,
                city=city,
                contract_ticker=contract_id,
                entry_price=price,
                old_contracts=old_contracts,
                old_stake_usd=old_contracts * rpc,
                old_realized_pnl=old_pnl,
                new_stake_usd=decision.final_stake_usd,
                new_contracts=new_contracts,
                new_realized_pnl=new_pnl,
                capped=decision.capped,
                reasoning=decision.reasoning,
            ))

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_report(rows: list[ReplayRow], bankroll_usd: float) -> str:
    if not rows:
        return "No settled trades to replay."

    df = pd.DataFrame([r.__dict__ for r in rows])
    total_old = df["old_realized_pnl"].sum()
    total_new = df["new_realized_pnl"].sum()

    outlier_idx = df["old_realized_pnl"].idxmax()
    outlier = df.loc[outlier_idx]
    df_excl = df.drop(index=outlier_idx)
    old_excl = df_excl["old_realized_pnl"].sum()
    new_excl = df_excl["new_realized_pnl"].sum()

    # ROI (P&L / dollars staked) is the fair comparison — the new sizer
    # deploys far less capital per trade by design, so raw P&L totals
    # alone understate how it performed on the capital it actually risked.
    old_roi = total_old / df["old_stake_usd"].sum() if df["old_stake_usd"].sum() else float("nan")
    new_roi = total_new / df["new_stake_usd"].sum() if df["new_stake_usd"].sum() else float("nan")
    old_roi_excl = old_excl / df_excl["old_stake_usd"].sum() if df_excl["old_stake_usd"].sum() else float("nan")
    new_roi_excl = new_excl / df_excl["new_stake_usd"].sum() if df_excl["new_stake_usd"].sum() else float("nan")

    # Max drawdown under new sizing.
    by_date = df.groupby("trade_date")["new_realized_pnl"].sum().sort_index()
    equity = bankroll_usd + by_date.cumsum()
    peak = equity.cummax()
    dd = (peak - equity)
    max_dd = float(dd.max())
    max_dd_date = dd.idxmax()

    by_date_old = df.groupby("trade_date")["old_realized_pnl"].sum().sort_index()
    equity_old = bankroll_usd + by_date_old.cumsum()
    dd_old = equity_old.cummax() - equity_old
    max_dd_old = float(dd_old.max())

    cheap = df[df["entry_price"] <= 0.10]
    cheap_excl_outlier = cheap.drop(index=outlier_idx, errors="ignore")

    daily_total_new = df.groupby("trade_date")["new_stake_usd"].sum().sort_index()
    worst_day = daily_total_new.idxmax()
    worst_day_usd = float(daily_total_new.max())

    n_capped = int(df["capped"].sum())

    lines = [
        "=" * 72,
        "  POSITION SIZER REPLAY — historical validation",
        "=" * 72,
        f"  Trades replayed        : {len(df)}",
        f"  Bankroll               : ${bankroll_usd:.2f}",
        "",
        "  -- Total P&L --",
        f"  Actual (old sizer)     : {total_old:+.2f}",
        f"  New sizer              : {total_new:+.2f}",
        f"  Actual, excl. outlier  : {old_excl:+.2f}",
        f"  New sizer, excl outlier: {new_excl:+.2f}",
        f"  Outlier trade          : {outlier['contract_ticker']} "
        f"({outlier['trade_date']}, {outlier['city']}) "
        f"old={outlier['old_realized_pnl']:+.2f} new={outlier['new_realized_pnl']:+.2f}",
        "",
        "  -- ROI (P&L / dollars staked) — the fair comparison, since the new",
        "     sizer deploys far less capital per trade by design --",
        f"  Actual (old sizer)     : {old_roi:+.1%}",
        f"  New sizer              : {new_roi:+.1%}",
        f"  Actual, excl. outlier   : {old_roi_excl:+.1%}",
        f"  New sizer, excl outlier: {new_roi_excl:+.1%}",
        "",
        "  -- Drawdown --",
        f"  Actual max drawdown    : ${max_dd_old:.2f}",
        f"  New sizer max drawdown : ${max_dd:.2f} (around {max_dd_date})",
        "",
        "  -- Cheap tail (entry <= $0.10) --",
        f"  n={len(cheap)}  old stake sum=${cheap['old_stake_usd'].sum():.2f}  "
        f"new stake sum=${cheap['new_stake_usd'].sum():.2f}",
        f"  old P&L=${cheap['old_realized_pnl'].sum():+.2f}  "
        f"new P&L=${cheap['new_realized_pnl'].sum():+.2f}",
        f"  excluding the outlier: n={len(cheap_excl_outlier)}  "
        f"old P&L=${cheap_excl_outlier['old_realized_pnl'].sum():+.2f}  "
        f"new P&L=${cheap_excl_outlier['new_realized_pnl'].sum():+.2f}",
        "",
        "  -- Exposure --",
        f"  Worst single trading day, new sizer  : ${worst_day_usd:.2f} on {worst_day} "
        f"({worst_day_usd / bankroll_usd:.1%} of bankroll)",
        f"  Per-trade cap bound on {n_capped}/{len(df)} trades",
        "",
        "  -- Per-trade stake, new sizer --",
        f"  mean=${df['new_stake_usd'].mean():.2f}  min=${df['new_stake_usd'].min():.2f}  "
        f"max=${df['new_stake_usd'].max():.2f}",
        f"  (was: mean=${df['old_stake_usd'].mean():.2f}  min=${df['old_stake_usd'].min():.2f}  "
        f"max=${df['old_stake_usd'].max():.2f})",
        "=" * 72,
    ]
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sizing_cfg = get_setting("risk.position_sizing", default={})
    bankroll_usd = float(sizing_cfg.get("bankroll_usd", 500.0))
    rows = replay()
    print(build_report(rows, bankroll_usd))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
