"""Tests for kelly.py, the spreader's kelly/SELL paths, and settle fees."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from deep_isobar.core.types import TradeSignal
from deep_isobar.trading.bracket_spreader import build_spread
from deep_isobar.trading.kelly import (
    correlation_haircut,
    effective_entry,
    kalshi_taker_fee,
    kelly_fraction,
    net_edge,
    risk_per_contract,
)
from deep_isobar.research.settle_paper_trades import _compute_pnl


# ── kelly.py math ────────────────────────────────────────────────────────────


def test_taker_fee_peaks_at_half():
    assert kalshi_taker_fee(0.5) == pytest.approx(0.0175)
    assert kalshi_taker_fee(0.05) == pytest.approx(0.07 * 0.05 * 0.95)
    assert kalshi_taker_fee(0.5) > kalshi_taker_fee(0.2) > kalshi_taker_fee(0.05)


def test_effective_entry_both_sides():
    fee = kalshi_taker_fee(0.30)
    assert effective_entry(0.30, "BUY") == pytest.approx(0.30 + fee)
    # SELL at bid 0.30 == buy NO at 0.70, plus the fee at the traded price.
    assert effective_entry(0.30, "SELL") == pytest.approx(0.70 + fee)


def test_net_edge_signs():
    # p=0.20 model vs ask 0.30: buying is negative EV, selling positive.
    assert net_edge(0.20, 0.30, "BUY") < 0
    assert net_edge(0.20, 0.30, "SELL") > 0


def test_kelly_fraction_hand_computed():
    # BUY: p=0.40, ask=0.30, fee=0.07*0.3*0.7=0.0147 → q=0.3147
    # f* = (0.40 − 0.3147)/(1 − 0.3147)
    expected = (0.40 - 0.3147) / (1 - 0.3147)
    assert kelly_fraction(0.40, 0.30, "BUY") == pytest.approx(expected, abs=1e-9)


def test_kelly_fraction_sell_side():
    # SELL: model p=0.17, bid=0.405 (the Dallas B95.5 case from 2026-07-02).
    # Win prob 0.83, effective cost = (1−0.405) + fee(0.405).
    q = (1 - 0.405) + kalshi_taker_fee(0.405)
    expected = (0.83 - q) / (1 - q)
    assert kelly_fraction(0.17, 0.405, "SELL") == pytest.approx(expected, abs=1e-9)
    assert expected > 0.5  # a fat edge — Kelly wants real size before fractioning


def test_kelly_fraction_zero_when_fee_eats_edge():
    # 1.5-point raw edge at mid prices dies to the 1.75-cent fee.
    assert kelly_fraction(0.515, 0.50, "BUY") == 0.0
    assert kelly_fraction(0.485, 0.50, "SELL") == 0.0


def test_correlation_haircut():
    assert correlation_haircut(1, 0.9) == 1.0
    assert correlation_haircut(5, 0.0) == 1.0
    assert correlation_haircut(5, 1.0) == pytest.approx(1 / 5)
    assert correlation_haircut(5, 0.6) == pytest.approx(1 / (1 + 4 * 0.6))
    assert correlation_haircut(3, -0.5) == 1.0  # negative rho clamped


def test_risk_per_contract():
    assert risk_per_contract(0.30, "BUY") == pytest.approx(0.30 + kalshi_taker_fee(0.30))
    assert risk_per_contract(0.30, "SELL") == pytest.approx(0.70 + kalshi_taker_fee(0.30))


# ── build_spread with kelly + SELL ───────────────────────────────────────────


def _signal(cid: str, side: str, model_p: float, market_p: float) -> TradeSignal:
    return TradeSignal(
        timestamp_utc=datetime.now(timezone.utc),
        contract_id=cid,
        city="Testville",
        target_date=date(2026, 7, 3),
        metric="high_temp_f",
        threshold_f=90,
        comparison_operator="ge",
        market_probability=market_p,
        model_probability=model_p,
        alpha=model_p - market_p,
        absolute_alpha=abs(model_p - market_p),
        signal_side=side,
        confidence_score=abs(model_p - market_p),
    )


def test_kelly_spread_includes_sell_signals():
    signals = [
        _signal("BUY1", "BUY", 0.45, 0.30),
        _signal("SELL1", "SELL", 0.17, 0.40),
    ]
    prices = {"BUY1": 0.31, "SELL1": 0.39}
    allocs = build_spread(
        signals, daily_exposure_cap_usd=50.0, max_contracts=3, min_alpha=0.10,
        allocation_method="kelly", entry_prices=prices,
        kelly_cfg={"bankroll_usd": 1000, "fraction": 0.25,
                   "avg_pairwise_correlation": 0.6, "n_correlated_bets": 5},
    )
    sides = {a.signal.contract_id: a.signal.signal_side for a in allocs}
    assert sides == {"BUY1": "BUY", "SELL1": "SELL"}
    for a in allocs:
        assert a.allocated_usd > 0
        assert a.contracts is not None and a.contracts > 0


def test_kelly_spread_drops_fee_dead_edges():
    signals = [_signal("THIN", "BUY", 0.515, 0.50)]
    allocs = build_spread(
        signals, 50.0, 3, 0.01, "kelly", entry_prices={"THIN": 0.50},
        kelly_cfg={"n_correlated_bets": 1},
    )
    assert allocs == []


def test_kelly_spread_respects_daily_cap():
    signals = [
        _signal("A", "BUY", 0.90, 0.30),   # huge edge → huge Kelly
        _signal("B", "SELL", 0.05, 0.60),  # huge edge → huge Kelly
    ]
    prices = {"A": 0.31, "B": 0.59}
    allocs = build_spread(
        signals, 50.0, 3, 0.10, "kelly", entry_prices=prices,
        kelly_cfg={"bankroll_usd": 10000, "fraction": 1.0,
                   "avg_pairwise_correlation": 0.0, "n_correlated_bets": 1},
    )
    assert sum(a.allocated_usd for a in allocs) == pytest.approx(50.0, abs=0.02)


def test_kelly_sizing_scales_with_haircut():
    signals = [_signal("A", "BUY", 0.45, 0.30)]
    prices = {"A": 0.31}
    base = {"bankroll_usd": 1000, "fraction": 0.25, "avg_pairwise_correlation": 0.6}
    solo = build_spread(signals, 500.0, 3, 0.10, "kelly", entry_prices=prices,
                        kelly_cfg={**base, "n_correlated_bets": 1})
    five = build_spread(signals, 500.0, 3, 0.10, "kelly", entry_prices=prices,
                        kelly_cfg={**base, "n_correlated_bets": 5})
    # allocated_usd is rounded to cents, so compare at cent precision.
    assert five[0].allocated_usd == pytest.approx(
        solo[0].allocated_usd / (1 + 4 * 0.6), abs=0.01
    )


def test_kelly_requires_entry_prices():
    with pytest.raises(ValueError, match="entry_prices"):
        build_spread([_signal("A", "BUY", 0.5, 0.3)], 50.0, 3, 0.1, "kelly")


def test_proportional_now_spreads_sells_and_fills_contracts():
    signals = [
        _signal("B1", "BUY", 0.50, 0.30),
        _signal("S1", "SELL", 0.10, 0.35),
    ]
    prices = {"B1": 0.31, "S1": 0.34}
    allocs = build_spread(signals, 50.0, 3, 0.10, "proportional",
                          entry_prices=prices)
    assert {a.signal.contract_id for a in allocs} == {"B1", "S1"}
    assert sum(a.allocated_usd for a in allocs) == pytest.approx(50.0)
    for a in allocs:
        rpc = risk_per_contract(prices[a.signal.contract_id], a.signal.signal_side)
        assert a.contracts == pytest.approx(round(a.allocated_usd / rpc, 2))


def test_equal_allocation_splits_cap():
    signals = [_signal(f"C{i}", "BUY", 0.5, 0.3) for i in range(3)]
    allocs = build_spread(signals, 50.0, 3, 0.10, "equal")
    assert sum(a.allocated_usd for a in allocs) == pytest.approx(50.0)
    assert all(a.allocated_usd in (16.66, 16.67, 16.68) for a in allocs)


# ── settlement fee model ─────────────────────────────────────────────────────


def test_settle_pnl_taker_fee_charged_both_ways():
    fee = 0.07 * 0.40 * 0.60  # 0.0168 per contract at entry 0.40
    win = _compute_pnl("BUY", 0.40, 1, 10.0)
    loss = _compute_pnl("BUY", 0.40, 0, 10.0)
    assert win == pytest.approx((0.60 - fee) * 10)
    assert loss == pytest.approx((-0.40 - fee) * 10)


def test_settle_pnl_sell_side():
    fee = 0.07 * 0.30 * 0.70
    win = _compute_pnl("SELL", 0.30, 0, 10.0)   # YES failed → short wins
    loss = _compute_pnl("SELL", 0.30, 1, 10.0)
    assert win == pytest.approx((0.30 - fee) * 10)
    assert loss == pytest.approx((-0.70 - fee) * 10)


def test_settle_between_outcome_convention():
    """YES iff floor <= actual < cap — must match probability_for_contract."""
    from deep_isobar.research.settle_paper_trades import _settle_open_trades
    import pandas as pd

    import numpy as np

    def _row(ticker, direction):
        # NaN (not "") for unfilled numerics — matches read_csv dtypes.
        return {
            "date": "2026-07-03", "city": "Testville", "contract_ticker": ticker,
            "direction": direction, "status": "OPEN", "entry_price": 0.40,
            "position_size": 10.0, "threshold_f": 95.0, "strike_type": "between",
            "floor_strike": 95, "cap_strike": 97, "realized_pnl": np.nan,
            "settled_temp": np.nan,
        }

    df = pd.DataFrame([_row("B95.5-in", "BUY"), _row("B95.5-sell", "SELL")])
    df, n = _settle_open_trades(df, date(2026, 7, 3), settled_temp=96.0,
                                city_name="Testville")
    assert n == 2
    # 96 is inside [95, 97) → YES settled: BUY wins, SELL loses.
    assert df.loc[df.contract_ticker == "B95.5-in", "status"].iloc[0] == "WIN"
    assert df.loc[df.contract_ticker == "B95.5-sell", "status"].iloc[0] == "LOSS"

    df2 = pd.DataFrame([_row("B95.5-out", "SELL")])
    df2, _ = _settle_open_trades(df2, date(2026, 7, 3), settled_temp=97.0,
                                 city_name="Testville")
    # 97 is outside [95, 97) → NO settled: the short wins.
    assert df2["status"].iloc[0] == "WIN"


def test_settle_pnl_fee_smaller_at_extremes():
    # Same 10-point win, but cheaper fee far from P=0.5.
    win_mid = _compute_pnl("BUY", 0.50, 1, 1.0)
    win_tail = _compute_pnl("BUY", 0.10, 1, 1.0)
    assert (1 - 0.5) - win_mid > (1 - 0.1) - win_tail
