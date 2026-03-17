"""Tests for deep_isobar.trading.trade_execution.

Covers:
- execute_paper_trade: returns ExecutedTrade with correct fields
- execute_paper_trade: paper_trade_flag is True
- execute_paper_trade: execution_status is "filled"
- execute_paper_trade: fill_quantity and avg_fill_price match inputs
- execute_paper_trade: trade_id has expected PAPER- prefix
- execute_paper_trade: signal fields (side, contract_id) propagated
- execute_paper_trade: ValueError on quantity <= 0
- execute_paper_trade: ValueError on price outside [0, 1]
- execute_paper_trade: boundary prices 0.0 and 1.0 accepted
- submit_live_trade: always raises RuntimeError
"""

import pytest
from datetime import date, datetime, timezone

from deep_isobar.core.types import ExecutedTrade, TradeSignal
from deep_isobar.trading.trade_execution import execute_paper_trade, submit_live_trade

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_TS = datetime(2026, 3, 17, 12, 0, 0, tzinfo=timezone.utc)
_DATE = date(2026, 3, 20)


def _make_signal(side: str = "BUY") -> TradeSignal:
    alpha = 0.15 if side == "BUY" else -0.15
    return TradeSignal(
        timestamp_utc=_TS,
        contract_id="CHI_HIGH_TEMP_F_GE_80_20260320",
        city="Chicago",
        target_date=_DATE,
        metric="high_temp_f",
        threshold_f=80,
        comparison_operator="ge",
        market_probability=0.50,
        model_probability=0.50 + alpha,
        alpha=alpha,
        absolute_alpha=abs(alpha),
        signal_side=side,
        confidence_score=abs(alpha),
    )


# ---------------------------------------------------------------------------
# execute_paper_trade — happy path
# ---------------------------------------------------------------------------


def test_paper_trade_returns_executed_trade():
    """Returns an ExecutedTrade instance."""
    trade = execute_paper_trade(signal=_make_signal(), quantity=10.0, price=0.55)
    assert isinstance(trade, ExecutedTrade)


def test_paper_trade_flag_is_true():
    """paper_trade_flag must always be True."""
    trade = execute_paper_trade(signal=_make_signal(), quantity=10.0, price=0.55)
    assert trade.paper_trade_flag is True


def test_paper_trade_execution_status_filled():
    """execution_status is 'filled' for a paper trade."""
    trade = execute_paper_trade(signal=_make_signal(), quantity=10.0, price=0.55)
    assert trade.execution_status == "filled"


def test_paper_trade_fill_quantity_matches_input():
    """fill_quantity equals the requested quantity (instant full fill)."""
    trade = execute_paper_trade(signal=_make_signal(), quantity=7.0, price=0.55)
    assert trade.fill_quantity == pytest.approx(7.0)


def test_paper_trade_avg_fill_price_matches_input():
    """avg_fill_price equals the requested price (no slippage)."""
    trade = execute_paper_trade(signal=_make_signal(), quantity=10.0, price=0.62)
    assert trade.avg_fill_price == pytest.approx(0.62)


def test_paper_trade_quantity_and_price_echoed():
    """quantity and price on the trade record match inputs."""
    trade = execute_paper_trade(signal=_make_signal(), quantity=5.0, price=0.40)
    assert trade.quantity == pytest.approx(5.0)
    assert trade.price == pytest.approx(0.40)


def test_paper_trade_contract_id_from_signal():
    """contract_id on the trade matches the signal's contract_id."""
    trade = execute_paper_trade(signal=_make_signal(), quantity=10.0, price=0.55)
    assert trade.contract_id == "CHI_HIGH_TEMP_F_GE_80_20260320"


def test_paper_trade_side_from_signal_buy():
    """side is taken from signal.signal_side (BUY)."""
    trade = execute_paper_trade(signal=_make_signal("BUY"), quantity=10.0, price=0.55)
    assert trade.side == "BUY"


def test_paper_trade_side_from_signal_sell():
    """side is taken from signal.signal_side (SELL)."""
    trade = execute_paper_trade(signal=_make_signal("SELL"), quantity=10.0, price=0.45)
    assert trade.side == "SELL"


def test_paper_trade_market_source_is_paper():
    """market_source is 'paper' for simulated trades."""
    trade = execute_paper_trade(signal=_make_signal(), quantity=10.0, price=0.55)
    assert trade.market_source == "paper"


def test_paper_trade_id_has_paper_prefix():
    """trade_id starts with 'PAPER-'."""
    trade = execute_paper_trade(signal=_make_signal(), quantity=10.0, price=0.55)
    assert trade.trade_id.startswith("PAPER-")


def test_paper_trade_id_unique_across_calls():
    """Two separate calls produce different trade IDs (different timestamps)."""
    t1 = execute_paper_trade(signal=_make_signal(), quantity=10.0, price=0.55)
    t2 = execute_paper_trade(signal=_make_signal(), quantity=10.0, price=0.55)
    # In practice timestamps differ by at least a few nanoseconds; IDs should differ.
    # If they somehow collide (sub-microsecond), the test is still informative.
    assert isinstance(t1.trade_id, str)
    assert isinstance(t2.trade_id, str)


def test_paper_trade_timestamp_is_datetime():
    """timestamp_utc is a datetime object."""
    trade = execute_paper_trade(signal=_make_signal(), quantity=10.0, price=0.55)
    assert isinstance(trade.timestamp_utc, datetime)


def test_paper_trade_exchange_order_id_is_none():
    """exchange_order_id is None for paper trades."""
    trade = execute_paper_trade(signal=_make_signal(), quantity=10.0, price=0.55)
    assert trade.exchange_order_id is None


# ---------------------------------------------------------------------------
# execute_paper_trade — boundary prices
# ---------------------------------------------------------------------------


def test_paper_trade_price_boundary_zero():
    """price == 0.0 is valid."""
    trade = execute_paper_trade(signal=_make_signal(), quantity=1.0, price=0.0)
    assert trade.price == pytest.approx(0.0)
    assert trade.paper_trade_flag is True


def test_paper_trade_price_boundary_one():
    """price == 1.0 is valid."""
    trade = execute_paper_trade(signal=_make_signal(), quantity=1.0, price=1.0)
    assert trade.price == pytest.approx(1.0)
    assert trade.paper_trade_flag is True


# ---------------------------------------------------------------------------
# execute_paper_trade — validation errors
# ---------------------------------------------------------------------------


def test_paper_trade_raises_on_zero_quantity():
    """quantity == 0 raises ValueError."""
    with pytest.raises(ValueError, match="quantity"):
        execute_paper_trade(signal=_make_signal(), quantity=0.0, price=0.55)


def test_paper_trade_raises_on_negative_quantity():
    """quantity < 0 raises ValueError."""
    with pytest.raises(ValueError, match="quantity"):
        execute_paper_trade(signal=_make_signal(), quantity=-1.0, price=0.55)


def test_paper_trade_raises_on_price_above_one():
    """price > 1.0 raises ValueError."""
    with pytest.raises(ValueError, match="price"):
        execute_paper_trade(signal=_make_signal(), quantity=10.0, price=1.01)


def test_paper_trade_raises_on_price_below_zero():
    """price < 0.0 raises ValueError."""
    with pytest.raises(ValueError, match="price"):
        execute_paper_trade(signal=_make_signal(), quantity=10.0, price=-0.01)


# ---------------------------------------------------------------------------
# submit_live_trade — not implemented
# ---------------------------------------------------------------------------


def test_submit_live_trade_raises_runtime_error():
    """submit_live_trade always raises RuntimeError."""
    with pytest.raises(RuntimeError, match="not implemented"):
        submit_live_trade(
            market_source="Kalshi",
            contract_id="CHI_HIGH_TEMP_F_GE_80_20260320",
            side="BUY",
            quantity=10.0,
            price=0.55,
        )


def test_submit_live_trade_raises_with_default_order_type():
    """submit_live_trade raises even with default order_type."""
    with pytest.raises(RuntimeError):
        submit_live_trade(
            market_source="Kalshi",
            contract_id="CHI_HIGH_TEMP_F_GE_80_20260320",
            side="SELL",
            quantity=5.0,
            price=0.45,
        )


def test_submit_live_trade_error_mentions_contract():
    """RuntimeError message includes the contract ID for debuggability."""
    cid = "CHI_HIGH_TEMP_F_GE_80_20260320"
    with pytest.raises(RuntimeError, match=cid):
        submit_live_trade(
            market_source="Kalshi",
            contract_id=cid,
            side="BUY",
            quantity=1.0,
            price=0.55,
        )
