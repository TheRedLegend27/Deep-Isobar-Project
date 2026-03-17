"""Tests for deep_isobar.trading.risk_manager.

Covers:
- approve BUY signal within all limits
- approve SELL signal within all limits
- reject HOLD signal
- reject proposed_quantity <= 0
- reject proposed_price out of [0, 1]
- reject position limit breach (BUY)
- reject position limit breach (SELL / short)
- reject daily exposure breach
- projected_position and daily_exposure_after computed correctly
- ValueError on negative limit arguments
- rejection_reason populated on every rejection
"""

import pytest
from datetime import date, datetime, timezone

from deep_isobar.core.types import RiskDecision, TradeSignal
from deep_isobar.trading.risk_manager import approve_trade

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_TS = datetime(2026, 3, 17, 12, 0, 0, tzinfo=timezone.utc)
_DATE = date(2026, 3, 20)


def _make_signal(
    side: str = "BUY",
    contract_id: str = "CHI_HIGH_TEMP_F_GE_80_20260320",
) -> TradeSignal:
    alpha = 0.15 if side == "BUY" else (-0.15 if side == "SELL" else 0.0)
    return TradeSignal(
        timestamp_utc=_TS,
        contract_id=contract_id,
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


def _approve(
    side: str = "BUY",
    quantity: float = 10.0,
    price: float = 0.55,
    current_position: float = 0.0,
    max_position: float = 50.0,
    daily_before: float = 0.0,
    max_daily: float = 1000.0,
) -> RiskDecision:
    """Convenience wrapper with safe defaults."""
    return approve_trade(
        signal=_make_signal(side),
        proposed_quantity=quantity,
        proposed_price=price,
        current_position=current_position,
        max_position_per_contract=max_position,
        daily_exposure_before=daily_before,
        max_daily_exposure=max_daily,
    )


# ---------------------------------------------------------------------------
# Approvals
# ---------------------------------------------------------------------------


def test_approve_buy_signal():
    """BUY within all limits → approved."""
    decision = _approve(side="BUY")
    assert isinstance(decision, RiskDecision)
    assert decision.approved_flag is True
    assert decision.rejection_reason is None


def test_approve_sell_signal():
    """SELL within all limits → approved."""
    decision = _approve(side="SELL")
    assert decision.approved_flag is True
    assert decision.rejection_reason is None


def test_approve_buy_proposed_side_echoed():
    """proposed_side on the decision matches the signal side."""
    decision = _approve(side="BUY")
    assert decision.proposed_side == "BUY"


def test_approve_sell_proposed_side_echoed():
    decision = _approve(side="SELL")
    assert decision.proposed_side == "SELL"


def test_approve_projected_position_buy():
    """BUY: projected = current + quantity."""
    decision = _approve(side="BUY", quantity=10.0, current_position=5.0)
    assert decision.projected_position == pytest.approx(15.0)


def test_approve_projected_position_sell():
    """SELL: projected = current − quantity."""
    decision = _approve(side="SELL", quantity=10.0, current_position=15.0)
    assert decision.projected_position == pytest.approx(5.0)


def test_approve_daily_exposure_after():
    """daily_exposure_after = daily_before + quantity * price."""
    decision = _approve(quantity=10.0, price=0.55, daily_before=100.0)
    assert decision.daily_exposure_after == pytest.approx(100.0 + 10.0 * 0.55)


def test_approve_price_boundary_zero():
    """price == 0.0 is a valid boundary."""
    decision = _approve(price=0.0)
    assert decision.approved_flag is True


def test_approve_price_boundary_one():
    """price == 1.0 is a valid boundary."""
    decision = _approve(price=1.0)
    assert decision.approved_flag is True


def test_approve_at_exact_position_limit():
    """Exactly at the position limit is allowed (≤, not <)."""
    # current=40, qty=10, max=50 → projected=50, which is == max → allowed
    decision = _approve(side="BUY", quantity=10.0, current_position=40.0, max_position=50.0)
    assert decision.approved_flag is True


def test_approve_at_exact_daily_limit():
    """Exactly at the daily exposure limit is allowed (≤, not <)."""
    # qty=10, price=0.50, daily_before=995.0 → after=1000.0 == max → allowed
    decision = _approve(quantity=10.0, price=0.50, daily_before=995.0, max_daily=1000.0)
    assert decision.approved_flag is True


# ---------------------------------------------------------------------------
# Rejections — HOLD
# ---------------------------------------------------------------------------


def test_reject_hold_signal():
    """HOLD is never executable."""
    decision = _approve(side="HOLD")
    assert decision.approved_flag is False
    assert "HOLD" in decision.rejection_reason


def test_reject_hold_current_position_preserved():
    """On HOLD rejection, current_position is preserved unchanged."""
    decision = _approve(side="HOLD", current_position=7.0)
    assert decision.current_position == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Rejections — invalid inputs
# ---------------------------------------------------------------------------


def test_reject_zero_quantity():
    """proposed_quantity == 0 is rejected."""
    decision = _approve(quantity=0.0)
    assert decision.approved_flag is False
    assert "quantity" in decision.rejection_reason.lower()


def test_reject_negative_quantity():
    """proposed_quantity < 0 is rejected."""
    decision = _approve(quantity=-5.0)
    assert decision.approved_flag is False
    assert "quantity" in decision.rejection_reason.lower()


def test_reject_price_above_one():
    """proposed_price > 1.0 is rejected."""
    decision = _approve(price=1.01)
    assert decision.approved_flag is False
    assert "price" in decision.rejection_reason.lower()


def test_reject_price_below_zero():
    """proposed_price < 0.0 is rejected."""
    decision = _approve(price=-0.01)
    assert decision.approved_flag is False
    assert "price" in decision.rejection_reason.lower()


# ---------------------------------------------------------------------------
# Rejections — position limit
# ---------------------------------------------------------------------------


def test_reject_position_limit_breach_buy():
    """BUY that would push position past max is rejected."""
    # current=45, qty=10, max=50 → projected=55 > 50
    decision = _approve(side="BUY", quantity=10.0, current_position=45.0, max_position=50.0)
    assert decision.approved_flag is False
    assert "position" in decision.rejection_reason.lower()


def test_reject_position_limit_breach_sell():
    """SELL that would push position below -max is rejected."""
    # current=0, qty=10, max=5 → projected=-10, abs(-10)=10 > 5
    decision = _approve(side="SELL", quantity=10.0, current_position=0.0, max_position=5.0)
    assert decision.approved_flag is False
    assert "position" in decision.rejection_reason.lower()


def test_reject_position_projected_position_still_populated():
    """projected_position is still computed and stored even on rejection."""
    decision = _approve(side="BUY", quantity=10.0, current_position=45.0, max_position=50.0)
    assert decision.projected_position == pytest.approx(55.0)


# ---------------------------------------------------------------------------
# Rejections — daily exposure
# ---------------------------------------------------------------------------


def test_reject_daily_exposure_breach():
    """Trade that would exceed max_daily_exposure is rejected."""
    # qty=10, price=0.60, daily_before=995 → after=1001 > 1000
    decision = _approve(quantity=10.0, price=0.60, daily_before=995.0, max_daily=1000.0)
    assert decision.approved_flag is False
    assert "exposure" in decision.rejection_reason.lower()


def test_reject_daily_exposure_after_still_populated():
    """daily_exposure_after is computed and stored even on rejection."""
    decision = _approve(quantity=10.0, price=0.60, daily_before=995.0, max_daily=1000.0)
    assert decision.daily_exposure_after == pytest.approx(995.0 + 10.0 * 0.60)


# ---------------------------------------------------------------------------
# Gate priority: HOLD checked before quantity/price gates
# ---------------------------------------------------------------------------


def test_hold_rejected_before_invalid_quantity():
    """HOLD is rejected first even if quantity is also invalid."""
    decision = approve_trade(
        signal=_make_signal("HOLD"),
        proposed_quantity=-5.0,
        proposed_price=0.55,
        current_position=0.0,
        max_position_per_contract=50.0,
        daily_exposure_before=0.0,
        max_daily_exposure=1000.0,
    )
    assert decision.approved_flag is False
    assert "HOLD" in decision.rejection_reason


# ---------------------------------------------------------------------------
# ValueError for misconfigured limits
# ---------------------------------------------------------------------------


def test_raises_on_negative_max_position():
    with pytest.raises(ValueError, match="max_position_per_contract"):
        approve_trade(
            signal=_make_signal("BUY"),
            proposed_quantity=10.0,
            proposed_price=0.55,
            current_position=0.0,
            max_position_per_contract=-1.0,
            daily_exposure_before=0.0,
            max_daily_exposure=1000.0,
        )


def test_raises_on_negative_daily_exposure_before():
    with pytest.raises(ValueError, match="daily_exposure_before"):
        approve_trade(
            signal=_make_signal("BUY"),
            proposed_quantity=10.0,
            proposed_price=0.55,
            current_position=0.0,
            max_position_per_contract=50.0,
            daily_exposure_before=-1.0,
            max_daily_exposure=1000.0,
        )


def test_raises_on_negative_max_daily_exposure():
    with pytest.raises(ValueError, match="max_daily_exposure"):
        approve_trade(
            signal=_make_signal("BUY"),
            proposed_quantity=10.0,
            proposed_price=0.55,
            current_position=0.0,
            max_position_per_contract=50.0,
            daily_exposure_before=0.0,
            max_daily_exposure=-100.0,
        )
