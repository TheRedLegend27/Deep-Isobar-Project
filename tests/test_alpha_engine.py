"""Tests for deep_isobar.trading.alpha_engine.

Covers:
- compute_alpha: positive, negative, zero, boundary values
- classify_signal: BUY, SELL, HOLD branches, exact threshold boundary
- build_trade_signal: full TradeSignal assembly, field mapping, optional flags
- ValueError for invalid probabilities (< 0, > 1)
- ValueError for invalid threshold (<= 0)
"""

import pytest
from datetime import date, datetime, timezone

from deep_isobar.core.types import MarketContract
from deep_isobar.trading.alpha_engine import (
    build_trade_signal,
    classify_signal,
    compute_alpha,
    compute_rank_score,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = datetime(2026, 3, 17, 12, 0, 0, tzinfo=timezone.utc)
_DATE = date(2026, 3, 20)


def _make_contract(**overrides) -> MarketContract:
    defaults = dict(
        contract_id="CHI_HIGH_TEMP_F_GE_80_20260320",
        market_source="Kalshi",
        city="Chicago",
        metric="high_temp_f",
        comparison_operator="ge",
        threshold_f=80,
        target_date=_DATE,
        settlement_source="NWS",
    )
    defaults.update(overrides)
    return MarketContract(**defaults)


# ---------------------------------------------------------------------------
# compute_alpha
# ---------------------------------------------------------------------------


def test_compute_alpha_positive():
    """Model above market → positive alpha."""
    assert compute_alpha(0.70, 0.50) == pytest.approx(0.20)


def test_compute_alpha_negative():
    """Model below market → negative alpha."""
    assert compute_alpha(0.30, 0.50) == pytest.approx(-0.20)


def test_compute_alpha_zero():
    """Equal probabilities → alpha == 0.0."""
    assert compute_alpha(0.60, 0.60) == pytest.approx(0.0)


def test_compute_alpha_boundary_values():
    """Boundary inputs 0.0 and 1.0 are valid."""
    assert compute_alpha(1.0, 0.0) == pytest.approx(1.0)
    assert compute_alpha(0.0, 1.0) == pytest.approx(-1.0)


def test_compute_alpha_invalid_model_probability_below_zero():
    with pytest.raises(ValueError, match="model_probability"):
        compute_alpha(-0.01, 0.5)


def test_compute_alpha_invalid_model_probability_above_one():
    with pytest.raises(ValueError, match="model_probability"):
        compute_alpha(1.01, 0.5)


def test_compute_alpha_invalid_market_probability_below_zero():
    with pytest.raises(ValueError, match="market_probability"):
        compute_alpha(0.5, -0.01)


def test_compute_alpha_invalid_market_probability_above_one():
    with pytest.raises(ValueError, match="market_probability"):
        compute_alpha(0.5, 1.01)


# ---------------------------------------------------------------------------
# classify_signal
# ---------------------------------------------------------------------------


def test_classify_signal_buy():
    """alpha > threshold → BUY."""
    assert classify_signal(alpha=0.15, threshold=0.10) == "BUY"


def test_classify_signal_sell():
    """alpha < -threshold → SELL."""
    assert classify_signal(alpha=-0.15, threshold=0.10) == "SELL"


def test_classify_signal_hold_positive():
    """alpha within [−threshold, threshold] → HOLD (positive side)."""
    assert classify_signal(alpha=0.05, threshold=0.10) == "HOLD"


def test_classify_signal_hold_negative():
    """alpha within [−threshold, threshold] → HOLD (negative side)."""
    assert classify_signal(alpha=-0.05, threshold=0.10) == "HOLD"


def test_classify_signal_hold_zero():
    """alpha == 0.0 → HOLD."""
    assert classify_signal(alpha=0.0, threshold=0.10) == "HOLD"


def test_classify_signal_exact_threshold_is_hold():
    """alpha exactly equal to threshold is NOT a BUY (strict >)."""
    assert classify_signal(alpha=0.10, threshold=0.10) == "HOLD"


def test_classify_signal_just_above_threshold_is_buy():
    """alpha one epsilon above threshold → BUY."""
    assert classify_signal(alpha=0.1001, threshold=0.10) == "BUY"


def test_classify_signal_exact_negative_threshold_is_hold():
    """alpha exactly equal to -threshold → HOLD (strict <)."""
    assert classify_signal(alpha=-0.10, threshold=0.10) == "HOLD"


def test_classify_signal_just_below_negative_threshold_is_sell():
    """alpha one epsilon below -threshold → SELL."""
    assert classify_signal(alpha=-0.1001, threshold=0.10) == "SELL"


def test_classify_signal_raises_on_zero_threshold():
    with pytest.raises(ValueError, match="threshold"):
        classify_signal(alpha=0.15, threshold=0.0)


def test_classify_signal_raises_on_negative_threshold():
    with pytest.raises(ValueError, match="threshold"):
        classify_signal(alpha=0.15, threshold=-0.05)


# ---------------------------------------------------------------------------
# build_trade_signal
# ---------------------------------------------------------------------------


def test_build_trade_signal_returns_trade_signal():
    """build_trade_signal returns a TradeSignal with correct type."""
    from deep_isobar.core.types import TradeSignal

    signal = build_trade_signal(
        timestamp_utc=_TS,
        contract=_make_contract(),
        model_probability=0.72,
        market_probability=0.55,
        signal_threshold=0.10,
        confidence_score=0.85,
    )
    assert isinstance(signal, TradeSignal)


def test_build_trade_signal_buy():
    """model >> market → signal_side BUY."""
    signal = build_trade_signal(
        timestamp_utc=_TS,
        contract=_make_contract(),
        model_probability=0.72,
        market_probability=0.55,
        signal_threshold=0.10,
        confidence_score=0.80,
    )
    assert signal.signal_side == "BUY"
    assert signal.alpha == pytest.approx(0.17)
    assert signal.absolute_alpha == pytest.approx(0.17)


def test_build_trade_signal_sell():
    """model << market → signal_side SELL."""
    signal = build_trade_signal(
        timestamp_utc=_TS,
        contract=_make_contract(),
        model_probability=0.35,
        market_probability=0.55,
        signal_threshold=0.10,
        confidence_score=0.75,
    )
    assert signal.signal_side == "SELL"
    assert signal.alpha == pytest.approx(-0.20)
    assert signal.absolute_alpha == pytest.approx(0.20)


def test_build_trade_signal_hold():
    """Probabilities close together → HOLD."""
    signal = build_trade_signal(
        timestamp_utc=_TS,
        contract=_make_contract(),
        model_probability=0.52,
        market_probability=0.55,
        signal_threshold=0.10,
        confidence_score=0.50,
    )
    assert signal.signal_side == "HOLD"


def test_build_trade_signal_field_mapping():
    """All contract fields are mapped correctly onto the TradeSignal."""
    contract = _make_contract(
        contract_id="CHI_HIGH_TEMP_F_GE_85_20260320",
        city="Chicago",
        metric="high_temp_f",
        comparison_operator="ge",
        threshold_f=85,
        target_date=_DATE,
    )
    signal = build_trade_signal(
        timestamp_utc=_TS,
        contract=contract,
        model_probability=0.65,
        market_probability=0.50,
        signal_threshold=0.10,
        confidence_score=0.70,
    )
    assert signal.contract_id == "CHI_HIGH_TEMP_F_GE_85_20260320"
    assert signal.city == "Chicago"
    assert signal.metric == "high_temp_f"
    assert signal.comparison_operator == "ge"
    assert signal.threshold_f == 85
    assert signal.target_date == _DATE
    assert signal.timestamp_utc == _TS


def test_build_trade_signal_optional_flags_default():
    """Optional boolean flags default to False; rank_score is auto-computed."""
    signal = build_trade_signal(
        timestamp_utc=_TS,
        contract=_make_contract(),
        model_probability=0.65,
        market_probability=0.50,
        signal_threshold=0.10,
        confidence_score=0.70,
    )
    assert signal.tail_opportunity_flag is False
    assert signal.forecast_shift_flag is False
    assert signal.stale_market_flag is False
    assert signal.microstructure_score is None
    # rank_score is now auto-computed from abs(alpha) = abs(0.65 - 0.50) = 0.15
    assert signal.rank_score == pytest.approx(0.15)
    assert signal.model_version == "v1"


def test_build_trade_signal_optional_flags_set():
    """Optional flags are forwarded correctly when provided."""
    signal = build_trade_signal(
        timestamp_utc=_TS,
        contract=_make_contract(),
        model_probability=0.65,
        market_probability=0.50,
        signal_threshold=0.10,
        confidence_score=0.70,
        tail_opportunity_flag=True,
        forecast_shift_flag=True,
        stale_market_flag=True,
        microstructure_score=0.88,
        rank_score=1.25,
        model_version="v2",
    )
    assert signal.tail_opportunity_flag is True
    assert signal.forecast_shift_flag is True
    assert signal.stale_market_flag is True
    assert signal.microstructure_score == pytest.approx(0.88)
    assert signal.rank_score == pytest.approx(1.25)
    assert signal.model_version == "v2"


def test_build_trade_signal_invalid_model_probability():
    with pytest.raises(ValueError, match="model_probability"):
        build_trade_signal(
            timestamp_utc=_TS,
            contract=_make_contract(),
            model_probability=1.5,
            market_probability=0.50,
            signal_threshold=0.10,
            confidence_score=0.70,
        )


def test_build_trade_signal_invalid_market_probability():
    with pytest.raises(ValueError, match="market_probability"):
        build_trade_signal(
            timestamp_utc=_TS,
            contract=_make_contract(),
            model_probability=0.65,
            market_probability=-0.1,
            signal_threshold=0.10,
            confidence_score=0.70,
        )


def test_build_trade_signal_invalid_threshold():
    with pytest.raises(ValueError, match="threshold"):
        build_trade_signal(
            timestamp_utc=_TS,
            contract=_make_contract(),
            model_probability=0.65,
            market_probability=0.50,
            signal_threshold=0.0,
            confidence_score=0.70,
        )


# ---------------------------------------------------------------------------
# compute_rank_score
# ---------------------------------------------------------------------------


def test_compute_rank_score_base_is_abs_alpha():
    """No extras → score == abs(alpha)."""
    assert compute_rank_score(alpha=0.20) == pytest.approx(0.20)
    assert compute_rank_score(alpha=-0.20) == pytest.approx(0.20)


def test_compute_rank_score_zero_alpha():
    """alpha == 0 with no bonuses → score == 0."""
    assert compute_rank_score(alpha=0.0) == pytest.approx(0.0)


def test_compute_rank_score_tail_rank_score_larger_wins():
    """tail_rank_score > abs(alpha) → tail score is the base."""
    score = compute_rank_score(alpha=0.20, tail_rank_score=0.40)
    assert score == pytest.approx(0.40)


def test_compute_rank_score_tail_rank_score_smaller_ignored():
    """tail_rank_score < abs(alpha) → abs(alpha) is still the base."""
    score = compute_rank_score(alpha=0.50, tail_rank_score=0.30)
    assert score == pytest.approx(0.50)


def test_compute_rank_score_tail_rank_score_equal():
    """tail_rank_score == abs(alpha) → no change."""
    score = compute_rank_score(alpha=0.30, tail_rank_score=0.30)
    assert score == pytest.approx(0.30)


def test_compute_rank_score_shift_bonus_added():
    """forecast_shift_flag=True adds shift_bonus to the base."""
    score = compute_rank_score(alpha=0.20, forecast_shift_flag=True, shift_bonus=0.05)
    assert score == pytest.approx(0.25)


def test_compute_rank_score_lag_bonus_added():
    """stale_market_flag=True adds lag_bonus."""
    score = compute_rank_score(alpha=0.20, stale_market_flag=True, lag_bonus=0.05)
    assert score == pytest.approx(0.25)


def test_compute_rank_score_both_flags_additive():
    """Both flags active → both bonuses stack."""
    score = compute_rank_score(
        alpha=0.20,
        forecast_shift_flag=True,
        stale_market_flag=True,
        shift_bonus=0.05,
        lag_bonus=0.05,
    )
    assert score == pytest.approx(0.30)


def test_compute_rank_score_microstructure_contribution():
    """microstructure_score adds microstructure_score * weight."""
    # 0.20 + 0.80 * 0.10 = 0.28
    score = compute_rank_score(alpha=0.20, microstructure_score=0.80, microstructure_weight=0.10)
    assert score == pytest.approx(0.28)


def test_compute_rank_score_all_components():
    """All components combine correctly."""
    # base = max(0.20, 0.30) = 0.30
    # + 0.05 (shift) + 0.05 (lag) + 0.60 * 0.10 (micro) = 0.46
    score = compute_rank_score(
        alpha=0.20,
        tail_rank_score=0.30,
        microstructure_score=0.60,
        forecast_shift_flag=True,
        stale_market_flag=True,
        shift_bonus=0.05,
        lag_bonus=0.05,
        microstructure_weight=0.10,
    )
    assert score == pytest.approx(0.46)


def test_compute_rank_score_zero_bonuses_no_effect():
    """Bonuses set to 0 and flags True → no change from base."""
    score = compute_rank_score(
        alpha=0.20,
        forecast_shift_flag=True,
        stale_market_flag=True,
        shift_bonus=0.0,
        lag_bonus=0.0,
    )
    assert score == pytest.approx(0.20)


def test_compute_rank_score_invalid_shift_bonus():
    with pytest.raises(ValueError, match="shift_bonus"):
        compute_rank_score(alpha=0.20, shift_bonus=-0.01)


def test_compute_rank_score_invalid_lag_bonus():
    with pytest.raises(ValueError, match="lag_bonus"):
        compute_rank_score(alpha=0.20, lag_bonus=-0.01)


def test_compute_rank_score_invalid_microstructure_weight():
    with pytest.raises(ValueError, match="microstructure_weight"):
        compute_rank_score(alpha=0.20, microstructure_weight=-0.01)


def test_compute_rank_score_invalid_microstructure_score():
    with pytest.raises(ValueError, match="microstructure_score"):
        compute_rank_score(alpha=0.20, microstructure_score=-0.01)


# ---------------------------------------------------------------------------
# build_trade_signal rank_score integration
# ---------------------------------------------------------------------------


def test_build_trade_signal_auto_rank_score_equals_abs_alpha():
    """Without flags, auto rank_score == abs(alpha)."""
    signal = build_trade_signal(
        timestamp_utc=_TS,
        contract=_make_contract(),
        model_probability=0.70,
        market_probability=0.50,
        signal_threshold=0.10,
        confidence_score=0.80,
    )
    assert signal.rank_score == pytest.approx(abs(signal.alpha))


def test_build_trade_signal_auto_rank_score_with_shift_flag():
    """forecast_shift_flag=True boosts auto rank_score by default shift_bonus (0.05)."""
    signal = build_trade_signal(
        timestamp_utc=_TS,
        contract=_make_contract(),
        model_probability=0.70,
        market_probability=0.50,
        signal_threshold=0.10,
        confidence_score=0.80,
        forecast_shift_flag=True,
    )
    # abs(alpha)=0.20, + 0.05 shift bonus = 0.25
    assert signal.rank_score == pytest.approx(0.25)


def test_build_trade_signal_explicit_rank_score_preserved():
    """Explicitly supplied rank_score is not overridden."""
    signal = build_trade_signal(
        timestamp_utc=_TS,
        contract=_make_contract(),
        model_probability=0.70,
        market_probability=0.50,
        signal_threshold=0.10,
        confidence_score=0.80,
        rank_score=9.99,
    )
    assert signal.rank_score == pytest.approx(9.99)


def test_build_trade_signal_rank_score_does_not_alter_alpha():
    """rank_score computation must never change alpha or signal_side."""
    signal = build_trade_signal(
        timestamp_utc=_TS,
        contract=_make_contract(),
        model_probability=0.70,
        market_probability=0.50,
        signal_threshold=0.10,
        confidence_score=0.80,
        forecast_shift_flag=True,
        stale_market_flag=True,
        microstructure_score=0.90,
    )
    assert signal.alpha == pytest.approx(0.20)
    assert signal.absolute_alpha == pytest.approx(0.20)
    assert signal.signal_side == "BUY"


# ---------------------------------------------------------------------------
# build_trade_signal — tail_multiplier integration
# ---------------------------------------------------------------------------


def test_build_trade_signal_tail_flag_boosts_rank_score():
    """tail_opportunity_flag=True with tail_multiplier=1.5 boosts rank_score.

    alpha = 0.70 - 0.50 = 0.20
    tail_rank_score = 0.20 * 1.5 = 0.30
    compute_rank_score base = max(0.20, 0.30) = 0.30
    rank_score = 0.30
    """
    signal = build_trade_signal(
        timestamp_utc=_TS,
        contract=_make_contract(),
        model_probability=0.70,
        market_probability=0.50,
        signal_threshold=0.10,
        confidence_score=0.80,
        tail_opportunity_flag=True,
        tail_multiplier=1.5,
    )
    assert signal.rank_score == pytest.approx(0.30)
    assert signal.tail_opportunity_flag is True


def test_build_trade_signal_tail_multiplier_one_no_extra_boost():
    """tail_multiplier=1.0 with tail_flag=True leaves rank_score == abs(alpha).

    abs(alpha)*1.0 == abs(alpha), so max(abs(alpha), abs(alpha)) = abs(alpha).
    """
    signal = build_trade_signal(
        timestamp_utc=_TS,
        contract=_make_contract(),
        model_probability=0.70,
        market_probability=0.50,
        signal_threshold=0.10,
        confidence_score=0.80,
        tail_opportunity_flag=True,
        tail_multiplier=1.0,
    )
    assert signal.rank_score == pytest.approx(abs(signal.alpha))


def test_build_trade_signal_tail_flag_false_no_boost():
    """tail_opportunity_flag=False (default): tail_multiplier has no effect."""
    signal_no_tail = build_trade_signal(
        timestamp_utc=_TS,
        contract=_make_contract(),
        model_probability=0.70,
        market_probability=0.50,
        signal_threshold=0.10,
        confidence_score=0.80,
        tail_opportunity_flag=False,
        tail_multiplier=2.0,  # ignored because flag is False
    )
    assert signal_no_tail.rank_score == pytest.approx(abs(signal_no_tail.alpha))


def test_build_trade_signal_tail_and_shift_combined():
    """Tail boost and shift bonus stack correctly.

    alpha = 0.20, tail_multiplier=1.5 → tail_rank_score=0.30
    base = max(0.20, 0.30) = 0.30
    + shift_bonus 0.05 → rank_score = 0.35
    """
    signal = build_trade_signal(
        timestamp_utc=_TS,
        contract=_make_contract(),
        model_probability=0.70,
        market_probability=0.50,
        signal_threshold=0.10,
        confidence_score=0.80,
        tail_opportunity_flag=True,
        tail_multiplier=1.5,
        forecast_shift_flag=True,
    )
    assert signal.rank_score == pytest.approx(0.35)


def test_build_trade_signal_tail_does_not_alter_alpha():
    """Tail boost must never modify raw alpha or signal_side."""
    signal = build_trade_signal(
        timestamp_utc=_TS,
        contract=_make_contract(),
        model_probability=0.70,
        market_probability=0.50,
        signal_threshold=0.10,
        confidence_score=0.80,
        tail_opportunity_flag=True,
        tail_multiplier=3.0,
    )
    assert signal.alpha == pytest.approx(0.20)
    assert signal.absolute_alpha == pytest.approx(0.20)
    assert signal.signal_side == "BUY"


def test_build_trade_signal_explicit_rank_score_overrides_tail():
    """Explicit rank_score skips all auto-computation including tail boost."""
    signal = build_trade_signal(
        timestamp_utc=_TS,
        contract=_make_contract(),
        model_probability=0.70,
        market_probability=0.50,
        signal_threshold=0.10,
        confidence_score=0.80,
        tail_opportunity_flag=True,
        tail_multiplier=1.5,
        rank_score=0.99,
    )
    assert signal.rank_score == pytest.approx(0.99)
