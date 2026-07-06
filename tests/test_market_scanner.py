"""Tests for deep_isobar.market.market_scanner.

Covers:
- evaluate_contract_opportunity: successful BUY / SELL / HOLD evaluation
- evaluate_contract_opportunity: missing threshold in surface raises KeyError
- evaluate_contract_opportunity: bad orderbook raises ValueError
- evaluate_contract_opportunity: timestamp defaults to now when not supplied
- evaluate_contract_opportunity: confidence_score == abs(alpha)
- evaluate_contract_opportunity: returned signal has a rank_score
- rank_trade_signals: rank_score used as primary key when available
- rank_trade_signals: fallback to absolute_alpha when rank_score is None
- rank_trade_signals: tie-breaking by absolute_alpha then contract_id
- rank_trade_signals: deterministic output regardless of input order
- rank_trade_signals: empty list handled
- rank_trade_signals: single-element list handled
"""

import pytest
from datetime import date, datetime, timezone

from deep_isobar.core.types import MarketContract, OrderBookSnapshot, TradeSignal
from deep_isobar.market.market_scanner import (
    evaluate_contract_opportunity,
    rank_trade_signals,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_DATE = date(2026, 3, 20)
_TS = datetime(2026, 3, 17, 12, 0, 0, tzinfo=timezone.utc)
_THRESHOLD = 0.10


def _make_contract(threshold_f: int = 80, **overrides) -> MarketContract:
    defaults = dict(
        contract_id=f"CHI_HIGH_TEMP_F_GE_{threshold_f}_20260320",
        market_source="Kalshi",
        city="Chicago",
        metric="high_temp_f",
        comparison_operator="ge",
        threshold_f=threshold_f,
        target_date=_DATE,
        settlement_source="NWS",
    )
    defaults.update(overrides)
    return MarketContract(**defaults)


def _make_orderbook(bid: float | None, ask: float | None, contract_id: str = "CHI_HIGH_TEMP_F_GE_80_20260320") -> OrderBookSnapshot:
    return OrderBookSnapshot(
        timestamp_utc=_TS,
        contract_id=contract_id,
        market_source="Kalshi",
        best_bid=bid,
        best_ask=ask,
    )


def _make_signal(
    absolute_alpha: float,
    contract_id: str = "X",
    rank_score: float | None = None,
) -> TradeSignal:
    """Convenience: build a minimal TradeSignal with a known absolute_alpha."""
    alpha = absolute_alpha  # positive alpha
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
        absolute_alpha=absolute_alpha,
        signal_side="BUY" if alpha > _THRESHOLD else "HOLD",
        confidence_score=absolute_alpha,
        rank_score=rank_score,
    )


# ---------------------------------------------------------------------------
# evaluate_contract_opportunity — successful evaluations
# ---------------------------------------------------------------------------


def test_evaluate_returns_trade_signal():
    """Returns a TradeSignal instance."""
    surface = {80: 0.70}
    ob = _make_orderbook(bid=55.0, ask=57.0)
    result = evaluate_contract_opportunity(
        contract=_make_contract(80),
        probability_surface=surface,
        orderbook=ob,
        signal_threshold=_THRESHOLD,
        timestamp_utc=_TS,
    )
    assert isinstance(result, TradeSignal)


def test_evaluate_buy_signal():
    """Model probability >> market probability → BUY."""
    # market mid = (40+42)/2 = 41 cents → 0.41
    # model = 0.70 → alpha = 0.29 > threshold 0.10 → BUY
    surface = {80: 0.70}
    ob = _make_orderbook(bid=40.0, ask=42.0)
    signal = evaluate_contract_opportunity(
        contract=_make_contract(80),
        probability_surface=surface,
        orderbook=ob,
        signal_threshold=_THRESHOLD,
        timestamp_utc=_TS,
    )
    assert signal.signal_side == "BUY"
    assert signal.alpha == pytest.approx(0.29, abs=1e-6)
    assert signal.model_probability == pytest.approx(0.70)
    assert signal.market_probability == pytest.approx(0.41)


def test_evaluate_sell_signal():
    """Model probability << market probability → SELL."""
    # market mid = (72+74)/2 = 73 cents → 0.73
    # model = 0.45 → alpha = -0.28 < -0.10 → SELL
    surface = {80: 0.45}
    ob = _make_orderbook(bid=72.0, ask=74.0)
    signal = evaluate_contract_opportunity(
        contract=_make_contract(80),
        probability_surface=surface,
        orderbook=ob,
        signal_threshold=_THRESHOLD,
        timestamp_utc=_TS,
    )
    assert signal.signal_side == "SELL"
    assert signal.alpha == pytest.approx(-0.28, abs=1e-6)


def test_evaluate_hold_signal():
    """Probabilities close together → HOLD."""
    # market mid = (59+61)/2 = 60 cents → 0.60
    # model = 0.62 → alpha = 0.02 < 0.10 → HOLD
    surface = {80: 0.62}
    ob = _make_orderbook(bid=59.0, ask=61.0)
    signal = evaluate_contract_opportunity(
        contract=_make_contract(80),
        probability_surface=surface,
        orderbook=ob,
        signal_threshold=_THRESHOLD,
        timestamp_utc=_TS,
    )
    assert signal.signal_side == "HOLD"


def test_evaluate_confidence_score_equals_abs_alpha():
    """confidence_score is set to abs(alpha) for the MVP."""
    surface = {80: 0.70}
    ob = _make_orderbook(bid=40.0, ask=42.0)
    signal = evaluate_contract_opportunity(
        contract=_make_contract(80),
        probability_surface=surface,
        orderbook=ob,
        signal_threshold=_THRESHOLD,
        timestamp_utc=_TS,
    )
    assert signal.confidence_score == pytest.approx(signal.absolute_alpha)


def test_evaluate_contract_fields_propagated():
    """Contract metadata is propagated correctly to the TradeSignal."""
    contract = _make_contract(
        threshold_f=85,
        contract_id="CHI_HIGH_TEMP_F_GE_85_20260320",
        city="Chicago",
        metric="high_temp_f",
        comparison_operator="ge",
    )
    surface = {85: 0.65}
    ob = _make_orderbook(bid=50.0, ask=52.0, contract_id="CHI_HIGH_TEMP_F_GE_85_20260320")
    signal = evaluate_contract_opportunity(
        contract=contract,
        probability_surface=surface,
        orderbook=ob,
        signal_threshold=_THRESHOLD,
        timestamp_utc=_TS,
    )
    assert signal.contract_id == "CHI_HIGH_TEMP_F_GE_85_20260320"
    assert signal.threshold_f == 85
    assert signal.city == "Chicago"
    assert signal.metric == "high_temp_f"
    assert signal.comparison_operator == "ge"
    assert signal.target_date == _DATE


def test_evaluate_timestamp_defaults_to_now(monkeypatch):
    """timestamp_utc defaults to datetime.now(timezone.utc) when not given."""
    fixed_ts = datetime(2026, 3, 17, 9, 0, 0, tzinfo=timezone.utc)

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_ts

    import deep_isobar.market.market_scanner as scanner_mod
    monkeypatch.setattr(scanner_mod, "datetime", _FakeDatetime)

    surface = {80: 0.65}
    ob = _make_orderbook(bid=50.0, ask=52.0)
    signal = evaluate_contract_opportunity(
        contract=_make_contract(80),
        probability_surface=surface,
        orderbook=ob,
        signal_threshold=_THRESHOLD,
        # no timestamp_utc → should default
    )
    assert signal.timestamp_utc == fixed_ts


def test_evaluate_decimal_priced_orderbook():
    """Orderbook with decimal bid/ask (already in [0,1]) is handled correctly."""
    # bid=0.55, ask=0.57 → both ≤ 1.0 → used as-is → mid=0.56
    surface = {80: 0.70}
    ob = _make_orderbook(bid=0.55, ask=0.57)
    signal = evaluate_contract_opportunity(
        contract=_make_contract(80),
        probability_surface=surface,
        orderbook=ob,
        signal_threshold=_THRESHOLD,
        timestamp_utc=_TS,
    )
    assert signal.market_probability == pytest.approx(0.56)
    assert signal.signal_side == "BUY"


# ---------------------------------------------------------------------------
# evaluate_contract_opportunity — error cases
# ---------------------------------------------------------------------------


def test_evaluate_raises_key_error_on_missing_threshold():
    """KeyError when contract.threshold_f is absent from probability_surface."""
    surface = {75: 0.70, 85: 0.40}   # 80 is missing
    ob = _make_orderbook(bid=55.0, ask=57.0)
    with pytest.raises(KeyError, match="80"):
        evaluate_contract_opportunity(
            contract=_make_contract(80),
            probability_surface=surface,
            orderbook=ob,
            signal_threshold=_THRESHOLD,
            timestamp_utc=_TS,
        )


def test_evaluate_raises_on_no_usable_price():
    """ValueError when orderbook has no bid, ask, or last_trade_price."""
    surface = {80: 0.65}
    ob = OrderBookSnapshot(
        timestamp_utc=_TS,
        contract_id="CHI_HIGH_TEMP_F_GE_80_20260320",
        market_source="Kalshi",
        best_bid=None,
        best_ask=None,
        last_trade_price=None,
    )
    with pytest.raises(ValueError):
        evaluate_contract_opportunity(
            contract=_make_contract(80),
            probability_surface=surface,
            orderbook=ob,
            signal_threshold=_THRESHOLD,
            timestamp_utc=_TS,
        )


def test_evaluate_raises_on_invalid_signal_threshold():
    """ValueError when signal_threshold <= 0."""
    surface = {80: 0.65}
    ob = _make_orderbook(bid=55.0, ask=57.0)
    with pytest.raises(ValueError, match="threshold"):
        evaluate_contract_opportunity(
            contract=_make_contract(80),
            probability_surface=surface,
            orderbook=ob,
            signal_threshold=0.0,
            timestamp_utc=_TS,
        )


# ---------------------------------------------------------------------------
# rank_trade_signals
# ---------------------------------------------------------------------------


def test_rank_descending_by_absolute_alpha():
    """Signals are returned in descending order of absolute_alpha."""
    signals = [
        _make_signal(0.05, "A"),
        _make_signal(0.25, "B"),
        _make_signal(0.15, "C"),
    ]
    ranked = rank_trade_signals(signals)
    assert [s.contract_id for s in ranked] == ["B", "C", "A"]


def test_rank_returns_new_list():
    """rank_trade_signals does not mutate the input list."""
    signals = [_make_signal(0.05, "A"), _make_signal(0.25, "B")]
    original_order = [s.contract_id for s in signals]
    rank_trade_signals(signals)
    assert [s.contract_id for s in signals] == original_order


def test_rank_empty_list():
    """Empty input returns empty list without error."""
    assert rank_trade_signals([]) == []


def test_rank_single_element():
    """Single-element list is returned unchanged."""
    signals = [_make_signal(0.20, "A")]
    ranked = rank_trade_signals(signals)
    assert len(ranked) == 1
    assert ranked[0].contract_id == "A"


def test_rank_tie_broken_by_contract_id():
    """Equal rank_score and absolute_alpha → tie-broken by contract_id ascending."""
    # All three have identical absolute_alpha=0.15, no rank_score
    signals = [
        _make_signal(0.15, "C"),
        _make_signal(0.15, "A"),
        _make_signal(0.15, "B"),
    ]
    ranked = rank_trade_signals(signals)
    assert [s.contract_id for s in ranked] == ["A", "B", "C"]


def test_rank_hold_and_buy_mixed():
    """HOLD signals (small alpha) rank below BUY/SELL signals."""
    signals = [
        _make_signal(0.02, "HOLD_1"),
        _make_signal(0.22, "BUY_1"),
        _make_signal(0.03, "HOLD_2"),
        _make_signal(0.18, "BUY_2"),
    ]
    ranked = rank_trade_signals(signals)
    assert ranked[0].contract_id == "BUY_1"
    assert ranked[1].contract_id == "BUY_2"
    # HOLDs at the end
    assert ranked[2].contract_id in ("HOLD_1", "HOLD_2")
    assert ranked[3].contract_id in ("HOLD_1", "HOLD_2")


# ---------------------------------------------------------------------------
# rank_trade_signals — rank_score-based ranking
# ---------------------------------------------------------------------------


def test_rank_by_rank_score_overrides_absolute_alpha():
    """rank_score takes priority over absolute_alpha when present."""
    # C has the smallest absolute_alpha but the largest rank_score → should be first
    signals = [
        _make_signal(0.30, "A", rank_score=0.30),
        _make_signal(0.20, "B", rank_score=0.25),
        _make_signal(0.10, "C", rank_score=0.40),
    ]
    ranked = rank_trade_signals(signals)
    assert [s.contract_id for s in ranked] == ["C", "A", "B"]


def test_rank_fallback_to_absolute_alpha_when_rank_score_none():
    """When all rank_scores are None, falls back to absolute_alpha ordering."""
    signals = [
        _make_signal(0.10, "A"),  # rank_score=None
        _make_signal(0.30, "B"),
        _make_signal(0.20, "C"),
    ]
    ranked = rank_trade_signals(signals)
    assert [s.contract_id for s in ranked] == ["B", "C", "A"]


def test_rank_rank_score_tie_broken_by_absolute_alpha():
    """Equal rank_scores → fall back to absolute_alpha descending."""
    signals = [
        _make_signal(0.10, "A", rank_score=0.50),
        _make_signal(0.30, "B", rank_score=0.50),
        _make_signal(0.20, "C", rank_score=0.50),
    ]
    ranked = rank_trade_signals(signals)
    # rank_scores equal → sort by absolute_alpha desc: B(0.30) > C(0.20) > A(0.10)
    assert [s.contract_id for s in ranked] == ["B", "C", "A"]


def test_rank_all_equal_broken_by_contract_id():
    """Equal rank_score and equal absolute_alpha → alphabetical contract_id."""
    signals = [
        _make_signal(0.20, "Z", rank_score=0.50),
        _make_signal(0.20, "A", rank_score=0.50),
        _make_signal(0.20, "M", rank_score=0.50),
    ]
    ranked = rank_trade_signals(signals)
    assert [s.contract_id for s in ranked] == ["A", "M", "Z"]


def test_rank_deterministic_regardless_of_input_order():
    """Ranked output is the same regardless of how inputs are ordered."""
    base = [
        _make_signal(0.30, "B", rank_score=0.30),
        _make_signal(0.10, "A", rank_score=0.50),
        _make_signal(0.20, "C", rank_score=0.20),
    ]
    import random
    shuffled = list(base)
    random.shuffle(shuffled)
    assert [s.contract_id for s in rank_trade_signals(base)] == \
           [s.contract_id for s in rank_trade_signals(shuffled)]


# ---------------------------------------------------------------------------
# evaluate_contract_opportunity — rank_score integration
# ---------------------------------------------------------------------------


def test_evaluate_produces_rank_score():
    """evaluate_contract_opportunity returns a signal with a non-None rank_score."""
    surface = {80: 0.70}
    ob = _make_orderbook(bid=40.0, ask=42.0)
    signal = evaluate_contract_opportunity(
        contract=_make_contract(80),
        probability_surface=surface,
        orderbook=ob,
        signal_threshold=_THRESHOLD,
        timestamp_utc=_TS,
    )
    assert signal.rank_score is not None
    assert signal.rank_score >= 0.0


def test_evaluate_rank_score_equals_abs_alpha_with_no_flags():
    """Without feature flags, rank_score == abs(alpha) (no bonuses applied)."""
    surface = {80: 0.70}
    ob = _make_orderbook(bid=40.0, ask=42.0)
    signal = evaluate_contract_opportunity(
        contract=_make_contract(80),
        probability_surface=surface,
        orderbook=ob,
        signal_threshold=_THRESHOLD,
        timestamp_utc=_TS,
    )
    # No feature flags → rank_score is just abs(alpha)
    assert signal.rank_score == pytest.approx(signal.absolute_alpha)


# ---------------------------------------------------------------------------
# evaluate_contract_opportunity — enhancement flag pass-through
# ---------------------------------------------------------------------------
# Shared setup for all enhancement tests:
#   model_probability = 0.70, market mid = (40+42)/2 = 41¢ → 0.41
#   alpha = 0.29, abs_alpha = 0.29


def test_evaluate_passes_forecast_shift_flag():
    """forecast_shift_flag=True is stored on the signal and adds shift_bonus to rank_score."""
    surface = {80: 0.70}
    ob = _make_orderbook(bid=40.0, ask=42.0)
    signal = evaluate_contract_opportunity(
        contract=_make_contract(80),
        probability_surface=surface,
        orderbook=ob,
        signal_threshold=_THRESHOLD,
        timestamp_utc=_TS,
        forecast_shift_flag=True,
    )
    assert signal.forecast_shift_flag is True
    # abs(alpha)=0.29 + shift_bonus=0.05 → 0.34
    assert signal.rank_score == pytest.approx(0.34, abs=1e-6)


def test_evaluate_passes_stale_market_flag():
    """stale_market_flag=True is stored on the signal and adds lag_bonus to rank_score."""
    surface = {80: 0.70}
    ob = _make_orderbook(bid=40.0, ask=42.0)
    signal = evaluate_contract_opportunity(
        contract=_make_contract(80),
        probability_surface=surface,
        orderbook=ob,
        signal_threshold=_THRESHOLD,
        timestamp_utc=_TS,
        stale_market_flag=True,
    )
    assert signal.stale_market_flag is True
    # abs(alpha)=0.29 + lag_bonus=0.05 → 0.34
    assert signal.rank_score == pytest.approx(0.34, abs=1e-6)


def test_evaluate_passes_microstructure_score():
    """microstructure_score is stored on the signal and contributes to rank_score."""
    surface = {80: 0.70}
    ob = _make_orderbook(bid=40.0, ask=42.0)
    signal = evaluate_contract_opportunity(
        contract=_make_contract(80),
        probability_surface=surface,
        orderbook=ob,
        signal_threshold=_THRESHOLD,
        timestamp_utc=_TS,
        microstructure_score=0.80,
    )
    assert signal.microstructure_score == pytest.approx(0.80)
    # abs(alpha)=0.29 + 0.80*0.10=0.08 → 0.37
    assert signal.rank_score == pytest.approx(0.37, abs=1e-6)


def test_evaluate_tail_flag_boosts_rank_score():
    """tail_opportunity_flag=True with tail_multiplier=1.5 boosts rank_score.

    tail_rank_score = 0.29 * 1.5 = 0.435
    rank_score = max(0.29, 0.435) = 0.435
    """
    surface = {80: 0.70}
    ob = _make_orderbook(bid=40.0, ask=42.0)
    signal = evaluate_contract_opportunity(
        contract=_make_contract(80),
        probability_surface=surface,
        orderbook=ob,
        signal_threshold=_THRESHOLD,
        timestamp_utc=_TS,
        tail_opportunity_flag=True,
        tail_multiplier=1.5,
    )
    assert signal.tail_opportunity_flag is True
    assert signal.rank_score == pytest.approx(0.435, abs=1e-6)


def test_evaluate_tail_does_not_alter_alpha():
    """Tail boost must not change alpha, absolute_alpha, or signal_side."""
    surface = {80: 0.70}
    ob = _make_orderbook(bid=40.0, ask=42.0)
    signal = evaluate_contract_opportunity(
        contract=_make_contract(80),
        probability_surface=surface,
        orderbook=ob,
        signal_threshold=_THRESHOLD,
        timestamp_utc=_TS,
        tail_opportunity_flag=True,
        tail_multiplier=3.0,
    )
    assert signal.alpha == pytest.approx(0.29, abs=1e-6)
    assert signal.absolute_alpha == pytest.approx(0.29, abs=1e-6)
    assert signal.signal_side == "BUY"


def test_evaluate_all_enhancement_flags_combined():
    """All enhancement flags active: rank_score accumulates all bonuses.

    alpha=0.29, tail_multiplier=1.5:
      tail_rank_score = 0.29 * 1.5 = 0.435
      base            = max(0.29, 0.435) = 0.435
      + shift_bonus   = 0.05
      + lag_bonus     = 0.05
      + micro_score*w = 0.80 * 0.10 = 0.08
      rank_score      = 0.615
    """
    surface = {80: 0.70}
    ob = _make_orderbook(bid=40.0, ask=42.0)
    signal = evaluate_contract_opportunity(
        contract=_make_contract(80),
        probability_surface=surface,
        orderbook=ob,
        signal_threshold=_THRESHOLD,
        timestamp_utc=_TS,
        forecast_shift_flag=True,
        stale_market_flag=True,
        microstructure_score=0.80,
        tail_opportunity_flag=True,
        tail_multiplier=1.5,
    )
    assert signal.forecast_shift_flag is True
    assert signal.stale_market_flag is True
    assert signal.microstructure_score == pytest.approx(0.80)
    assert signal.tail_opportunity_flag is True
    assert signal.rank_score == pytest.approx(0.615, abs=1e-6)
    # Raw alpha unchanged
    assert signal.alpha == pytest.approx(0.29, abs=1e-6)


# ── probability-surface keying (2026-07-04 collision bug) ────────────────────


def test_same_threshold_contracts_get_distinct_probabilities():
    """A 'T98' tail and a 98-99 bracket share threshold_f=98.  With
    contract_id keys each signal must use its own probability — threshold
    keys silently collided and flipped the NY 2026-07-04 trade's side."""
    from datetime import datetime, timezone

    from deep_isobar.core.types import MarketContract, OrderBookSnapshot
    from deep_isobar.market.market_scanner import evaluate_contract_opportunity

    now = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)

    def contract(cid, **kw):
        return MarketContract(
            contract_id=cid, market_source="Kalshi", city="New York",
            target_date=now.date(), metric="high_temp_f", threshold_f=98,
            comparison_operator="ge", settlement_source="NWS", **kw,
        )

    tail = contract("KXHIGHNY-26JUL04-T98", strike_type="less", cap_strike=98)
    bracket = contract("KXHIGHNY-26JUL04-B98.5", strike_type="between",
                       floor_strike=98, cap_strike=99)
    surface = {tail.contract_id: 0.915, bracket.contract_id: 0.054}
    book = OrderBookSnapshot(
        timestamp_utc=now, contract_id="x", market_source="Kalshi",
        best_bid=0.63, best_ask=0.65,
    )

    s_tail = evaluate_contract_opportunity(tail, surface, book, 0.10, now)
    s_bracket = evaluate_contract_opportunity(bracket, surface, book, 0.10, now)

    assert s_tail.model_probability == 0.915
    assert s_bracket.model_probability == 0.054
    # Model 0.915 vs market 0.64 → BUY side, never SELL.
    assert s_tail.signal_side == "BUY"
