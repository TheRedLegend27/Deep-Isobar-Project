"""Tests for deep_isobar.market.market_scanner.

Covers:
- evaluate_contract_opportunity: successful BUY / SELL / HOLD evaluation
- evaluate_contract_opportunity: missing threshold in surface raises KeyError
- evaluate_contract_opportunity: bad orderbook raises ValueError
- evaluate_contract_opportunity: timestamp defaults to now when not supplied
- evaluate_contract_opportunity: confidence_score == abs(alpha)
- rank_trade_signals: descending order by absolute_alpha
- rank_trade_signals: stable sort on equal absolute_alpha
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


def _make_signal(absolute_alpha: float, contract_id: str = "X") -> TradeSignal:
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


def test_rank_stable_on_equal_alpha():
    """Signals with equal absolute_alpha preserve relative input order."""
    # All three have identical absolute_alpha=0.15
    signals = [
        _make_signal(0.15, "A"),
        _make_signal(0.15, "B"),
        _make_signal(0.15, "C"),
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
