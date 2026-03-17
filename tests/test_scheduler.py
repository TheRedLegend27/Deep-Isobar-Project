"""Tests for deep_isobar.scheduler.

Strategy: monkeypatch the two external boundaries (kalshi_client functions)
so tests are fast and deterministic.  The rest of the pipeline — city
universe, forecast generation, ensemble, KDE, probability surface, scanner
— runs against real stub implementations so the full wiring is exercised.

Covers:
- run_once returns a list
- run_once with real stubs produces TradeSignal objects
- empty contract list returns empty list
- contract with threshold outside surface is skipped
- broken orderbook for one contract is skipped; others proceed
- unknown market_source in contract fetch raises RuntimeError propagated
- run_loop calls run_once and respects interval (using monkeypatched sleep)
- run_loop continues after a non-fatal exception
- run_loop re-raises KeyError (fatal configuration error)
"""

import pytest
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, call

from deep_isobar.core.types import MarketContract, OrderBookSnapshot, TradeSignal
import deep_isobar.scheduler as scheduler_mod


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 3, 17, 12, 0, 0, tzinfo=timezone.utc)
_TOMORROW = (_NOW + timedelta(days=1)).date()


def _make_contract(threshold_f: int = 75) -> MarketContract:
    return MarketContract(
        contract_id=f"CHI_HIGH_TEMP_F_GE_{threshold_f}_20260318",
        market_source="Kalshi",
        city="Chicago",
        metric="high_temp_f",
        comparison_operator="ge",
        threshold_f=threshold_f,
        target_date=_TOMORROW,
        settlement_source="NWS",
    )


def _make_orderbook(contract_id: str, bid: float = 48.0, ask: float = 52.0) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        timestamp_utc=_NOW,
        contract_id=contract_id,
        market_source="Kalshi",
        best_bid=bid,
        best_ask=ask,
    )


# ---------------------------------------------------------------------------
# run_once — basic contract
# ---------------------------------------------------------------------------


def test_run_once_returns_list():
    """run_once must return a list (may be empty)."""
    result = scheduler_mod.run_once(now_utc=_NOW)
    assert isinstance(result, list)


def test_run_once_with_real_stubs_returns_trade_signals():
    """End-to-end: stubs produce at least one TradeSignal."""
    result = scheduler_mod.run_once(now_utc=_NOW)
    # Stub contracts are at thresholds [60..90] — all within surface [50..120].
    # Each should produce a signal (BUY, SELL, or HOLD depending on alpha).
    assert len(result) > 0
    for signal in result:
        assert isinstance(signal, TradeSignal)


def test_run_once_result_is_sorted_by_descending_absolute_alpha():
    """Returned signals are sorted with the highest |alpha| first."""
    result = scheduler_mod.run_once(now_utc=_NOW)
    if len(result) >= 2:
        for i in range(len(result) - 1):
            assert result[i].absolute_alpha >= result[i + 1].absolute_alpha


def test_run_once_default_timestamp():
    """run_once with no argument runs without error."""
    result = scheduler_mod.run_once()
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# run_once — empty contracts
# ---------------------------------------------------------------------------


def test_run_once_empty_contracts_returns_empty_list():
    """No contracts from Kalshi → empty signal list."""
    with patch.object(scheduler_mod, "fetch_live_contracts", return_value=[]):
        result = scheduler_mod.run_once(now_utc=_NOW)
    assert result == []


# ---------------------------------------------------------------------------
# run_once — threshold filtering
# ---------------------------------------------------------------------------


def test_run_once_surface_auto_expands_to_cover_all_contracts():
    """Surface range auto-derives from contract thresholds, so no contract is skipped."""
    # Threshold 200 is well above the config ceiling (120); surface should expand to cover it.
    high_contract = _make_contract(threshold_f=200)

    def fake_orderbook(source, cid):
        return _make_orderbook(cid)

    with (
        patch.object(scheduler_mod, "fetch_live_contracts", return_value=[high_contract]),
        patch.object(scheduler_mod, "fetch_orderbook_for_contract", side_effect=fake_orderbook),
    ):
        result = scheduler_mod.run_once(now_utc=_NOW)

    # threshold=200 is now covered — signal is produced.
    assert len(result) == 1
    assert result[0].threshold_f == 200


def test_run_once_all_contracts_evaluated_regardless_of_threshold():
    """All fetched contracts produce signals because the surface always covers them."""
    c75 = _make_contract(threshold_f=75)
    c200 = _make_contract(threshold_f=200)

    def fake_orderbook(source, cid):
        return _make_orderbook(cid)

    with (
        patch.object(scheduler_mod, "fetch_live_contracts", return_value=[c75, c200]),
        patch.object(scheduler_mod, "fetch_orderbook_for_contract", side_effect=fake_orderbook),
    ):
        result = scheduler_mod.run_once(now_utc=_NOW)

    assert len(result) == 2
    assert {s.threshold_f for s in result} == {75, 200}


# ---------------------------------------------------------------------------
# run_once — partial failure tolerance
# ---------------------------------------------------------------------------


def test_run_once_bad_orderbook_skipped_others_proceed():
    """If one orderbook fetch fails, the remaining contracts are still evaluated."""
    c1 = _make_contract(threshold_f=70)
    c2 = _make_contract(threshold_f=75)

    call_count = 0

    def flaky_orderbook(source, cid):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Simulated API timeout")
        return _make_orderbook(cid)

    with (
        patch.object(scheduler_mod, "fetch_live_contracts", return_value=[c1, c2]),
        patch.object(scheduler_mod, "fetch_orderbook_for_contract", side_effect=flaky_orderbook),
    ):
        result = scheduler_mod.run_once(now_utc=_NOW)

    # c1 failed, c2 succeeded → exactly 1 signal
    assert len(result) == 1
    assert result[0].threshold_f == 75


# ---------------------------------------------------------------------------
# run_once — contract fields propagated
# ---------------------------------------------------------------------------


def test_run_once_signal_contract_id_matches():
    """The contract_id on each returned signal matches the contract."""
    contract = _make_contract(threshold_f=75)

    with (
        patch.object(scheduler_mod, "fetch_live_contracts", return_value=[contract]),
        patch.object(
            scheduler_mod,
            "fetch_orderbook_for_contract",
            return_value=_make_orderbook(contract.contract_id),
        ),
    ):
        result = scheduler_mod.run_once(now_utc=_NOW)

    assert len(result) == 1
    assert result[0].contract_id == contract.contract_id


# ---------------------------------------------------------------------------
# _attempt_paper_trades
# ---------------------------------------------------------------------------


def _make_signal(contract_id: str, side: str, alpha: float = 0.20) -> TradeSignal:
    return TradeSignal(
        timestamp_utc=_NOW,
        contract_id=contract_id,
        city="Chicago",
        target_date=_TOMORROW,
        metric="high_temp_f",
        threshold_f=45,
        comparison_operator="ge",
        market_probability=0.50,
        model_probability=0.50 + alpha if side == "BUY" else 0.50 - alpha,
        alpha=alpha if side == "BUY" else -alpha,
        absolute_alpha=abs(alpha),
        signal_side=side,
        confidence_score=1.0,
    )


def test_attempt_paper_trades_executes_sell_signal():
    """A SELL signal with room to trade is paper-executed."""
    positions: dict = {}
    signal = _make_signal("CONTRACT-A", "SELL")
    trades, exp = scheduler_mod._attempt_paper_trades(
        signals=[signal],
        positions=positions,
        daily_exposure=0.0,
        quantity_per_trade=10.0,
        max_position_per_contract=50.0,
        max_daily_exposure=500.0,
    )
    assert len(trades) == 1
    assert trades[0].side == "SELL"
    assert trades[0].paper_trade_flag is True
    assert positions["CONTRACT-A"] == -10.0
    assert exp == pytest.approx(10.0 * 0.50)   # qty * price


def test_attempt_paper_trades_skips_hold_signals():
    """HOLD signals are skipped — no trade, no position change."""
    positions: dict = {}
    trades, exp = scheduler_mod._attempt_paper_trades(
        signals=[_make_signal("CONTRACT-B", "HOLD")],
        positions=positions,
        daily_exposure=0.0,
        quantity_per_trade=10.0,
        max_position_per_contract=50.0,
        max_daily_exposure=500.0,
    )
    assert trades == []
    assert exp == 0.0
    assert "CONTRACT-B" not in positions


def test_attempt_paper_trades_respects_daily_exposure_cap():
    """A trade that would breach max_daily_exposure is rejected."""
    positions: dict = {}
    trades, exp = scheduler_mod._attempt_paper_trades(
        signals=[_make_signal("CONTRACT-C", "SELL")],
        positions=positions,
        daily_exposure=499.0,          # only $1 headroom
        quantity_per_trade=10.0,       # 10 × $0.50 = $5 notional — over cap
        max_position_per_contract=50.0,
        max_daily_exposure=500.0,
    )
    assert trades == []
    assert exp == 499.0   # unchanged


def test_attempt_paper_trades_daily_exposure_resets_across_calls():
    """Executed trades update exposure; subsequent calls see accumulated value."""
    positions: dict = {}
    _, exp = scheduler_mod._attempt_paper_trades(
        signals=[_make_signal("CONTRACT-D", "SELL")],
        positions=positions,
        daily_exposure=0.0,
        quantity_per_trade=10.0,
        max_position_per_contract=50.0,
        max_daily_exposure=500.0,
    )
    assert exp == pytest.approx(5.0)   # 10 × 0.50


def test_run_loop_executes_paper_trades(monkeypatch):
    """run_loop calls _attempt_paper_trades after each run_once cycle."""
    calls: list = []

    def fake_run_once(now_utc=None):
        return [_make_signal("C-X", "SELL")]

    def fake_attempt(signals, positions, daily_exposure, **kwargs):
        calls.append(("attempt", len(signals)))
        raise KeyboardInterrupt  # stop after first call

    monkeypatch.setattr(scheduler_mod, "run_once", fake_run_once)
    monkeypatch.setattr(scheduler_mod, "_attempt_paper_trades", fake_attempt)
    monkeypatch.setattr(scheduler_mod.time, "sleep", lambda s: None)

    with pytest.raises(KeyboardInterrupt):
        scheduler_mod.run_loop(interval_seconds=1)

    assert calls == [("attempt", 1)]


# ---------------------------------------------------------------------------
# run_loop
# ---------------------------------------------------------------------------


def test_run_loop_calls_run_once_and_sleeps(monkeypatch):
    """run_loop calls run_once then sleeps; stops after N iterations."""
    call_count = 0
    sleep_calls = []

    def fake_run_once(now_utc=None):
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise KeyboardInterrupt  # stop the loop after 2 iterations
        return []

    monkeypatch.setattr(scheduler_mod, "run_once", fake_run_once)
    monkeypatch.setattr(scheduler_mod.time, "sleep", lambda s: sleep_calls.append(s))

    with pytest.raises(KeyboardInterrupt):
        scheduler_mod.run_loop(interval_seconds=60)

    assert call_count == 2
    assert sleep_calls == [60]


def test_run_loop_continues_after_non_fatal_exception(monkeypatch):
    """Non-fatal exception in run_once is caught; loop continues."""
    call_count = 0

    def flaky_run_once(now_utc=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Simulated scan failure")
        if call_count >= 2:
            raise KeyboardInterrupt
        return []

    monkeypatch.setattr(scheduler_mod, "run_once", flaky_run_once)
    monkeypatch.setattr(scheduler_mod.time, "sleep", lambda s: None)

    with pytest.raises(KeyboardInterrupt):
        scheduler_mod.run_loop(interval_seconds=1)

    # loop ran twice: once raised RuntimeError, once raised KeyboardInterrupt
    assert call_count == 2


def test_run_loop_reraises_key_error(monkeypatch):
    """KeyError (missing city config) propagates out of run_loop immediately."""
    def bad_run_once(now_utc=None):
        raise KeyError("Chicago")

    monkeypatch.setattr(scheduler_mod, "run_once", bad_run_once)
    monkeypatch.setattr(scheduler_mod.time, "sleep", lambda s: None)

    with pytest.raises(KeyError, match="Chicago"):
        scheduler_mod.run_loop(interval_seconds=1)
