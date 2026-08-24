"""Tests for live order execution (trading/trade_execution.submit_live_trade).

The safety properties under test, in gate order:

1. Bad inputs raise ``ValueError`` before anything else runs.
2. ``runtime.paper_trade`` != False refuses with LiveTradingDisabledError
   — the default config state can never place a live order.
3. Position limit enforced against risk.max_position_per_contract.
4. Kill switch is the FINAL gate: engaged switch refuses even when every
   other gate passes, and no exchange call is ever made.
5. BUY maps to yes-side at price; SELL maps to no-side at 1−price.
6. The idempotency key is passed through / auto-generated.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from deep_isobar.ops import kill_switch
from deep_isobar.trading import trade_execution
from deep_isobar.trading.trade_execution import (
    LiveTradingDisabledError,
    submit_live_trade,
)

_ORDER_OK = {"order_id": "abc-123", "status": "resting"}


def _settings(paper_trade, max_pos=100):
    """get_setting stub covering the two keys submit_live_trade reads."""
    values = {
        "runtime.paper_trade": paper_trade,
        "risk.max_position_per_contract": max_pos,
    }
    return lambda key, default=None: values.get(key, default)


def _call(**overrides):
    kwargs = dict(
        market_source="Kalshi",
        contract_id="KXHIGHLAX-26AUG18-B77.5",
        side="BUY",
        quantity=10,
        price=0.32,
    )
    kwargs.update(overrides)
    return submit_live_trade(**kwargs)


@pytest.fixture(autouse=True)
def _switch_disengaged(monkeypatch):
    monkeypatch.setattr(kill_switch, "is_engaged", lambda: False)


class TestInputValidation:
    @pytest.mark.parametrize("bad", [
        dict(market_source="Polymarket"),
        dict(side="HOLD"),
        dict(quantity=0),
        dict(quantity=-5),
        dict(quantity=2.5),
        dict(price=0.0),
        dict(price=1.0),
        dict(price=1.4),
    ])
    def test_bad_inputs_raise_value_error(self, bad):
        with pytest.raises(ValueError):
            _call(**bad)


class TestPaperTradeGate:
    def test_paper_trade_true_refuses(self):
        with patch.object(trade_execution, "get_setting", _settings(True)):
            with pytest.raises(LiveTradingDisabledError):
                _call()

    def test_paper_trade_absent_refuses(self):
        # Missing key falls back to the default True — absent config is paper.
        with patch.object(trade_execution, "get_setting", lambda k, d=None: d):
            with pytest.raises(LiveTradingDisabledError):
                _call()

    def test_paper_trade_falsy_but_not_false_refuses(self):
        # 0, "", None are NOT an explicit go-live decision.
        for sneaky in (0, "", None, "false"):
            with patch.object(trade_execution, "get_setting", _settings(sneaky)):
                with pytest.raises(LiveTradingDisabledError):
                    _call()

    def test_no_exchange_call_when_refused(self):
        with patch.object(trade_execution, "get_setting", _settings(True)), \
             patch.object(trade_execution.kalshi_client, "create_order") as create:
            with pytest.raises(LiveTradingDisabledError):
                _call()
        create.assert_not_called()


class TestPositionLimit:
    def test_over_limit_refuses(self):
        with patch.object(trade_execution, "get_setting", _settings(False, max_pos=50)):
            with pytest.raises(ValueError, match="max_position_per_contract"):
                _call(quantity=51)

    def test_at_limit_allowed(self):
        with patch.object(trade_execution, "get_setting", _settings(False, max_pos=50)), \
             patch.object(trade_execution.kalshi_client, "create_order",
                          return_value=_ORDER_OK):
            trade = _call(quantity=50)
        assert trade.quantity == 50.0


class TestKillSwitchFinalGate:
    def test_engaged_switch_refuses_and_never_calls_exchange(self, monkeypatch):
        monkeypatch.setattr(kill_switch, "is_engaged", lambda: True)
        monkeypatch.setattr(
            kill_switch, "get_state",
            lambda: kill_switch.KillSwitchState(
                engaged=True, reason="test", source="test", detail=None,
            ),
        )
        with patch.object(trade_execution, "get_setting", _settings(False)), \
             patch.object(trade_execution.kalshi_client, "create_order") as create:
            with pytest.raises(kill_switch.KillSwitchEngagedError):
                _call()
        create.assert_not_called()


class TestSideMapping:
    def test_buy_maps_to_yes_at_price(self):
        with patch.object(trade_execution, "get_setting", _settings(False)), \
             patch.object(trade_execution.kalshi_client, "create_order",
                          return_value=_ORDER_OK) as create:
            _call(side="BUY", price=0.32)
        kwargs = create.call_args.kwargs
        assert kwargs["side"] == "yes"
        assert kwargs["action"] == "buy"
        assert kwargs["price_cents"] == 32

    def test_sell_maps_to_no_at_complement(self):
        with patch.object(trade_execution, "get_setting", _settings(False)), \
             patch.object(trade_execution.kalshi_client, "create_order",
                          return_value=_ORDER_OK) as create:
            _call(side="SELL", price=0.57)
        kwargs = create.call_args.kwargs
        assert kwargs["side"] == "no"
        assert kwargs["action"] == "buy"
        assert kwargs["price_cents"] == 43


class TestExecutedTradeRecord:
    def test_resting_order_record(self):
        with patch.object(trade_execution, "get_setting", _settings(False)), \
             patch.object(trade_execution.kalshi_client, "create_order",
                          return_value=_ORDER_OK):
            trade = _call()
        assert trade.paper_trade_flag is False
        assert trade.execution_status == "open"
        assert trade.exchange_order_id == "abc-123"
        assert trade.fill_quantity is None
        assert trade.trade_id.startswith("LIVE-")

    def test_executed_order_record(self):
        executed = {"order_id": "abc-456", "status": "executed"}
        with patch.object(trade_execution, "get_setting", _settings(False)), \
             patch.object(trade_execution.kalshi_client, "create_order",
                          return_value=executed):
            trade = _call(quantity=7)
        assert trade.execution_status == "filled"
        assert trade.fill_quantity == 7.0
        assert trade.avg_fill_price == trade.price

    def test_client_order_id_passthrough(self):
        with patch.object(trade_execution, "get_setting", _settings(False)), \
             patch.object(trade_execution.kalshi_client, "create_order",
                          return_value=_ORDER_OK) as create:
            trade = _call(client_order_id="retry-key-1")
        assert create.call_args.kwargs["client_order_id"] == "retry-key-1"
        assert trade.trade_id == "LIVE-retry-ke"

    def test_generated_client_order_ids_differ(self):
        with patch.object(trade_execution, "get_setting", _settings(False)), \
             patch.object(trade_execution.kalshi_client, "create_order",
                          return_value=_ORDER_OK) as create:
            _call()
            _call()
        first, second = (c.kwargs["client_order_id"] for c in create.call_args_list)
        assert first != second


class TestCreateOrderValidation:
    """The low-level client validates before any network traffic."""

    @pytest.mark.parametrize("bad", [
        dict(action="hold"),
        dict(side="maybe"),
        dict(count=0),
        dict(count=2.5),
        dict(price_cents=0),
        dict(price_cents=100),
        dict(price_cents=0.5),
        dict(client_order_id=""),
        dict(order_type="stop"),
    ])
    def test_bad_args_raise_value_error(self, bad):
        from deep_isobar.market import kalshi_client
        kwargs = dict(
            ticker="KXHIGHLAX-26AUG18-B77.5",
            action="buy", side="yes", count=10,
            price_cents=32, client_order_id="k1",
        )
        kwargs.update(bad)
        with pytest.raises(ValueError):
            kalshi_client.create_order(**kwargs)

    def test_stub_mode_refuses(self):
        from deep_isobar.market import kalshi_client
        with patch.object(kalshi_client, "_use_stub_mode", return_value=True):
            with pytest.raises(RuntimeError, match="stub"):
                kalshi_client.create_order(
                    ticker="T", action="buy", side="yes", count=1,
                    price_cents=50, client_order_id="k1",
                )
