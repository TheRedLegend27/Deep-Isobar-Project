"""Tests for deep_isobar.market.kalshi_client.

Strategy:
- Parsing functions (_parse_ticker, _parse_contract, orderbook helpers) are
  tested directly with fixture dicts — no network or credentials required.
- Stub-mode public interface tests run without mocking (no credentials in CI).
- Live-mode tests patch _load_credentials and _kalshi_get to avoid any real
  network call or RSA key material.

Kalshi v2 response formats assumed:
- Orderbook: {"orderbook_fp": {"yes_dollars": [["0.15","100.00"]], "no_dollars": [...]}}
  Arrays are sorted ASCENDING — best bid/ask is the LAST element.
  Prices are dollar strings in [0, 1].
- Markets: ticker like "KXHIGHCHI-26MAR18-B51.5", timestamps in "created_time",
  "latest_expiration_time".

Covers:
- _parse_ticker: valid (decimal & integer brackets), malformed, unknown month,
  invalid calendar date, KX-prefix series, case-insensitive
- _parse_contract: valid market dict, unknown series, missing ticker, closed
  market, optional timestamp fields using Kalshi v2 field names
- _best_bid_from_yes: single level, multiple levels (max of all), empty, malformed
- _best_ask_from_no: complementary arithmetic (1.0 - no_bid), multiple levels,
  empty, malformed, symmetric 50/50
- _parse_orderbook_response: full, no-yes side, no-no side, crossed market,
  sizes from last element, missing orderbook_fp key
- fetch_live_contracts stub: list, MarketContract types, Chicago fields,
  wrong source raises, case-insensitive source
- fetch_orderbook_for_contract stub: snapshot type, bid/ask values, wrong source
- fetch_live_contracts live (mocked): parses response, skips bad tickers,
  empty API → stub fallback, RuntimeError → stub fallback
- fetch_orderbook_for_contract live (mocked): parses response, error → stub
- stub_mode config: forces stub even with credentials
"""

import pytest
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

from deep_isobar.core.types import MarketContract, OrderBookSnapshot
import deep_isobar.market.kalshi_client as kc

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 3, 17, 12, 0, 0, tzinfo=timezone.utc)
_FAKE_CREDENTIALS = ("key-id-123", MagicMock())


def _make_market_dict(
    ticker: str = "KXHIGHCHI-26MAR18-B51.5",
    status: str = "active",
    title: str = "Chicago High Temp",
    created_time: str | None = "2026-03-15T00:00:00Z",
    latest_expiration_time: str | None = "2026-03-18T23:00:00Z",
) -> dict:
    """Build a minimal Kalshi v2 market API dict."""
    return {
        "ticker": ticker,
        "status": status,
        "rules_primary": title,
        "created_time": created_time,
        "latest_expiration_time": latest_expiration_time,
        "yes_bid_dollars": "0.1500",
        "yes_ask_dollars": "0.1800",
        "last_price_dollars": "0.1600",
    }


def _make_orderbook_response(
    yes_levels: list | None = None,
    no_levels: list | None = None,
) -> dict:
    """Build a Kalshi v2 orderbook_fp API response.

    Prices are dollar strings, arrays sorted ascending (best bid is last).
    Default: yes best bid = 0.15, no best bid = 0.83 → yes ask = 0.17.
    """
    return {
        "orderbook_fp": {
            "yes_dollars": yes_levels if yes_levels is not None else [["0.15", "100.00"]],
            "no_dollars": no_levels if no_levels is not None else [["0.83", "80.00"]],
        }
    }


# ---------------------------------------------------------------------------
# _parse_ticker
# ---------------------------------------------------------------------------


def test_parse_ticker_kx_prefix_valid():
    result = kc._parse_ticker("KXHIGHCHI-26MAR18-B51.5")
    assert result is not None
    assert result["series"] == "KXHIGHCHI"
    assert result["target_date"] == date(2026, 3, 18)
    assert result["threshold_f"] == 52   # round(51.5) = 52


def test_parse_ticker_integer_bracket():
    result = kc._parse_ticker("HIGHCHI-26MAR18-B90")
    assert result is not None
    assert result["series"] == "HIGHCHI"
    assert result["threshold_f"] == 90


def test_parse_ticker_decimal_bracket_rounded():
    """B75.5 rounds to 76."""
    result = kc._parse_ticker("KXHIGHCHI-26MAR18-B75.5")
    assert result is not None
    assert result["threshold_f"] == 76


def test_parse_ticker_case_insensitive():
    result = kc._parse_ticker("kxhighchi-26mar18-b51.5")
    assert result is not None
    assert result["series"] == "KXHIGHCHI"


def test_parse_ticker_malformed_returns_none():
    assert kc._parse_ticker("NOT-A-TICKER") is None


def test_parse_ticker_missing_b_prefix_returns_none():
    assert kc._parse_ticker("KXHIGHCHI-26MAR18-51.5") is None


def test_parse_ticker_unknown_month_returns_none():
    assert kc._parse_ticker("KXHIGHCHI-26XYZ18-B51.5") is None


def test_parse_ticker_invalid_date_returns_none():
    assert kc._parse_ticker("KXHIGHCHI-26MAR99-B51.5") is None


def test_parse_ticker_all_months_parse():
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    for i, m in enumerate(months, start=1):
        result = kc._parse_ticker(f"KXHIGHCHI-26{m}15-B80")
        assert result is not None, f"month {m} failed"
        assert result["target_date"].month == i


# ---------------------------------------------------------------------------
# _parse_contract
# ---------------------------------------------------------------------------


def test_parse_contract_valid_returns_market_contract():
    contract = kc._parse_contract(_make_market_dict(), _NOW)
    assert isinstance(contract, MarketContract)


def test_parse_contract_kx_series_fields():
    contract = kc._parse_contract(_make_market_dict("KXHIGHCHI-26MAR18-B51.5"), _NOW)
    assert contract.city == "Chicago"
    assert contract.metric == "high_temp_f"
    assert contract.comparison_operator == "ge"
    assert contract.threshold_f == 52
    assert contract.target_date == date(2026, 3, 18)
    assert contract.settlement_source == "NWS"
    assert contract.market_source == "Kalshi"
    assert contract.contract_id == "KXHIGHCHI-26MAR18-B51.5"


def test_parse_contract_unknown_series_returns_none():
    assert kc._parse_contract(_make_market_dict("UNKNOWN-26MAR18-B90"), _NOW) is None


def test_parse_contract_missing_ticker_returns_none():
    market = _make_market_dict()
    market["ticker"] = ""
    assert kc._parse_contract(market, _NOW) is None


def test_parse_contract_closed_status_is_inactive():
    contract = kc._parse_contract(_make_market_dict(status="closed"), _NOW)
    assert contract is not None
    assert contract.active is False


def test_parse_contract_open_status_is_active():
    contract = kc._parse_contract(_make_market_dict(status="open"), _NOW)
    assert contract.active is True


def test_parse_contract_timestamps_from_v2_field_names():
    """Kalshi v2 uses created_time and latest_expiration_time."""
    market = _make_market_dict(
        created_time="2026-03-15T00:00:00Z",
        latest_expiration_time="2026-03-18T23:00:00Z",
    )
    contract = kc._parse_contract(market, _NOW)
    assert contract.listed_at_utc is not None
    assert contract.listed_at_utc.day == 15
    assert contract.expires_at_utc is not None
    assert contract.expires_at_utc.day == 18


def test_parse_contract_missing_timestamps_are_none():
    market = _make_market_dict(created_time=None, latest_expiration_time=None)
    contract = kc._parse_contract(market, _NOW)
    assert contract.listed_at_utc is None
    assert contract.expires_at_utc is None


def test_parse_contract_bare_series_also_recognised():
    """HIGHCHI (without KX prefix) is still in the series map."""
    contract = kc._parse_contract(_make_market_dict("HIGHCHI-26MAR18-B90"), _NOW)
    assert contract is not None
    assert contract.city == "Chicago"


# ---------------------------------------------------------------------------
# _best_bid_from_yes — dollar string prices, ascending sort
# ---------------------------------------------------------------------------


def test_best_bid_from_yes_single_level():
    assert kc._best_bid_from_yes([["0.15", "100.00"]]) == pytest.approx(0.15)


def test_best_bid_from_yes_multiple_levels_returns_max():
    # Ascending sorted; max() picks the highest regardless of order
    levels = [["0.13", "50.00"], ["0.14", "50.00"], ["0.15", "100.00"]]
    assert kc._best_bid_from_yes(levels) == pytest.approx(0.15)


def test_best_bid_from_yes_empty_returns_none():
    assert kc._best_bid_from_yes([]) is None


def test_best_bid_from_yes_malformed_returns_none():
    assert kc._best_bid_from_yes([["bad", "data"]]) is None


# ---------------------------------------------------------------------------
# _best_ask_from_no — complementary pricing: YES ask = 1.0 - NO bid
# ---------------------------------------------------------------------------


def test_best_ask_from_no_single_level():
    # NO bid = 0.83 → YES ask = 1.0 - 0.83 = 0.17
    assert kc._best_ask_from_no([["0.83", "80.00"]]) == pytest.approx(0.17)


def test_best_ask_from_no_multiple_levels_uses_best():
    levels = [["0.80", "50.00"], ["0.82", "30.00"], ["0.83", "80.00"]]
    assert kc._best_ask_from_no(levels) == pytest.approx(0.17)


def test_best_ask_from_no_symmetric_market():
    # NO bid = 0.50 → YES ask = 0.50
    assert kc._best_ask_from_no([["0.50", "100.00"]]) == pytest.approx(0.50)


def test_best_ask_from_no_empty_returns_none():
    assert kc._best_ask_from_no([]) is None


def test_best_ask_from_no_malformed_returns_none():
    assert kc._best_ask_from_no([["x"]]) is None


# ---------------------------------------------------------------------------
# _parse_orderbook_response
# ---------------------------------------------------------------------------


def test_parse_orderbook_response_full_snapshot():
    data = _make_orderbook_response(
        yes_levels=[["0.15", "100.00"]],
        no_levels=[["0.83", "80.00"]],
    )
    snap = kc._parse_orderbook_response("KXHIGHCHI-26MAR18-B51.5", data, _NOW)
    assert isinstance(snap, OrderBookSnapshot)
    assert snap.best_bid == pytest.approx(0.15)
    assert snap.best_ask == pytest.approx(0.17)   # 1.0 - 0.83
    assert snap.contract_id == "KXHIGHCHI-26MAR18-B51.5"
    assert snap.market_source == "Kalshi"


def test_parse_orderbook_response_empty_yes_bid_is_none():
    data = _make_orderbook_response(yes_levels=[], no_levels=[["0.83", "80.00"]])
    snap = kc._parse_orderbook_response("X", data, _NOW)
    assert snap.best_bid is None
    assert snap.best_ask == pytest.approx(0.17)


def test_parse_orderbook_response_empty_no_ask_is_none():
    data = _make_orderbook_response(yes_levels=[["0.15", "100.00"]], no_levels=[])
    snap = kc._parse_orderbook_response("X", data, _NOW)
    assert snap.best_bid == pytest.approx(0.15)
    assert snap.best_ask is None


def test_parse_orderbook_response_crossed_market_drops_ask():
    """bid=0.90, no_bid=0.15 → yes_ask=0.85 < bid → crossed → ask dropped."""
    data = _make_orderbook_response(
        yes_levels=[["0.90", "10.00"]],
        no_levels=[["0.15", "10.00"]],  # yes_ask = 1.0 - 0.15 = 0.85 < 0.90
    )
    snap = kc._parse_orderbook_response("X", data, _NOW)
    assert snap.best_bid == pytest.approx(0.90)
    assert snap.best_ask is None


def test_parse_orderbook_response_sizes_from_last_element():
    """Arrays sorted ascending — best bid/ask size is at index [-1]."""
    data = _make_orderbook_response(
        yes_levels=[["0.13", "50.00"], ["0.15", "200.00"]],  # best bid at [-1]
        no_levels=[["0.80", "30.00"], ["0.83", "100.00"]],   # best no bid at [-1]
    )
    snap = kc._parse_orderbook_response("X", data, _NOW)
    assert snap.bid_size == pytest.approx(200.0)
    assert snap.ask_size == pytest.approx(100.0)


def test_parse_orderbook_response_missing_orderbook_fp_key():
    snap = kc._parse_orderbook_response("X", {}, _NOW)
    assert snap.best_bid is None
    assert snap.best_ask is None


def test_parse_orderbook_response_empty_both_sides():
    data = {"orderbook_fp": {"yes_dollars": [], "no_dollars": []}}
    snap = kc._parse_orderbook_response("X", data, _NOW)
    assert snap.best_bid is None
    assert snap.best_ask is None


# ---------------------------------------------------------------------------
# Public interface — stub mode (no credentials in CI)
# ---------------------------------------------------------------------------


@patch.object(kc, "_load_credentials", return_value=None)
def test_fetch_live_contracts_returns_list(_):
    result = kc.fetch_live_contracts("Kalshi")
    assert isinstance(result, list)


@patch.object(kc, "_load_credentials", return_value=None)
def test_fetch_live_contracts_returns_market_contracts(_):
    result = kc.fetch_live_contracts("Kalshi")
    assert len(result) > 0
    for c in result:
        assert isinstance(c, MarketContract)


@patch.object(kc, "_load_credentials", return_value=None)
def test_fetch_live_contracts_stub_chicago_high_temp_fields(_):
    for c in kc.fetch_live_contracts("Kalshi"):
        assert c.city == "Chicago"
        assert c.metric == "high_temp_f"
        assert c.comparison_operator == "ge"
        assert c.market_source == "Kalshi"
        assert isinstance(c.threshold_f, int)
        assert isinstance(c.target_date, date)


def test_fetch_live_contracts_wrong_source_raises():
    with pytest.raises(RuntimeError, match="not supported"):
        kc.fetch_live_contracts("Polymarket")


def test_fetch_live_contracts_case_insensitive_source():
    assert isinstance(kc.fetch_live_contracts("kalshi"), list)


@patch.object(kc, "_load_credentials", return_value=None)
def test_fetch_orderbook_stub_returns_snapshot(_):
    snap = kc.fetch_orderbook_for_contract("Kalshi", "CHI_HIGH_TEMP_F_GE_75_20260318")
    assert isinstance(snap, OrderBookSnapshot)


@patch.object(kc, "_load_credentials", return_value=None)
def test_fetch_orderbook_stub_bid_ask_values(_):
    snap = kc.fetch_orderbook_for_contract("Kalshi", "CHI_HIGH_TEMP_F_GE_75_20260318")
    assert snap.best_bid == pytest.approx(kc._STUB_BID)
    assert snap.best_ask == pytest.approx(kc._STUB_ASK)


def test_fetch_orderbook_wrong_source_raises():
    with pytest.raises(RuntimeError, match="not supported"):
        kc.fetch_orderbook_for_contract("Polymarket", "X")


# ---------------------------------------------------------------------------
# Public interface — live mode (mocked _load_credentials + _kalshi_get)
# ---------------------------------------------------------------------------


@patch.object(kc, "_load_credentials", return_value=_FAKE_CREDENTIALS)
@patch.object(kc, "_kalshi_get")
def test_fetch_live_contracts_live_parses_api_response(mock_get, _):
    mock_get.return_value = {
        "markets": [
            _make_market_dict("KXHIGHCHI-26MAR18-B90"),
            _make_market_dict("KXHIGHCHI-26MAR18-B75"),
        ],
        "cursor": None,
    }
    result = kc.fetch_live_contracts("Kalshi")
    assert len(result) == 2
    thresholds = {c.threshold_f for c in result}
    assert thresholds == {90, 75}
    assert all(c.city == "Chicago" for c in result)


@patch.object(kc, "_load_credentials", return_value=_FAKE_CREDENTIALS)
@patch.object(kc, "_kalshi_get")
def test_fetch_live_contracts_skips_unparseable_tickers(mock_get, _):
    mock_get.return_value = {
        "markets": [
            _make_market_dict("KXHIGHCHI-26MAR18-B90"),
            _make_market_dict("GARBAGE-TICKER"),
        ],
        "cursor": None,
    }
    result = kc.fetch_live_contracts("Kalshi")
    assert len(result) == 1
    assert result[0].threshold_f == 90


@patch.object(kc, "_load_credentials", return_value=_FAKE_CREDENTIALS)
@patch.object(kc, "_kalshi_get")
def test_fetch_live_contracts_empty_response_falls_back_to_stub(mock_get, _):
    mock_get.return_value = {"markets": [], "cursor": None}
    result = kc.fetch_live_contracts("Kalshi")
    assert len(result) == len(kc._STUB_THRESHOLDS)


@patch.object(kc, "_load_credentials", return_value=_FAKE_CREDENTIALS)
@patch.object(kc, "_kalshi_get", side_effect=RuntimeError("network timeout"))
def test_fetch_live_contracts_api_error_falls_back_to_stub(_, __):
    result = kc.fetch_live_contracts("Kalshi")
    assert len(result) == len(kc._STUB_THRESHOLDS)


@patch.object(kc, "_load_credentials", return_value=_FAKE_CREDENTIALS)
@patch.object(kc, "_kalshi_get")
def test_fetch_orderbook_live_parses_dollar_prices(mock_get, _):
    mock_get.return_value = _make_orderbook_response(
        yes_levels=[["0.30", "200.00"]],
        no_levels=[["0.68", "150.00"]],
    )
    snap = kc.fetch_orderbook_for_contract("Kalshi", "KXHIGHCHI-26MAR18-B51.5")
    assert isinstance(snap, OrderBookSnapshot)
    assert snap.best_bid == pytest.approx(0.30)
    assert snap.best_ask == pytest.approx(0.32)   # 1.0 - 0.68
    assert snap.contract_id == "KXHIGHCHI-26MAR18-B51.5"


@patch.object(kc, "_load_credentials", return_value=_FAKE_CREDENTIALS)
@patch.object(kc, "_kalshi_get", side_effect=RuntimeError("API down"))
def test_fetch_orderbook_api_error_falls_back_to_stub(_, __):
    snap = kc.fetch_orderbook_for_contract("Kalshi", "KXHIGHCHI-26MAR18-B51.5")
    assert snap.best_bid == pytest.approx(kc._STUB_BID)
    assert snap.best_ask == pytest.approx(kc._STUB_ASK)


# ---------------------------------------------------------------------------
# stub_mode config override
# ---------------------------------------------------------------------------


@patch.object(kc, "_load_credentials", return_value=_FAKE_CREDENTIALS)
@patch("deep_isobar.market.kalshi_client.get_setting")
def test_stub_mode_config_forces_stub_despite_credentials(mock_setting, _):
    def side_effect(key, default=None):
        if key == "markets.kalshi.stub_mode":
            return True
        return default

    mock_setting.side_effect = side_effect
    result = kc.fetch_live_contracts("Kalshi")
    assert len(result) == len(kc._STUB_THRESHOLDS)
    assert all(c.city == "Chicago" for c in result)
