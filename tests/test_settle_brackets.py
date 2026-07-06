"""Settlement of bracket (between) contracts + pending-date backlog.

Regression tests for the 2026-07-05 findings: (1) B-tickers were excluded
from settlement entirely ("mark manually"), leaving the bracket book
permanently OPEN; (2) between grading used an exclusive cap while Kalshi
pays inclusively; (3) settle only looked at rows dated *today*, whose ACIS
observation never exists at 18:00, so nothing ever settled.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from deep_isobar.research.settle_paper_trades import _settle_open_trades


def _frame(rows) -> pd.DataFrame:
    base = {
        "date": "2026-07-04", "city": "Chicago", "direction": "SELL",
        "alpha": -0.15, "model_prob": 0.15, "market_prob": 0.30,
        "entry_price": 0.30, "position_size": 10.0, "status": "OPEN",
        "realized_pnl": None, "settled_temp": None, "threshold_f": 87.0,
        "strike_type": "between", "floor_strike": 87, "cap_strike": 88,
        "contract_ticker": "KXHIGHCHI-26JUL04-B87.5",
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def test_bracket_sell_wins_when_high_misses_bracket():
    df = _frame([{}])
    out, n = _settle_open_trades(df, date(2026, 7, 4), settled_temp=85.0,
                                 city_name="Chicago")
    assert n == 1
    assert out.loc[0, "status"] == "WIN"
    assert float(out.loc[0, "realized_pnl"]) > 0


@pytest.mark.parametrize("high", [87.0, 88.0])
def test_bracket_cap_and_floor_both_pay_yes(high):
    """88 must count as inside 87-88 — the exclusive-cap bug called it out."""
    df = _frame([{"direction": "BUY", "entry_price": 0.30}])
    out, n = _settle_open_trades(df, date(2026, 7, 4), settled_temp=high,
                                 city_name="Chicago")
    assert n == 1
    assert out.loc[0, "status"] == "WIN"


def test_bracket_sell_loses_inside_bracket():
    df = _frame([{}])
    out, _ = _settle_open_trades(df, date(2026, 7, 4), settled_temp=88.0,
                                 city_name="Chicago")
    assert out.loc[0, "status"] == "LOSS"
    assert float(out.loc[0, "realized_pnl"]) < 0


def test_void_rows_left_untouched():
    df = _frame([{"status": "VOID", "realized_pnl": 0.0}])
    out, n = _settle_open_trades(df, date(2026, 7, 4), settled_temp=88.0,
                                 city_name="Chicago")
    assert n == 0
    assert out.loc[0, "status"] == "VOID"


def test_other_dates_not_touched():
    df = _frame([{"date": "2026-07-05"}])
    out, n = _settle_open_trades(df, date(2026, 7, 4), settled_temp=88.0,
                                 city_name="Chicago")
    assert n == 0
    assert out.loc[0, "status"] == "OPEN"
