"""Tests for the maker-fill simulator's conservative semantics."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from deep_isobar.lab.maker_fills import _maker_limit, simulate_resting_order


def _books(rows):
    t0 = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)
    return pd.DataFrame([
        {"snapshot_utc": t0.replace(hour=12 + i), "contract_id": "C",
         "best_bid": b, "best_ask": a}
        for i, (b, a) in enumerate(rows)
    ])


PLACED = datetime(2026, 7, 6, 12, 30, tzinfo=timezone.utc)


def test_buy_fills_only_on_trade_through():
    # Resting BUY at 0.30: ask touching 0.30 is NOT a fill; ask 0.29 is.
    books = _books([(0.28, 0.32), (0.29, 0.30), (0.28, 0.29)])
    filled, when = simulate_resting_order(books, PLACED, "BUY", 0.30)
    assert filled
    assert when == books["snapshot_utc"].iloc[2]

    touch_only = _books([(0.28, 0.32), (0.29, 0.30), (0.29, 0.30)])
    filled, _ = simulate_resting_order(touch_only, PLACED, "BUY", 0.30)
    assert not filled


def test_sell_fills_only_when_bid_trades_through():
    books = _books([(0.28, 0.32), (0.30, 0.34), (0.31, 0.35)])
    filled, _ = simulate_resting_order(books, PLACED, "SELL", 0.30)
    assert filled
    filled, _ = simulate_resting_order(books, PLACED, "SELL", 0.31)
    assert not filled  # bid only ever touches 0.31


def test_snapshots_before_placement_ignored():
    # The trade-through happens BEFORE we placed — must not count.
    books = _books([(0.10, 0.20)])  # ask 0.20 < limit, but at 12:00 < placed
    filled, _ = simulate_resting_order(books, PLACED, "BUY", 0.30)
    assert not filled


def test_maker_limit_policies():
    assert _maker_limit("BUY", 0.28, 0.32, "join") == 0.28
    assert _maker_limit("SELL", 0.28, 0.32, "join") == 0.32
    assert _maker_limit("BUY", 0.28, 0.32, "improve") == pytest.approx(0.29)
    assert _maker_limit("SELL", 0.28, 0.32, "improve") == pytest.approx(0.31)
    # One-tick spread: "improve" would cross → refuse (that's a taker order).
    assert _maker_limit("BUY", 0.30, 0.31, "improve") is None
    assert _maker_limit("BUY", float("nan"), 0.31, "join") is None
