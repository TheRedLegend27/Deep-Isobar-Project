"""Tests for deep_isobar.research.daily_scorecard (pure computations)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from deep_isobar.research.daily_scorecard import (
    _pit_sparkline,
    _trend_arrow,
    load_settled_trades,
    realized_outcome,
    render_markdown,
    window_stats,
)


# ── realized_outcome ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "direction,status,expected",
    [
        ("BUY", "WIN", 1),
        ("BUY", "LOSS", 0),
        ("SELL", "WIN", 0),
        ("SELL", "LOSS", 1),
        ("BUY", "OPEN", None),
        ("HOLD", "WIN", None),
    ],
)
def test_realized_outcome(direction, status, expected):
    assert realized_outcome(direction, status) == expected


# ── window_stats ─────────────────────────────────────────────────────────────


def _trades_frame() -> pd.DataFrame:
    """6 settled trades over 3 days; model closer to truth than market."""
    rows = [
        # date, direction, status, model_prob, market_prob, pnl
        (date(2026, 7, 1), "BUY",  "WIN",  0.70, 0.55, 4.5),
        (date(2026, 7, 1), "SELL", "LOSS", 0.30, 0.40, -6.0),
        (date(2026, 7, 2), "BUY",  "WIN",  0.65, 0.50, 5.0),
        (date(2026, 7, 2), "BUY",  "LOSS", 0.60, 0.45, -4.5),
        (date(2026, 7, 3), "SELL", "WIN",  0.20, 0.35, 3.5),
        (date(2026, 7, 3), "BUY",  "WIN",  0.75, 0.60, 4.0),
    ]
    df = pd.DataFrame(
        rows,
        columns=["date", "direction", "status", "model_prob", "market_prob", "realized_pnl"],
    )
    df["alpha"] = df["model_prob"] - df["market_prob"]
    df["entry_price"] = df["market_prob"]
    df["city"] = "Chicago"
    df["contract_ticker"] = [f"T{i}" for i in range(len(df))]
    df["settled_temp"] = 90
    df["outcome"] = [realized_outcome(d, s) for d, s in zip(df["direction"], df["status"])]
    return df


def test_window_stats_basic():
    stats = window_stats(_trades_frame(), days=30, asof=date(2026, 7, 3))
    assert stats["n"] == 6
    assert stats["days"] == 3
    assert stats["pnl"] == pytest.approx(6.5)
    assert stats["win_rate"] == pytest.approx(4 / 6)


def test_window_stats_respects_window():
    stats = window_stats(_trades_frame(), days=1, asof=date(2026, 7, 3))
    assert stats["n"] == 2
    assert stats["pnl"] == pytest.approx(7.5)


def test_window_stats_empty():
    assert window_stats(_trades_frame(), days=7, asof=date(2026, 8, 1)) == {"n": 0}


def test_brier_edge_sign():
    """Model probs are closer to outcomes than market probs → positive edge."""
    stats = window_stats(_trades_frame(), days=30, asof=date(2026, 7, 3))
    # outcomes: BUY WIN→1, SELL LOSS→1, BUY WIN→1, BUY LOSS→0, SELL WIN→0, BUY WIN→1
    outcomes = np.array([1, 1, 1, 0, 0, 1], dtype=float)
    model = np.array([0.70, 0.30, 0.65, 0.60, 0.20, 0.75])
    market = np.array([0.55, 0.40, 0.50, 0.45, 0.35, 0.60])
    assert stats["model_brier"] == pytest.approx(np.mean((model - outcomes) ** 2))
    assert stats["market_brier"] == pytest.approx(np.mean((market - outcomes) ** 2))
    assert stats["brier_edge"] == pytest.approx(stats["market_brier"] - stats["model_brier"])


# ── load_settled_trades ──────────────────────────────────────────────────────


def test_load_settled_trades_missing_csv(tmp_path):
    df = load_settled_trades(tmp_path / "nope.csv")
    assert df.empty


def test_load_settled_trades_filters_and_parses(tmp_path):
    csv = tmp_path / "paper_trades.csv"
    csv.write_text(
        "date,city,contract_ticker,direction,alpha,model_prob,market_prob,"
        "entry_price,status,realized_pnl,settled_temp\n"
        "2026-07-01,Chicago,T1,BUY,0.15,0.70,0.55,0.55,WIN,4.5,95\n"
        "2026-07-01,Chicago,T2,BUY,0.12,0.60,0.48,0.48,NO_SIGNAL,,\n"
        "2026-07-02,Boston,T3,SELL,-0.11,0.30,0.41,0.41,OPEN,,\n",
        encoding="utf-8",
    )
    df = load_settled_trades(csv)
    assert len(df) == 1
    assert df["outcome"].iloc[0] == 1
    assert df["date"].iloc[0] == date(2026, 7, 1)


# ── rendering helpers ────────────────────────────────────────────────────────


def test_trend_arrow_directions():
    assert _trend_arrow(0.8, 0.6, higher_is_better=True) == "↑"
    assert _trend_arrow(0.4, 0.6, higher_is_better=True) == "↓"
    assert _trend_arrow(0.4, 0.6, higher_is_better=False) == "↑"
    assert _trend_arrow(float("nan"), 0.6) == "→"


def test_pit_sparkline_flat_and_empty():
    assert _pit_sparkline(np.array([])) == "n/a"
    flat = _pit_sparkline(np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
    assert len(set(flat)) == 1  # perfectly flat PIT → identical blocks


def test_render_markdown_smoke():
    df = _trades_frame()
    latest = df[df["date"] == date(2026, 7, 3)]
    s7 = window_stats(df, 7, asof=date(2026, 7, 3))
    s30 = window_stats(df, 30, asof=date(2026, 7, 3))
    cal = {
        "city": "Chicago", "station_id": "KMDW", "n_days": 30,
        "crps": 1.05, "mae": 1.4, "nbm_mae": 1.5, "nbm_divergence_f": 0.8,
        "mean_sigma": 1.6, "modal_prob": 0.44,
        "pit_hist": np.array([0.18, 0.22, 0.20, 0.22, 0.18]),
        "spread_source": "members", "spread_days": 12, "spread_coverage": 0.4,
        "params_age_days": 0,
    }
    md = render_markdown(date(2026, 7, 3), latest, date(2026, 7, 3), s7, s30, [cal])
    assert "Deep Isobar scorecard" in md
    assert "beating the market" in md
    assert "Chicago (KMDW)" in md
    assert "+7.50" in md  # day P&L of the latest settlement day


def test_render_markdown_reading_uses_recent_window():
    """Reading should reflect the 7d edge, not a stale 30d edge of the opposite sign."""
    empty = pd.DataFrame(columns=["city", "contract_ticker", "direction", "alpha",
                                  "entry_price", "settled_temp", "status", "realized_pnl"])
    s7 = {
        "n": 50, "pnl": 100.0, "win_rate": 0.6,
        "model_brier": 0.17, "market_brier": 0.22,
        "brier_edge": 0.05, "mean_abs_alpha": 0.18,
    }
    s30 = {
        "n": 120, "pnl": 300.0, "win_rate": 0.5,
        "model_brier": 0.23, "market_brier": 0.22,
        "brier_edge": -0.01, "mean_abs_alpha": 0.18,
    }
    md = render_markdown(date(2026, 7, 3), empty, None, s7, s30, [])
    assert "beating the market" in md
    assert "treat new" not in md


def test_render_markdown_handles_no_trades():
    empty = pd.DataFrame(columns=["city", "contract_ticker", "direction", "alpha",
                                  "entry_price", "settled_temp", "status", "realized_pnl"])
    md = render_markdown(date(2026, 7, 3), empty, None, {"n": 0}, {"n": 0}, [])
    assert "No settled trades" in md
