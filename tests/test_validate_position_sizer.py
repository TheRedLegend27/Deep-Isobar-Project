"""Smoke tests for the position-sizer historical replay script.

Uses a small synthetic CSV (not the real data/paper_trades.csv) so this
stays fast and deterministic — the real historical replay is run and
reported on separately, this just proves the replay/report machinery
itself is correct on a known input.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from deep_isobar.research.validate_position_sizer import build_report, replay

_COLUMNS = [
    "date", "contract_ticker", "direction", "alpha", "model_prob", "market_prob",
    "entry_price", "position_size", "status", "realized_pnl", "threshold_f",
    "strike_type", "city", "anomaly_confidence", "anomaly_flags", "ens_spread_var",
]


def _row(**overrides) -> dict:
    base = dict(
        date="2026-07-01", contract_ticker="X-T90", direction="BUY",
        alpha=0.20, model_prob=0.55, market_prob=0.35, entry_price=0.36,
        position_size=10.0, status="WIN", realized_pnl=6.0, threshold_f=90,
        strike_type="greater", city="Testville",
        anomaly_confidence="", anomaly_flags="", ens_spread_var="",
    )
    base.update(overrides)
    return base


@pytest.fixture()
def synthetic_csv(tmp_path) -> Path:
    rows = [
        _row(date="2026-07-01", contract_ticker="X-T90", realized_pnl=6.0, status="WIN"),
        _row(date="2026-07-02", contract_ticker="X-T91", realized_pnl=-4.0, status="LOSS",
             model_prob=0.45, market_prob=0.35, entry_price=0.36),
        _row(date="2026-07-03", contract_ticker="X-T92", realized_pnl=8.0, status="WIN",
             entry_price=0.05, model_prob=0.30, market_prob=0.03),  # cheap tail
        # An OPEN row must be excluded entirely from the replay.
        _row(date="2026-07-04", contract_ticker="X-T93", status="OPEN", realized_pnl=""),
    ]
    path = tmp_path / "paper_trades.csv"
    pd.DataFrame(rows, columns=_COLUMNS).to_csv(path, index=False)
    return path


def test_replay_only_includes_settled_trades(synthetic_csv):
    rows = replay(synthetic_csv)
    assert len(rows) == 3
    assert all(r.contract_ticker != "X-T93" for r in rows)


def test_replay_stakes_are_bounded_by_config_caps(synthetic_csv):
    rows = replay(synthetic_csv)
    for r in rows:
        assert r.new_stake_usd >= 0.0
        # Single trading city in this fixture — city cap is the binding one.
        assert r.new_stake_usd <= 500.0  # sane upper bound, never runs away


def test_replay_new_pnl_is_zero_when_stake_is_zero(synthetic_csv):
    rows = replay(synthetic_csv)
    for r in rows:
        if r.new_stake_usd == 0.0:
            assert r.new_realized_pnl == 0.0


def test_replay_cheap_tail_row_is_present_and_haircut(synthetic_csv):
    rows = replay(synthetic_csv)
    cheap = [r for r in rows if r.contract_ticker == "X-T92"]
    assert len(cheap) == 1
    normal = [r for r in rows if r.contract_ticker == "X-T90"][0]
    tail = cheap[0]
    # The tail-entry haircut note only fires for the cheap contract.
    assert "tail" in tail.reasoning
    assert "tail" not in normal.reasoning


def test_build_report_runs_on_replay_output(synthetic_csv):
    rows = replay(synthetic_csv)
    report = build_report(rows, bankroll_usd=500.0)
    assert "POSITION SIZER REPLAY" in report
    assert "Trades replayed        : 3" in report


def test_build_report_empty_input():
    assert "No settled trades" in build_report([], bankroll_usd=500.0)


def test_replay_track_record_has_no_lookahead(synthetic_csv):
    """The 2026-07-03 trade's track record must only see 07-01 and 07-02,
    never the same day or later — this is what protects the live sizer
    from lookahead bias too."""
    df = pd.read_csv(synthetic_csv)
    df["date"] = pd.to_datetime(df["date"]).dt.date

    from deep_isobar.trading.position_sizer import build_station_track_record

    settled = df[df["status"].isin(["WIN", "LOSS"])].copy()
    record = build_station_track_record(settled, "Testville", asof=date(2026, 7, 3))
    assert record is not None
    assert record.n_trades == 2  # only 07-01 and 07-02, not 07-03 itself
