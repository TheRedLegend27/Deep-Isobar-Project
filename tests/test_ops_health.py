"""Tests for deep_isobar.ops.health — the silent-failure alarms.

Every check is driven with synthetic files under tmp_path; no check may
depend on the real data/ tree or the wall clock.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from deep_isobar.ops.health import (
    ALARM,
    OK,
    SKIP,
    HealthCheck,
    alarms,
    check_orderbook_freshness,
    check_params_age,
    check_scorecard_gaps,
    check_session_activity,
    check_settlement_currency,
    check_stub_books,
    render_health_section,
)

_NOON = datetime(2026, 7, 12, 12, 0)
_TOMORROW = date(2026, 7, 13)


def _write_csv(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ── session_activity ─────────────────────────────────────────────────────────


def test_session_skipped_before_judgement_time(tmp_path):
    early = datetime(2026, 7, 12, 10, 45)
    check = check_session_activity(early, tmp_path / "daily_log.csv")
    assert check.status == SKIP


def test_session_alarm_when_log_missing(tmp_path):
    check = check_session_activity(_NOON, tmp_path / "daily_log.csv")
    assert check.status == ALARM


def test_session_alarm_when_no_rows_for_target_date(tmp_path):
    # Rows exist, but only for older target dates — the Jul 8-10 shape.
    log = _write_csv(tmp_path / "daily_log.csv", [
        {"date": "2026-07-11", "contract_ticker": "X-T90", "status": "OPEN"},
    ])
    check = check_session_activity(_NOON, log)
    assert check.status == ALARM
    assert str(_TOMORROW) in check.detail


def test_session_ok_counts_evaluations_and_signals(tmp_path):
    log = _write_csv(tmp_path / "daily_log.csv", [
        {"date": str(_TOMORROW), "contract_ticker": "X-T90", "status": "OPEN"},
        {"date": str(_TOMORROW), "contract_ticker": "X-T92", "status": "NO_SIGNAL"},
        {"date": str(_TOMORROW), "contract_ticker": "X-T94", "status": "OPEN"},
    ])
    check = check_session_activity(_NOON, log)
    assert check.status == OK
    assert "3 contracts" in check.detail
    assert "2 signals" in check.detail


# ── settlement_currency ──────────────────────────────────────────────────────


def test_settlement_skip_without_trades_file(tmp_path):
    check = check_settlement_currency(_NOON, tmp_path / "paper_trades.csv")
    assert check.status == SKIP


def test_settlement_ok_with_fresh_open_positions(tmp_path):
    trades = _write_csv(tmp_path / "paper_trades.csv", [
        {"date": "2026-07-11", "status": "WIN"},
        {"date": str(_TOMORROW), "status": "OPEN"},  # tomorrow's target — normal
    ])
    check = check_settlement_currency(_NOON, trades)
    assert check.status == OK


def test_settlement_alarm_when_open_past_48h(tmp_path):
    trades = _write_csv(tmp_path / "paper_trades.csv", [
        {"date": "2026-07-09", "status": "OPEN"},
        {"date": "2026-07-10", "status": "OPEN"},
        {"date": "2026-07-11", "status": "WIN"},
    ])
    check = check_settlement_currency(_NOON, trades)
    assert check.status == ALARM
    assert "2 position(s)" in check.detail
    assert "2026-07-09" in check.detail


def test_settlement_open_exactly_yesterday_is_not_stale(tmp_path):
    # Settle runs the evening of the target date; a >48h rule means
    # yesterday's OPEN row is not yet an alarm (CLI reports can lag).
    trades = _write_csv(tmp_path / "paper_trades.csv", [
        {"date": "2026-07-11", "status": "OPEN"},
    ])
    assert check_settlement_currency(_NOON, trades).status == OK


# ── orderbook_freshness ──────────────────────────────────────────────────────


def _book_file(history_dir: Path, day: str, name: str, mtime: datetime,
               rows: list[dict] | None = None) -> Path:
    day_dir = history_dir / f"date={day}"
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / name
    pd.DataFrame(rows or [{"best_bid": 0.3, "best_ask": 0.4}]).to_parquet(path)
    ts = mtime.timestamp()
    os.utime(path, (ts, ts))
    return path


def test_freshness_alarm_when_no_history(tmp_path):
    check = check_orderbook_freshness(_NOON, tmp_path / "market_history")
    assert check.status == ALARM


def test_freshness_ok_within_window(tmp_path):
    hist = tmp_path / "market_history"
    _book_file(hist, "2026-07-12", "books_155000.parquet", _NOON - timedelta(minutes=10))
    assert check_orderbook_freshness(_NOON, hist).status == OK


def test_freshness_alarm_when_collector_silent_in_window(tmp_path):
    hist = tmp_path / "market_history"
    _book_file(hist, "2026-07-12", "books_120000.parquet", _NOON - timedelta(hours=3))
    check = check_orderbook_freshness(_NOON, hist)
    assert check.status == ALARM
    assert "180 window-min" in check.detail


def test_freshness_tolerates_overnight_gap(tmp_path):
    # 06:20: the last snapshot is from yesterday 20:55.  Only 5 window-min
    # elapsed yesterday + 20 today — the overnight pause doesn't count.
    now = datetime(2026, 7, 12, 6, 20)
    hist = tmp_path / "market_history"
    _book_file(hist, "2026-07-11", "books_205500.parquet",
               datetime(2026, 7, 11, 20, 55))
    assert check_orderbook_freshness(now, hist).status == OK


def test_freshness_alarm_when_yesterday_died_midday(tmp_path):
    # 06:20 next day: the last snapshot is from yesterday noon — the
    # collector was dead for the back half of yesterday's window, and the
    # overnight pause must not launder that silence.
    now = datetime(2026, 7, 12, 6, 20)
    hist = tmp_path / "market_history"
    _book_file(hist, "2026-07-11", "books_160000.parquet",
               datetime(2026, 7, 11, 12, 0))
    assert check_orderbook_freshness(now, hist).status == ALARM


# ── stub_books ───────────────────────────────────────────────────────────────


def _stub_rows(n: int = 10) -> list[dict]:
    return [{"best_bid": 0.48, "best_ask": 0.52} for _ in range(n)]


def _live_rows(n: int = 10) -> list[dict]:
    return [{"best_bid": 0.30 + i * 0.01, "best_ask": 0.35 + i * 0.01} for i in range(n)]


def test_stub_books_skip_without_files(tmp_path):
    assert check_stub_books(tmp_path / "market_history").status == SKIP


def test_stub_books_ok_on_live_prices(tmp_path):
    hist = tmp_path / "market_history"
    _book_file(hist, "2026-07-12", "books_120000.parquet", _NOON, _live_rows())
    assert check_stub_books(hist).status == OK


def test_stub_books_alarm_on_48_52_snapshot(tmp_path):
    hist = tmp_path / "market_history"
    _book_file(hist, "2026-07-12", "books_120000.parquet", _NOON, _stub_rows())
    check = check_stub_books(hist)
    assert check.status == ALARM
    assert "books_120000.parquet" in check.detail


def test_stub_books_ignores_tiny_snapshots_and_incidental_matches(tmp_path):
    hist = tmp_path / "market_history"
    # 3 rows at 48/52 could be one real contract quoted there — too small.
    _book_file(hist, "2026-07-12", "books_120000.parquet", _NOON, _stub_rows(3))
    # A single 48/52 row among live books is not stub-shaped.
    _book_file(hist, "2026-07-12", "books_121000.parquet", _NOON,
               _live_rows(9) + _stub_rows(1))
    assert check_stub_books(hist).status == OK


# ── params_age ───────────────────────────────────────────────────────────────


class _City:
    def __init__(self, station_id: str, low: bool = False):
        self.station_id = station_id
        self.kalshi_low_series = "KXLOW" if low else None


def _write_params(params_dir: Path, key: str, fitted_at: datetime) -> None:
    # Matches the EMOSParams JSON schema closely enough for load_params.
    params_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "station_id": key,
        "model_names": ["GFS"],
        "a0": 0.0,
        "a": [1.0],
        "c": 1.0,
        "d": 0.5,
        "sigma_floor_f": 0.5,
        "fitted_at_utc": fitted_at.isoformat(timespec="seconds"),
    }
    (params_dir / f"{key}_emos.json").write_text(json.dumps(payload), encoding="utf-8")


_NOW_UTC = datetime(2026, 7, 12, 16, 0, tzinfo=timezone.utc)


def test_params_age_ok_when_fresh(tmp_path):
    _write_params(tmp_path, "KTST", _NOW_UTC - timedelta(hours=6))
    check = check_params_age(_NOW_UTC, params_dir=tmp_path, cities=[_City("KTST")])
    assert check.status == OK


def test_params_age_alarm_when_refit_missed(tmp_path):
    _write_params(tmp_path, "KTST", _NOW_UTC - timedelta(hours=40))
    check = check_params_age(_NOW_UTC, params_dir=tmp_path, cities=[_City("KTST")])
    assert check.status == ALARM
    assert "KTST" in check.detail


def test_params_age_checks_low_params_too(tmp_path):
    _write_params(tmp_path, "KTST", _NOW_UTC - timedelta(hours=2))
    _write_params(tmp_path, "KTST_low", _NOW_UTC - timedelta(hours=40))
    check = check_params_age(
        _NOW_UTC, params_dir=tmp_path, cities=[_City("KTST", low=True)]
    )
    assert check.status == ALARM
    assert "KTST_low" in check.detail


def test_params_age_skip_when_nothing_fitted(tmp_path):
    check = check_params_age(_NOW_UTC, params_dir=tmp_path, cities=[_City("KTST")])
    assert check.status == SKIP


# ── scorecard_gaps ───────────────────────────────────────────────────────────


def test_scorecard_gaps_ok(tmp_path):
    for i in range(1, 4):
        d = _NOON.date() - timedelta(days=i)
        (tmp_path / f"scorecard_{d}.md").write_text("x", encoding="utf-8")
    assert check_scorecard_gaps(_NOON, tmp_path).status == OK


def test_scorecard_gaps_alarm_lists_missing_days(tmp_path):
    (tmp_path / "scorecard_2026-07-11.md").write_text("x", encoding="utf-8")
    check = check_scorecard_gaps(_NOON, tmp_path)
    assert check.status == ALARM
    assert "2026-07-10" in check.detail and "2026-07-09" in check.detail
    assert "2026-07-11" not in check.detail


# ── rendering / aggregation ──────────────────────────────────────────────────


def test_render_health_section_leads_with_alarm_banner():
    checks = [
        HealthCheck("session_activity", ALARM, "0 contracts evaluated"),
        HealthCheck("stub_books", OK, "clean"),
        HealthCheck("params_age", SKIP, "no fitted params found"),
    ]
    md = render_health_section(checks)
    assert md.startswith("## Ops health")
    assert "🚨 1 invariant(s) broken" in md
    assert "✅ `stub_books`" in md
    assert "⏸ `params_age`" in md
    assert alarms(checks) == [checks[0]]


def test_render_health_section_all_green_has_no_banner():
    md = render_health_section([HealthCheck("stub_books", OK, "clean")])
    assert "🚨" not in md
