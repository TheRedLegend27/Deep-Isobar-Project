"""Tests: as-of store lookahead enforcement, audit log, xlsx exports."""

from __future__ import annotations

import io
import sqlite3
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from deep_isobar.dashboard.audit import fetch_audit, init_audit_table, record_audit
from deep_isobar.dashboard.exports import xlsx_response
from deep_isobar.lab.asof_store import AsOfStore


# ── AsOfStore ────────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path):
    training = tmp_path / "emos_training"
    training.mkdir(parents=True)
    pd.DataFrame({
        "date": [date(2026, 6, 1), date(2026, 6, 2), date(2026, 6, 3)],
        "GFS": [80.0, 82.0, 84.0],
        "NBM": [79.0, 81.0, 83.0],
        "actual_f": [81.0, 83.0, None],   # June 3 not yet settled
        "ens_spread_var": [2.0, 3.0, 4.0],
    }).to_parquet(training / "KTST_t1_training.parquet", index=False)
    pd.DataFrame({
        "date": [date(2026, 6, 1), date(2026, 6, 2), date(2026, 6, 3)],
        "ens_spread_var": [2.0, 3.0, 4.0],
    }).to_parquet(training / "KTST_t1_spread.parquet", index=False)

    day = tmp_path / "market_history" / "date=2026-06-02"
    day.mkdir(parents=True)
    pd.DataFrame({
        "snapshot_utc": [
            datetime(2026, 6, 2, 12, 0, tzinfo=timezone.utc),
            datetime(2026, 6, 2, 18, 0, tzinfo=timezone.utc),
        ],
        "city": ["Chicago", "Chicago"],
        "contract_id": ["C1", "C1"],
        "best_bid": [0.4, 0.5],
        "best_ask": [0.45, 0.55],
    }).to_parquet(day / "books_120000.parquet", index=False)

    return AsOfStore(data_dir=tmp_path)


def test_forecasts_include_asof_day(store):
    """The T+1 forecast for day D was issued on D−1 → knowable on D."""
    fx = store.forecasts("KTST", as_of=date(2026, 6, 2))
    assert list(fx["date"]) == [date(2026, 6, 1), date(2026, 6, 2)]
    assert "actual_f" not in fx.columns  # truth never leaks through forecasts


def test_settlements_are_strictly_before_asof(store):
    """The high for day D posts the evening of D → NOT knowable morning-of."""
    obs = store.settlements("KTST", as_of=date(2026, 6, 2))
    assert list(obs["date"]) == [date(2026, 6, 1)]


def test_spread_history_cutoff(store):
    sp = store.spread_history("KTST", as_of=date(2026, 6, 2))
    assert list(sp["ens_spread_var"]) == [2.0, 3.0]


def test_market_books_timestamp_cutoff(store):
    cutoff = datetime(2026, 6, 2, 15, 0, tzinfo=timezone.utc)
    books = store.market_books(as_of=cutoff)
    assert len(books) == 1
    assert books["best_bid"].iloc[0] == 0.4


def test_market_books_reject_naive_datetime(store):
    with pytest.raises(ValueError, match="timezone-aware"):
        store.market_books(as_of=datetime(2026, 6, 2, 15, 0))


def test_missing_station_raises(store):
    with pytest.raises(FileNotFoundError):
        store.forecasts("KNOPE", as_of=date(2026, 6, 2))


# ── audit log ────────────────────────────────────────────────────────────────


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_audit_table(conn)
    return conn


def test_audit_roundtrip_most_recent_first():
    conn = _conn()
    record_audit(conn, "kaden", "login.success", ip="1.2.3.4")
    record_audit(conn, "kaden", "settings.patch", detail="alpha 0.10 -> 0.12")
    rows = fetch_audit(conn)
    assert [r["action"] for r in rows] == ["settings.patch", "login.success"]
    assert rows[0]["detail"] == "alpha 0.10 -> 0.12"
    assert rows[1]["ip"] == "1.2.3.4"


def test_audit_filters_by_username():
    conn = _conn()
    record_audit(conn, "kaden", "login.success")
    record_audit(conn, "mathguy", "login.failure")
    rows = fetch_audit(conn, username="mathguy")
    assert len(rows) == 1
    assert rows[0]["action"] == "login.failure"


def test_audit_anonymous_and_never_raises():
    conn = _conn()
    record_audit(conn, "", "login.failure")
    assert fetch_audit(conn)[0]["username"] == "<anonymous>"
    conn.close()
    record_audit(conn, "x", "y")  # closed connection — must not raise


# ── xlsx export ──────────────────────────────────────────────────────────────


def test_xlsx_response_roundtrip():
    df = pd.DataFrame({"city": ["Chicago", "Boston"], "pnl": [4.5, -2.0]})
    resp = xlsx_response(df, "trades_test")
    assert "spreadsheetml" in resp.media_type
    assert 'filename="trades_test_' in resp.headers["content-disposition"]

    # StreamingResponse wraps a BytesIO; drain it and read back via pandas.
    buf = io.BytesIO()
    import asyncio

    async def _drain():
        async for chunk in resp.body_iterator:
            buf.write(chunk if isinstance(chunk, bytes) else chunk.encode())

    asyncio.run(_drain())
    buf.seek(0)
    out = pd.read_excel(buf)
    assert list(out.columns) == ["city", "pnl"]
    assert out["pnl"].tolist() == [4.5, -2.0]
