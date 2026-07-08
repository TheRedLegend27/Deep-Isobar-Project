"""Tests for the GEFS backfill worker (network mocked)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from deep_isobar.lab import gefs_backfill as gb


# ── extract: local-day sample filtering ──────────────────────────────────────


def test_extract_filters_to_local_day_and_aggregates(monkeypatch, tmp_path):
    """Only samples valid on the station's LOCAL target date count, and the
    aggregate is max (highs) / min (lows) over those samples."""
    fetched: list[int] = []

    def fake_download(grib_url, idx_url, dest):
        Path(dest).write_bytes(b"x")
        fetched.append(int(grib_url[-3:]))

    # value = fhour so we can see exactly which samples aggregated
    def fake_extract(path, lat, lon):
        return float(str(path).rsplit("_f", 1)[1][:3])

    import deep_isobar.data.historical_forecast_ingest as hfi
    monkeypatch.setattr(hfi, "_download_snippet", fake_download)
    monkeypatch.setattr(hfi, "_extract_fahrenheit", fake_extract)

    # Chicago (UTC-5 in July): local 2026-06-15 spans 05Z(15th)–05Z(16th),
    # so from the 00z run of the 14th that's fhours 30..48 within our 24-48
    # window → samples 30,33,36,39,42,45,48.
    v = gb.extract_member_daily(41.786, 272.248, "America/Chicago",
                                date(2026, 6, 15), "gep01", tmp_path, agg="max")
    assert v == 48.0
    v_min = gb.extract_member_daily(41.786, 272.248, "America/Chicago",
                                    date(2026, 6, 15), "gep01", tmp_path, agg="min")
    assert v_min == 30.0
    assert min(fetched) >= 24 and max(fetched) <= 48
    assert not list(tmp_path.glob("*.grib2"))  # snippets always cleaned up


def test_extract_returns_none_on_sparse_decode(monkeypatch, tmp_path):
    import deep_isobar.data.historical_forecast_ingest as hfi
    monkeypatch.setattr(hfi, "_download_snippet",
                        lambda *a: (_ for _ in ()).throw(RuntimeError("404")))
    monkeypatch.setattr(hfi, "_extract_fahrenheit", lambda *a: 0.0)
    v = gb.extract_member_daily(41.786, 272.248, "America/Chicago",
                                date(2026, 6, 15), "gep01", tmp_path)
    assert v is None


# ── summarize ────────────────────────────────────────────────────────────────


def test_summarize_members_matches_live_spread_definition(tmp_path):
    df = pd.DataFrame({
        "date": [date(2026, 6, 1)] * 3 + [date(2026, 6, 2)] * 12,
        "member": [f"m{i}" for i in range(3)] + [f"m{i}" for i in range(12)],
        "value_f": [80.0, 82.0, 84.0] + [70.0 + i for i in range(12)],
    })
    p = tmp_path / "KTST_high_202606_members.parquet"
    df.to_parquet(p, index=False)
    out = gb.summarize_members(p)
    # Day 1 has only 3 members → dropped (n_members >= 10 guard).
    assert list(out["date"]) == [date(2026, 6, 2)]
    assert out["ens_spread_var"].iloc[0] == pytest.approx(
        pd.Series(range(12), dtype=float).var(ddof=1)
    )


# ── queue mechanics ──────────────────────────────────────────────────────────


def _task_file(queue: Path, name: str) -> Path:
    queue.mkdir(parents=True, exist_ok=True)
    p = queue / name
    p.write_text(json.dumps({"station_id": "KTST", "lat": 1.0, "lon_360": 2.0,
                             "timezone": "America/Chicago", "metric": "high",
                             "month": "2026-06"}))
    return p


def test_claim_is_exclusive(tmp_path):
    q = tmp_path / "queue"
    _task_file(q, "a.json")
    first = gb.claim_task(q)
    assert first is not None
    assert gb.claim_task(q) is None          # nothing left to claim
    assert not (q / "a.json").exists()       # moved into claimed/


def test_worker_loop_success_and_failure(tmp_path, monkeypatch):
    q = tmp_path / "queue"
    _task_file(q, "good.json")
    _task_file(q, "bad.json")

    def fake_run(task, out_dir):
        if "bad" in task.get("month", "") or task.get("_bad"):
            raise RuntimeError("boom")
        return tmp_path / "out.parquet"

    # Make one task fail deterministically
    bad = json.loads((q / "bad.json").read_text())
    bad["_bad"] = True
    (q / "bad.json").write_text(json.dumps(bad))

    monkeypatch.setattr(gb, "run_task", fake_run)
    done = gb.worker_loop(q, tmp_path / "out")
    assert done == 1
    failed = list((q / "failed").glob("*.json"))
    assert len(failed) == 1
    assert "boom" in json.loads(failed[0].read_text())["error"]
    assert not list((q / "claimed").glob("*.json"))  # nothing left claimed
