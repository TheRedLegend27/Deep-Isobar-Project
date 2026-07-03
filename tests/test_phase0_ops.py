"""Tests for Phase 0 ops: preflight breakers, backup rotation, interval jobs."""

from __future__ import annotations

import tarfile
from datetime import datetime, timedelta, timezone

import pytest

from deep_isobar.calibration.emos import EMOSParams
from deep_isobar.ops.backup import run_backup
from deep_isobar.supervisor import _interval_job_due
from deep_isobar.trading.preflight import run_preflight

MODELS = ["GFS", "ECMWF", "ICON", "GEM", "NBM"]


def _params(age_days: float = 0.0) -> EMOSParams:
    fitted = datetime.now(timezone.utc) - timedelta(days=age_days)
    return EMOSParams(
        station_id="KTST", model_names=MODELS,
        a0=0.0, a=[0.2] * 5, c=1.0, d=0.5,
        fitted_at_utc=fitted.isoformat(timespec="seconds"),
    )


def _ok_kwargs(**overrides) -> dict:
    base = dict(
        city_name="Testville",
        effective_mean=90.0,
        effective_std=1.7,
        dist_source="EMOS",
        nbm_max_f=89.5,
        emos_params=_params(),
        n_contracts=10,
        market_is_live=True,
        hist_min_f=45.0,
        hist_max_f=98.0,
    )
    base.update(overrides)
    return base


# ── preflight ────────────────────────────────────────────────────────────────


def test_preflight_passes_sane_day():
    result = run_preflight(**_ok_kwargs())
    assert result.ok
    assert result.failures == []


def test_preflight_blocks_stub_market():
    result = run_preflight(**_ok_kwargs(market_is_live=False))
    assert not result.ok
    assert any("STUB" in f for f in result.failures)


def test_preflight_blocks_insane_mean():
    # The 2026-07-02 incident: 103.8°F Philadelphia against history max 98.
    result = run_preflight(**_ok_kwargs(effective_mean=120.0))
    assert not result.ok
    assert any("outside sane bounds" in f for f in result.failures)


def test_preflight_blocks_nbm_hard_divergence():
    result = run_preflight(**_ok_kwargs(effective_mean=96.5, nbm_max_f=89.5))
    assert not result.ok
    assert any("NBM" in f for f in result.failures)


def test_preflight_blocks_stale_params():
    result = run_preflight(**_ok_kwargs(emos_params=_params(age_days=5)))
    assert not result.ok
    assert any("days old" in f for f in result.failures)


def test_preflight_blocks_broken_sigma_and_thin_market():
    result = run_preflight(**_ok_kwargs(effective_std=25.0, n_contracts=1))
    assert not result.ok
    assert any("sigma" in f for f in result.failures)
    assert any("contracts" in f for f in result.failures)


def test_preflight_missing_history_warns_but_passes():
    result = run_preflight(**_ok_kwargs(hist_min_f=None, hist_max_f=None))
    assert result.ok
    assert any("bounds" in w for w in result.warnings)


def test_preflight_legacy_path_skips_params_check():
    result = run_preflight(**_ok_kwargs(dist_source="legacy", emos_params=None,
                                        nbm_max_f=None))
    assert result.ok


# ── backup ───────────────────────────────────────────────────────────────────


def test_backup_archives_and_rotates(tmp_path):
    data = tmp_path / "data"
    (data / "paper_trades").mkdir(parents=True)
    (data / "paper_trades" / "paper_trades.csv").write_text("date\n2026-07-03\n")
    (data / "emos_params").mkdir()
    (data / "emos_params" / "KTST_emos.json").write_text("{}")
    (data / "logs").mkdir()
    (data / "logs" / "noise.log").write_text("x" * 1000)

    dest = tmp_path / "backups"
    first = run_backup(dest_dir=dest, keep=2, data_dir=data)
    assert first is not None and first.exists()

    with tarfile.open(first) as tar:
        names = tar.getnames()
    assert any("paper_trades.csv" in n for n in names)
    assert any("KTST_emos.json" in n for n in names)
    assert not any("noise.log" in n for n in names)  # logs excluded

    # Rotation: keep=2 → creating two more leaves exactly 2 archives.
    run_backup(dest_dir=dest, keep=2, data_dir=data)
    run_backup(dest_dir=dest, keep=2, data_dir=data)
    assert len(list(dest.glob("deep_isobar_data_*.tar.gz"))) == 2


def test_backup_empty_data_returns_none(tmp_path):
    assert run_backup(dest_dir=tmp_path / "b", keep=3, data_dir=tmp_path / "nope") is None


# ── supervisor interval jobs ─────────────────────────────────────────────────


def _job(**kw):
    return {"name": "collector", "module": "x", "every_minutes": 10, **kw}


def test_interval_due_when_never_run():
    now = datetime(2026, 7, 3, 12, 0)
    assert _interval_job_due(_job(), {"last_run": {}}, now)


def test_interval_respects_spacing():
    now = datetime(2026, 7, 3, 12, 0)
    recent = {"last_run": {"collector": "2026-07-03T11:55:00"}}
    stale = {"last_run": {"collector": "2026-07-03T11:45:00"}}
    assert not _interval_job_due(_job(), recent, now)
    assert _interval_job_due(_job(), stale, now)


def test_interval_respects_window():
    job = _job(window="06:00-21:00")
    state = {"last_run": {}}
    assert not _interval_job_due(job, state, datetime(2026, 7, 3, 5, 30))
    assert _interval_job_due(job, state, datetime(2026, 7, 3, 6, 0))
    assert not _interval_job_due(job, state, datetime(2026, 7, 3, 21, 30))


def test_interval_handles_legacy_date_format():
    # A daily-format date in state (pre-upgrade) must not crash — treat as due.
    now = datetime(2026, 7, 3, 12, 0)
    assert _interval_job_due(_job(), {"last_run": {"collector": "2026-07-02"}}, now)
