"""Live trend-variance plumbing tests — per-station fit options, run-view
banking, and the session-time trend diff.

The Previous Runs API serves previous_dayN only for PAST timestamps (verified
live 2026-07-19: for tomorrow it mirrors the current run), so live trend is
computed against run views banked by yesterday's session, exactly like the
T+1 spread history.
"""

from __future__ import annotations

import io
import json
from datetime import date, timedelta

import pandas as pd
import pytest

from deep_isobar.calibration import emos_training as et
from deep_isobar.calibration.emos_training import (
    EMOS_MODELS,
    _fit_option,
    _record_run_views,
    _runview_path,
    fetch_live_trend_f,
)


# ── per-station fit options ──────────────────────────────────────────────────


def _patch_settings(monkeypatch, settings: dict):
    def fake_get_setting(key, default=None):
        return settings.get(key, default)

    monkeypatch.setattr("deep_isobar.config.get_setting", fake_get_setting)


def test_fit_option_station_override_beats_global(monkeypatch):
    _patch_settings(monkeypatch, {
        "emos.trend_variance": False,
        "emos.station_overrides": {"KDFW": {"trend_variance": True}},
    })
    assert _fit_option("KDFW", "trend_variance", default=False) is True
    assert _fit_option("KBOS", "trend_variance", default=False) is False


def test_fit_option_low_key_is_independent(monkeypatch):
    _patch_settings(monkeypatch, {
        "emos.station_overrides": {"KDFW": {"trend_variance": True}},
    })
    assert _fit_option("KDFW_low", "trend_variance", default=False) is False


def test_fit_option_falls_back_to_global(monkeypatch):
    _patch_settings(monkeypatch, {"emos.nonneg_weights": True})
    assert _fit_option("KDFW", "nonneg_weights", default=False) is True


# ── run-view banking ─────────────────────────────────────────────────────────


def test_record_run_views_dedups_same_day(tmp_path):
    t = date.today() + timedelta(days=1)
    _record_run_views("KTST", {t: 90.0}, training_dir=tmp_path)
    _record_run_views("KTST", {t: 91.5}, training_dir=tmp_path)  # re-run wins
    df = pd.read_parquet(_runview_path("KTST", tmp_path))
    assert len(df) == 1
    assert df.iloc[0]["mean_view_f"] == 91.5


# ── live trend diff (stubbed API) ────────────────────────────────────────────


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_urlopen_factory(views_by_model: dict[str, float]):
    """Fake Previous Runs response: tomorrow's hourly trace per model."""
    tomorrow = date.today() + timedelta(days=1)
    times = [f"{tomorrow}T{h:02d}:00" for h in range(24)]
    hourly = {"time": times}
    for om_id, peak in views_by_model.items():
        # Flat trace at the peak — daily max == peak.
        hourly[f"temperature_2m_{om_id}"] = [peak] * 24
    payload = json.dumps({"hourly": hourly}).encode()

    def fake_urlopen(req, timeout=None):
        return _FakeResponse(payload)

    return fake_urlopen


def test_trend_none_on_bootstrap_then_diff_next_day(tmp_path, monkeypatch):
    monkeypatch.setattr(et, "_TRAINING_DIR", tmp_path)
    tomorrow = date.today() + timedelta(days=1)
    views = {om: 95.0 for om in EMOS_MODELS.values()}
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen_factory(views))

    # Bootstrap day: nothing recorded before today -> None, but views banked.
    assert fetch_live_trend_f(0.0, 0.0, "America/Chicago", tomorrow, "KTST") is None
    banked = pd.read_parquet(_runview_path("KTST", tmp_path))
    assert banked.iloc[0]["mean_view_f"] == pytest.approx(95.0)

    # Simulate yesterday's session having recorded a 90.0 view of tomorrow.
    banked["recorded_date"] = date.today() - timedelta(days=1)
    banked["mean_view_f"] = 90.0
    banked.to_parquet(_runview_path("KTST", tmp_path), index=False)

    trend = fetch_live_trend_f(0.0, 0.0, "America/Chicago", tomorrow, "KTST")
    assert trend == pytest.approx(5.0)


def test_trend_ignores_todays_own_record(tmp_path, monkeypatch):
    # Two calls the same day (retry, manual re-run) must not diff run vs itself.
    monkeypatch.setattr(et, "_TRAINING_DIR", tmp_path)
    tomorrow = date.today() + timedelta(days=1)
    views = {om: 95.0 for om in EMOS_MODELS.values()}
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen_factory(views))

    assert fetch_live_trend_f(0.0, 0.0, "America/Chicago", tomorrow, "KTST") is None
    assert fetch_live_trend_f(0.0, 0.0, "America/Chicago", tomorrow, "KTST") is None
