"""Tests for deep_isobar.data.ensemble_ingest (Open-Meteo responses mocked)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from deep_isobar.data import ensemble_ingest
from deep_isobar.data.ensemble_ingest import (
    fetch_member_daily_maxes,
    load_t1_spread_history,
    pooled_member_variance,
    record_t1_spread,
)


# ── pooled_member_variance ───────────────────────────────────────────────────


def test_pooled_variance_averages_within_model_variances():
    members = {
        "GEFS": [80.0, 82.0, 84.0],   # var = 4.0
        "EPS":  [79.0, 81.0],         # var = 2.0
    }
    assert pooled_member_variance(members) == pytest.approx(3.0)


def test_pooled_variance_skips_single_member_models():
    members = {"GEFS": [80.0], "EPS": [79.0, 81.0]}
    assert pooled_member_variance(members) == pytest.approx(2.0)


def test_pooled_variance_none_when_no_usable_model():
    assert pooled_member_variance({}) is None
    assert pooled_member_variance({"GEFS": [80.0]}) is None


def test_pooled_variance_excludes_cross_model_offset():
    """Two internally tight ensembles far apart must NOT read as spread."""
    members = {"GEFS": [80.0, 80.0, 80.0], "EPS": [90.0, 90.0, 90.0]}
    assert pooled_member_variance(members) == pytest.approx(0.0)


# ── fetch_member_daily_maxes ─────────────────────────────────────────────────


def _live_payload() -> dict:
    return {
        "daily": {
            "time": ["2026-06-30", "2026-07-01", "2026-07-02"],
            # GEFS: control + 2 members
            "temperature_2m_max_ncep_gefs025": [88.0, 90.0, 85.0],
            "temperature_2m_max_member01_ncep_gefs025": [88.5, 91.0, 84.0],
            "temperature_2m_max_member02_ncep_gefs025": [87.5, 89.0, 86.0],
            # EPS: control + 1 member, with a null on the target day
            "temperature_2m_max_ecmwf_ifs025_ensemble": [88.2, 90.5, 85.5],
            "temperature_2m_max_member01_ecmwf_ifs025_ensemble": [88.0, None, 85.0],
        }
    }


def test_fetch_member_daily_maxes_parses_members(monkeypatch):
    monkeypatch.setattr(ensemble_ingest, "_http_get_json", lambda url: _live_payload())
    out = fetch_member_daily_maxes(41.8, -87.8, "America/Chicago", date(2026, 7, 1))
    assert sorted(out["GEFS"]) == [89.0, 90.0, 91.0]
    assert out["EPS"] == [90.5]  # null member dropped


def test_fetch_member_daily_maxes_target_outside_range(monkeypatch):
    monkeypatch.setattr(ensemble_ingest, "_http_get_json", lambda url: _live_payload())
    with pytest.raises(ValueError, match="not in response"):
        fetch_member_daily_maxes(41.8, -87.8, "America/Chicago", date(2026, 7, 9))


def test_fetch_member_daily_maxes_ignores_unrequested_model_keys(monkeypatch):
    payload = _live_payload()
    payload["daily"]["temperature_2m_max_bogus_model"] = [70.0, 70.0, 70.0]
    monkeypatch.setattr(ensemble_ingest, "_http_get_json", lambda url: payload)
    out = fetch_member_daily_maxes(41.8, -87.8, "America/Chicago", date(2026, 7, 1))
    assert set(out) == {"GEFS", "EPS"}


# ── T+1 spread history (accumulate-forward) ─────────────────────────────────


def test_record_and_load_spread_history_roundtrip(tmp_path):
    record_t1_spread("KTST", date(2026, 7, 1), 3.5, spread_dir=tmp_path)
    record_t1_spread("KTST", date(2026, 7, 2), 4.25, spread_dir=tmp_path)
    df = load_t1_spread_history("KTST", spread_dir=tmp_path)
    assert list(df["date"]) == [date(2026, 7, 1), date(2026, 7, 2)]
    assert list(df["ens_spread_var"]) == [3.5, 4.25]


def test_record_same_date_overwrites(tmp_path):
    record_t1_spread("KTST", date(2026, 7, 1), 3.5, spread_dir=tmp_path)
    record_t1_spread("KTST", date(2026, 7, 1), 5.0, spread_dir=tmp_path)
    df = load_t1_spread_history("KTST", spread_dir=tmp_path)
    assert len(df) == 1
    assert df["ens_spread_var"].iloc[0] == 5.0


def test_load_missing_history_is_empty(tmp_path):
    df = load_t1_spread_history("KNOPE", spread_dir=tmp_path)
    assert df.empty
    assert list(df.columns) == ["date", "ens_spread_var"]


def test_history_is_per_station(tmp_path):
    record_t1_spread("KAAA", date(2026, 7, 1), 1.0, spread_dir=tmp_path)
    record_t1_spread("KBBB", date(2026, 7, 1), 2.0, spread_dir=tmp_path)
    assert load_t1_spread_history("KAAA", spread_dir=tmp_path)["ens_spread_var"].iloc[0] == 1.0
    assert load_t1_spread_history("KBBB", spread_dir=tmp_path)["ens_spread_var"].iloc[0] == 2.0
