"""Tests for research/flip_gate.py — the numeric trade-flip gate."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from deep_isobar.core.types import CityProfile
from deep_isobar.research.flip_gate import (
    evaluate_city,
    liquidity_stats,
    render_report,
)

CFG = {
    "min_calibration_days": 21,
    "max_crps_f": 1.00,
    "require_beats_nbm": True,
    "max_median_spread": 0.06,
    "min_two_sided_frac": 0.75,
    "min_series_volume_24h": 100.0,
    "min_track_trades": 10,
}


def _city(name: str = "Miami") -> CityProfile:
    return CityProfile(
        city=name, city_code="MIA", station_id="KMIA",
        timezone="America/New_York", settlement_source="NWS",
        variance_multiplier=1.0, mean_bias_correction_f=0.0,
        kde_bandwidth=5.5,
    )


GOOD_CALIB = {"n_days": 30, "crps": 0.60, "mae": 0.80, "nbm_mae": 2.11}
GOOD_LIQ = {
    "n_snapshots": 200, "n_quotes": 2400, "two_sided_frac": 0.92,
    "median_spread": 0.04, "median_volume_24h": 850.0,
}
NO_TRACK = {"n": 0, "pnl": 0.0}


def test_all_green_passes():
    r = evaluate_city(_city(), GOOD_CALIB, GOOD_LIQ, NO_TRACK, CFG)
    assert r.passed
    assert all(c.ok for c in r.criteria)


def test_missing_calibration_fails():
    r = evaluate_city(_city(), None, GOOD_LIQ, NO_TRACK, CFG)
    assert not r.passed
    assert any(c.name == "calibration" and not c.ok for c in r.criteria)


def test_high_crps_fails():
    calib = {**GOOD_CALIB, "crps": 1.21}
    r = evaluate_city(_city(), calib, GOOD_LIQ, NO_TRACK, CFG)
    assert not r.passed
    assert any(c.name == "crps" and not c.ok for c in r.criteria)


def test_losing_nbm_fails():
    calib = {**GOOD_CALIB, "mae": 2.50, "nbm_mae": 2.11}
    r = evaluate_city(_city(), calib, GOOD_LIQ, NO_TRACK, CFG)
    assert not r.passed
    assert any(c.name == "beats_nbm" and not c.ok for c in r.criteria)


def test_wide_spread_or_thin_volume_fails():
    r = evaluate_city(
        _city(), GOOD_CALIB, {**GOOD_LIQ, "median_spread": 0.09}, NO_TRACK, CFG
    )
    assert not r.passed
    r = evaluate_city(
        _city(), GOOD_CALIB, {**GOOD_LIQ, "median_volume_24h": 12.0}, NO_TRACK, CFG
    )
    assert not r.passed


def test_no_books_fails():
    r = evaluate_city(_city(), GOOD_CALIB, None, NO_TRACK, CFG)
    assert not r.passed
    assert any(c.name == "liquidity" and not c.ok for c in r.criteria)


def test_losing_track_record_blocks_reflip():
    # The LA case: calibration and liquidity both fine, paper book negative.
    track = {"n": 16, "pnl": -19.33}
    r = evaluate_city(_city("Los Angeles"), GOOD_CALIB, GOOD_LIQ, track, CFG)
    assert not r.passed
    assert any(c.name == "track_record" and not c.ok for c in r.criteria)


def test_thin_track_record_does_not_block():
    track = {"n": 3, "pnl": -2.0}  # below min_track_trades — not evidence
    r = evaluate_city(_city(), GOOD_CALIB, GOOD_LIQ, track, CFG)
    assert r.passed


def test_positive_track_record_passes():
    track = {"n": 40, "pnl": 120.0}
    r = evaluate_city(_city(), GOOD_CALIB, GOOD_LIQ, track, CFG)
    assert r.passed


def test_liquidity_stats_from_parquet(tmp_path):
    day = tmp_path / f"date={date(2026, 8, 22)}"
    day.mkdir()
    df = pd.DataFrame({
        "snapshot_utc": ["t1"] * 3 + ["t2"] * 3,
        "series": ["KXHIGHMIA"] * 6,
        "metric": ["high"] * 6,
        "best_bid": [0.10, 0.40, None, 0.12, 0.38, 0.02],
        "best_ask": [0.13, 0.44, 0.50, 0.15, 0.44, 0.05],
        "volume_24h": [100.0, 250.0, 0.0, 120.0, 260.0, 10.0],
    })
    df.to_parquet(day / "books_120000.parquet")

    stats = liquidity_stats(
        "KXHIGHMIA", history_dir=tmp_path, days=3, asof=date(2026, 8, 22)
    )
    assert stats is not None
    assert stats["n_snapshots"] == 2
    assert stats["two_sided_frac"] == pytest.approx(5 / 6)
    # spreads on two-sided rows: .03,.04,.03,.06,.03 → median .03
    assert stats["median_spread"] == pytest.approx(0.03)
    # per-snapshot volume: 350, 390 → median 370
    assert stats["median_volume_24h"] == pytest.approx(370.0)
    # other series / empty window → None
    assert liquidity_stats("KXHIGHCHI", history_dir=tmp_path, days=3,
                           asof=date(2026, 8, 22)) is None


def test_render_report_lists_passers_first():
    good = evaluate_city(_city("Austin"), GOOD_CALIB, GOOD_LIQ, NO_TRACK, CFG)
    bad = evaluate_city(_city("Zurich"), None, None, NO_TRACK, CFG)
    text = render_report([bad, good])
    assert "1 of 2 candidates pass" in text
    assert text.index("Austin") < text.index("Zurich")
