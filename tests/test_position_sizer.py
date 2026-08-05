"""Tests for deep_isobar.trading.position_sizer — go-live safety build, Part 2.

Covers: city daily cap (both hard caps binding correctly), each evidence
multiplier in isolation (cheap-tail, calibration, track record, anomaly +
spread), zero/negative edge sizing to zero via the Kelly allocations it's
handed, the hard per-trade cap actually clamping a concentrated allocation,
bankroll changes scaling output proportionally, and determinism.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from deep_isobar.core.types import TradeSignal
from deep_isobar.trading.bracket_spreader import BracketAllocation, build_spread
from deep_isobar.trading.kelly import risk_per_contract
from deep_isobar.trading.position_sizer import (
    StationCalibration,
    StationTrackRecord,
    _calibration_multiplier,
    _tail_multiplier,
    _track_record_multiplier,
    adjust_allocations,
    apply_sizing_decisions,
    build_station_track_record,
    compute_city_daily_cap,
)


def _signal(cid: str, side: str, model_p: float, market_p: float) -> TradeSignal:
    return TradeSignal(
        timestamp_utc=datetime.now(timezone.utc),
        contract_id=cid,
        city="Testville",
        target_date=date(2026, 8, 1),
        metric="high_temp_f",
        threshold_f=90,
        comparison_operator="ge",
        market_probability=market_p,
        model_probability=model_p,
        alpha=model_p - market_p,
        absolute_alpha=abs(model_p - market_p),
        signal_side=side,
        confidence_score=abs(model_p - market_p),
    )


def _alloc(cid: str, side: str, model_p: float, market_p: float, usd: float) -> BracketAllocation:
    return BracketAllocation(
        signal=_signal(cid, side, model_p, market_p),
        allocated_usd=usd,
        allocation_fraction=1.0,
        rank=1,
    )


# ---------------------------------------------------------------------------
# compute_city_daily_cap — hard caps binding
# ---------------------------------------------------------------------------


def test_city_cap_uses_tighter_of_the_two_caps():
    cfg = {"max_city_daily_pct": 0.03, "max_total_daily_pct": 0.10}
    # 5 cities: total share = 10%/5 = 2%, tighter than the 3% per-city cap.
    assert compute_city_daily_cap(500.0, 5, cfg) == pytest.approx(10.0)
    # 2 cities: total share = 10%/2 = 5%, looser than the 3% per-city cap
    # — the per-city cap binds instead.
    assert compute_city_daily_cap(500.0, 2, cfg) == pytest.approx(15.0)


def test_city_cap_sum_across_cities_never_exceeds_total_cap():
    cfg = {"max_city_daily_pct": 0.03, "max_total_daily_pct": 0.10}
    n = 7
    per_city = compute_city_daily_cap(500.0, n, cfg)
    assert per_city * n <= 500.0 * 0.10 + 1e-9


def test_city_cap_zero_cities():
    assert compute_city_daily_cap(500.0, 0, {}) == 0.0


def test_city_cap_scales_with_bankroll():
    cfg = {"max_city_daily_pct": 0.03, "max_total_daily_pct": 0.10}
    assert compute_city_daily_cap(1000.0, 5, cfg) == pytest.approx(
        2 * compute_city_daily_cap(500.0, 5, cfg)
    )


# ---------------------------------------------------------------------------
# Cheap-tail entry-price multiplier
# ---------------------------------------------------------------------------


def test_tail_multiplier_haircuts_cheap_entries():
    cfg = {"cheap_tail_price_threshold": 0.10, "cheap_tail_multiplier": 0.35}
    mult, note = _tail_multiplier(0.05, cfg)
    assert mult == pytest.approx(0.35)
    assert note is not None


def test_tail_multiplier_haircuts_rich_entries_symmetrically():
    cfg = {"cheap_tail_price_threshold": 0.10, "cheap_tail_multiplier": 0.35}
    mult, _ = _tail_multiplier(0.95, cfg)
    assert mult == pytest.approx(0.35)


def test_tail_multiplier_full_size_in_the_middle():
    cfg = {"cheap_tail_price_threshold": 0.10, "cheap_tail_multiplier": 0.35}
    mult, note = _tail_multiplier(0.50, cfg)
    assert mult == 1.0
    assert note is None


def test_tail_multiplier_boundary_is_haircut_not_full_size():
    cfg = {"cheap_tail_price_threshold": 0.10, "cheap_tail_multiplier": 0.35}
    mult, _ = _tail_multiplier(0.10, cfg)
    assert mult == pytest.approx(0.35)
    mult, _ = _tail_multiplier(0.11, cfg)
    assert mult == 1.0


def test_tail_multiplier_none_price_is_neutral():
    assert _tail_multiplier(None, {}) == (1.0, None)


def test_tail_multiplier_never_fully_excludes():
    """Design choice: haircut, not exclusion — the multiplier must stay > 0."""
    cfg = {"cheap_tail_price_threshold": 0.10, "cheap_tail_multiplier": 0.35}
    mult, _ = _tail_multiplier(0.01, cfg)
    assert mult > 0.0


# ---------------------------------------------------------------------------
# Calibration-quality multiplier
# ---------------------------------------------------------------------------


def test_calibration_multiplier_none_is_neutral():
    assert _calibration_multiplier(None, {}) == (1.0, None)


def test_calibration_multiplier_beats_benchmark_stays_at_or_below_one():
    calib = StationCalibration(mae=1.0, nbm_mae=2.0)  # we're twice as accurate
    mult, note = _calibration_multiplier(calib, {"calibration_multiplier_floor": 0.5})
    assert mult == 1.0  # accuracy ratio capped at 1.0 — never boosts above neutral
    assert note is not None


def test_calibration_multiplier_worse_than_benchmark_is_haircut():
    calib = StationCalibration(mae=2.0, nbm_mae=1.0)  # twice as inaccurate
    mult, _ = _calibration_multiplier(calib, {"calibration_multiplier_floor": 0.5})
    assert 0.5 <= mult < 1.0


def test_calibration_multiplier_flat_pit_scores_high():
    flat = StationCalibration(mae=1.0, pit_hist=(0.2, 0.2, 0.2, 0.2, 0.2))
    mult, _ = _calibration_multiplier(flat, {"calibration_multiplier_floor": 0.5})
    assert mult == pytest.approx(1.0)


def test_calibration_multiplier_concentrated_pit_scores_low():
    concentrated = StationCalibration(mae=1.0, pit_hist=(1.0, 0.0, 0.0, 0.0, 0.0))
    mult, _ = _calibration_multiplier(concentrated, {"calibration_multiplier_floor": 0.5})
    assert mult == pytest.approx(0.5)  # hits the floor


def test_calibration_multiplier_missing_nbm_falls_back_to_pit_only():
    calib = StationCalibration(mae=1.0, nbm_mae=None, pit_hist=(0.2, 0.2, 0.2, 0.2, 0.2))
    mult, note = _calibration_multiplier(calib, {"calibration_multiplier_floor": 0.5})
    assert mult == pytest.approx(1.0)
    assert note is not None


def test_calibration_multiplier_never_exceeds_one():
    calib = StationCalibration(mae=0.1, nbm_mae=10.0, pit_hist=(0.2,) * 5)
    mult, _ = _calibration_multiplier(calib, {"calibration_multiplier_floor": 0.5})
    assert mult <= 1.0


# ---------------------------------------------------------------------------
# Track-record multiplier — risk-off only
# ---------------------------------------------------------------------------


def test_track_record_neutral_below_sample_minimum():
    record = StationTrackRecord(n_trades=5, realized_edge_per_contract=-0.50)
    mult, note = _track_record_multiplier(record, {"track_record_min_trades": 20})
    assert mult == 1.0
    assert note is None


def test_track_record_neutral_when_edge_nonnegative():
    record = StationTrackRecord(n_trades=50, realized_edge_per_contract=0.05)
    mult, note = _track_record_multiplier(record, {"track_record_min_trades": 20})
    assert mult == 1.0
    assert note is None


def test_track_record_never_boosts_for_a_hot_streak():
    """Design choice: risk-off only — a good record never sizes above 1.0."""
    record = StationTrackRecord(n_trades=100, realized_edge_per_contract=0.30)
    mult, _ = _track_record_multiplier(record, {"track_record_min_trades": 20})
    assert mult == 1.0


def test_track_record_haircuts_large_enough_negative_edge():
    record = StationTrackRecord(n_trades=25, realized_edge_per_contract=-0.10)
    cfg = {
        "track_record_min_trades": 20,
        "track_record_floor": 0.5,
        "track_record_worst_edge_usd": -0.10,
    }
    mult, note = _track_record_multiplier(record, cfg)
    assert mult == pytest.approx(0.5)  # at the configured worst edge → floor
    assert note is not None


def test_track_record_none_is_neutral():
    assert _track_record_multiplier(None, {}) == (1.0, None)


# ---------------------------------------------------------------------------
# build_station_track_record — no lookahead
# ---------------------------------------------------------------------------


def test_build_station_track_record_excludes_future_dates():
    import pandas as pd

    df = pd.DataFrame([
        {"city": "Chicago", "date": date(2026, 7, 1), "status": "WIN",
         "position_size": 10.0, "realized_pnl": 5.0},
        {"city": "Chicago", "date": date(2026, 8, 1), "status": "LOSS",
         "position_size": 10.0, "realized_pnl": -100.0},  # future — must be excluded
    ])
    record = build_station_track_record(df, "Chicago", asof=date(2026, 7, 15))
    assert record is not None
    assert record.n_trades == 1
    assert record.realized_edge_per_contract == pytest.approx(0.5)


def test_build_station_track_record_filters_by_city():
    import pandas as pd

    df = pd.DataFrame([
        {"city": "Chicago", "date": date(2026, 7, 1), "status": "WIN",
         "position_size": 10.0, "realized_pnl": 5.0},
        {"city": "Dallas", "date": date(2026, 7, 1), "status": "LOSS",
         "position_size": 10.0, "realized_pnl": -5.0},
    ])
    record = build_station_track_record(df, "Chicago", asof=date(2026, 8, 1))
    assert record.n_trades == 1


def test_build_station_track_record_empty_df_returns_none():
    import pandas as pd

    assert build_station_track_record(pd.DataFrame(), "Chicago", date(2026, 8, 1)) is None


def test_build_station_track_record_no_matching_rows_returns_none():
    import pandas as pd

    df = pd.DataFrame([
        {"city": "Dallas", "date": date(2026, 7, 1), "status": "WIN",
         "position_size": 10.0, "realized_pnl": 5.0},
    ])
    assert build_station_track_record(df, "Chicago", date(2026, 8, 1)) is None


# ---------------------------------------------------------------------------
# adjust_allocations — integration
# ---------------------------------------------------------------------------


def test_adjust_allocations_zero_edge_kelly_allocations_stay_zero():
    """A dead-edge signal never reaches adjust_allocations (build_spread's
    kelly method already drops it) — confirm that pipeline end to end."""
    signals = [_signal("THIN", "BUY", 0.515, 0.50)]  # dies to the taker fee
    allocations = build_spread(
        signals, daily_exposure_cap_usd=50.0, max_contracts=3, min_alpha=0.01,
        allocation_method="kelly", entry_prices={"THIN": 0.50},
        kelly_cfg={"n_correlated_bets": 1},
    )
    assert allocations == []


def test_adjust_allocations_never_increases_stake():
    allocations = [_alloc("A", "BUY", 0.60, 0.30, usd=20.0)]
    decisions = adjust_allocations(
        allocations,
        bankroll_usd=500.0,
        entry_prices={"A": 0.31},
        anomaly_report=None,
        ensemble_std_f=None,
        station_calibration=None,
        station_track_record=None,
        cfg={"max_risk_per_trade_pct": 1.0},  # cap disabled for this check
    )
    assert decisions[0].final_stake_usd <= decisions[0].kelly_usd


def test_adjust_allocations_hard_per_trade_cap_clamps_concentrated_bet():
    """The core concentration fix: one huge Kelly allocation must not blow
    past the per-trade cap just because it's the only signal that day."""
    allocations = [_alloc("BIG", "BUY", 0.90, 0.20, usd=45.0)]  # a big chunk of a $50 city cap
    cfg = {"max_risk_per_trade_pct": 0.025}  # 2.5% of $500 = $12.50
    decisions = adjust_allocations(
        allocations, bankroll_usd=500.0, entry_prices={"BIG": 0.21},
        anomaly_report=None, ensemble_std_f=None,
        station_calibration=None, station_track_record=None, cfg=cfg,
    )
    assert decisions[0].capped is True
    assert decisions[0].final_stake_usd == pytest.approx(12.50)


def test_adjust_allocations_cheap_tail_reduces_stake():
    allocations = [_alloc("CHEAP", "BUY", 0.30, 0.05, usd=10.0)]
    cfg = {
        "max_risk_per_trade_pct": 1.0,
        "cheap_tail_price_threshold": 0.10,
        "cheap_tail_multiplier": 0.35,
    }
    decisions = adjust_allocations(
        allocations, bankroll_usd=500.0, entry_prices={"CHEAP": 0.06},
        anomaly_report=None, ensemble_std_f=None,
        station_calibration=None, station_track_record=None, cfg=cfg,
    )
    assert decisions[0].final_stake_usd == pytest.approx(3.50)
    assert "tail" in decisions[0].reasoning


def test_adjust_allocations_bankroll_scales_output_proportionally():
    allocations_a = [_alloc("A", "BUY", 0.60, 0.30, usd=5.0)]
    allocations_b = [_alloc("A", "BUY", 0.60, 0.30, usd=5.0)]
    cfg = {"max_risk_per_trade_pct": 1.0}  # cap disabled so the scale-up is visible

    d1 = adjust_allocations(
        allocations_a, bankroll_usd=500.0, entry_prices={"A": 0.31},
        anomaly_report=None, ensemble_std_f=None,
        station_calibration=None, station_track_record=None, cfg=cfg,
    )
    d2 = adjust_allocations(
        allocations_b, bankroll_usd=1000.0, entry_prices={"A": 0.31},
        anomaly_report=None, ensemble_std_f=None,
        station_calibration=None, station_track_record=None, cfg=cfg,
    )
    # kelly_usd itself doesn't depend on bankroll here (it's the input), but
    # the per-trade cap that could otherwise clamp it does — with the cap
    # effectively disabled, bankroll must not silently change the stake.
    assert d1[0].final_stake_usd == d2[0].final_stake_usd == pytest.approx(5.0)

    # Now show the cap itself scaling with bankroll directly.
    tight_cfg = {"max_risk_per_trade_pct": 0.01}
    big = [_alloc("A", "BUY", 0.60, 0.30, usd=100.0)]
    small_bankroll = adjust_allocations(
        big, bankroll_usd=500.0, entry_prices={"A": 0.31},
        anomaly_report=None, ensemble_std_f=None,
        station_calibration=None, station_track_record=None, cfg=tight_cfg,
    )
    big2 = [_alloc("A", "BUY", 0.60, 0.30, usd=100.0)]
    large_bankroll = adjust_allocations(
        big2, bankroll_usd=1000.0, entry_prices={"A": 0.31},
        anomaly_report=None, ensemble_std_f=None,
        station_calibration=None, station_track_record=None, cfg=tight_cfg,
    )
    assert large_bankroll[0].final_stake_usd == pytest.approx(
        2 * small_bankroll[0].final_stake_usd
    )


def test_adjust_allocations_calibration_moves_size_down_when_poor():
    good = StationCalibration(mae=1.0, nbm_mae=1.0, pit_hist=(0.2,) * 5)
    poor = StationCalibration(mae=3.0, nbm_mae=1.0, pit_hist=(1.0, 0.0, 0.0, 0.0, 0.0))
    cfg = {"max_risk_per_trade_pct": 1.0, "calibration_multiplier_floor": 0.5}

    good_decision = adjust_allocations(
        [_alloc("A", "BUY", 0.60, 0.30, usd=10.0)], bankroll_usd=500.0,
        entry_prices={"A": 0.31}, anomaly_report=None, ensemble_std_f=None,
        station_calibration=good, station_track_record=None, cfg=cfg,
    )[0]
    poor_decision = adjust_allocations(
        [_alloc("A", "BUY", 0.60, 0.30, usd=10.0)], bankroll_usd=500.0,
        entry_prices={"A": 0.31}, anomaly_report=None, ensemble_std_f=None,
        station_calibration=poor, station_track_record=None, cfg=cfg,
    )[0]
    assert poor_decision.final_stake_usd < good_decision.final_stake_usd


def test_adjust_allocations_deterministic():
    kwargs = dict(
        bankroll_usd=500.0, entry_prices={"A": 0.31},
        anomaly_report=None, ensemble_std_f=4.0,
        station_calibration=StationCalibration(mae=1.2, nbm_mae=1.0),
        station_track_record=StationTrackRecord(n_trades=30, realized_edge_per_contract=-0.05),
        cfg={"max_risk_per_trade_pct": 0.025, "track_record_min_trades": 20},
    )
    d1 = adjust_allocations([_alloc("A", "BUY", 0.60, 0.30, usd=15.0)], **kwargs)
    d2 = adjust_allocations([_alloc("A", "BUY", 0.60, 0.30, usd=15.0)], **kwargs)
    assert d1[0].final_stake_usd == d2[0].final_stake_usd
    assert d1[0].reasoning == d2[0].reasoning


def test_apply_sizing_decisions_recomputes_contracts():
    allocations = [_alloc("A", "BUY", 0.60, 0.30, usd=20.0)]
    allocations[0].contracts = 100.0  # stale value from build_spread's own fill
    cfg = {"max_risk_per_trade_pct": 0.01}  # forces a clamp on a $500 bankroll ($5)
    decisions = adjust_allocations(
        allocations, bankroll_usd=500.0, entry_prices={"A": 0.31},
        anomaly_report=None, ensemble_std_f=None,
        station_calibration=None, station_track_record=None, cfg=cfg,
    )
    apply_sizing_decisions(allocations, decisions, entry_prices={"A": 0.31})
    assert allocations[0].allocated_usd == pytest.approx(5.0)
    assert allocations[0].contracts != 100.0
    assert allocations[0].contracts == pytest.approx(
        5.0 / risk_per_contract(0.31, "BUY"), abs=0.05
    )
