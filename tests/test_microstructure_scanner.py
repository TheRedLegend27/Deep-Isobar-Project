"""Tests for deep_isobar.market.microstructure_scanner.

Covers:
- compute_spread: normal case, missing bid, missing ask, bid > ask,
  prices out of [0, 1]
- compute_liquidity_score: all fields present, partial fields, all absent,
  zero fields
- compute_staleness_seconds: normal, zero staleness, negative staleness raises,
  mismatched timezone awareness raises
- compute_microstructure_score: composite direction, deterministic outputs,
  missing bid/ask degrades spread sub-score instead of raising, fully stale
  market, high-quality market
"""

import pytest
from datetime import datetime, timezone, timedelta

from deep_isobar.core.types import OrderBookSnapshot
from deep_isobar.market.microstructure_scanner import (
    compute_spread,
    compute_liquidity_score,
    compute_staleness_seconds,
    compute_microstructure_score,
    MAX_SPREAD,
    MAX_STALENESS_SECONDS,
    LIQUIDITY_CAP,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 3, 17, 12, 0, 0, tzinfo=timezone.utc)
_TS = datetime(2026, 3, 17, 11, 0, 0, tzinfo=timezone.utc)  # 1 hour before _NOW


def _snap(
    bid=0.45,
    ask=0.55,
    bid_size=100.0,
    ask_size=100.0,
    volume_24h=5000.0,
    open_interest=10000.0,
    timestamp_utc=_TS,
) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        timestamp_utc=timestamp_utc,
        contract_id="TEST-CONTRACT",
        market_source="Kalshi",
        best_bid=bid,
        best_ask=ask,
        bid_size=bid_size,
        ask_size=ask_size,
        volume_24h=volume_24h,
        open_interest=open_interest,
    )


# ---------------------------------------------------------------------------
# compute_spread
# ---------------------------------------------------------------------------


class TestComputeSpread:
    def test_normal_spread(self):
        """Spread = ask - bid."""
        snap = _snap(bid=0.47, ask=0.53)
        assert compute_spread(snap) == pytest.approx(0.06)

    def test_zero_spread(self):
        """bid == ask → spread is 0.0."""
        snap = _snap(bid=0.50, ask=0.50)
        assert compute_spread(snap) == pytest.approx(0.0)

    def test_spread_at_boundary(self):
        """bid=0.0, ask=1.0 → maximum possible spread."""
        snap = _snap(bid=0.0, ask=1.0)
        assert compute_spread(snap) == pytest.approx(1.0)

    def test_missing_bid_raises(self):
        """ValueError when best_bid is None."""
        snap = _snap(bid=None, ask=0.55)
        with pytest.raises(ValueError, match="best_bid is None"):
            compute_spread(snap)

    def test_missing_ask_raises(self):
        """ValueError when best_ask is None."""
        snap = _snap(bid=0.45, ask=None)
        with pytest.raises(ValueError, match="best_ask is None"):
            compute_spread(snap)

    def test_bid_above_1_raises(self):
        """ValueError when best_bid > 1.0."""
        snap = _snap(bid=1.5, ask=1.8)
        with pytest.raises(ValueError, match="best_bid"):
            compute_spread(snap)

    def test_ask_above_1_raises(self):
        """ValueError when best_ask > 1.0."""
        snap = _snap(bid=0.45, ask=1.1)
        with pytest.raises(ValueError, match="best_ask"):
            compute_spread(snap)

    def test_negative_bid_raises(self):
        """ValueError when best_bid < 0."""
        snap = _snap(bid=-0.1, ask=0.5)
        with pytest.raises(ValueError, match="best_bid"):
            compute_spread(snap)

    def test_crossed_book_raises(self):
        """ValueError when best_bid > best_ask (crossed book)."""
        snap = _snap(bid=0.60, ask=0.50)
        with pytest.raises(ValueError, match="crossed"):
            compute_spread(snap)


# ---------------------------------------------------------------------------
# compute_liquidity_score
# ---------------------------------------------------------------------------


class TestComputeLiquidityScore:
    def test_all_fields_present(self):
        """Score = bid_size + ask_size + volume_24h*0.01 + open_interest*0.01."""
        snap = _snap(bid_size=100.0, ask_size=150.0, volume_24h=2000.0, open_interest=5000.0)
        expected = 100.0 + 150.0 + 2000.0 * 0.01 + 5000.0 * 0.01
        assert compute_liquidity_score(snap) == pytest.approx(expected)

    def test_only_bid_ask_size(self):
        """Missing volume and OI → only sizes contribute."""
        snap = OrderBookSnapshot(
            timestamp_utc=_TS, contract_id="X", market_source="K",
            best_bid=0.45, best_ask=0.55,
            bid_size=50.0, ask_size=75.0,
            volume_24h=None, open_interest=None,
        )
        assert compute_liquidity_score(snap) == pytest.approx(125.0)

    def test_all_liquidity_fields_absent(self):
        """All optional fields None → score is 0.0."""
        snap = OrderBookSnapshot(
            timestamp_utc=_TS, contract_id="X", market_source="K",
            best_bid=0.45, best_ask=0.55,
        )
        assert compute_liquidity_score(snap) == pytest.approx(0.0)

    def test_zero_values_give_zero(self):
        """Explicit zero for every field → score is 0.0."""
        snap = OrderBookSnapshot(
            timestamp_utc=_TS, contract_id="X", market_source="K",
            best_bid=0.45, best_ask=0.55,
            bid_size=0.0, ask_size=0.0, volume_24h=0.0, open_interest=0.0,
        )
        assert compute_liquidity_score(snap) == pytest.approx(0.0)

    def test_missing_bid_ask_does_not_crash(self):
        """Missing bid/ask prices do not affect liquidity score."""
        snap = OrderBookSnapshot(
            timestamp_utc=_TS, contract_id="X", market_source="K",
            best_bid=None, best_ask=None,
            bid_size=50.0, ask_size=50.0,
        )
        # Only sizes contribute
        assert compute_liquidity_score(snap) == pytest.approx(100.0)

    def test_higher_fields_higher_score(self):
        """Doubling all liquidity fields doubles the score."""
        snap_low = _snap(bid_size=50.0, ask_size=50.0, volume_24h=1000.0, open_interest=1000.0)
        snap_high = _snap(bid_size=100.0, ask_size=100.0, volume_24h=2000.0, open_interest=2000.0)
        assert compute_liquidity_score(snap_high) == pytest.approx(
            compute_liquidity_score(snap_low) * 2
        )


# ---------------------------------------------------------------------------
# compute_staleness_seconds
# ---------------------------------------------------------------------------


class TestComputeStalenessSeconds:
    def test_normal_staleness(self):
        """Staleness = seconds between snapshot and now."""
        snap = _snap(timestamp_utc=datetime(2026, 3, 17, 11, 50, 0, tzinfo=timezone.utc))
        now = datetime(2026, 3, 17, 12, 0, 0, tzinfo=timezone.utc)
        assert compute_staleness_seconds(snap, now) == pytest.approx(600.0)

    def test_zero_staleness(self):
        """now_utc == snapshot.timestamp_utc → staleness is 0."""
        snap = _snap(timestamp_utc=_NOW)
        assert compute_staleness_seconds(snap, _NOW) == pytest.approx(0.0)

    def test_one_hour_staleness(self):
        """Snapshot from 1 hour ago → 3600 seconds."""
        snap = _snap(timestamp_utc=_TS)
        assert compute_staleness_seconds(snap, _NOW) == pytest.approx(3600.0)

    def test_negative_staleness_raises(self):
        """now_utc earlier than snapshot → ValueError."""
        snap = _snap(timestamp_utc=_NOW)
        earlier = _NOW - timedelta(seconds=1)
        with pytest.raises(ValueError, match="earlier"):
            compute_staleness_seconds(snap, earlier)

    def test_mismatched_timezone_aware_naive_raises(self):
        """Mixing tz-aware and naive datetimes raises ValueError."""
        snap = _snap(timestamp_utc=_TS)  # tz-aware
        naive_now = datetime(2026, 3, 17, 12, 0, 0)  # naive
        with pytest.raises(ValueError, match="timezone"):
            compute_staleness_seconds(snap, naive_now)

    def test_both_naive_works(self):
        """Both naive datetimes are accepted."""
        ts_naive = datetime(2026, 3, 17, 11, 0, 0)
        now_naive = datetime(2026, 3, 17, 12, 0, 0)
        snap = OrderBookSnapshot(
            timestamp_utc=ts_naive, contract_id="X", market_source="K",
            best_bid=0.45, best_ask=0.55,
        )
        assert compute_staleness_seconds(snap, now_naive) == pytest.approx(3600.0)

    def test_fractional_seconds(self):
        """Sub-second precision is preserved."""
        ts = datetime(2026, 3, 17, 11, 59, 59, 500000, tzinfo=timezone.utc)
        now = datetime(2026, 3, 17, 12, 0, 0, tzinfo=timezone.utc)
        snap = OrderBookSnapshot(
            timestamp_utc=ts, contract_id="X", market_source="K",
            best_bid=0.45, best_ask=0.55,
        )
        assert compute_staleness_seconds(snap, now) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# compute_microstructure_score
# ---------------------------------------------------------------------------


class TestComputeMicrostructureScore:
    def test_returns_float_in_unit_interval(self):
        """Score is always in [0.0, 1.0]."""
        snap = _snap()
        score = compute_microstructure_score(snap, _NOW)
        assert 0.0 <= score <= 1.0

    def test_deterministic(self):
        """Same inputs always produce the same score."""
        snap = _snap()
        assert compute_microstructure_score(snap, _NOW) == compute_microstructure_score(snap, _NOW)

    def test_tight_spread_better_than_wide(self):
        """Tighter spread → higher composite score (healthier)."""
        snap_tight = _snap(bid=0.49, ask=0.51)  # spread=0.02
        snap_wide = _snap(bid=0.30, ask=0.70)   # spread=0.40
        score_tight = compute_microstructure_score(snap_tight, _NOW)
        score_wide = compute_microstructure_score(snap_wide, _NOW)
        assert score_tight > score_wide

    def test_fresh_better_than_stale(self):
        """Fresher snapshot → higher composite score."""
        ts_fresh = _NOW - timedelta(seconds=60)
        ts_stale = _NOW - timedelta(seconds=3500)
        snap_fresh = _snap(timestamp_utc=ts_fresh)
        snap_stale = _snap(timestamp_utc=ts_stale)
        score_fresh = compute_microstructure_score(snap_fresh, _NOW)
        score_stale = compute_microstructure_score(snap_stale, _NOW)
        assert score_fresh > score_stale

    def test_liquid_better_than_illiquid(self):
        """Higher liquidity → higher composite score."""
        snap_liquid = _snap(bid_size=500.0, ask_size=500.0, volume_24h=50000.0)
        snap_illiquid = OrderBookSnapshot(
            timestamp_utc=_TS, contract_id="X", market_source="K",
            best_bid=0.45, best_ask=0.55,
        )
        score_liquid = compute_microstructure_score(snap_liquid, _NOW)
        score_illiquid = compute_microstructure_score(snap_illiquid, _NOW)
        assert score_liquid > score_illiquid

    def test_missing_bid_ask_degrades_spread_score(self):
        """Missing bid/ask → spread_score treated as 0 (not a raised exception)."""
        snap = OrderBookSnapshot(
            timestamp_utc=_TS, contract_id="X", market_source="K",
            best_bid=None, best_ask=None,
            bid_size=100.0, ask_size=100.0,
        )
        score = compute_microstructure_score(snap, _NOW)
        # spread_score = 0; other sub-scores may be > 0
        assert 0.0 <= score <= 1.0

    def test_fully_stale_market_penalised(self):
        """Snapshot at MAX_STALENESS_SECONDS age → freshness_score == 0."""
        ts_very_old = _NOW - timedelta(seconds=MAX_STALENESS_SECONDS)
        snap = _snap(bid=0.49, ask=0.51, timestamp_utc=ts_very_old)
        score = compute_microstructure_score(snap, _NOW)
        # freshness_score == 0 → composite can be at most 2/3
        assert score <= (2.0 / 3.0) + 1e-9

    def test_spread_at_max_gives_zero_spread_sub_score(self):
        """Spread == MAX_SPREAD → spread_score == 0."""
        snap = _snap(bid=0.0, ask=MAX_SPREAD, timestamp_utc=_NOW)
        # With staleness=0, freshness_score=1; spread_score=0
        score = compute_microstructure_score(snap, _NOW)
        # score = (0 + 1 + liquidity_norm) / 3
        liq_norm = min(compute_liquidity_score(snap) / LIQUIDITY_CAP, 1.0)
        expected = (0.0 + 1.0 + liq_norm) / 3.0
        assert score == pytest.approx(expected)

    def test_hand_calculated_composite(self):
        """Verify composite formula against a manually worked example.

        Setup:
        - spread = 0.10  → spread_score = 1 - 0.10/0.50 = 0.80
        - staleness = 360s → freshness_score = 1 - 360/3600 = 0.90
        - liquidity = 0 (no size/volume/OI) → liquidity_norm = 0.0
        - composite = (0.80 + 0.90 + 0.0) / 3 ≈ 0.5667
        """
        ts = _NOW - timedelta(seconds=360)
        snap = OrderBookSnapshot(
            timestamp_utc=ts, contract_id="X", market_source="K",
            best_bid=0.45, best_ask=0.55,
        )
        score = compute_microstructure_score(snap, _NOW)
        assert score == pytest.approx((0.80 + 0.90 + 0.0) / 3.0, abs=1e-6)

    def test_now_earlier_than_snapshot_raises(self):
        """now_utc before snapshot.timestamp_utc raises ValueError."""
        snap = _snap(timestamp_utc=_NOW)
        with pytest.raises(ValueError):
            compute_microstructure_score(snap, _NOW - timedelta(seconds=1))

    def test_crossed_book_raises_not_silently_swallowed(self):
        """Crossed book (bid > ask) propagates ValueError rather than being masked."""
        snap = _snap(bid=0.60, ask=0.50)  # crossed: bid > ask
        with pytest.raises(ValueError, match="crossed"):
            compute_microstructure_score(snap, _NOW)
