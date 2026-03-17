"""Tests for deep_isobar.market.market_lag_detection.

Covers:
- significant shift + unchanged market -> True
- significant shift + small move below threshold -> True
- significant shift + move exactly at threshold -> False (boundary)
- significant shift + sufficient move -> False
- insignificant shift -> always False regardless of market move
- invalid market_probability_before (out of [0, 1])
- invalid market_probability_after (out of [0, 1])
- invalid min_expected_response (negative)
- zero min_expected_response (any non-zero market move satisfies it)
- default min_expected_response used when not supplied
"""

import pytest
from datetime import date, datetime, timezone

from deep_isobar.core.types import ForecastShiftEvent
from deep_isobar.market.market_lag_detection import detect_market_lag

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_TARGET = date(2026, 3, 18)
_RUN_PREV = datetime(2026, 3, 17, 0, 0, 0, tzinfo=timezone.utc)
_RUN_CURR = datetime(2026, 3, 17, 6, 0, 0, tzinfo=timezone.utc)


def _shift(significant: bool = True, shift_f: float = 5.0) -> ForecastShiftEvent:
    return ForecastShiftEvent(
        city="Chicago",
        model_name="GFS",
        metric="high_temp_f",
        previous_run_time_utc=_RUN_PREV,
        current_run_time_utc=_RUN_CURR,
        target_date=_TARGET,
        previous_value_f=72.0,
        current_value_f=72.0 + shift_f,
        shift_f=shift_f,
        absolute_shift_f=abs(shift_f),
        shift_direction="warmer" if shift_f >= 0 else "cooler",
        significant_shift_flag=significant,
    )


# ---------------------------------------------------------------------------
# Core decision logic
# ---------------------------------------------------------------------------


class TestDetectMarketLagDecisionLogic:
    def test_significant_shift_unchanged_market_returns_true(self):
        """Significant shift + market fully unchanged → lagging."""
        assert detect_market_lag(_shift(significant=True), 0.40, 0.40) is True

    def test_significant_shift_small_move_returns_true(self):
        """Significant shift + market moved less than threshold → lagging."""
        # response = 0.03, threshold = 0.05 → True
        assert detect_market_lag(_shift(significant=True), 0.40, 0.43, min_expected_response=0.05) is True

    def test_significant_shift_move_at_threshold_returns_false(self):
        """Market moved exactly at threshold → not lagging (response < threshold is False at equality)."""
        # Use 0.0 → 0.05: abs difference is exactly 0.05 in float (no representation drift)
        # 0.05 < 0.05 is False → lagging=False
        assert detect_market_lag(_shift(significant=True), 0.0, 0.05, min_expected_response=0.05) is False

    def test_significant_shift_move_above_threshold_returns_false(self):
        """Significant shift + market moved enough → not lagging."""
        # response = 0.10, threshold = 0.05 → False
        assert detect_market_lag(_shift(significant=True), 0.40, 0.50, min_expected_response=0.05) is False

    def test_insignificant_shift_unchanged_market_returns_false(self):
        """Insignificant forecast shift → always False regardless of market."""
        assert detect_market_lag(_shift(significant=False), 0.40, 0.40) is False

    def test_insignificant_shift_with_large_market_move_returns_false(self):
        """Insignificant shift → False even if market moved a lot."""
        assert detect_market_lag(_shift(significant=False), 0.30, 0.70) is False

    def test_insignificant_shift_with_tiny_market_move_returns_false(self):
        """Insignificant shift → always False."""
        assert detect_market_lag(_shift(significant=False), 0.50, 0.501) is False

    def test_market_moved_in_opposite_direction_still_counts(self):
        """Market response is absolute — direction does not matter."""
        # prob went from 0.50 to 0.44 → response=0.06 ≥ 0.05 → not lagging
        assert detect_market_lag(_shift(significant=True), 0.50, 0.44, min_expected_response=0.05) is False


# ---------------------------------------------------------------------------
# Default min_expected_response
# ---------------------------------------------------------------------------


class TestDefaultMinExpectedResponse:
    def test_default_is_0_05(self):
        """Default min_expected_response=0.05: move of 0.04 → True."""
        # Without kwarg the default 0.05 should apply
        result_with_default = detect_market_lag(_shift(), 0.40, 0.44)
        result_explicit = detect_market_lag(_shift(), 0.40, 0.44, min_expected_response=0.05)
        assert result_with_default == result_explicit

    def test_default_sufficient_move(self):
        """Default 0.05: move of 0.06 → False."""
        assert detect_market_lag(_shift(), 0.40, 0.46) is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_zero_min_expected_response_any_move_satisfies(self):
        """With threshold=0, even a 0.0001 move is sufficient → False."""
        assert detect_market_lag(_shift(significant=True), 0.50, 0.5001, min_expected_response=0.0) is False

    def test_zero_min_expected_response_no_move_not_lagging(self):
        """With threshold=0, no market move → response=0.0 < 0.0 is False → not lagging."""
        assert detect_market_lag(_shift(significant=True), 0.50, 0.50, min_expected_response=0.0) is False

    def test_probabilities_at_boundary_0(self):
        """Probabilities of exactly 0.0 are valid."""
        assert detect_market_lag(_shift(significant=True), 0.0, 0.0) is True

    def test_probabilities_at_boundary_1(self):
        """Probabilities of exactly 1.0 are valid."""
        assert detect_market_lag(_shift(significant=True), 1.0, 1.0) is True


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_market_probability_before_above_1_raises(self):
        """ValueError for market_probability_before > 1."""
        with pytest.raises(ValueError, match="market_probability_before"):
            detect_market_lag(_shift(), 1.1, 0.50)

    def test_market_probability_before_negative_raises(self):
        """ValueError for market_probability_before < 0."""
        with pytest.raises(ValueError, match="market_probability_before"):
            detect_market_lag(_shift(), -0.01, 0.50)

    def test_market_probability_after_above_1_raises(self):
        """ValueError for market_probability_after > 1."""
        with pytest.raises(ValueError, match="market_probability_after"):
            detect_market_lag(_shift(), 0.50, 1.1)

    def test_market_probability_after_negative_raises(self):
        """ValueError for market_probability_after < 0."""
        with pytest.raises(ValueError, match="market_probability_after"):
            detect_market_lag(_shift(), 0.50, -0.01)

    def test_negative_min_expected_response_raises(self):
        """ValueError for negative min_expected_response."""
        with pytest.raises(ValueError, match="min_expected_response"):
            detect_market_lag(_shift(), 0.40, 0.50, min_expected_response=-0.01)

    def test_zero_min_expected_response_is_valid(self):
        """min_expected_response=0.0 is a valid (if unusual) threshold."""
        detect_market_lag(_shift(), 0.40, 0.50, min_expected_response=0.0)  # must not raise
