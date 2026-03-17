"""Tests for deep_isobar.models.probability_engine.

Covers:
- probability_ge_normal: bounds, known values, symmetry, edge cases
- probability_le_normal: bounds, known values, symmetry, edge cases
- Complementary relationship: ge(t) + le(t) == 1.0 at any threshold
- ValueError for std_f <= 0
"""

import pytest

from deep_isobar.models.probability_engine import (
    probability_ge_normal,
    probability_le_normal,
)


# ---------------------------------------------------------------------------
# probability_ge_normal
# ---------------------------------------------------------------------------


def test_ge_normal_at_mean_is_half():
    """P(T >= mean) should be exactly 0.5 for a symmetric distribution."""
    assert probability_ge_normal(mean_f=75.0, std_f=5.0, threshold_f=75.0) == pytest.approx(0.5)


def test_ge_normal_above_mean_less_than_half():
    """Threshold above the mean → probability less than 0.5."""
    p = probability_ge_normal(mean_f=75.0, std_f=5.0, threshold_f=80.0)
    assert p < 0.5


def test_ge_normal_below_mean_greater_than_half():
    """Threshold below the mean → probability greater than 0.5."""
    p = probability_ge_normal(mean_f=75.0, std_f=5.0, threshold_f=70.0)
    assert p > 0.5


def test_ge_normal_known_value():
    """P(T >= 85 | μ=84, σ=2) ≈ 0.308 (from standard normal table)."""
    p = probability_ge_normal(mean_f=84.0, std_f=2.0, threshold_f=85.0)
    assert p == pytest.approx(0.3085, abs=1e-3)


def test_ge_normal_extreme_high_threshold_near_zero():
    """A threshold far above the mean should yield probability near 0."""
    p = probability_ge_normal(mean_f=75.0, std_f=5.0, threshold_f=150.0)
    assert p < 0.001


def test_ge_normal_extreme_low_threshold_near_one():
    """A threshold far below the mean should yield probability near 1."""
    p = probability_ge_normal(mean_f=75.0, std_f=5.0, threshold_f=0.0)
    assert p > 0.999


def test_ge_normal_output_bounds():
    """Output must always be in [0.0, 1.0]."""
    p = probability_ge_normal(mean_f=84.0, std_f=2.0, threshold_f=85.0)
    assert 0.0 <= p <= 1.0


def test_ge_normal_raises_on_zero_std():
    """std_f == 0 must raise ValueError."""
    with pytest.raises(ValueError, match="std_f"):
        probability_ge_normal(mean_f=75.0, std_f=0.0, threshold_f=75.0)


def test_ge_normal_raises_on_negative_std():
    """Negative std_f must raise ValueError."""
    with pytest.raises(ValueError, match="std_f"):
        probability_ge_normal(mean_f=75.0, std_f=-3.0, threshold_f=75.0)


# ---------------------------------------------------------------------------
# probability_le_normal
# ---------------------------------------------------------------------------


def test_le_normal_at_mean_is_half():
    """P(T <= mean) should be exactly 0.5 for a symmetric distribution."""
    assert probability_le_normal(mean_f=75.0, std_f=5.0, threshold_f=75.0) == pytest.approx(0.5)


def test_le_normal_above_mean_greater_than_half():
    """Threshold above the mean → le probability greater than 0.5."""
    p = probability_le_normal(mean_f=75.0, std_f=5.0, threshold_f=80.0)
    assert p > 0.5


def test_le_normal_below_mean_less_than_half():
    """Threshold below the mean → le probability less than 0.5."""
    p = probability_le_normal(mean_f=75.0, std_f=5.0, threshold_f=70.0)
    assert p < 0.5


def test_le_normal_known_value():
    """P(T <= 85 | μ=84, σ=2) ≈ 0.692."""
    p = probability_le_normal(mean_f=84.0, std_f=2.0, threshold_f=85.0)
    assert p == pytest.approx(0.6915, abs=1e-3)


def test_le_normal_output_bounds():
    """Output must always be in [0.0, 1.0]."""
    p = probability_le_normal(mean_f=84.0, std_f=2.0, threshold_f=85.0)
    assert 0.0 <= p <= 1.0


def test_le_normal_raises_on_zero_std():
    """std_f == 0 must raise ValueError."""
    with pytest.raises(ValueError, match="std_f"):
        probability_le_normal(mean_f=75.0, std_f=0.0, threshold_f=75.0)


def test_le_normal_raises_on_negative_std():
    """Negative std_f must raise ValueError."""
    with pytest.raises(ValueError, match="std_f"):
        probability_le_normal(mean_f=75.0, std_f=-1.0, threshold_f=75.0)


# ---------------------------------------------------------------------------
# Complementary relationship
# ---------------------------------------------------------------------------


def test_ge_and_le_sum_to_one():
    """ge(t) + le(t) must equal 1.0 at any threshold (continuous distribution)."""
    mean_f, std_f, threshold_f = 80.0, 4.0, 83.0
    ge = probability_ge_normal(mean_f, std_f, threshold_f)
    le = probability_le_normal(mean_f, std_f, threshold_f)
    assert ge + le == pytest.approx(1.0, abs=1e-9)


def test_ge_and_le_sum_at_mean():
    """ge(mean) + le(mean) == 1.0."""
    mean_f, std_f = 72.0, 3.5
    ge = probability_ge_normal(mean_f, std_f, mean_f)
    le = probability_le_normal(mean_f, std_f, mean_f)
    assert ge + le == pytest.approx(1.0, abs=1e-9)


def test_ge_symmetry():
    """P(T >= mean + d) == P(T <= mean - d) for a symmetric distribution."""
    mean_f, std_f, d = 75.0, 5.0, 3.0
    ge = probability_ge_normal(mean_f, std_f, mean_f + d)
    le = probability_le_normal(mean_f, std_f, mean_f - d)
    assert ge == pytest.approx(le, abs=1e-9)
