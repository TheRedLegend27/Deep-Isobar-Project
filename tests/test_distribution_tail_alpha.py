"""Tests for deep_isobar.trading.distribution_tail_alpha.

Covers:
- is_tail_threshold: tail detected (True)
- is_tail_threshold: not a tail (False)
- is_tail_threshold: threshold exactly at the cutoff boundary (True)
- is_tail_threshold: invalid adjusted_std_f raises ValueError
- is_tail_threshold: invalid (negative) z_score_cutoff raises ValueError
- compute_tail_alpha_boost: tail_flag False returns abs(alpha)
- compute_tail_alpha_boost: tail_flag True returns abs(alpha) * multiplier
- compute_tail_alpha_boost: alpha = 0 returns 0
- compute_tail_alpha_boost: invalid (negative) tail_multiplier raises ValueError
"""

import pytest
from deep_isobar.trading.distribution_tail_alpha import (
    compute_tail_alpha_boost,
    is_tail_threshold,
)


# ---------------------------------------------------------------------------
# is_tail_threshold — tail detected
# ---------------------------------------------------------------------------


def test_is_tail_threshold_true_above_cutoff():
    """Threshold far from the mean (z >> cutoff) → True."""
    # mean=45, threshold=20, std=8  → z = |20-45|/8 = 3.125 ≥ 1.5
    assert is_tail_threshold(
        ensemble_mean_f=45.0,
        threshold_f=20,
        adjusted_std_f=8.0,
        z_score_cutoff=1.5,
    ) is True


def test_is_tail_threshold_true_negative_direction():
    """Tail can be above the mean too (positive z direction)."""
    # mean=45, threshold=60, std=5  → z = |60-45|/5 = 3.0 ≥ 1.5
    assert is_tail_threshold(
        ensemble_mean_f=45.0,
        threshold_f=60,
        adjusted_std_f=5.0,
        z_score_cutoff=1.5,
    ) is True


# ---------------------------------------------------------------------------
# is_tail_threshold — not a tail
# ---------------------------------------------------------------------------


def test_is_tail_threshold_false_close_to_mean():
    """Threshold near the mean (z < cutoff) → False."""
    # mean=45, threshold=47, std=8  → z = 2/8 = 0.25 < 1.5
    assert is_tail_threshold(
        ensemble_mean_f=45.0,
        threshold_f=47,
        adjusted_std_f=8.0,
        z_score_cutoff=1.5,
    ) is False


def test_is_tail_threshold_false_threshold_equals_mean():
    """Threshold exactly at the mean → z = 0 → False."""
    assert is_tail_threshold(
        ensemble_mean_f=45.0,
        threshold_f=45,
        adjusted_std_f=8.0,
    ) is False


# ---------------------------------------------------------------------------
# is_tail_threshold — exact cutoff boundary
# ---------------------------------------------------------------------------


def test_is_tail_threshold_exact_cutoff_boundary():
    """Threshold at exactly z = cutoff → True (>= is inclusive)."""
    # mean=40, std=10, cutoff=1.5 → boundary threshold = 40 + 1.5*10 = 55
    assert is_tail_threshold(
        ensemble_mean_f=40.0,
        threshold_f=55,
        adjusted_std_f=10.0,
        z_score_cutoff=1.5,
    ) is True


def test_is_tail_threshold_just_below_cutoff():
    """One degree inside the cutoff → False."""
    # z = |54 - 40| / 10 = 1.4 < 1.5
    assert is_tail_threshold(
        ensemble_mean_f=40.0,
        threshold_f=54,
        adjusted_std_f=10.0,
        z_score_cutoff=1.5,
    ) is False


# ---------------------------------------------------------------------------
# is_tail_threshold — cutoff = 0
# ---------------------------------------------------------------------------


def test_is_tail_threshold_zero_cutoff_always_true():
    """z_score_cutoff=0 means every threshold is a tail (z >= 0 always)."""
    assert is_tail_threshold(
        ensemble_mean_f=45.0,
        threshold_f=45,
        adjusted_std_f=8.0,
        z_score_cutoff=0.0,
    ) is True


# ---------------------------------------------------------------------------
# is_tail_threshold — validation errors
# ---------------------------------------------------------------------------


def test_is_tail_threshold_invalid_std_zero():
    """adjusted_std_f = 0 raises ValueError."""
    with pytest.raises(ValueError, match="adjusted_std_f"):
        is_tail_threshold(
            ensemble_mean_f=45.0,
            threshold_f=30,
            adjusted_std_f=0.0,
        )


def test_is_tail_threshold_invalid_std_negative():
    """adjusted_std_f < 0 raises ValueError."""
    with pytest.raises(ValueError, match="adjusted_std_f"):
        is_tail_threshold(
            ensemble_mean_f=45.0,
            threshold_f=30,
            adjusted_std_f=-3.0,
        )


def test_is_tail_threshold_invalid_cutoff_negative():
    """Negative z_score_cutoff raises ValueError."""
    with pytest.raises(ValueError, match="z_score_cutoff"):
        is_tail_threshold(
            ensemble_mean_f=45.0,
            threshold_f=30,
            adjusted_std_f=8.0,
            z_score_cutoff=-1.0,
        )


# ---------------------------------------------------------------------------
# compute_tail_alpha_boost — tail_flag False
# ---------------------------------------------------------------------------


def test_boost_tail_flag_false_returns_abs_alpha_positive():
    """tail_flag=False: score == abs(alpha) for positive alpha."""
    score = compute_tail_alpha_boost(alpha=0.25, tail_multiplier=2.0, tail_flag=False)
    assert score == pytest.approx(0.25)


def test_boost_tail_flag_false_returns_abs_alpha_negative():
    """tail_flag=False: score == abs(alpha) for negative alpha (SELL signal)."""
    score = compute_tail_alpha_boost(alpha=-0.20, tail_multiplier=1.5, tail_flag=False)
    assert score == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# compute_tail_alpha_boost — tail_flag True
# ---------------------------------------------------------------------------


def test_boost_tail_flag_true_applies_multiplier():
    """tail_flag=True: score == abs(alpha) * tail_multiplier."""
    score = compute_tail_alpha_boost(alpha=0.30, tail_multiplier=1.5, tail_flag=True)
    assert score == pytest.approx(0.45)


def test_boost_tail_flag_true_negative_alpha():
    """tail_flag=True with negative alpha (SELL): score uses abs(alpha)."""
    score = compute_tail_alpha_boost(alpha=-0.30, tail_multiplier=2.0, tail_flag=True)
    assert score == pytest.approx(0.60)


def test_boost_tail_flag_true_multiplier_one_no_change():
    """tail_multiplier=1.0 leaves the score unchanged (no boost)."""
    score = compute_tail_alpha_boost(alpha=0.15, tail_multiplier=1.0, tail_flag=True)
    assert score == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# compute_tail_alpha_boost — zero alpha
# ---------------------------------------------------------------------------


def test_boost_zero_alpha_flag_false():
    """alpha=0, tail_flag=False → score=0."""
    assert compute_tail_alpha_boost(alpha=0.0, tail_multiplier=2.0, tail_flag=False) == 0.0


def test_boost_zero_alpha_flag_true():
    """alpha=0, tail_flag=True → score=0 regardless of multiplier."""
    assert compute_tail_alpha_boost(alpha=0.0, tail_multiplier=3.0, tail_flag=True) == 0.0


# ---------------------------------------------------------------------------
# compute_tail_alpha_boost — tail_multiplier = 0
# ---------------------------------------------------------------------------


def test_boost_multiplier_zero_flag_true():
    """tail_multiplier=0 with tail_flag=True → score=0 (boosted to nothing)."""
    score = compute_tail_alpha_boost(alpha=0.50, tail_multiplier=0.0, tail_flag=True)
    assert score == 0.0


# ---------------------------------------------------------------------------
# compute_tail_alpha_boost — validation errors
# ---------------------------------------------------------------------------


def test_boost_invalid_negative_multiplier():
    """Negative tail_multiplier raises ValueError."""
    with pytest.raises(ValueError, match="tail_multiplier"):
        compute_tail_alpha_boost(alpha=0.20, tail_multiplier=-1.0, tail_flag=True)
