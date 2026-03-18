"""Distribution tail alpha helpers for Deep Isobar.

Provides two pure utility functions for identifying whether a contract threshold
falls in the tail of the forecast distribution, and for computing an optional
ranking boost for tail opportunities.

These functions do **not** generate trade signals and do **not** modify raw alpha
values.  They are designed to be called by higher-level modules (e.g. alpha_engine
or market_scanner) that want to prioritise tail contracts in their ranking pass.

Tail detection uses a simple z-score rule:

    z = |threshold_f − ensemble_mean_f| / adjusted_std_f

A threshold is classified as a tail opportunity when ``z >= z_score_cutoff``
(default 1.5).

Ranking boost:

    boosted_score = |alpha|                      # tail_flag is False
    boosted_score = |alpha| * tail_multiplier    # tail_flag is True

The boost is a **ranking score**, not a change to raw alpha.  Raw alpha is always
preserved unchanged.

Typical usage::

    from deep_isobar.trading.distribution_tail_alpha import (
        is_tail_threshold,
        compute_tail_alpha_boost,
    )

    tail = is_tail_threshold(
        ensemble_mean_f=45.0,
        threshold_f=30,
        adjusted_std_f=8.0,
    )
    score = compute_tail_alpha_boost(alpha=0.25, tail_multiplier=1.5, tail_flag=tail)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def is_tail_threshold(
    ensemble_mean_f: float,
    threshold_f: int,
    adjusted_std_f: float,
    z_score_cutoff: float = 1.5,
) -> bool:
    """Return True if *threshold_f* lies in the tail of the forecast distribution.

    Uses a z-score rule:

        z = |threshold_f − ensemble_mean_f| / adjusted_std_f

    The threshold is classified as a tail opportunity when ``z >= z_score_cutoff``.

    Args:
        ensemble_mean_f: Bias-corrected ensemble mean temperature in °F.
        threshold_f: Integer contract threshold in °F.
        adjusted_std_f: Ensemble standard deviation after variance multiplier
            has been applied.  Must be strictly positive.
        z_score_cutoff: Minimum z-score required to classify the threshold as a
            tail opportunity.  Must be >= 0.  Defaults to 1.5.

    Returns:
        ``True`` if the threshold is in the tail, ``False`` otherwise.

    Raises:
        ValueError: If *adjusted_std_f* is not strictly positive.
        ValueError: If *z_score_cutoff* is negative.
    """
    if adjusted_std_f <= 0:
        raise ValueError(
            f"adjusted_std_f must be > 0, got {adjusted_std_f}"
        )
    if z_score_cutoff < 0:
        raise ValueError(
            f"z_score_cutoff must be >= 0, got {z_score_cutoff}"
        )

    z = abs(threshold_f - ensemble_mean_f) / adjusted_std_f
    result = z >= z_score_cutoff

    logger.debug(
        "is_tail_threshold  mean=%.2f threshold=%s std=%.4f z=%.4f cutoff=%.2f → %s",
        ensemble_mean_f, threshold_f, adjusted_std_f, z, z_score_cutoff, result,
    )

    return result


def compute_tail_alpha_boost(
    alpha: float,
    tail_multiplier: float,
    tail_flag: bool,
) -> float:
    """Return a ranking score that optionally boosts tail-opportunity contracts.

    This function returns a **ranking score**, not a modified raw alpha.
    Raw alpha is always preserved unchanged in the caller.

    Behaviour:

    - If *tail_flag* is ``False``: returns ``|alpha|``.
    - If *tail_flag* is ``True``: returns ``|alpha| * tail_multiplier``.

    BUY and SELL opportunities are treated symmetrically because the score is
    based on absolute alpha.

    Args:
        alpha: Raw signed alpha (model_probability − market_probability).
            May be positive, negative, or zero.
        tail_multiplier: Multiplier applied to |alpha| when *tail_flag* is
            ``True``.  Must be >= 0.  A value of ``1.0`` leaves the score
            unchanged; ``1.5`` increases it by 50 %.
        tail_flag: Whether this contract is classified as a tail opportunity
            (as returned by :func:`is_tail_threshold`).

    Returns:
        Non-negative ranking score.

    Raises:
        ValueError: If *tail_multiplier* is negative.
    """
    if tail_multiplier < 0:
        raise ValueError(
            f"tail_multiplier must be >= 0, got {tail_multiplier}"
        )

    abs_alpha = abs(alpha)

    if tail_flag:
        score = abs_alpha * tail_multiplier
    else:
        score = abs_alpha

    logger.debug(
        "compute_tail_alpha_boost  alpha=%.4f tail_flag=%s multiplier=%.2f → score=%.4f",
        alpha, tail_flag, tail_multiplier, score,
    )

    return score
