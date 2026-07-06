"""Normal distribution probability engine for Deep Isobar.

Provides threshold-based probability calculations using the normal (Gaussian)
distribution.  These functions are the numeric core of the probability surface
and serve as the fallback when KDE is not available.

Canonical interface (from INTERFACES.md)::

    probability_ge_normal(mean_f, std_f, threshold_f) -> float  # P(T >= threshold)
    probability_le_normal(mean_f, std_f, threshold_f) -> float  # P(T <= threshold)

Kalshi strike-type aware interface::

    probability_for_contract(strike_type, floor_strike, cap_strike, mean_f, std_f)

Both functions require ``std_f > 0`` and return values in ``[0.0, 1.0]``.

Continuity corrections
----------------------
NWS Climatological Reports record daily high temperatures as whole integers
(°F).  All contract boundary probabilities therefore apply a ±0.5°F
continuity correction so that, e.g., P(T < 66) is computed as
``norm.cdf(65.5, mean, std)`` rather than ``norm.cdf(66, mean, std)``.
"""

from __future__ import annotations

import logging
from typing import Optional

from scipy.stats import norm

logger = logging.getLogger(__name__)


def probability_ge_normal(mean_f: float, std_f: float, threshold_f: float) -> float:
    """Compute P(T >= threshold) under a normal distribution.

    Args:
        mean_f: Distribution mean in °F.
        std_f: Distribution standard deviation in °F.  Must be > 0.
        threshold_f: Temperature threshold in °F.

    Returns:
        Probability in ``[0.0, 1.0]``.

    Raises:
        ValueError: If *std_f* is not positive.

    Example::

        prob = probability_ge_normal(mean_f=75.0, std_f=5.0, threshold_f=80.0)
        # → P(T >= 80 | μ=75, σ=5) ≈ 0.159
    """
    if std_f <= 0:
        raise ValueError(f"std_f must be positive, got {std_f}")

    prob = float(max(0.0, min(1.0, 1.0 - norm.cdf(threshold_f, loc=mean_f, scale=std_f))))
    logger.debug(
        "probability_ge_normal  mean=%.1f std=%.2f threshold=%.1f → %.4f",
        mean_f, std_f, threshold_f, prob,
    )
    return prob


def probability_for_contract(
    strike_type: str,
    floor_strike: Optional[int],
    cap_strike: Optional[int],
    mean_f: float,
    std_f: float,
) -> float:
    """Compute model YES-probability for a Kalshi KXHIGHCHI-style contract.

    Applies ±0.5°F continuity corrections because NWS reports temperatures
    as whole integers.

    Settlement rules by ``strike_type``:

    - ``"less"``    : YES if ``actual < cap_strike``
      → ``P(X < cap) = norm.cdf(cap − 0.5, mean, std)``
    - ``"greater"`` : YES if ``actual > floor_strike``
      → ``P(X > floor) = 1 − norm.cdf(floor + 0.5, mean, std)``

    Settlement rules by ``strike_type``:

    - ``"less"``    : YES if ``actual < cap_strike``
      → ``P(X < cap) = norm.cdf(cap − 0.5, mean, std)``
    - ``"greater"`` : YES if ``actual > floor_strike``
      → ``P(X > floor) = 1 − norm.cdf(floor + 0.5, mean, std)``
    - ``"between"`` : YES if ``floor_strike ≤ actual ≤ cap_strike`` (cap
      INCLUSIVE — Kalshi's B82.5 = "82-83°" pays on 82 and 83)
      → ``P(floor ≤ X ≤ cap) = norm.cdf(cap + 0.5) − norm.cdf(floor − 0.5)``

    Args:
        strike_type: Kalshi settlement type — ``"less"``, ``"greater"``, or
            ``"between"``.
        floor_strike: Lower boundary in °F; required for ``"greater"`` and
            ``"between"``.
        cap_strike: Upper boundary in °F; required for ``"less"`` and
            ``"between"``.
        mean_f: Ensemble forecast mean in °F.
        std_f: Ensemble forecast standard deviation in °F.  Must be > 0.

    Returns:
        YES-probability in ``[0.0, 1.0]``.

    Raises:
        ValueError: If *std_f* is not positive, *strike_type* is unrecognised,
            or a required boundary is ``None``.

    Example::

        # T60 contract (less, cap=60): P(actual < 60)
        p = probability_for_contract("less", None, 60, mean_f=65.0, std_f=5.5)
        # → norm.cdf(59.5, 65.0, 5.5) ≈ 0.159

        # T67 contract (greater, floor=67): P(actual > 67)
        p = probability_for_contract("greater", 67, None, mean_f=65.0, std_f=5.5)
        # → 1 - norm.cdf(67.5, 65.0, 5.5) ≈ 0.325

        # B65.5-style bracket (between, floor=65, cap=66): P(65 ≤ actual ≤ 66)
        p = probability_for_contract("between", 65, 66, mean_f=67.0, std_f=5.5)
        # → norm.cdf(66.5, 67.0, 5.5) - norm.cdf(64.5, 67.0, 5.5) ≈ 0.14
    """
    if std_f <= 0:
        raise ValueError(f"std_f must be positive, got {std_f}")

    st = strike_type.lower()
    if st == "less":
        if cap_strike is None:
            raise ValueError("cap_strike is required for strike_type='less'")
        # P(actual < cap) ≈ P(X ≤ cap − 1) with integer rounding → cdf(cap − 0.5)
        prob = float(norm.cdf(cap_strike - 0.5, loc=mean_f, scale=std_f))
    elif st == "greater":
        if floor_strike is None:
            raise ValueError("floor_strike is required for strike_type='greater'")
        # P(actual > floor) ≈ P(X ≥ floor + 1) → 1 − cdf(floor + 0.5)
        prob = float(1.0 - norm.cdf(floor_strike + 0.5, loc=mean_f, scale=std_f))
    elif st == "between":
        if floor_strike is None:
            raise ValueError("floor_strike is required for strike_type='between'")
        if cap_strike is None:
            raise ValueError("cap_strike is required for strike_type='between'")
        # P(floor ≤ actual ≤ cap) — CAP IS INCLUSIVE.  Verified against the
        # live API 2026-07-05: B82.5 = "82-83°" with floor=82, cap=83, YES
        # pays on 82 AND 83.  The previous exclusive-cap formula counted
        # only half of every 2-degree bracket, making all brackets look
        # overpriced and skewing the whole book toward bracket SELLs.
        prob = float(
            norm.cdf(cap_strike + 0.5, loc=mean_f, scale=std_f)
            - norm.cdf(floor_strike - 0.5, loc=mean_f, scale=std_f)
        )
    else:
        raise ValueError(
            f"Unsupported strike_type {strike_type!r}. "
            "Expected 'less', 'greater', or 'between'."
        )

    prob = max(0.0, min(1.0, prob))
    logger.debug(
        "probability_for_contract  strike_type=%s floor=%s cap=%s "
        "mean=%.1f std=%.2f → %.4f",
        strike_type, floor_strike, cap_strike, mean_f, std_f, prob,
    )
    return prob


def probability_le_normal(mean_f: float, std_f: float, threshold_f: float) -> float:
    """Compute P(T <= threshold) under a normal distribution.

    Args:
        mean_f: Distribution mean in °F.
        std_f: Distribution standard deviation in °F.  Must be > 0.
        threshold_f: Temperature threshold in °F.

    Returns:
        Probability in ``[0.0, 1.0]``.

    Raises:
        ValueError: If *std_f* is not positive.

    Example::

        prob = probability_le_normal(mean_f=75.0, std_f=5.0, threshold_f=70.0)
        # → P(T <= 70 | μ=75, σ=5) ≈ 0.159
    """
    if std_f <= 0:
        raise ValueError(f"std_f must be positive, got {std_f}")

    prob = float(max(0.0, min(1.0, norm.cdf(threshold_f, loc=mean_f, scale=std_f))))
    logger.debug(
        "probability_le_normal  mean=%.1f std=%.2f threshold=%.1f → %.4f",
        mean_f, std_f, threshold_f, prob,
    )
    return prob
