"""Market lag detection for Deep Isobar.

Detects when a meaningful forecast shift has not been reflected in market
prices.  This is a first-pass, intentionally simple heuristic — it returns a
plain boolean rather than a magnitude score so callers can use it as a
gating flag without needing to understand an arbitrary scale.

The function is deterministic and pure: given the same inputs it always
returns the same output.

MVP logic
---------
A market is considered "lagging" when **both** conditions hold:

1. The forecast shift was significant (``shift_event.significant_shift_flag``
   is ``True``).
2. The market probability barely moved relative to the shift
   (``|market_probability_after - market_probability_before|
   < min_expected_response``).

If the forecast shift was not significant the market cannot be said to be
lagging — return ``False``.  If the market has moved enough to reflect the
updated forecast — return ``False``.

Future improvements
-------------------
Later versions may incorporate:

- Scaling ``min_expected_response`` by ``shift_event.absolute_shift_f`` so
  larger temperature moves require larger market reactions.
- Microstructure context (a wide-spread/illiquid market may legitimately
  react slowly).
- Contract distance from the ensemble mean to weight tail vs. near-the-money
  contracts differently.

Example usage::

    from datetime import date, datetime, timezone
    from deep_isobar.core.types import ForecastShiftEvent
    from deep_isobar.market.market_lag_detection import detect_market_lag

    shift = ForecastShiftEvent(
        city="Chicago",
        model_name="GFS",
        metric="high_temp_f",
        previous_run_time_utc=datetime(2026, 3, 17, 0, 0, tzinfo=timezone.utc),
        current_run_time_utc=datetime(2026, 3, 17, 6, 0, tzinfo=timezone.utc),
        target_date=date(2026, 3, 18),
        previous_value_f=72.0,
        current_value_f=78.0,
        shift_f=6.0,
        absolute_shift_f=6.0,
        shift_direction="warmer",
        significant_shift_flag=True,
    )

    lagging = detect_market_lag(
        shift_event=shift,
        market_probability_before=0.40,
        market_probability_after=0.41,  # barely moved
        min_expected_response=0.05,
    )
    print(lagging)  # True
"""

from __future__ import annotations

import logging

from deep_isobar.core.types import ForecastShiftEvent

logger = logging.getLogger(__name__)


def detect_market_lag(
    shift_event: ForecastShiftEvent,
    market_probability_before: float,
    market_probability_after: float,
    min_expected_response: float = 0.05,
) -> bool:
    """Detect whether a market has failed to react to a meaningful forecast shift.

    This is an intentionally simple first-pass detector.  It uses
    ``shift_event.significant_shift_flag`` as the gate and compares the
    absolute market probability change against ``min_expected_response``.

    Decision table:

    +--------------------------+------------------+--------+
    | significant_shift_flag   | market_response  | result |
    +==========================+==================+========+
    | ``False``                | any              | False  |
    | ``True``                 | < threshold      | True   |
    | ``True``                 | >= threshold     | False  |
    +--------------------------+------------------+--------+

    Args:
        shift_event: A :class:`~deep_isobar.core.types.ForecastShiftEvent`
            produced by the forecast shift detection module.  Only
            ``significant_shift_flag`` is read by this function.
        market_probability_before: Market-implied probability immediately
            before the forecast shift was published.  Must be in ``[0.0, 1.0]``.
        market_probability_after: Market-implied probability after the forecast
            shift was published.  Must be in ``[0.0, 1.0]``.
        min_expected_response: Minimum absolute change in market probability
            that constitutes a "sufficient" reaction to a significant forecast
            shift.  Must be non-negative.  Defaults to ``0.05`` (5 percentage
            points).

    Returns:
        ``True`` if the market appears to be lagging behind the forecast shift;
        ``False`` otherwise.

    Raises:
        ValueError: If ``market_probability_before`` is outside ``[0.0, 1.0]``.
        ValueError: If ``market_probability_after`` is outside ``[0.0, 1.0]``.
        ValueError: If ``min_expected_response`` is negative.

    Example::

        lagging = detect_market_lag(
            shift_event,
            market_probability_before=0.40,
            market_probability_after=0.41,  # barely moved
        )
        # True — significant shift but market barely moved
    """
    # ── Validation ────────────────────────────────────────────────────────
    if not (0.0 <= market_probability_before <= 1.0):
        raise ValueError(
            f"market_probability_before={market_probability_before!r} is outside "
            "[0.0, 1.0]; market probabilities must be decimal fractions"
        )
    if not (0.0 <= market_probability_after <= 1.0):
        raise ValueError(
            f"market_probability_after={market_probability_after!r} is outside "
            "[0.0, 1.0]; market probabilities must be decimal fractions"
        )
    if min_expected_response < 0.0:
        raise ValueError(
            f"min_expected_response={min_expected_response!r} is negative; "
            "it must be a non-negative float"
        )

    # ── Gate: insignificant forecast shift → no lag possible ──────────────
    if not shift_event.significant_shift_flag:
        logger.debug(
            "detect_market_lag: city=%s model=%s target=%s — "
            "significant_shift_flag=False, returning False",
            shift_event.city,
            shift_event.model_name,
            shift_event.target_date,
        )
        return False

    # ── Market response check ─────────────────────────────────────────────
    market_response = abs(market_probability_after - market_probability_before)
    lagging = market_response < min_expected_response

    logger.info(
        "detect_market_lag: city=%s model=%s target=%s metric=%s "
        "shift_f=%.2f significant=True "
        "prob_before=%.4f prob_after=%.4f response=%.4f threshold=%.4f lagging=%s",
        shift_event.city,
        shift_event.model_name,
        shift_event.target_date,
        shift_event.metric,
        shift_event.absolute_shift_f,
        market_probability_before,
        market_probability_after,
        market_response,
        min_expected_response,
        lagging,
    )
    return lagging
