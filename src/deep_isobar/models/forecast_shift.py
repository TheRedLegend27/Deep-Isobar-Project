"""Forecast Shift Detection — run-to-run forecast comparison for Deep Isobar.

Compares two :class:`~deep_isobar.core.types.ForecastPoint` instances
from successive model runs and produces a
:class:`~deep_isobar.core.types.ForecastShiftEvent` describing the
magnitude, direction, and statistical significance of the change.

Dependencies:
    - ``deep_isobar.core.types.ForecastPoint``
    - ``deep_isobar.core.types.ForecastShiftEvent``

Typical usage::

    from datetime import date, datetime, timezone
    from deep_isobar.core.types import ForecastPoint
    from deep_isobar.models.forecast_shift import compute_forecast_shift

    prev = ForecastPoint(
        city="Chicago", station_id="KORD", model_name="GFS",
        run_time_utc=datetime(2026, 3, 16, 6, tzinfo=timezone.utc),
        target_date=date(2026, 3, 17), metric="high_temp_f",
        forecast_value_f=70.0, lead_hours=30, source_name="NOAA",
    )
    curr = ForecastPoint(
        city="Chicago", station_id="KORD", model_name="GFS",
        run_time_utc=datetime(2026, 3, 16, 12, tzinfo=timezone.utc),
        target_date=date(2026, 3, 17), metric="high_temp_f",
        forecast_value_f=73.5, lead_hours=24, source_name="NOAA",
    )

    event = compute_forecast_shift(prev, curr)
    # event.shift_f == 3.5, event.shift_direction == "up",
    # event.significant_shift_flag == True  (3.5 >= default 2.0)
"""

from __future__ import annotations

import logging

from deep_isobar.core.types import ForecastPoint, ForecastShiftEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_forecast_shift(
    previous: ForecastPoint,
    current: ForecastPoint,
    significance_threshold_f: float = 2.0,
) -> ForecastShiftEvent:
    """Compare two forecast runs and detect a shift event.

    Both *previous* and *current* must refer to the **same** city,
    model, target date, and metric.  If any of these fields differ a
    ``ValueError`` is raised with a message listing every mismatched
    field.

    Args:
        previous: The earlier forecast point.
        current: The later forecast point.
        significance_threshold_f: Absolute shift (°F) at or above which
            the shift is flagged as significant.  Defaults to ``2.0``.

    Returns:
        A populated :class:`~deep_isobar.core.types.ForecastShiftEvent`.

    Raises:
        ValueError: If *previous* and *current* disagree on city,
            model_name, target_date, or metric.
    """
    _validate_compatible(previous, current)

    shift_f: float = current.forecast_value_f - previous.forecast_value_f
    absolute_shift_f: float = abs(shift_f)

    if shift_f > 0:
        shift_direction = "up"
    elif shift_f < 0:
        shift_direction = "down"
    else:
        shift_direction = "flat"

    significant_shift_flag: bool = absolute_shift_f >= significance_threshold_f

    event = ForecastShiftEvent(
        city=current.city,
        model_name=current.model_name,
        metric=current.metric,
        previous_run_time_utc=previous.run_time_utc,
        current_run_time_utc=current.run_time_utc,
        target_date=current.target_date,
        previous_value_f=previous.forecast_value_f,
        current_value_f=current.forecast_value_f,
        shift_f=shift_f,
        absolute_shift_f=absolute_shift_f,
        shift_direction=shift_direction,
        significant_shift_flag=significant_shift_flag,
    )

    logger.info(
        "compute_forecast_shift  city=%s model=%s target=%s metric=%s  "
        "shift=%.1f°F direction=%s significant=%s",
        event.city,
        event.model_name,
        event.target_date,
        event.metric,
        shift_f,
        shift_direction,
        significant_shift_flag,
    )

    return event


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_compatible(
    previous: ForecastPoint,
    current: ForecastPoint,
) -> None:
    """Raise ``ValueError`` if the two points are not comparable.

    Checks that ``city``, ``model_name``, ``target_date``, and
    ``metric`` match.  The error message lists every field that
    differs so the caller can diagnose the issue quickly.

    Args:
        previous: The earlier forecast point.
        current: The later forecast point.

    Raises:
        ValueError: With a detailed message listing mismatched fields.
    """
    mismatches: list[str] = []

    if previous.city != current.city:
        mismatches.append(
            f"city ('{previous.city}' vs '{current.city}')"
        )
    if previous.model_name != current.model_name:
        mismatches.append(
            f"model_name ('{previous.model_name}' vs '{current.model_name}')"
        )
    if previous.target_date != current.target_date:
        mismatches.append(
            f"target_date ({previous.target_date} vs {current.target_date})"
        )
    if previous.metric != current.metric:
        mismatches.append(
            f"metric ('{previous.metric}' vs '{current.metric}')"
        )

    if mismatches:
        joined = "; ".join(mismatches)
        raise ValueError(
            f"Incompatible forecast points — mismatched fields: {joined}"
        )