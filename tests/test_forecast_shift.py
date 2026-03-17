"""Tests for deep_isobar.models.forecast_shift.

Covers:
    - directional shifts (up / down / flat)
    - significance flagging at, above, and below the threshold
    - custom significance thresholds
    - full ForecastShiftEvent field population
    - mismatch validation for city, model_name, target_date, metric
    - edge cases: large shifts, negative temps, zero threshold
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from deep_isobar.core.types import ForecastPoint, ForecastShiftEvent
from deep_isobar.models.forecast_shift import compute_forecast_shift


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PREV_RUN = datetime(2026, 3, 16, 6, 0, 0, tzinfo=timezone.utc)
_CURR_RUN = datetime(2026, 3, 16, 12, 0, 0, tzinfo=timezone.utc)
_TARGET = date(2026, 3, 17)


def _make_point(
    *,
    city: str = "Chicago",
    station_id: str = "KORD",
    model_name: str = "GFS",
    run_time_utc: datetime = _PREV_RUN,
    target_date: date = _TARGET,
    metric: str = "high_temp_f",
    forecast_value_f: float = 70.0,
    lead_hours: int = 24,
    source_name: str = "NOAA",
) -> ForecastPoint:
    """Convenience factory for building test ForecastPoints."""
    return ForecastPoint(
        city=city,
        station_id=station_id,
        model_name=model_name,
        run_time_utc=run_time_utc,
        target_date=target_date,
        metric=metric,
        forecast_value_f=forecast_value_f,
        lead_hours=lead_hours,
        source_name=source_name,
    )


# ---------------------------------------------------------------------------
# Direction tests
# ---------------------------------------------------------------------------


class TestShiftDirection:
    """Verify up / down / flat direction classification."""

    def test_basic_upward_shift(self) -> None:
        prev = _make_point(forecast_value_f=70.0, run_time_utc=_PREV_RUN)
        curr = _make_point(forecast_value_f=73.0, run_time_utc=_CURR_RUN)
        event = compute_forecast_shift(prev, curr)

        assert event.shift_direction == "up"
        assert event.shift_f == pytest.approx(3.0)
        assert event.absolute_shift_f == pytest.approx(3.0)

    def test_basic_downward_shift(self) -> None:
        prev = _make_point(forecast_value_f=70.0, run_time_utc=_PREV_RUN)
        curr = _make_point(forecast_value_f=67.0, run_time_utc=_CURR_RUN)
        event = compute_forecast_shift(prev, curr)

        assert event.shift_direction == "down"
        assert event.shift_f == pytest.approx(-3.0)
        assert event.absolute_shift_f == pytest.approx(3.0)

    def test_flat_shift(self) -> None:
        prev = _make_point(forecast_value_f=70.0, run_time_utc=_PREV_RUN)
        curr = _make_point(forecast_value_f=70.0, run_time_utc=_CURR_RUN)
        event = compute_forecast_shift(prev, curr)

        assert event.shift_direction == "flat"
        assert event.shift_f == pytest.approx(0.0)
        assert event.absolute_shift_f == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Significance tests
# ---------------------------------------------------------------------------


class TestSignificance:
    """Verify the significant_shift_flag with various thresholds."""

    def test_significant_shift_at_threshold(self) -> None:
        prev = _make_point(forecast_value_f=70.0, run_time_utc=_PREV_RUN)
        curr = _make_point(forecast_value_f=72.0, run_time_utc=_CURR_RUN)
        event = compute_forecast_shift(prev, curr, significance_threshold_f=2.0)

        assert event.significant_shift_flag is True

    def test_significant_shift_above_threshold(self) -> None:
        prev = _make_point(forecast_value_f=70.0, run_time_utc=_PREV_RUN)
        curr = _make_point(forecast_value_f=75.0, run_time_utc=_CURR_RUN)
        event = compute_forecast_shift(prev, curr, significance_threshold_f=2.0)

        assert event.significant_shift_flag is True

    def test_insignificant_shift_below_threshold(self) -> None:
        prev = _make_point(forecast_value_f=70.0, run_time_utc=_PREV_RUN)
        curr = _make_point(forecast_value_f=71.0, run_time_utc=_CURR_RUN)
        event = compute_forecast_shift(prev, curr, significance_threshold_f=2.0)

        assert event.significant_shift_flag is False

    def test_custom_significance_threshold(self) -> None:
        prev = _make_point(forecast_value_f=70.0, run_time_utc=_PREV_RUN)
        curr = _make_point(forecast_value_f=74.0, run_time_utc=_CURR_RUN)

        # 4°F shift — insignificant with a 5°F threshold
        event = compute_forecast_shift(prev, curr, significance_threshold_f=5.0)
        assert event.significant_shift_flag is False

        # 4°F shift — significant with a 3°F threshold
        event = compute_forecast_shift(prev, curr, significance_threshold_f=3.0)
        assert event.significant_shift_flag is True


# ---------------------------------------------------------------------------
# Field population
# ---------------------------------------------------------------------------


class TestEventFields:
    """Ensure all ForecastShiftEvent fields are populated correctly."""

    def test_event_fields_populated(self) -> None:
        prev = _make_point(forecast_value_f=68.0, run_time_utc=_PREV_RUN)
        curr = _make_point(forecast_value_f=73.0, run_time_utc=_CURR_RUN)
        event = compute_forecast_shift(prev, curr)

        assert isinstance(event, ForecastShiftEvent)
        assert event.city == "Chicago"
        assert event.model_name == "GFS"
        assert event.metric == "high_temp_f"
        assert event.previous_run_time_utc == _PREV_RUN
        assert event.current_run_time_utc == _CURR_RUN
        assert event.target_date == _TARGET
        assert event.previous_value_f == pytest.approx(68.0)
        assert event.current_value_f == pytest.approx(73.0)
        assert event.shift_f == pytest.approx(5.0)
        assert event.absolute_shift_f == pytest.approx(5.0)
        assert event.shift_direction == "up"
        assert event.significant_shift_flag is True


# ---------------------------------------------------------------------------
# Validation / error tests
# ---------------------------------------------------------------------------


class TestValidation:
    """Ensure ValueError is raised for incompatible forecast points."""

    def test_mismatch_city_raises(self) -> None:
        prev = _make_point(city="Chicago", run_time_utc=_PREV_RUN)
        curr = _make_point(city="Dallas", run_time_utc=_CURR_RUN)

        with pytest.raises(ValueError, match="city"):
            compute_forecast_shift(prev, curr)

    def test_mismatch_model_raises(self) -> None:
        prev = _make_point(model_name="GFS", run_time_utc=_PREV_RUN)
        curr = _make_point(model_name="ECMWF", run_time_utc=_CURR_RUN)

        with pytest.raises(ValueError, match="model_name"):
            compute_forecast_shift(prev, curr)

    def test_mismatch_target_date_raises(self) -> None:
        prev = _make_point(target_date=date(2026, 3, 17), run_time_utc=_PREV_RUN)
        curr = _make_point(target_date=date(2026, 3, 18), run_time_utc=_CURR_RUN)

        with pytest.raises(ValueError, match="target_date"):
            compute_forecast_shift(prev, curr)

    def test_mismatch_metric_raises(self) -> None:
        prev = _make_point(metric="high_temp_f", run_time_utc=_PREV_RUN)
        curr = _make_point(metric="low_temp_f", run_time_utc=_CURR_RUN)

        with pytest.raises(ValueError, match="metric"):
            compute_forecast_shift(prev, curr)

    def test_multiple_mismatches_all_listed(self) -> None:
        """When several fields differ, the error should mention them all."""
        prev = _make_point(city="Chicago", model_name="GFS", run_time_utc=_PREV_RUN)
        curr = _make_point(city="Dallas", model_name="ECMWF", run_time_utc=_CURR_RUN)

        with pytest.raises(ValueError, match="city") as exc_info:
            compute_forecast_shift(prev, curr)

        assert "model_name" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Cover unusual but valid inputs."""

    def test_large_shift(self) -> None:
        prev = _make_point(forecast_value_f=30.0, run_time_utc=_PREV_RUN)
        curr = _make_point(forecast_value_f=130.0, run_time_utc=_CURR_RUN)
        event = compute_forecast_shift(prev, curr)

        assert event.shift_f == pytest.approx(100.0)
        assert event.absolute_shift_f == pytest.approx(100.0)
        assert event.shift_direction == "up"
        assert event.significant_shift_flag is True

    def test_negative_forecast_values(self) -> None:
        prev = _make_point(forecast_value_f=-5.0, run_time_utc=_PREV_RUN)
        curr = _make_point(forecast_value_f=-10.0, run_time_utc=_CURR_RUN)
        event = compute_forecast_shift(prev, curr)

        assert event.shift_f == pytest.approx(-5.0)
        assert event.absolute_shift_f == pytest.approx(5.0)
        assert event.shift_direction == "down"

    def test_zero_threshold(self) -> None:
        """With threshold=0.0, any non-flat shift should be significant."""
        prev = _make_point(forecast_value_f=70.0, run_time_utc=_PREV_RUN)
        curr = _make_point(forecast_value_f=70.1, run_time_utc=_CURR_RUN)
        event = compute_forecast_shift(prev, curr, significance_threshold_f=0.0)

        assert event.significant_shift_flag is True
        assert event.shift_direction == "up"

    def test_zero_threshold_flat_is_significant(self) -> None:
        """With threshold=0.0, even flat (0.0 >= 0.0) is significant."""
        prev = _make_point(forecast_value_f=70.0, run_time_utc=_PREV_RUN)
        curr = _make_point(forecast_value_f=70.0, run_time_utc=_CURR_RUN)
        event = compute_forecast_shift(prev, curr, significance_threshold_f=0.0)

        assert event.significant_shift_flag is True
        assert event.shift_direction == "flat"
