"""Tests for deep_isobar.models.forecast_generation.

Covers the three public functions and key private helpers with both
happy-path and edge-case scenarios.  ``get_city_profile`` is mocked
to avoid filesystem / config dependencies.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import patch

import pytest

from deep_isobar.core.types import CityProfile, ForecastPoint
from deep_isobar.models.forecast_generation import (
    VALID_METRICS,
    _compute_lead_hours,
    _compute_stub_forecast_value,
    _validate_metric,
    fetch_forecasts_for_city,
    fetch_model_forecast,
    register_forecast_run,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_PROFILE = CityProfile(
    city="Chicago",
    city_code="CHI",
    station_id="KORD",
    timezone="America/Chicago",
    settlement_source="NWS",
    variance_multiplier=1.2,
    mean_bias_correction_f=0.5,
    kde_bandwidth=0.8,
)

_RUN_TIME = datetime(2026, 3, 16, 12, 0, 0, tzinfo=timezone.utc)
_TARGET_DATE = date(2026, 3, 17)


# ── _validate_metric ─────────────────────────────────────────────────────


class TestValidateMetric:
    """Tests for the private ``_validate_metric`` helper."""

    def test_high_temp_f_accepted(self) -> None:
        _validate_metric("high_temp_f")  # should not raise

    def test_low_temp_f_accepted(self) -> None:
        _validate_metric("low_temp_f")  # should not raise

    def test_invalid_metric_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid metric"):
            _validate_metric("wind_speed_mph")

    def test_case_sensitive(self) -> None:
        with pytest.raises(ValueError, match="Invalid metric"):
            _validate_metric("HIGH_TEMP_F")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid metric"):
            _validate_metric("")


# ── _compute_lead_hours ──────────────────────────────────────────────────


class TestComputeLeadHours:
    """Tests for the private ``_compute_lead_hours`` helper."""

    def test_positive_lead(self) -> None:
        run = datetime(2026, 3, 16, 12, 0, 0, tzinfo=timezone.utc)
        target = date(2026, 3, 17)
        assert _compute_lead_hours(run, target) == 12

    def test_zero_lead(self) -> None:
        run = datetime(2026, 3, 17, 0, 0, 0, tzinfo=timezone.utc)
        target = date(2026, 3, 17)
        assert _compute_lead_hours(run, target) == 0

    def test_negative_clamped_to_zero(self) -> None:
        run = datetime(2026, 3, 17, 6, 0, 0, tzinfo=timezone.utc)
        target = date(2026, 3, 17)
        assert _compute_lead_hours(run, target) == 0

    def test_naive_run_time_treated_as_utc(self) -> None:
        run = datetime(2026, 3, 16, 0, 0, 0)  # naive
        target = date(2026, 3, 17)
        assert _compute_lead_hours(run, target) == 24


# ── _compute_stub_forecast_value ─────────────────────────────────────────


class TestComputeStubForecastValue:
    """Tests for the deterministic stub forecast value generator."""

    def test_deterministic(self) -> None:
        v1 = _compute_stub_forecast_value("NYC", "GFS", _TARGET_DATE, "high_temp_f")
        v2 = _compute_stub_forecast_value("NYC", "GFS", _TARGET_DATE, "high_temp_f")
        assert v1 == v2

    def test_different_inputs_different_values(self) -> None:
        v_gfs = _compute_stub_forecast_value("NYC", "GFS", _TARGET_DATE, "high_temp_f")
        v_ecmwf = _compute_stub_forecast_value("NYC", "ECMWF", _TARGET_DATE, "high_temp_f")
        assert v_gfs != v_ecmwf

    def test_value_in_range(self) -> None:
        val = _compute_stub_forecast_value("Denver", "NAM", _TARGET_DATE, "low_temp_f")
        assert 55.0 <= val <= 95.0

    def test_different_metrics_differ(self) -> None:
        v_high = _compute_stub_forecast_value("Chicago", "GFS", _TARGET_DATE, "high_temp_f")
        v_low = _compute_stub_forecast_value("Chicago", "GFS", _TARGET_DATE, "low_temp_f")
        assert v_high != v_low


# ── fetch_model_forecast ─────────────────────────────────────────────────


class TestFetchModelForecast:
    """Tests for ``fetch_model_forecast``."""

    def test_returns_forecast_point(self) -> None:
        result = fetch_model_forecast(
            city="Chicago",
            station_id="KORD",
            model_name="GFS",
            run_time_utc=_RUN_TIME,
            target_date=_TARGET_DATE,
            metric="high_temp_f",
        )
        assert isinstance(result, ForecastPoint)

    def test_fields_match_inputs(self) -> None:
        result = fetch_model_forecast(
            city="Chicago",
            station_id="KORD",
            model_name="GFS",
            run_time_utc=_RUN_TIME,
            target_date=_TARGET_DATE,
            metric="high_temp_f",
        )
        assert result.city == "Chicago"
        assert result.station_id == "KORD"
        assert result.model_name == "GFS"
        assert result.run_time_utc == _RUN_TIME
        assert result.target_date == _TARGET_DATE
        assert result.metric == "high_temp_f"
        assert result.source_name == "GFS"

    def test_forecast_value_is_float(self) -> None:
        result = fetch_model_forecast(
            city="Dallas",
            station_id="KDFW",
            model_name="ECMWF",
            run_time_utc=_RUN_TIME,
            target_date=_TARGET_DATE,
            metric="low_temp_f",
        )
        assert isinstance(result.forecast_value_f, float)

    def test_lead_hours_non_negative(self) -> None:
        result = fetch_model_forecast(
            city="Chicago",
            station_id="KORD",
            model_name="GFS",
            run_time_utc=_RUN_TIME,
            target_date=_TARGET_DATE,
            metric="high_temp_f",
        )
        assert result.lead_hours >= 0

    def test_invalid_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid metric"):
            fetch_model_forecast(
                city="Chicago",
                station_id="KORD",
                model_name="GFS",
                run_time_utc=_RUN_TIME,
                target_date=_TARGET_DATE,
                metric="precipitation_mm",
            )

    def test_low_temp_f_accepted(self) -> None:
        result = fetch_model_forecast(
            city="Denver",
            station_id="KDEN",
            model_name="NAM",
            run_time_utc=_RUN_TIME,
            target_date=_TARGET_DATE,
            metric="low_temp_f",
        )
        assert result.metric == "low_temp_f"

    def test_deterministic_value(self) -> None:
        """Same inputs should always produce the same forecast value."""
        r1 = fetch_model_forecast(
            city="Chicago", station_id="KORD", model_name="GFS",
            run_time_utc=_RUN_TIME, target_date=_TARGET_DATE,
            metric="high_temp_f",
        )
        r2 = fetch_model_forecast(
            city="Chicago", station_id="KORD", model_name="GFS",
            run_time_utc=_RUN_TIME, target_date=_TARGET_DATE,
            metric="high_temp_f",
        )
        assert r1.forecast_value_f == r2.forecast_value_f


# ── fetch_forecasts_for_city ─────────────────────────────────────────────


class TestFetchForecastsForCity:
    """Tests for ``fetch_forecasts_for_city``."""

    @patch(
        "deep_isobar.models.forecast_generation.get_city_profile",
        return_value=_SAMPLE_PROFILE,
    )
    def test_returns_list_of_forecast_points(self, mock_gcp) -> None:
        results = fetch_forecasts_for_city(
            city="Chicago",
            target_date=_TARGET_DATE,
            metric="high_temp_f",
            model_names=["GFS", "ECMWF"],
        )
        assert len(results) == 2
        assert all(isinstance(r, ForecastPoint) for r in results)

    @patch(
        "deep_isobar.models.forecast_generation.get_city_profile",
        return_value=_SAMPLE_PROFILE,
    )
    def test_station_id_from_profile(self, mock_gcp) -> None:
        results = fetch_forecasts_for_city(
            city="Chicago",
            target_date=_TARGET_DATE,
            metric="high_temp_f",
            model_names=["GFS"],
        )
        assert results[0].station_id == "KORD"

    @patch(
        "deep_isobar.models.forecast_generation.get_city_profile",
        return_value=_SAMPLE_PROFILE,
    )
    def test_each_model_represented(self, mock_gcp) -> None:
        models = ["GFS", "ECMWF", "NAM"]
        results = fetch_forecasts_for_city(
            city="Chicago",
            target_date=_TARGET_DATE,
            metric="high_temp_f",
            model_names=models,
        )
        assert [r.model_name for r in results] == models

    def test_invalid_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid metric"):
            fetch_forecasts_for_city(
                city="Chicago",
                target_date=_TARGET_DATE,
                metric="humidity_pct",
                model_names=["GFS"],
            )

    def test_empty_model_names_raises(self) -> None:
        with pytest.raises(ValueError, match="model_names must not be empty"):
            fetch_forecasts_for_city(
                city="Chicago",
                target_date=_TARGET_DATE,
                metric="high_temp_f",
                model_names=[],
            )

    @patch(
        "deep_isobar.models.forecast_generation.get_city_profile",
        side_effect=KeyError("City not found: 'Atlantis'"),
    )
    def test_unknown_city_propagates_key_error(self, mock_gcp) -> None:
        with pytest.raises(KeyError, match="Atlantis"):
            fetch_forecasts_for_city(
                city="Atlantis",
                target_date=_TARGET_DATE,
                metric="high_temp_f",
                model_names=["GFS"],
            )


# ── register_forecast_run ────────────────────────────────────────────────


class TestRegisterForecastRun:
    """Tests for ``register_forecast_run``."""

    def test_returns_dict(self) -> None:
        result = register_forecast_run(
            model_name="GFS",
            run_time_utc=_RUN_TIME,
            cycle_label="12z",
            source_name="NOAA",
        )
        assert isinstance(result, dict)

    def test_contains_required_keys(self) -> None:
        result = register_forecast_run(
            model_name="GFS",
            run_time_utc=_RUN_TIME,
            cycle_label="12z",
            source_name="NOAA",
        )
        expected_keys = {
            "model_name",
            "run_time_utc",
            "cycle_label",
            "source_name",
            "ingest_status",
            "registered_at_utc",
        }
        assert set(result.keys()) == expected_keys

    def test_values_match_inputs(self) -> None:
        result = register_forecast_run(
            model_name="ECMWF",
            run_time_utc=_RUN_TIME,
            cycle_label="00z",
            source_name="ECMWF_API",
        )
        assert result["model_name"] == "ECMWF"
        assert result["cycle_label"] == "00z"
        assert result["source_name"] == "ECMWF_API"

    def test_ingest_status_is_complete(self) -> None:
        result = register_forecast_run(
            model_name="GFS",
            run_time_utc=_RUN_TIME,
            cycle_label="12z",
            source_name="NOAA",
        )
        assert result["ingest_status"] == "complete"

    def test_run_time_utc_is_iso_string(self) -> None:
        result = register_forecast_run(
            model_name="GFS",
            run_time_utc=_RUN_TIME,
            cycle_label="12z",
            source_name="NOAA",
        )
        assert isinstance(result["run_time_utc"], str)
        # Should be parseable back
        parsed = datetime.fromisoformat(result["run_time_utc"])
        assert parsed == _RUN_TIME

    def test_registered_at_utc_is_iso_string(self) -> None:
        result = register_forecast_run(
            model_name="GFS",
            run_time_utc=_RUN_TIME,
            cycle_label="12z",
            source_name="NOAA",
        )
        assert isinstance(result["registered_at_utc"], str)
        # Should be parseable
        datetime.fromisoformat(result["registered_at_utc"])


# ── VALID_METRICS constant ───────────────────────────────────────────────


class TestValidMetrics:
    """Sanity checks on the exported constant."""

    def test_contains_expected_values(self) -> None:
        assert "high_temp_f" in VALID_METRICS
        assert "low_temp_f" in VALID_METRICS

    def test_no_unexpected_values(self) -> None:
        assert len(VALID_METRICS) == 2
