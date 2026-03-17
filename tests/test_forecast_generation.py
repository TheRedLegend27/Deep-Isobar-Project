"""Tests for deep_isobar.models.forecast_generation.

Organisation
------------
- Shared fixtures and helpers at the top
- Tests grouped by function under clear class names
- All tests that touch ``fetch_model_forecast`` or ``fetch_forecasts_for_city``
  either force stub mode (via ``monkeypatch``) or mock ``_fetch_live_value`` /
  ``requests.get`` directly, so the suite never makes real network calls.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from deep_isobar.core.types import CityProfile, ForecastPoint
from deep_isobar.models import forecast_generation as fg
from deep_isobar.models.forecast_generation import (
    VALID_METRICS,
    _CITY_COORDS,
    _OPEN_METEO_MODELS,
    _SOURCE_LIVE,
    _SOURCE_STUB,
    _compute_lead_hours,
    _compute_stub_forecast_value,
    _fetch_open_meteo,
    _is_stub_mode,
    _validate_metric,
    fetch_forecasts_for_city,
    fetch_model_forecast,
    register_forecast_run,
)

# ---------------------------------------------------------------------------
# Shared fixtures / constants
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


def _make_open_meteo_response(
    target_date: date,
    value: float,
    daily_var: str = "temperature_2m_max",
) -> dict[str, Any]:
    """Build a minimal Open-Meteo-shaped JSON response."""
    return {
        "daily": {
            "time": [target_date.isoformat()],
            daily_var: [value],
        }
    }


# ---------------------------------------------------------------------------
# _validate_metric
# ---------------------------------------------------------------------------


class TestValidateMetric:
    def test_high_temp_f_accepted(self) -> None:
        _validate_metric("high_temp_f")

    def test_low_temp_f_accepted(self) -> None:
        _validate_metric("low_temp_f")

    def test_invalid_metric_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid metric"):
            _validate_metric("wind_speed_mph")

    def test_case_sensitive(self) -> None:
        with pytest.raises(ValueError, match="Invalid metric"):
            _validate_metric("HIGH_TEMP_F")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid metric"):
            _validate_metric("")


# ---------------------------------------------------------------------------
# _compute_lead_hours
# ---------------------------------------------------------------------------


class TestComputeLeadHours:
    def test_positive_lead(self) -> None:
        run = datetime(2026, 3, 16, 12, 0, 0, tzinfo=timezone.utc)
        assert _compute_lead_hours(run, date(2026, 3, 17)) == 12

    def test_zero_lead(self) -> None:
        run = datetime(2026, 3, 17, 0, 0, 0, tzinfo=timezone.utc)
        assert _compute_lead_hours(run, date(2026, 3, 17)) == 0

    def test_negative_clamped_to_zero(self) -> None:
        run = datetime(2026, 3, 17, 6, 0, 0, tzinfo=timezone.utc)
        assert _compute_lead_hours(run, date(2026, 3, 17)) == 0

    def test_naive_run_time_treated_as_utc(self) -> None:
        run = datetime(2026, 3, 16, 0, 0, 0)  # naive
        assert _compute_lead_hours(run, date(2026, 3, 17)) == 24


# ---------------------------------------------------------------------------
# _compute_stub_forecast_value
# ---------------------------------------------------------------------------


class TestComputeStubForecastValue:
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


# ---------------------------------------------------------------------------
# _is_stub_mode
# ---------------------------------------------------------------------------


class TestIsStubMode:
    def test_false_by_default(self, monkeypatch) -> None:
        monkeypatch.delenv("FORECAST_STUB_MODE", raising=False)
        assert _is_stub_mode() is False

    @pytest.mark.parametrize("val", ["1", "true", "True", "TRUE", "yes", "YES"])
    def test_truthy_values(self, monkeypatch, val) -> None:
        monkeypatch.setenv("FORECAST_STUB_MODE", val)
        assert _is_stub_mode() is True

    @pytest.mark.parametrize("val", ["0", "false", "no", "", "off"])
    def test_falsy_values(self, monkeypatch, val) -> None:
        monkeypatch.setenv("FORECAST_STUB_MODE", val)
        assert _is_stub_mode() is False


# ---------------------------------------------------------------------------
# _fetch_open_meteo — unit tests (mock requests.get)
# ---------------------------------------------------------------------------


class TestFetchOpenMeteo:
    """Test the Open-Meteo HTTP fetch layer in isolation."""

    def _mock_get(self, response_data: dict) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = response_data
        mock = MagicMock(return_value=mock_resp)
        return mock

    def test_parses_high_temp_correctly(self) -> None:
        data = _make_open_meteo_response(_TARGET_DATE, 42.5, "temperature_2m_max")
        with patch("requests.get", self._mock_get(data)):
            val = _fetch_open_meteo(41.78, -87.75, "America/Chicago",
                                    "gfs_seamless", _TARGET_DATE, "high_temp_f")
        assert val == pytest.approx(42.5)

    def test_parses_low_temp_correctly(self) -> None:
        data = _make_open_meteo_response(_TARGET_DATE, 28.3, "temperature_2m_min")
        with patch("requests.get", self._mock_get(data)):
            val = _fetch_open_meteo(41.78, -87.75, "America/Chicago",
                                    "gfs_seamless", _TARGET_DATE, "low_temp_f")
        assert val == pytest.approx(28.3)

    def test_returns_none_when_date_not_in_response(self) -> None:
        # Response covers a different date
        data = _make_open_meteo_response(date(2026, 3, 20), 55.0, "temperature_2m_max")
        with patch("requests.get", self._mock_get(data)):
            val = _fetch_open_meteo(41.78, -87.75, "America/Chicago",
                                    "gfs_seamless", _TARGET_DATE, "high_temp_f")
        assert val is None

    def test_returns_none_when_value_is_null(self) -> None:
        data = {
            "daily": {
                "time": [_TARGET_DATE.isoformat()],
                "temperature_2m_max": [None],
            }
        }
        with patch("requests.get", self._mock_get(data)):
            val = _fetch_open_meteo(41.78, -87.75, "America/Chicago",
                                    "gfs_seamless", _TARGET_DATE, "high_temp_f")
        assert val is None

    def test_raises_on_api_error_response(self) -> None:
        data = {"error": True, "reason": "Invalid model code"}
        with patch("requests.get", self._mock_get(data)):
            with pytest.raises(ValueError, match="Open-Meteo API error"):
                _fetch_open_meteo(41.78, -87.75, "America/Chicago",
                                  "bad_model", _TARGET_DATE, "high_temp_f")

    def test_value_is_rounded_to_one_decimal(self) -> None:
        data = _make_open_meteo_response(_TARGET_DATE, 72.678, "temperature_2m_max")
        with patch("requests.get", self._mock_get(data)):
            val = _fetch_open_meteo(41.78, -87.75, "America/Chicago",
                                    "gfs_seamless", _TARGET_DATE, "high_temp_f")
        assert val == 72.7

    def test_requests_fahrenheit_unit(self) -> None:
        data = _make_open_meteo_response(_TARGET_DATE, 50.0, "temperature_2m_max")
        mock_get = self._mock_get(data)
        with patch("requests.get", mock_get):
            _fetch_open_meteo(41.78, -87.75, "America/Chicago",
                              "gfs_seamless", _TARGET_DATE, "high_temp_f")
        call_kwargs = mock_get.call_args
        params = call_kwargs[1].get("params") or call_kwargs[0][1]
        assert params["temperature_unit"] == "fahrenheit"


# ---------------------------------------------------------------------------
# fetch_model_forecast — stub mode forced via env var
# ---------------------------------------------------------------------------


class TestFetchModelForecastStubMode:
    """Tests with FORECAST_STUB_MODE=1 — no network calls made."""

    def test_returns_forecast_point(self, monkeypatch) -> None:
        monkeypatch.setenv("FORECAST_STUB_MODE", "1")
        result = fetch_model_forecast(
            city="Chicago", station_id="KORD", model_name="GFS",
            run_time_utc=_RUN_TIME, target_date=_TARGET_DATE, metric="high_temp_f",
        )
        assert isinstance(result, ForecastPoint)

    def test_fields_match_inputs(self, monkeypatch) -> None:
        monkeypatch.setenv("FORECAST_STUB_MODE", "1")
        result = fetch_model_forecast(
            city="Chicago", station_id="KORD", model_name="GFS",
            run_time_utc=_RUN_TIME, target_date=_TARGET_DATE, metric="high_temp_f",
        )
        assert result.city == "Chicago"
        assert result.station_id == "KORD"
        assert result.model_name == "GFS"
        assert result.run_time_utc == _RUN_TIME
        assert result.target_date == _TARGET_DATE
        assert result.metric == "high_temp_f"

    def test_source_name_is_stub(self, monkeypatch) -> None:
        monkeypatch.setenv("FORECAST_STUB_MODE", "1")
        result = fetch_model_forecast(
            city="Chicago", station_id="KORD", model_name="GFS",
            run_time_utc=_RUN_TIME, target_date=_TARGET_DATE, metric="high_temp_f",
        )
        assert result.source_name == _SOURCE_STUB

    def test_forecast_value_is_float(self, monkeypatch) -> None:
        monkeypatch.setenv("FORECAST_STUB_MODE", "1")
        result = fetch_model_forecast(
            city="Dallas", station_id="KDFW", model_name="ECMWF",
            run_time_utc=_RUN_TIME, target_date=_TARGET_DATE, metric="low_temp_f",
        )
        assert isinstance(result.forecast_value_f, float)

    def test_lead_hours_non_negative(self, monkeypatch) -> None:
        monkeypatch.setenv("FORECAST_STUB_MODE", "1")
        result = fetch_model_forecast(
            city="Chicago", station_id="KORD", model_name="GFS",
            run_time_utc=_RUN_TIME, target_date=_TARGET_DATE, metric="high_temp_f",
        )
        assert result.lead_hours >= 0

    def test_deterministic_in_stub_mode(self, monkeypatch) -> None:
        monkeypatch.setenv("FORECAST_STUB_MODE", "1")
        r1 = fetch_model_forecast(
            city="Chicago", station_id="KORD", model_name="GFS",
            run_time_utc=_RUN_TIME, target_date=_TARGET_DATE, metric="high_temp_f",
        )
        r2 = fetch_model_forecast(
            city="Chicago", station_id="KORD", model_name="GFS",
            run_time_utc=_RUN_TIME, target_date=_TARGET_DATE, metric="high_temp_f",
        )
        assert r1.forecast_value_f == r2.forecast_value_f

    def test_low_temp_f_accepted(self, monkeypatch) -> None:
        monkeypatch.setenv("FORECAST_STUB_MODE", "1")
        result = fetch_model_forecast(
            city="Denver", station_id="KDEN", model_name="NAM",
            run_time_utc=_RUN_TIME, target_date=_TARGET_DATE, metric="low_temp_f",
        )
        assert result.metric == "low_temp_f"

    def test_invalid_metric_raises(self, monkeypatch) -> None:
        monkeypatch.setenv("FORECAST_STUB_MODE", "1")
        with pytest.raises(ValueError, match="Invalid metric"):
            fetch_model_forecast(
                city="Chicago", station_id="KORD", model_name="GFS",
                run_time_utc=_RUN_TIME, target_date=_TARGET_DATE,
                metric="precipitation_mm",
            )


# ---------------------------------------------------------------------------
# fetch_model_forecast — live mode (mock _fetch_live_value)
# ---------------------------------------------------------------------------


class TestFetchModelForecastLiveMode:
    """Tests for live-mode behaviour via a mocked _fetch_live_value."""

    def test_source_name_is_open_meteo_on_live_success(self, monkeypatch) -> None:
        monkeypatch.delenv("FORECAST_STUB_MODE", raising=False)
        with patch.object(fg, "_fetch_live_value", return_value=38.5):
            result = fetch_model_forecast(
                city="Chicago", station_id="KORD", model_name="GFS",
                run_time_utc=_RUN_TIME, target_date=_TARGET_DATE, metric="high_temp_f",
            )
        assert result.source_name == _SOURCE_LIVE
        assert result.forecast_value_f == pytest.approx(38.5)

    def test_falls_back_to_stub_when_live_returns_none(self, monkeypatch) -> None:
        monkeypatch.delenv("FORECAST_STUB_MODE", raising=False)
        with patch.object(fg, "_fetch_live_value", return_value=None):
            result = fetch_model_forecast(
                city="Chicago", station_id="KORD", model_name="GFS",
                run_time_utc=_RUN_TIME, target_date=_TARGET_DATE, metric="high_temp_f",
            )
        assert result.source_name == _SOURCE_STUB
        assert isinstance(result.forecast_value_f, float)

    def test_stub_fallback_is_deterministic(self, monkeypatch) -> None:
        monkeypatch.delenv("FORECAST_STUB_MODE", raising=False)
        with patch.object(fg, "_fetch_live_value", return_value=None):
            r1 = fetch_model_forecast(
                city="Chicago", station_id="KORD", model_name="GFS",
                run_time_utc=_RUN_TIME, target_date=_TARGET_DATE, metric="high_temp_f",
            )
            r2 = fetch_model_forecast(
                city="Chicago", station_id="KORD", model_name="GFS",
                run_time_utc=_RUN_TIME, target_date=_TARGET_DATE, metric="high_temp_f",
            )
        assert r1.forecast_value_f == r2.forecast_value_f

    def test_live_value_is_fahrenheit_from_api(self, monkeypatch) -> None:
        # The API returns °F directly (temperature_unit=fahrenheit param).
        # Verify no secondary conversion is applied.
        monkeypatch.delenv("FORECAST_STUB_MODE", raising=False)
        raw_f = 41.2
        with patch.object(fg, "_fetch_live_value", return_value=raw_f):
            result = fetch_model_forecast(
                city="Chicago", station_id="KORD", model_name="GFS",
                run_time_utc=_RUN_TIME, target_date=_TARGET_DATE, metric="high_temp_f",
            )
        assert result.forecast_value_f == pytest.approx(raw_f)


# ---------------------------------------------------------------------------
# fetch_model_forecast — city / model not in lookup tables
# ---------------------------------------------------------------------------


class TestFetchModelForecastFallbackTriggers:
    """Verify fallback fires for cities/models absent from lookup tables."""

    def test_unknown_city_falls_back(self, monkeypatch) -> None:
        monkeypatch.delenv("FORECAST_STUB_MODE", raising=False)
        # "Atlantis" is not in _CITY_COORDS — live fetch should be skipped
        result = fetch_model_forecast(
            city="Atlantis", station_id="KATL", model_name="GFS",
            run_time_utc=_RUN_TIME, target_date=_TARGET_DATE, metric="high_temp_f",
        )
        assert result.source_name == _SOURCE_STUB

    def test_unknown_model_falls_back(self, monkeypatch) -> None:
        monkeypatch.delenv("FORECAST_STUB_MODE", raising=False)
        # "ICON" is not in _OPEN_METEO_MODELS
        result = fetch_model_forecast(
            city="Chicago", station_id="KORD", model_name="ICON",
            run_time_utc=_RUN_TIME, target_date=_TARGET_DATE, metric="high_temp_f",
        )
        assert result.source_name == _SOURCE_STUB


# ---------------------------------------------------------------------------
# fetch_forecasts_for_city
# ---------------------------------------------------------------------------


class TestFetchForecastsForCity:
    @patch(
        "deep_isobar.models.forecast_generation.get_city_profile",
        return_value=_SAMPLE_PROFILE,
    )
    def test_returns_one_point_per_model(self, mock_gcp, monkeypatch) -> None:
        monkeypatch.setenv("FORECAST_STUB_MODE", "1")
        results = fetch_forecasts_for_city(
            city="Chicago", target_date=_TARGET_DATE,
            metric="high_temp_f", model_names=["GFS", "ECMWF"],
        )
        assert len(results) == 2
        assert all(isinstance(r, ForecastPoint) for r in results)

    @patch(
        "deep_isobar.models.forecast_generation.get_city_profile",
        return_value=_SAMPLE_PROFILE,
    )
    def test_station_id_from_profile(self, mock_gcp, monkeypatch) -> None:
        monkeypatch.setenv("FORECAST_STUB_MODE", "1")
        results = fetch_forecasts_for_city(
            city="Chicago", target_date=_TARGET_DATE,
            metric="high_temp_f", model_names=["GFS"],
        )
        assert results[0].station_id == "KORD"

    @patch(
        "deep_isobar.models.forecast_generation.get_city_profile",
        return_value=_SAMPLE_PROFILE,
    )
    def test_each_model_represented_in_order(self, mock_gcp, monkeypatch) -> None:
        monkeypatch.setenv("FORECAST_STUB_MODE", "1")
        models = ["GFS", "ECMWF", "NAM"]
        results = fetch_forecasts_for_city(
            city="Chicago", target_date=_TARGET_DATE,
            metric="high_temp_f", model_names=models,
        )
        assert [r.model_name for r in results] == models

    def test_invalid_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid metric"):
            fetch_forecasts_for_city(
                city="Chicago", target_date=_TARGET_DATE,
                metric="humidity_pct", model_names=["GFS"],
            )

    def test_empty_model_names_raises(self) -> None:
        with pytest.raises(ValueError, match="model_names must not be empty"):
            fetch_forecasts_for_city(
                city="Chicago", target_date=_TARGET_DATE,
                metric="high_temp_f", model_names=[],
            )

    @patch(
        "deep_isobar.models.forecast_generation.get_city_profile",
        side_effect=KeyError("City not found: 'Atlantis'"),
    )
    def test_unknown_city_propagates_key_error(self, mock_gcp) -> None:
        with pytest.raises(KeyError, match="Atlantis"):
            fetch_forecasts_for_city(
                city="Atlantis", target_date=_TARGET_DATE,
                metric="high_temp_f", model_names=["GFS"],
            )

    @patch(
        "deep_isobar.models.forecast_generation.get_city_profile",
        return_value=_SAMPLE_PROFILE,
    )
    def test_partial_model_failure_still_returns_all_models(
        self, mock_gcp, monkeypatch
    ) -> None:
        """If one model's live fetch returns None, it falls back; others are unaffected.

        _fetch_live_value is documented to never raise — it catches internally and
        returns None. Patching it to return None is the correct simulation of failure.
        """
        monkeypatch.delenv("FORECAST_STUB_MODE", raising=False)

        def live_side_effect(city, model_name, target_date, metric):
            # GFS succeeds; ECMWF returns None (simulated network failure caught inside)
            return 38.5 if model_name == "GFS" else None

        with patch.object(fg, "_fetch_live_value", side_effect=live_side_effect):
            results = fetch_forecasts_for_city(
                city="Chicago", target_date=_TARGET_DATE,
                metric="high_temp_f", model_names=["GFS", "ECMWF"],
            )

        assert len(results) == 2
        gfs, ecmwf = results
        assert gfs.source_name == _SOURCE_LIVE
        assert gfs.forecast_value_f == pytest.approx(38.5)
        # ECMWF fell back to stub
        assert ecmwf.source_name == _SOURCE_STUB

    @patch(
        "deep_isobar.models.forecast_generation.get_city_profile",
        return_value=_SAMPLE_PROFILE,
    )
    def test_partial_model_failure_realistic_path(
        self, mock_gcp, monkeypatch
    ) -> None:
        """_fetch_live_value returns None for failed model — rest succeed."""
        monkeypatch.delenv("FORECAST_STUB_MODE", raising=False)

        def live_side_effect(city, model_name, target_date, metric):
            return 38.5 if model_name == "GFS" else None

        with patch.object(fg, "_fetch_live_value", side_effect=live_side_effect):
            results = fetch_forecasts_for_city(
                city="Chicago", target_date=_TARGET_DATE,
                metric="high_temp_f", model_names=["GFS", "ECMWF", "NAM"],
            )

        assert len(results) == 3
        assert results[0].source_name == _SOURCE_LIVE   # GFS live
        assert results[1].source_name == _SOURCE_STUB   # ECMWF fallback
        assert results[2].source_name == _SOURCE_STUB   # NAM fallback


# ---------------------------------------------------------------------------
# register_forecast_run
# ---------------------------------------------------------------------------


class TestRegisterForecastRun:
    def test_returns_dict(self) -> None:
        assert isinstance(
            register_forecast_run("GFS", _RUN_TIME, "12z", "open-meteo"), dict
        )

    def test_contains_required_keys(self) -> None:
        result = register_forecast_run("GFS", _RUN_TIME, "12z", "open-meteo")
        assert set(result.keys()) == {
            "model_name", "run_time_utc", "cycle_label",
            "source_name", "ingest_status", "registered_at_utc",
        }

    def test_values_match_inputs(self) -> None:
        result = register_forecast_run("ECMWF", _RUN_TIME, "00z", "open-meteo")
        assert result["model_name"] == "ECMWF"
        assert result["cycle_label"] == "00z"
        assert result["source_name"] == "open-meteo"

    def test_ingest_status_is_complete(self) -> None:
        result = register_forecast_run("GFS", _RUN_TIME, "12z", "open-meteo")
        assert result["ingest_status"] == "complete"

    def test_run_time_utc_is_iso_string(self) -> None:
        result = register_forecast_run("GFS", _RUN_TIME, "12z", "open-meteo")
        assert isinstance(result["run_time_utc"], str)
        assert datetime.fromisoformat(result["run_time_utc"]) == _RUN_TIME

    def test_registered_at_utc_is_iso_string(self) -> None:
        result = register_forecast_run("GFS", _RUN_TIME, "12z", "open-meteo")
        assert isinstance(result["registered_at_utc"], str)
        datetime.fromisoformat(result["registered_at_utc"])

    def test_stub_source_name_accepted(self) -> None:
        result = register_forecast_run("GFS", _RUN_TIME, "12z", "stub")
        assert result["source_name"] == "stub"


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_valid_metrics_contains_expected_values(self) -> None:
        assert "high_temp_f" in VALID_METRICS
        assert "low_temp_f" in VALID_METRICS
        assert len(VALID_METRICS) == 2

    def test_chicago_in_city_coords(self) -> None:
        assert "Chicago" in _CITY_COORDS
        lat, lon, tz = _CITY_COORDS["Chicago"]
        assert 40.0 < lat < 43.0   # latitude band for Chicago
        assert -89.0 < lon < -86.0  # longitude band
        assert "America" in tz

    def test_all_standard_models_in_open_meteo_map(self) -> None:
        for model in ("GFS", "ECMWF", "NAM"):
            assert model in _OPEN_METEO_MODELS, f"{model} missing from _OPEN_METEO_MODELS"

    def test_open_meteo_model_codes_are_strings(self) -> None:
        for model, code in _OPEN_METEO_MODELS.items():
            assert isinstance(code, str), f"code for {model} must be a string"
            assert len(code) > 0
