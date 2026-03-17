"""Tests for deep_isobar.models.temperature_ensemble.

Covers:
- Equal weighting when city_profile has no model weights
- Weighted average when city_profile has model weights
- Unknown model not in weight map falls back to 1.0 contribution
- bias_corrected_mean_f includes mean_bias_correction_f
- heat_bias_adjustment_f applied for metric="high_temp_f"
- cold_bias_adjustment_f applied for metric="low_temp_f"
- adjusted_std_f == ensemble_std_f * variance_multiplier
- contributing_models is a sorted unique list
- model_count matches len(contributing_models)
- ValueError on empty forecasts
- ValueError on invalid metric
- ValueError on ForecastPoint with mismatched target_date
- ValueError on ForecastPoint with mismatched metric
- methodology "equal_weight_normal" vs "weighted_normal"
"""

import pytest
from datetime import date, datetime

from deep_isobar.core.types import CityProfile, ForecastPoint
from deep_isobar.models.temperature_ensemble import build_temperature_ensemble

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TARGET_DATE = date(2026, 3, 17)
RUN_TIME = datetime(2026, 3, 17, 6, 0, 0)


def _make_profile(
    model_weight_gfs=None,
    model_weight_ecmwf=None,
    model_weight_nam=None,
    variance_multiplier=1.0,
    mean_bias_correction_f=0.0,
    heat_bias_adjustment_f=0.0,
    cold_bias_adjustment_f=0.0,
) -> CityProfile:
    return CityProfile(
        city="Test City",
        city_code="TST",
        station_id="KTST",
        timezone="America/Chicago",
        settlement_source="NWS",
        variance_multiplier=variance_multiplier,
        mean_bias_correction_f=mean_bias_correction_f,
        kde_bandwidth=1.0,
        model_weight_gfs=model_weight_gfs,
        model_weight_ecmwf=model_weight_ecmwf,
        model_weight_nam=model_weight_nam,
        heat_bias_adjustment_f=heat_bias_adjustment_f,
        cold_bias_adjustment_f=cold_bias_adjustment_f,
    )


def _make_fp(model_name: str, value: float, metric: str = "high_temp_f", target_date: date = TARGET_DATE) -> ForecastPoint:
    return ForecastPoint(
        city="Test City",
        station_id="KTST",
        model_name=model_name,
        run_time_utc=RUN_TIME,
        target_date=target_date,
        metric=metric,
        forecast_value_f=value,
        lead_hours=24,
        source_name="TEST",
    )


# ---------------------------------------------------------------------------
# Equal weighting
# ---------------------------------------------------------------------------


def test_equal_weighting_methodology():
    """All None weights → methodology == 'equal_weight_normal'."""
    profile = _make_profile()
    forecasts = [_make_fp("GFS", 70.0), _make_fp("ECMWF", 74.0)]
    result = build_temperature_ensemble(profile, forecasts, TARGET_DATE, "high_temp_f", RUN_TIME)
    assert result.methodology == "equal_weight_normal"


def test_equal_weighting_mean():
    """Equal weights → ensemble_mean_f is the arithmetic mean."""
    profile = _make_profile()
    forecasts = [_make_fp("GFS", 70.0), _make_fp("ECMWF", 74.0), _make_fp("NAM", 72.0)]
    result = build_temperature_ensemble(profile, forecasts, TARGET_DATE, "high_temp_f", RUN_TIME)
    assert result.ensemble_mean_f == pytest.approx((70.0 + 74.0 + 72.0) / 3)


# ---------------------------------------------------------------------------
# Weighted average
# ---------------------------------------------------------------------------


def test_weighted_methodology():
    """Profile weights set → methodology == 'weighted_normal'."""
    profile = _make_profile(model_weight_gfs=0.6, model_weight_ecmwf=0.4)
    forecasts = [_make_fp("GFS", 70.0), _make_fp("ECMWF", 80.0)]
    result = build_temperature_ensemble(profile, forecasts, TARGET_DATE, "high_temp_f", RUN_TIME)
    assert result.methodology == "weighted_normal"


def test_weighted_mean():
    """GFS=0.6, ECMWF=0.4 → weighted mean = 0.6*70 + 0.4*80 = 74."""
    profile = _make_profile(model_weight_gfs=0.6, model_weight_ecmwf=0.4)
    forecasts = [_make_fp("GFS", 70.0), _make_fp("ECMWF", 80.0)]
    result = build_temperature_ensemble(profile, forecasts, TARGET_DATE, "high_temp_f", RUN_TIME)
    assert result.ensemble_mean_f == pytest.approx(0.6 * 70.0 + 0.4 * 80.0)


# ---------------------------------------------------------------------------
# Unknown model fallback
# ---------------------------------------------------------------------------


def test_unknown_model_fallback_weight():
    """A model not in GFS/ECMWF/NAM falls back to raw weight 1.0, still normalized."""
    # GFS weight=0.5, CUSTOM fallback=1.0 → raw=[0.5, 1.0] → normalized=[1/3, 2/3]
    profile = _make_profile(model_weight_gfs=0.5)
    forecasts = [_make_fp("GFS", 60.0), _make_fp("CUSTOM_MODEL", 90.0)]
    result = build_temperature_ensemble(profile, forecasts, TARGET_DATE, "high_temp_f", RUN_TIME)
    expected_mean = (1 / 3) * 60.0 + (2 / 3) * 90.0
    assert result.ensemble_mean_f == pytest.approx(expected_mean)
    assert result.methodology == "weighted_normal"


# ---------------------------------------------------------------------------
# Bias correction
# ---------------------------------------------------------------------------


def test_mean_bias_correction_applied():
    """bias_corrected_mean_f includes mean_bias_correction_f."""
    profile = _make_profile(mean_bias_correction_f=2.5)
    forecasts = [_make_fp("GFS", 70.0)]
    result = build_temperature_ensemble(profile, forecasts, TARGET_DATE, "high_temp_f", RUN_TIME)
    assert result.bias_corrected_mean_f == pytest.approx(70.0 + 2.5)


def test_heat_bias_applied_for_high_temp():
    """heat_bias_adjustment_f is added for metric='high_temp_f'."""
    profile = _make_profile(mean_bias_correction_f=1.0, heat_bias_adjustment_f=0.8)
    forecasts = [_make_fp("GFS", 70.0, metric="high_temp_f")]
    result = build_temperature_ensemble(profile, forecasts, TARGET_DATE, "high_temp_f", RUN_TIME)
    assert result.bias_corrected_mean_f == pytest.approx(70.0 + 1.0 + 0.8)


def test_cold_bias_applied_for_low_temp():
    """cold_bias_adjustment_f is added for metric='low_temp_f'."""
    profile = _make_profile(mean_bias_correction_f=1.0, cold_bias_adjustment_f=-0.5)
    forecasts = [_make_fp("GFS", 50.0, metric="low_temp_f")]
    result = build_temperature_ensemble(profile, forecasts, TARGET_DATE, "low_temp_f", RUN_TIME)
    assert result.bias_corrected_mean_f == pytest.approx(50.0 + 1.0 + (-0.5))


def test_heat_bias_not_applied_for_low_temp():
    """heat_bias_adjustment_f must NOT be added for metric='low_temp_f'."""
    profile = _make_profile(heat_bias_adjustment_f=5.0, cold_bias_adjustment_f=0.0)
    forecasts = [_make_fp("GFS", 50.0, metric="low_temp_f")]
    result = build_temperature_ensemble(profile, forecasts, TARGET_DATE, "low_temp_f", RUN_TIME)
    assert result.bias_corrected_mean_f == pytest.approx(50.0)  # no heat bias


# ---------------------------------------------------------------------------
# Variance adjustment
# ---------------------------------------------------------------------------


def test_adjusted_std_equals_std_times_multiplier():
    """adjusted_std_f == ensemble_std_f * variance_multiplier."""
    profile = _make_profile(variance_multiplier=1.4)
    forecasts = [_make_fp("GFS", 70.0), _make_fp("ECMWF", 76.0)]
    result = build_temperature_ensemble(profile, forecasts, TARGET_DATE, "high_temp_f", RUN_TIME)
    assert result.adjusted_std_f == pytest.approx(result.ensemble_std_f * 1.4)


def test_variance_multiplier_stored():
    """variance_multiplier field echoes city_profile.variance_multiplier."""
    profile = _make_profile(variance_multiplier=1.2)
    forecasts = [_make_fp("GFS", 70.0)]
    result = build_temperature_ensemble(profile, forecasts, TARGET_DATE, "high_temp_f", RUN_TIME)
    assert result.variance_multiplier == 1.2


# ---------------------------------------------------------------------------
# Contributing models
# ---------------------------------------------------------------------------


def test_contributing_models_sorted_unique():
    """contributing_models is a sorted unique list of model names."""
    profile = _make_profile()
    # Duplicate GFS intentional — should appear only once
    forecasts = [_make_fp("NAM", 70.0), _make_fp("GFS", 72.0), _make_fp("GFS", 71.0)]
    result = build_temperature_ensemble(profile, forecasts, TARGET_DATE, "high_temp_f", RUN_TIME)
    assert result.contributing_models == ["GFS", "NAM"]


def test_model_count_matches_contributing():
    """model_count == len(contributing_models)."""
    profile = _make_profile()
    forecasts = [_make_fp("GFS", 70.0), _make_fp("ECMWF", 72.0), _make_fp("NAM", 71.0)]
    result = build_temperature_ensemble(profile, forecasts, TARGET_DATE, "high_temp_f", RUN_TIME)
    assert result.model_count == len(result.contributing_models)


# ---------------------------------------------------------------------------
# ValueError cases
# ---------------------------------------------------------------------------


def test_raises_on_empty_forecasts():
    """ValueError raised when forecasts list is empty."""
    profile = _make_profile()
    with pytest.raises(ValueError, match="empty"):
        build_temperature_ensemble(profile, [], TARGET_DATE, "high_temp_f", RUN_TIME)


def test_raises_on_invalid_metric():
    """ValueError raised for an unrecognized metric string."""
    profile = _make_profile()
    forecasts = [_make_fp("GFS", 70.0)]
    with pytest.raises(ValueError, match="metric"):
        build_temperature_ensemble(profile, forecasts, TARGET_DATE, "max_temp_c", RUN_TIME)


def test_raises_on_mismatched_target_date():
    """ValueError raised when a ForecastPoint has a different target_date."""
    profile = _make_profile()
    wrong_date = date(2026, 3, 18)
    forecasts = [_make_fp("GFS", 70.0), _make_fp("ECMWF", 72.0, target_date=wrong_date)]
    with pytest.raises(ValueError, match="target_date"):
        build_temperature_ensemble(profile, forecasts, TARGET_DATE, "high_temp_f", RUN_TIME)


def test_raises_on_mismatched_metric_in_forecast():
    """ValueError raised when a ForecastPoint metric differs from requested metric."""
    profile = _make_profile()
    forecasts = [_make_fp("GFS", 70.0), _make_fp("ECMWF", 40.0, metric="low_temp_f")]
    with pytest.raises(ValueError, match="metric"):
        build_temperature_ensemble(profile, forecasts, TARGET_DATE, "high_temp_f", RUN_TIME)
