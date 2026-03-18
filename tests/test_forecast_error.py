"""Tests for src/deep_isobar/models/forecast_error.py."""

import math
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from deep_isobar.models.forecast_error import (
    compute_forecast_error,
    summarize_forecast_error_by_city_model,
)

# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

_D1 = date(2026, 1, 1)
_D2 = date(2026, 1, 2)


def _forecast_df(**overrides) -> pd.DataFrame:
    """Minimal valid forecast_df (2 models, 1 city, 1 date, high_temp_f)."""
    base = {
        "city": ["Chicago", "Chicago"],
        "model_name": ["GFS", "ECMWF"],
        "target_date": [_D1, _D1],
        "metric": ["high_temp_f", "high_temp_f"],
        "forecast_value_f": [35.0, 38.0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _actual_df(**overrides) -> pd.DataFrame:
    """Minimal valid actual_df (1 city, 1 date, wide format)."""
    base = {
        "city": ["Chicago"],
        "target_date": [_D1],
        "high_temp_f": [33.0],
        "low_temp_f": [20.0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# compute_forecast_error — error calculation
# ---------------------------------------------------------------------------


class TestComputeForecastErrorCalculation:
    def test_error_f_is_forecast_minus_actual(self):
        result = compute_forecast_error(_forecast_df(), _actual_df())
        gfs = result[result["model_name"] == "GFS"].iloc[0]
        # 35.0 - 33.0 = 2.0
        assert gfs["error_f"] == pytest.approx(2.0)

    def test_absolute_error_f(self):
        result = compute_forecast_error(_forecast_df(), _actual_df())
        gfs = result[result["model_name"] == "GFS"].iloc[0]
        ecmwf = result[result["model_name"] == "ECMWF"].iloc[0]
        assert gfs["absolute_error_f"] == pytest.approx(2.0)   # |35 - 33|
        assert ecmwf["absolute_error_f"] == pytest.approx(5.0)  # |38 - 33|

    def test_squared_error(self):
        result = compute_forecast_error(_forecast_df(), _actual_df())
        gfs = result[result["model_name"] == "GFS"].iloc[0]
        ecmwf = result[result["model_name"] == "ECMWF"].iloc[0]
        assert gfs["squared_error"] == pytest.approx(4.0)   # 2²
        assert ecmwf["squared_error"] == pytest.approx(25.0)  # 5²

    def test_negative_error_is_cold_bias(self):
        # forecast below actual → negative signed error, positive absolute
        fc = _forecast_df(forecast_value_f=[28.0, 30.0])
        result = compute_forecast_error(fc, _actual_df())
        gfs = result[result["model_name"] == "GFS"].iloc[0]
        assert gfs["error_f"] == pytest.approx(-5.0)        # 28 - 33
        assert gfs["absolute_error_f"] == pytest.approx(5.0)
        assert gfs["squared_error"] == pytest.approx(25.0)

    def test_zero_error_when_perfect_forecast(self):
        fc = _forecast_df(forecast_value_f=[33.0, 33.0])
        result = compute_forecast_error(fc, _actual_df())
        for _, row in result.iterrows():
            assert row["error_f"] == pytest.approx(0.0)
            assert row["absolute_error_f"] == pytest.approx(0.0)
            assert row["squared_error"] == pytest.approx(0.0)

    def test_actual_value_f_reflects_high_temp_metric(self):
        # high_temp_f=33, low_temp_f=20; metric=high_temp_f → actual_value_f=33
        result = compute_forecast_error(_forecast_df(), _actual_df())
        assert (result["actual_value_f"] == 33.0).all()

    def test_low_temp_f_metric_uses_low_actual(self):
        fc = _forecast_df(
            metric=["low_temp_f", "low_temp_f"],
            forecast_value_f=[22.0, 21.0],
        )
        result = compute_forecast_error(fc, _actual_df())
        assert (result["actual_value_f"] == 20.0).all()

    def test_output_contains_all_required_columns(self):
        result = compute_forecast_error(_forecast_df(), _actual_df())
        for col in [
            "city", "model_name", "target_date", "metric",
            "forecast_value_f", "actual_value_f",
            "error_f", "absolute_error_f", "squared_error",
        ]:
            assert col in result.columns, f"Missing required column: {col}"


# ---------------------------------------------------------------------------
# compute_forecast_error — join behaviour
# ---------------------------------------------------------------------------


class TestComputeForecastErrorJoin:
    def test_multiple_cities_joined_independently(self):
        fc = pd.DataFrame({
            "city": ["Chicago", "New York"],
            "model_name": ["GFS", "GFS"],
            "target_date": [_D1, _D1],
            "metric": ["high_temp_f", "high_temp_f"],
            "forecast_value_f": [35.0, 40.0],
        })
        actual = pd.DataFrame({
            "city": ["Chicago", "New York"],
            "target_date": [_D1, _D1],
            "high_temp_f": [33.0, 37.0],
        })
        result = compute_forecast_error(fc, actual)
        assert len(result) == 2
        chi = result[result["city"] == "Chicago"].iloc[0]
        nyc = result[result["city"] == "New York"].iloc[0]
        assert chi["error_f"] == pytest.approx(2.0)
        assert nyc["error_f"] == pytest.approx(3.0)

    def test_multiple_run_dates_produce_multiple_error_rows(self):
        # Two GFS runs for same city/target_date — both should appear
        fc = pd.DataFrame({
            "city": ["Chicago", "Chicago"],
            "model_name": ["GFS", "GFS"],
            "target_date": [_D1, _D1],
            "metric": ["high_temp_f", "high_temp_f"],
            "forecast_value_f": [35.0, 36.0],
            "run_time_utc": [
                datetime(2025, 12, 31, 0, tzinfo=timezone.utc),
                datetime(2025, 12, 31, 12, tzinfo=timezone.utc),
            ],
        })
        result = compute_forecast_error(fc, _actual_df())
        assert len(result) == 2

    def test_unmatched_forecast_rows_are_dropped(self):
        # actual has no Chicago data on _D2
        actual = _actual_df(target_date=[_D2])
        result = compute_forecast_error(_forecast_df(), actual)
        assert result.empty

    def test_no_matching_city_returns_empty(self):
        actual = pd.DataFrame({
            "city": ["Dallas"],
            "target_date": [_D1],
            "high_temp_f": [55.0],
        })
        result = compute_forecast_error(_forecast_df(), actual)
        assert result.empty

    def test_no_match_empty_result_has_required_columns(self):
        # When the join produces zero rows the result must still carry the
        # required error columns so downstream code can inspect .columns safely.
        actual = _actual_df(target_date=[_D2])  # no overlap with forecast _D1
        result = compute_forecast_error(_forecast_df(), actual)
        assert result.empty
        for col in ["error_f", "absolute_error_f", "squared_error", "actual_value_f"]:
            assert col in result.columns, f"Missing column in empty result: {col}"

    def test_extra_forecast_columns_preserved(self):
        fc = _forecast_df()
        fc["run_time_utc"] = datetime(2026, 1, 1, 0, tzinfo=timezone.utc)
        fc["lead_hours"] = 24
        result = compute_forecast_error(fc, _actual_df())
        assert "run_time_utc" in result.columns
        assert "lead_hours" in result.columns

    def test_actual_null_temp_rows_excluded(self):
        # actual has NaN for high_temp_f — that row should not contribute
        actual = pd.DataFrame({
            "city": ["Chicago"],
            "target_date": [_D1],
            "high_temp_f": [float("nan")],
            "low_temp_f": [20.0],
        })
        fc = _forecast_df(metric=["high_temp_f", "high_temp_f"])
        result = compute_forecast_error(fc, actual)
        # high_temp_f is null → no match for high_temp_f metric
        assert result.empty

    def test_mixed_metrics_joined_correctly(self):
        fc = pd.DataFrame({
            "city": ["Chicago", "Chicago"],
            "model_name": ["GFS", "GFS"],
            "target_date": [_D1, _D1],
            "metric": ["high_temp_f", "low_temp_f"],
            "forecast_value_f": [35.0, 22.0],
        })
        result = compute_forecast_error(fc, _actual_df())
        assert len(result) == 2
        high = result[result["metric"] == "high_temp_f"].iloc[0]
        low = result[result["metric"] == "low_temp_f"].iloc[0]
        assert high["actual_value_f"] == pytest.approx(33.0)
        assert low["actual_value_f"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# compute_forecast_error — empty / validation
# ---------------------------------------------------------------------------


class TestComputeForecastErrorEdgeCases:
    def test_empty_forecast_df_returns_empty(self):
        empty = pd.DataFrame(
            columns=["city", "model_name", "target_date", "metric", "forecast_value_f"]
        )
        result = compute_forecast_error(empty, _actual_df())
        assert result.empty

    def test_empty_actual_df_returns_empty(self):
        empty = pd.DataFrame(columns=["city", "target_date", "high_temp_f"])
        result = compute_forecast_error(_forecast_df(), empty)
        assert result.empty

    def test_missing_forecast_column_raises(self):
        bad = _forecast_df().drop(columns=["forecast_value_f"])
        with pytest.raises(ValueError, match="forecast_df"):
            compute_forecast_error(bad, _actual_df())

    def test_missing_actual_city_column_raises(self):
        bad = _actual_df().drop(columns=["city"])
        with pytest.raises(ValueError, match="actual_df"):
            compute_forecast_error(_forecast_df(), bad)

    def test_actual_without_any_temp_column_raises(self):
        bad = _actual_df().drop(columns=["high_temp_f", "low_temp_f"])
        with pytest.raises(ValueError, match="at least one of"):
            compute_forecast_error(_forecast_df(), bad)

    def test_actual_with_only_high_temp_column_works(self):
        actual = _actual_df().drop(columns=["low_temp_f"])
        result = compute_forecast_error(_forecast_df(), actual)
        assert not result.empty
        assert (result["actual_value_f"] == 33.0).all()

    def test_actual_df_long_form_is_accepted(self):
        # actual_df already in long format (metric + actual_value_f) — should work
        long_actual = pd.DataFrame({
            "city": ["Chicago"],
            "target_date": [_D1],
            "metric": ["high_temp_f"],
            "actual_value_f": [33.0],
        })
        result = compute_forecast_error(_forecast_df(), long_actual)
        assert not result.empty
        assert (result["actual_value_f"] == 33.0).all()

    def test_actual_df_long_form_error_values_correct(self):
        # GFS forecast 35, actual 33 → error 2; ECMWF forecast 38 → error 5
        long_actual = pd.DataFrame({
            "city": ["Chicago"],
            "target_date": [_D1],
            "metric": ["high_temp_f"],
            "actual_value_f": [33.0],
        })
        result = compute_forecast_error(_forecast_df(), long_actual)
        gfs = result[result["model_name"] == "GFS"].iloc[0]
        ecmwf = result[result["model_name"] == "ECMWF"].iloc[0]
        assert gfs["error_f"] == pytest.approx(2.0)
        assert ecmwf["error_f"] == pytest.approx(5.0)

    def test_unsupported_metric_raises(self):
        # metric not in {high_temp_f, low_temp_f} → ValueError
        bad_fc = _forecast_df(metric=["feels_like_f", "feels_like_f"])
        with pytest.raises(ValueError, match="unsupported metric"):
            compute_forecast_error(bad_fc, _actual_df())

    def test_duplicate_actual_rows_do_not_fan_out_merge(self):
        # actual_df has two identical rows for Chicago/_D1 — output must not double
        actual = pd.DataFrame({
            "city": ["Chicago", "Chicago"],
            "target_date": [_D1, _D1],
            "high_temp_f": [33.0, 33.0],
        })
        result = compute_forecast_error(_forecast_df(), actual)
        # forecast has 2 rows (GFS + ECMWF); each should match exactly once
        assert len(result) == 2


# ---------------------------------------------------------------------------
# summarize_forecast_error_by_city_model
# ---------------------------------------------------------------------------


def _make_error_df() -> pd.DataFrame:
    """Three rows: two GFS runs + one ECMWF run for Chicago/high_temp_f."""
    return pd.DataFrame({
        "city": ["Chicago", "Chicago", "Chicago"],
        "model_name": ["GFS", "GFS", "ECMWF"],
        "metric": ["high_temp_f", "high_temp_f", "high_temp_f"],
        "forecast_value_f": [35.0, 33.0, 38.0],
        "actual_value_f": [33.0, 33.0, 33.0],
        "error_f": [2.0, 0.0, 5.0],
        "absolute_error_f": [2.0, 0.0, 5.0],
        "squared_error": [4.0, 0.0, 25.0],
    })


class TestSummarizeForecastError:
    def test_sample_count(self):
        summary = summarize_forecast_error_by_city_model(_make_error_df())
        gfs = summary[summary["model_name"] == "GFS"].iloc[0]
        ecmwf = summary[summary["model_name"] == "ECMWF"].iloc[0]
        assert gfs["sample_count"] == 2
        assert ecmwf["sample_count"] == 1

    def test_mean_error_f(self):
        summary = summarize_forecast_error_by_city_model(_make_error_df())
        gfs = summary[summary["model_name"] == "GFS"].iloc[0]
        assert gfs["mean_error_f"] == pytest.approx(1.0)  # (2 + 0) / 2

    def test_mean_absolute_error_f(self):
        summary = summarize_forecast_error_by_city_model(_make_error_df())
        gfs = summary[summary["model_name"] == "GFS"].iloc[0]
        assert gfs["mean_absolute_error_f"] == pytest.approx(1.0)  # (2 + 0) / 2

    def test_rmse_f(self):
        summary = summarize_forecast_error_by_city_model(_make_error_df())
        gfs = summary[summary["model_name"] == "GFS"].iloc[0]
        # MSE = (4 + 0) / 2 = 2.0, RMSE = sqrt(2)
        assert gfs["rmse_f"] == pytest.approx(math.sqrt(2.0))

    def test_rmse_equals_sqrt_of_mean_squared_error(self):
        # Single row: error=3, squared=9, MSE=9, RMSE=3
        error_df = pd.DataFrame({
            "city": ["Chicago"],
            "model_name": ["GFS"],
            "metric": ["high_temp_f"],
            "error_f": [3.0],
            "absolute_error_f": [3.0],
            "squared_error": [9.0],
        })
        summary = summarize_forecast_error_by_city_model(error_df)
        assert summary.iloc[0]["rmse_f"] == pytest.approx(3.0)

    def test_grouped_by_city_model_metric(self):
        summary = summarize_forecast_error_by_city_model(_make_error_df())
        # GFS + ECMWF, both for Chicago/high_temp_f → 2 groups
        assert len(summary) == 2

    def test_multi_city_grouping(self):
        error_df = pd.DataFrame({
            "city": ["Chicago", "New York"],
            "model_name": ["GFS", "GFS"],
            "metric": ["high_temp_f", "high_temp_f"],
            "error_f": [2.0, -1.0],
            "absolute_error_f": [2.0, 1.0],
            "squared_error": [4.0, 1.0],
        })
        summary = summarize_forecast_error_by_city_model(error_df)
        assert len(summary) == 2
        chi = summary[summary["city"] == "Chicago"].iloc[0]
        nyc = summary[summary["city"] == "New York"].iloc[0]
        assert chi["mean_error_f"] == pytest.approx(2.0)
        assert nyc["mean_error_f"] == pytest.approx(-1.0)

    def test_output_columns(self):
        summary = summarize_forecast_error_by_city_model(_make_error_df())
        for col in [
            "city", "model_name", "metric",
            "sample_count", "mean_error_f", "mean_absolute_error_f", "rmse_f",
        ]:
            assert col in summary.columns, f"Missing column: {col}"

    def test_mean_squared_error_not_in_output(self):
        # Intermediate column should be dropped before returning
        summary = summarize_forecast_error_by_city_model(_make_error_df())
        assert "mean_squared_error" not in summary.columns

    def test_empty_input_returns_empty(self):
        empty = pd.DataFrame(
            columns=["city", "model_name", "metric", "error_f", "absolute_error_f", "squared_error"]
        )
        result = summarize_forecast_error_by_city_model(empty)
        assert result.empty

    def test_missing_required_column_raises(self):
        bad = _make_error_df().drop(columns=["squared_error"])
        with pytest.raises(ValueError, match="error_df"):
            summarize_forecast_error_by_city_model(bad)

    def test_round_trip_compute_then_summarize(self):
        """compute_forecast_error output feeds cleanly into summarize."""
        fc = pd.DataFrame({
            "city": ["Chicago", "Chicago"],
            "model_name": ["GFS", "GFS"],
            "target_date": [_D1, _D2],
            "metric": ["high_temp_f", "high_temp_f"],
            "forecast_value_f": [35.0, 37.0],
        })
        actual = pd.DataFrame({
            "city": ["Chicago", "Chicago"],
            "target_date": [_D1, _D2],
            "high_temp_f": [33.0, 36.0],
        })
        error_df = compute_forecast_error(fc, actual)
        summary = summarize_forecast_error_by_city_model(error_df)
        assert len(summary) == 1
        row = summary.iloc[0]
        assert row["sample_count"] == 2
        assert row["mean_error_f"] == pytest.approx(1.5)      # (2 + 1) / 2
        assert row["mean_absolute_error_f"] == pytest.approx(1.5)
        assert row["rmse_f"] == pytest.approx(math.sqrt((4 + 1) / 2))
