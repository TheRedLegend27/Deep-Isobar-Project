"""Tests for deep_isobar.research.mispriced_weather_markets."""

from __future__ import annotations

import pytest
import pandas as pd

from deep_isobar.research.mispriced_weather_markets import (
    score_cities_by_alpha_opportunity,
    score_thresholds_by_profitability,
    score_market_types_by_profitability,
    recommend_city_profile_params,
    _CITY_SCORE_COLS,
    _THRESHOLD_SCORE_COLS,
    _MARKET_TYPE_COLS,
    _PARAM_REC_COLS,
    _BASELINE_STD_F,
    _KDE_BANDWIDTH_FACTOR,
    _KDE_BANDWIDTH_FLOOR,
    _MIN_VARIANCE_MULTIPLIER,
    _MAX_VARIANCE_MULTIPLIER,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _alpha_row(city, side, abs_alpha, metric="high_temp_f", threshold_f=80,
               comparison_operator="ge", rank_score=None,
               tail_flag=False, micro_score=None):
    """Return a single alpha_opportunity row as a dict."""
    return {
        "city": city,
        "signal_side": side,
        "absolute_alpha": abs_alpha,
        "metric": metric,
        "threshold_f": threshold_f,
        "comparison_operator": comparison_operator,
        "rank_score": rank_score,
        "tail_opportunity_flag": tail_flag,
        "microstructure_score": micro_score,
    }


def _make_alpha_df(rows):
    return pd.DataFrame(rows)


def _error_row(city, model, metric, error_f, abs_error_f, sq_error):
    return {
        "city": city,
        "model_name": model,
        "metric": metric,
        "error_f": error_f,
        "absolute_error_f": abs_error_f,
        "squared_error": sq_error,
    }


def _make_error_df(rows):
    return pd.DataFrame(rows)


# ===========================================================================
# score_cities_by_alpha_opportunity
# ===========================================================================


class TestScoreCitiesByAlphaOpportunity:
    def test_empty_input_returns_correct_columns(self):
        result = score_cities_by_alpha_opportunity(pd.DataFrame())
        assert list(result.columns) == _CITY_SCORE_COLS
        assert len(result) == 0

    def test_missing_required_column_raises(self):
        df = pd.DataFrame({"city": ["Chicago"], "signal_side": ["BUY"]})
        with pytest.raises(ValueError, match="missing required columns"):
            score_cities_by_alpha_opportunity(df)

    def test_all_hold_rows_returns_empty(self):
        df = _make_alpha_df([
            _alpha_row("Chicago", "HOLD", 0.05),
            _alpha_row("Chicago", "HOLD", 0.08),
        ])
        result = score_cities_by_alpha_opportunity(df)
        assert len(result) == 0
        assert list(result.columns) == _CITY_SCORE_COLS

    def test_single_city_single_signal(self):
        df = _make_alpha_df([_alpha_row("Chicago", "BUY", 0.15)])
        result = score_cities_by_alpha_opportunity(df)
        assert len(result) == 1
        assert result.iloc[0]["city"] == "Chicago"
        assert result.iloc[0]["trade_count"] == 1
        assert result.iloc[0]["mean_absolute_alpha"] == pytest.approx(0.15)
        assert result.iloc[0]["city_alpha_score"] == pytest.approx(0.15)

    def test_multiple_cities_sorted_descending(self):
        df = _make_alpha_df([
            _alpha_row("Phoenix", "BUY", 0.10),
            _alpha_row("Phoenix", "SELL", 0.12),
            _alpha_row("Chicago", "BUY", 0.20),
            _alpha_row("Chicago", "SELL", 0.22),
        ])
        result = score_cities_by_alpha_opportunity(df)
        # Chicago mean_abs_alpha = (0.20+0.22)/2 = 0.21
        # Phoenix mean_abs_alpha = (0.10+0.12)/2 = 0.11
        assert result.iloc[0]["city"] == "Chicago"
        assert result.iloc[1]["city"] == "Phoenix"

    def test_hold_rows_excluded_from_trade_count(self):
        df = _make_alpha_df([
            _alpha_row("Chicago", "BUY", 0.20),
            _alpha_row("Chicago", "HOLD", 0.01),
            _alpha_row("Chicago", "HOLD", 0.02),
        ])
        result = score_cities_by_alpha_opportunity(df)
        assert result.iloc[0]["trade_count"] == 1  # only the BUY row

    def test_tail_opp_rate_calculated_correctly(self):
        df = _make_alpha_df([
            _alpha_row("Chicago", "BUY", 0.20, tail_flag=True),
            _alpha_row("Chicago", "BUY", 0.18, tail_flag=True),
            _alpha_row("Chicago", "SELL", 0.15, tail_flag=False),
            _alpha_row("Chicago", "SELL", 0.12, tail_flag=False),
        ])
        result = score_cities_by_alpha_opportunity(df)
        assert result.iloc[0]["tail_opp_rate"] == pytest.approx(0.5)

    def test_tail_opp_rate_zero_when_column_absent(self):
        df = pd.DataFrame({
            "city": ["Chicago", "Chicago"],
            "signal_side": ["BUY", "SELL"],
            "absolute_alpha": [0.20, 0.15],
        })
        result = score_cities_by_alpha_opportunity(df)
        assert result.iloc[0]["tail_opp_rate"] == pytest.approx(0.0)

    def test_mean_rank_score_populated_when_present(self):
        df = _make_alpha_df([
            _alpha_row("Chicago", "BUY", 0.20, rank_score=0.30),
            _alpha_row("Chicago", "SELL", 0.15, rank_score=0.25),
        ])
        result = score_cities_by_alpha_opportunity(df)
        assert result.iloc[0]["mean_rank_score"] == pytest.approx(0.275)

    def test_mean_rank_score_none_when_column_absent(self):
        df = pd.DataFrame({
            "city": ["Chicago"],
            "signal_side": ["BUY"],
            "absolute_alpha": [0.20],
        })
        result = score_cities_by_alpha_opportunity(df)
        assert result.iloc[0]["mean_rank_score"] is None

    def test_city_alpha_score_equals_mean_absolute_alpha(self):
        df = _make_alpha_df([
            _alpha_row("Chicago", "BUY", 0.20),
            _alpha_row("Chicago", "BUY", 0.30),
        ])
        result = score_cities_by_alpha_opportunity(df)
        assert result.iloc[0]["city_alpha_score"] == pytest.approx(0.25)
        assert result.iloc[0]["mean_absolute_alpha"] == pytest.approx(0.25)


# ===========================================================================
# score_thresholds_by_profitability
# ===========================================================================


class TestScoreThresholdsByProfitability:
    def test_empty_input_returns_correct_columns(self):
        result = score_thresholds_by_profitability(pd.DataFrame())
        assert list(result.columns) == _THRESHOLD_SCORE_COLS
        assert len(result) == 0

    def test_missing_required_column_raises(self):
        df = pd.DataFrame({"city": ["Chicago"], "absolute_alpha": [0.2]})
        with pytest.raises(ValueError, match="missing required columns"):
            score_thresholds_by_profitability(df)

    def test_all_hold_returns_empty(self):
        df = _make_alpha_df([_alpha_row("Chicago", "HOLD", 0.05)])
        result = score_thresholds_by_profitability(df)
        assert len(result) == 0

    def test_ranks_by_mean_absolute_alpha_descending(self):
        df = _make_alpha_df([
            _alpha_row("Chicago", "BUY", 0.30, metric="high_temp_f", threshold_f=85),
            _alpha_row("Chicago", "BUY", 0.10, metric="high_temp_f", threshold_f=80),
        ])
        result = score_thresholds_by_profitability(df)
        assert result.iloc[0]["threshold_f"] == 85
        assert result.iloc[1]["threshold_f"] == 80

    def test_groups_by_city_metric_threshold_operator(self):
        df = _make_alpha_df([
            _alpha_row("Chicago", "BUY", 0.20, metric="high_temp_f", threshold_f=80, comparison_operator="ge"),
            _alpha_row("Chicago", "BUY", 0.15, metric="high_temp_f", threshold_f=80, comparison_operator="ge"),
            _alpha_row("Chicago", "BUY", 0.25, metric="low_temp_f", threshold_f=40, comparison_operator="le"),
        ])
        result = score_thresholds_by_profitability(df)
        assert len(result) == 2  # two distinct (metric, threshold_f, operator) groups
        assert result.iloc[0]["metric"] == "low_temp_f"  # higher mean alpha

    def test_optional_columns_absent_fills_none(self):
        df = pd.DataFrame({
            "city": ["Chicago"],
            "signal_side": ["BUY"],
            "absolute_alpha": [0.20],
        })
        result = score_thresholds_by_profitability(df)
        assert result.iloc[0]["metric"] is None
        assert result.iloc[0]["threshold_f"] is None
        assert result.iloc[0]["comparison_operator"] is None

    def test_tail_opp_rate_aggregated_per_group(self):
        df = _make_alpha_df([
            _alpha_row("Chicago", "BUY", 0.20, threshold_f=85, tail_flag=True),
            _alpha_row("Chicago", "BUY", 0.20, threshold_f=85, tail_flag=True),
            _alpha_row("Chicago", "BUY", 0.20, threshold_f=80, tail_flag=False),
        ])
        result = score_thresholds_by_profitability(df)
        th85_row = result[result["threshold_f"] == 85].iloc[0]
        th80_row = result[result["threshold_f"] == 80].iloc[0]
        assert th85_row["tail_opp_rate"] == pytest.approx(1.0)
        assert th80_row["tail_opp_rate"] == pytest.approx(0.0)


# ===========================================================================
# score_market_types_by_profitability
# ===========================================================================


class TestScoreMarketTypesByProfitability:
    def test_empty_input_returns_correct_columns(self):
        result = score_market_types_by_profitability(pd.DataFrame())
        assert list(result.columns) == _MARKET_TYPE_COLS
        assert len(result) == 0

    def test_missing_required_column_raises(self):
        df = pd.DataFrame({"city": ["Chicago"], "signal_side": ["BUY"]})
        with pytest.raises(ValueError, match="missing required columns"):
            score_market_types_by_profitability(df)

    def test_all_hold_returns_empty(self):
        df = _make_alpha_df([_alpha_row("Chicago", "HOLD", 0.05)])
        result = score_market_types_by_profitability(df)
        assert len(result) == 0

    def test_ranked_by_mean_absolute_alpha_descending(self):
        df = _make_alpha_df([
            _alpha_row("Chicago", "BUY", 0.25, metric="high_temp_f"),
            _alpha_row("Chicago", "BUY", 0.10, metric="low_temp_f"),
        ])
        result = score_market_types_by_profitability(df)
        assert result.iloc[0]["metric"] == "high_temp_f"
        assert result.iloc[1]["metric"] == "low_temp_f"

    def test_mean_microstructure_score_populated(self):
        df = _make_alpha_df([
            _alpha_row("Chicago", "BUY", 0.20, metric="high_temp_f", micro_score=0.80),
            _alpha_row("Chicago", "BUY", 0.20, metric="high_temp_f", micro_score=0.60),
        ])
        result = score_market_types_by_profitability(df)
        assert result.iloc[0]["mean_microstructure_score"] == pytest.approx(0.70)

    def test_mean_microstructure_score_none_when_column_absent(self):
        df = pd.DataFrame({
            "city": ["Chicago"],
            "signal_side": ["BUY"],
            "absolute_alpha": [0.20],
            "metric": ["high_temp_f"],
        })
        result = score_market_types_by_profitability(df)
        assert result.iloc[0]["mean_microstructure_score"] is None

    def test_metric_none_when_column_absent(self):
        df = pd.DataFrame({
            "city": ["Chicago"],
            "signal_side": ["BUY"],
            "absolute_alpha": [0.20],
        })
        result = score_market_types_by_profitability(df)
        assert result.iloc[0]["metric"] is None

    def test_groups_two_cities_two_metrics(self):
        df = _make_alpha_df([
            _alpha_row("Chicago", "BUY", 0.20, metric="high_temp_f"),
            _alpha_row("Chicago", "BUY", 0.10, metric="low_temp_f"),
            _alpha_row("Phoenix", "BUY", 0.30, metric="high_temp_f"),
        ])
        result = score_market_types_by_profitability(df)
        assert len(result) == 3


# ===========================================================================
# recommend_city_profile_params
# ===========================================================================


class TestRecommendCityProfileParams:
    def test_empty_input_returns_correct_columns(self):
        result = recommend_city_profile_params(pd.DataFrame())
        assert list(result.columns) == _PARAM_REC_COLS
        assert len(result) == 0

    def test_missing_required_column_raises(self):
        df = pd.DataFrame({"city": ["Chicago"], "model_name": ["GFS"]})
        with pytest.raises(ValueError, match="missing required columns"):
            recommend_city_profile_params(df)

    def test_bias_correction_is_negative_mean_error(self):
        # model runs 2°F warm → mean_error_f = +2.0 → correction = -2.0
        df = _make_error_df([
            _error_row("Chicago", "GFS", "high_temp_f", 2.0, 2.0, 4.0),
            _error_row("Chicago", "GFS", "high_temp_f", 2.0, 2.0, 4.0),
        ])
        result = recommend_city_profile_params(df)
        assert result.iloc[0]["recommended_mean_bias_correction_f"] == pytest.approx(-2.0)

    def test_bias_correction_positive_when_model_runs_cold(self):
        # model runs 3°F cold → mean_error_f = -3.0 → correction = +3.0
        df = _make_error_df([
            _error_row("Chicago", "GFS", "low_temp_f", -3.0, 3.0, 9.0),
        ])
        result = recommend_city_profile_params(df)
        assert result.iloc[0]["recommended_mean_bias_correction_f"] == pytest.approx(3.0)

    def test_rmse_computed_correctly(self):
        # squared_error values: 4 and 16 → mean = 10 → rmse = sqrt(10)
        df = _make_error_df([
            _error_row("Chicago", "GFS", "high_temp_f", 2.0, 2.0, 4.0),
            _error_row("Chicago", "GFS", "high_temp_f", 4.0, 4.0, 16.0),
        ])
        result = recommend_city_profile_params(df)
        assert result.iloc[0]["rmse_f"] == pytest.approx(10.0 ** 0.5, rel=1e-6)

    def test_variance_multiplier_normal_range(self):
        # rmse = 5.0 → multiplier = 5.0 / 5.0 = 1.0
        df = _make_error_df([
            _error_row("Chicago", "GFS", "high_temp_f", 0.0, 5.0, 25.0),
        ])
        result = recommend_city_profile_params(df)
        assert result.iloc[0]["recommended_variance_multiplier"] == pytest.approx(1.0)

    def test_variance_multiplier_clamped_at_floor(self):
        # Very small rmse: 0.5°F → 0.5/5.0 = 0.1 → clamped to 0.5
        df = _make_error_df([
            _error_row("Chicago", "GFS", "high_temp_f", 0.5, 0.5, 0.25),
        ])
        result = recommend_city_profile_params(df)
        assert result.iloc[0]["recommended_variance_multiplier"] == pytest.approx(
            _MIN_VARIANCE_MULTIPLIER
        )

    def test_variance_multiplier_clamped_at_ceiling(self):
        # Very large rmse: 30°F → 30/5.0 = 6.0 → clamped to 3.0
        df = _make_error_df([
            _error_row("Chicago", "GFS", "high_temp_f", 30.0, 30.0, 900.0),
        ])
        result = recommend_city_profile_params(df)
        assert result.iloc[0]["recommended_variance_multiplier"] == pytest.approx(
            _MAX_VARIANCE_MULTIPLIER
        )

    def test_kde_bandwidth_floor(self):
        # rmse = 1.0 → 1.0 * 0.3 = 0.3 < floor (1.0) → bandwidth = 1.0
        df = _make_error_df([
            _error_row("Chicago", "GFS", "high_temp_f", 1.0, 1.0, 1.0),
        ])
        result = recommend_city_profile_params(df)
        assert result.iloc[0]["recommended_kde_bandwidth"] == pytest.approx(
            _KDE_BANDWIDTH_FLOOR
        )

    def test_kde_bandwidth_above_floor(self):
        # rmse = 10.0 → 10.0 * 0.3 = 3.0 > floor → bandwidth = 3.0
        df = _make_error_df([
            _error_row("Chicago", "GFS", "high_temp_f", 0.0, 10.0, 100.0),
        ])
        result = recommend_city_profile_params(df)
        assert result.iloc[0]["recommended_kde_bandwidth"] == pytest.approx(
            10.0 * _KDE_BANDWIDTH_FACTOR
        )

    def test_groups_by_city_model_metric(self):
        df = _make_error_df([
            _error_row("Chicago", "GFS", "high_temp_f", 1.0, 1.0, 1.0),
            _error_row("Chicago", "ECMWF", "high_temp_f", 2.0, 2.0, 4.0),
            _error_row("Phoenix", "GFS", "high_temp_f", 3.0, 3.0, 9.0),
        ])
        result = recommend_city_profile_params(df)
        assert len(result) == 3
        assert set(result["city"]) == {"Chicago", "Phoenix"}
        assert set(result[result["city"] == "Chicago"]["model_name"]) == {"GFS", "ECMWF"}

    def test_sample_count_correct(self):
        df = _make_error_df([
            _error_row("Chicago", "GFS", "high_temp_f", 1.0, 1.0, 1.0),
            _error_row("Chicago", "GFS", "high_temp_f", 2.0, 2.0, 4.0),
            _error_row("Chicago", "GFS", "high_temp_f", 3.0, 3.0, 9.0),
        ])
        result = recommend_city_profile_params(df)
        assert result.iloc[0]["sample_count"] == 3

    def test_output_sorted_by_city_model_metric(self):
        df = _make_error_df([
            _error_row("Phoenix", "GFS", "high_temp_f", 1.0, 1.0, 1.0),
            _error_row("Chicago", "GFS", "low_temp_f", 2.0, 2.0, 4.0),
            _error_row("Chicago", "GFS", "high_temp_f", 3.0, 3.0, 9.0),
        ])
        result = recommend_city_profile_params(df)
        assert list(result["city"]) == ["Chicago", "Chicago", "Phoenix"]
        assert list(result[result["city"] == "Chicago"]["metric"]) == [
            "high_temp_f",
            "low_temp_f",
        ]
