"""Mispriced weather markets analyzer for Deep Isobar.

Identifies the best cities, thresholds, and market types for trading by
analysing historical alpha opportunity logs and forecast error data.

This module is a **research tool** for Phase 10 targeting and optimisation.
It reads historical DataFrames produced by the alpha engine and the forecast
error pipeline — it does not connect to live exchange APIs, the scheduler, or
the execution layer.

Public interface::

    score_cities_by_alpha_opportunity(alpha_opportunity_df) -> pd.DataFrame
    score_thresholds_by_profitability(alpha_opportunity_df) -> pd.DataFrame
    score_market_types_by_profitability(alpha_opportunity_df) -> pd.DataFrame
    recommend_city_profile_params(forecast_error_df) -> pd.DataFrame

Typical workflow::

    alpha_df = pd.read_parquet("data/features/alpha_opportunity/")
    error_df = pd.read_parquet("data/features/forecast_error_history/")

    city_scores      = score_cities_by_alpha_opportunity(alpha_df)
    threshold_scores = score_thresholds_by_profitability(alpha_df)
    market_scores    = score_market_types_by_profitability(alpha_df)
    param_recs       = recommend_city_profile_params(error_df)

Input schemas
-------------
``alpha_opportunity_df`` — conforming to the ``alpha_opportunity`` feature store
(DATA_SCHEMA.md §17):

    Required: ``city``, ``absolute_alpha``, ``signal_side``
    Optional (used when present): ``metric``, ``threshold_f``,
        ``comparison_operator``, ``rank_score``,
        ``tail_opportunity_flag``, ``microstructure_score``

``forecast_error_df`` — conforming to the ``forecast_error_history`` feature
store (DATA_SCHEMA.md §9):

    Required: ``city``, ``model_name``, ``metric``, ``error_f``,
        ``absolute_error_f``, ``squared_error``

Output schemas
--------------
``score_cities_by_alpha_opportunity``:

    city, trade_count, mean_absolute_alpha, mean_rank_score,
    tail_opp_rate, city_alpha_score

``score_thresholds_by_profitability``:

    city, metric, threshold_f, comparison_operator, trade_count,
    mean_absolute_alpha, tail_opp_rate, threshold_score

``score_market_types_by_profitability``:

    city, metric, trade_count, mean_absolute_alpha,
    mean_microstructure_score, market_type_score

``recommend_city_profile_params``:

    city, model_name, metric, sample_count, mean_error_f, rmse_f,
    recommended_mean_bias_correction_f,
    recommended_variance_multiplier,
    recommended_kde_bandwidth

Parameter recommendation formulas
----------------------------------
``recommended_mean_bias_correction_f = -mean_error_f``
    Negative of the observed mean signed error.  If the model runs
    consistently +2 °F warm, the correction is -2 °F.

``recommended_variance_multiplier = clamp(rmse_f / 5.0, 0.5, 3.0)``
    A high RMSE relative to the 5 °F baseline suggests the ensemble spread
    should be widened.  Clamped to [0.5, 3.0] to prevent extreme values.

``recommended_kde_bandwidth = max(1.0, rmse_f * 0.3)``
    Wider KDE bandwidth when forecast uncertainty is higher.
    Floor of 1.0 °F prevents over-smoothing.
"""

from __future__ import annotations

import logging
import math

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Required columns for each public function's input.
_ALPHA_OPP_REQUIRED: frozenset[str] = frozenset({"city", "absolute_alpha", "signal_side"})
_ERROR_REQUIRED: frozenset[str] = frozenset(
    {"city", "model_name", "metric", "error_f", "absolute_error_f", "squared_error"}
)

# Assumed baseline ensemble standard deviation (°F) used to scale the
# variance_multiplier recommendation.  5 °F is a reasonable mid-range prior
# for US city temperature ensemble spreads.
_BASELINE_STD_F: float = 5.0

# kde_bandwidth = max(floor, rmse_f * factor)
_KDE_BANDWIDTH_FACTOR: float = 0.3
_KDE_BANDWIDTH_FLOOR: float = 1.0

# Clamps applied to the variance_multiplier recommendation.
_MIN_VARIANCE_MULTIPLIER: float = 0.5
_MAX_VARIANCE_MULTIPLIER: float = 3.0

# Canonical output column orders.
_CITY_SCORE_COLS: list[str] = [
    "city",
    "trade_count",
    "mean_absolute_alpha",
    "mean_rank_score",
    "tail_opp_rate",
    "city_alpha_score",
]
_THRESHOLD_SCORE_COLS: list[str] = [
    "city",
    "metric",
    "threshold_f",
    "comparison_operator",
    "trade_count",
    "mean_absolute_alpha",
    "tail_opp_rate",
    "threshold_score",
]
_MARKET_TYPE_COLS: list[str] = [
    "city",
    "metric",
    "trade_count",
    "mean_absolute_alpha",
    "mean_microstructure_score",
    "market_type_score",
]
_PARAM_REC_COLS: list[str] = [
    "city",
    "model_name",
    "metric",
    "sample_count",
    "mean_error_f",
    "rmse_f",
    "recommended_mean_bias_correction_f",
    "recommended_variance_multiplier",
    "recommended_kde_bandwidth",
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_columns(
    df: pd.DataFrame,
    required: frozenset[str],
    df_name: str,
) -> None:
    """Raise ValueError if *df* is missing any of the *required* columns."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{df_name} is missing required columns: {sorted(missing)}"
        )


def _add_tail_opp_rate(
    result: pd.DataFrame,
    executed: pd.DataFrame,
    group_keys: list[str],
) -> pd.DataFrame:
    """Merge ``tail_opp_rate`` into *result* from *executed*.

    When ``tail_opportunity_flag`` is absent from *executed*, adds a column
    of ``0.0`` values.  NaN flags are treated as ``False``.
    """
    if "tail_opportunity_flag" not in executed.columns:
        result["tail_opp_rate"] = 0.0
        return result

    tail_rates = (
        executed.assign(
            _tail=lambda df: df["tail_opportunity_flag"].fillna(False).astype(bool)
        )
        .groupby(group_keys, sort=False)["_tail"]
        .mean()
        .rename("tail_opp_rate")
        .reset_index()
    )
    result = result.merge(tail_rates, on=group_keys, how="left")
    result["tail_opp_rate"] = result["tail_opp_rate"].fillna(0.0)
    return result


def _add_mean_rank_score(
    result: pd.DataFrame,
    executed: pd.DataFrame,
    group_keys: list[str],
) -> pd.DataFrame:
    """Merge ``mean_rank_score`` into *result* from *executed*.

    When ``rank_score`` is absent, adds a column of ``None`` values.
    NaN rank_score rows are excluded from the mean (missing data, not zero).
    """
    if "rank_score" not in executed.columns:
        result["mean_rank_score"] = None
        return result

    rank_means = (
        executed.groupby(group_keys, sort=False)["rank_score"]
        .mean()
        .rename("mean_rank_score")
        .reset_index()
    )
    result = result.merge(rank_means, on=group_keys, how="left")
    return result


def _add_mean_microstructure_score(
    result: pd.DataFrame,
    executed: pd.DataFrame,
    group_keys: list[str],
) -> pd.DataFrame:
    """Merge ``mean_microstructure_score`` into *result* from *executed*.

    When ``microstructure_score`` is absent, adds a column of ``None`` values.
    """
    if "microstructure_score" not in executed.columns:
        result["mean_microstructure_score"] = None
        return result

    micro_means = (
        executed.groupby(group_keys, sort=False)["microstructure_score"]
        .mean()
        .rename("mean_microstructure_score")
        .reset_index()
    )
    result = result.merge(micro_means, on=group_keys, how="left")
    return result


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def score_cities_by_alpha_opportunity(
    alpha_opportunity_df: pd.DataFrame,
) -> pd.DataFrame:
    """Rank cities by their historical alpha opportunity quality.

    Aggregates ``alpha_opportunity`` records by ``city`` and computes a
    composite ``city_alpha_score``.  Only non-HOLD signals (rows where
    ``signal_side != "HOLD"``) are counted — HOLD rows represent no
    actionable mispricing and are excluded from all metrics.

    Scoring formula::

        city_alpha_score = mean_absolute_alpha

    ``mean_rank_score`` is included as a supplementary column.  Callers may
    use it as an alternative primary sort when all historical records contain
    ``rank_score`` data.

    Args:
        alpha_opportunity_df: Historical alpha opportunity records conforming
            to ``alpha_opportunity`` schema (DATA_SCHEMA.md §17).
            Required columns: ``city``, ``absolute_alpha``, ``signal_side``.
            Optional columns used when present: ``rank_score``,
            ``tail_opportunity_flag``.

    Returns:
        DataFrame with one row per city, sorted descending by
        ``city_alpha_score``.  Columns: ``city``, ``trade_count``,
        ``mean_absolute_alpha``, ``mean_rank_score``, ``tail_opp_rate``,
        ``city_alpha_score``.

        Returns an empty DataFrame (with those columns) when the input is
        empty or contains no non-HOLD rows.

    Raises:
        ValueError: If required columns are missing from the input.

    Example::

        scores = score_cities_by_alpha_opportunity(alpha_df)
        best_city = scores.iloc[0]["city"]  # highest-alpha city
    """
    if alpha_opportunity_df.empty:
        logger.info("score_cities_by_alpha_opportunity: empty input")
        return pd.DataFrame(columns=_CITY_SCORE_COLS)

    _validate_columns(alpha_opportunity_df, _ALPHA_OPP_REQUIRED, "alpha_opportunity_df")

    executed = alpha_opportunity_df[alpha_opportunity_df["signal_side"] != "HOLD"]
    if executed.empty:
        logger.info(
            "score_cities_by_alpha_opportunity: no non-HOLD rows — returning empty result"
        )
        return pd.DataFrame(columns=_CITY_SCORE_COLS)

    group_keys = ["city"]
    result = (
        executed.groupby(group_keys, sort=True)
        .agg(
            trade_count=("absolute_alpha", "count"),
            mean_absolute_alpha=("absolute_alpha", "mean"),
        )
        .reset_index()
    )

    result = _add_mean_rank_score(result, executed, group_keys)
    result = _add_tail_opp_rate(result, executed, group_keys)

    result["city_alpha_score"] = result["mean_absolute_alpha"]
    result = result.sort_values("city_alpha_score", ascending=False).reset_index(drop=True)

    logger.info(
        "score_cities_by_alpha_opportunity: scored %d cities from %d executed signals",
        len(result),
        len(executed),
    )
    return result[_CITY_SCORE_COLS]


def score_thresholds_by_profitability(
    alpha_opportunity_df: pd.DataFrame,
) -> pd.DataFrame:
    """Rank city-threshold combinations by historical alpha profitability.

    Groups by ``(city, metric, threshold_f, comparison_operator)`` and
    computes a ``threshold_score`` per group.  This identifies which specific
    contract shapes (e.g. Chicago high_temp_f >= 85 °F) have historically
    offered the most mispricing.

    Scoring formula::

        threshold_score = mean_absolute_alpha

    Only non-HOLD signals are included.  Optional grouping columns
    (``metric``, ``threshold_f``, ``comparison_operator``) are used when
    present in the input; if absent they are set to ``None`` in the output
    and the function effectively degenerates to a city-level score.

    Args:
        alpha_opportunity_df: Historical alpha opportunity records.
            Required columns: ``city``, ``absolute_alpha``, ``signal_side``.
            Optional columns used when present: ``metric``, ``threshold_f``,
            ``comparison_operator``, ``tail_opportunity_flag``.

    Returns:
        DataFrame sorted descending by ``threshold_score``.  Columns:
        ``city``, ``metric``, ``threshold_f``, ``comparison_operator``,
        ``trade_count``, ``mean_absolute_alpha``, ``tail_opp_rate``,
        ``threshold_score``.

        Returns empty DataFrame (with those columns) when input is empty.

    Raises:
        ValueError: If required columns are missing.

    Example::

        scores = score_thresholds_by_profitability(alpha_df)
        best_row = scores.iloc[0]
        print(f"{best_row.city} {best_row.metric} >= {best_row.threshold_f}")
    """
    if alpha_opportunity_df.empty:
        logger.info("score_thresholds_by_profitability: empty input")
        return pd.DataFrame(columns=_THRESHOLD_SCORE_COLS)

    _validate_columns(alpha_opportunity_df, _ALPHA_OPP_REQUIRED, "alpha_opportunity_df")

    executed = alpha_opportunity_df[alpha_opportunity_df["signal_side"] != "HOLD"]
    if executed.empty:
        logger.info(
            "score_thresholds_by_profitability: no non-HOLD rows — returning empty result"
        )
        return pd.DataFrame(columns=_THRESHOLD_SCORE_COLS)

    group_keys = ["city"]
    for col in ("metric", "threshold_f", "comparison_operator"):
        if col in executed.columns:
            group_keys.append(col)

    result = (
        executed.groupby(group_keys, sort=True)
        .agg(
            trade_count=("absolute_alpha", "count"),
            mean_absolute_alpha=("absolute_alpha", "mean"),
        )
        .reset_index()
    )

    result = _add_tail_opp_rate(result, executed, group_keys)

    result["threshold_score"] = result["mean_absolute_alpha"]
    result = result.sort_values("threshold_score", ascending=False).reset_index(drop=True)

    # Ensure all expected output columns exist (fill absent ones with None).
    for col in _THRESHOLD_SCORE_COLS:
        if col not in result.columns:
            result[col] = None

    logger.info(
        "score_thresholds_by_profitability: scored %d city/threshold groups",
        len(result),
    )
    return result[_THRESHOLD_SCORE_COLS]


def score_market_types_by_profitability(
    alpha_opportunity_df: pd.DataFrame,
) -> pd.DataFrame:
    """Rank market types (city × metric) by alpha profitability.

    Groups by ``(city, metric)`` — e.g. ``("Chicago", "high_temp_f")`` vs
    ``("Chicago", "low_temp_f")`` — and scores each combination.
    Microstructure quality is included as a supplementary signal when
    ``microstructure_score`` is present in the input.

    Scoring formula::

        market_type_score = mean_absolute_alpha

    ``mean_microstructure_score`` is included as a supplementary column.
    High ``mean_absolute_alpha`` combined with high ``mean_microstructure_score``
    indicates a market that is both mispriced and tradeable.

    Only non-HOLD signals are included.  If ``metric`` is absent from the
    input the function groups by ``city`` only and sets ``metric`` to ``None``
    in the output.

    Args:
        alpha_opportunity_df: Historical alpha opportunity records.
            Required columns: ``city``, ``absolute_alpha``, ``signal_side``.
            Optional columns used when present: ``metric``,
            ``microstructure_score``.

    Returns:
        DataFrame sorted descending by ``market_type_score``.  Columns:
        ``city``, ``metric``, ``trade_count``, ``mean_absolute_alpha``,
        ``mean_microstructure_score``, ``market_type_score``.

        Returns empty DataFrame (with those columns) when input is empty.

    Raises:
        ValueError: If required columns are missing.

    Example::

        scores = score_market_types_by_profitability(alpha_df)
        print(scores[["city", "metric", "market_type_score"]])
    """
    if alpha_opportunity_df.empty:
        logger.info("score_market_types_by_profitability: empty input")
        return pd.DataFrame(columns=_MARKET_TYPE_COLS)

    _validate_columns(alpha_opportunity_df, _ALPHA_OPP_REQUIRED, "alpha_opportunity_df")

    executed = alpha_opportunity_df[alpha_opportunity_df["signal_side"] != "HOLD"]
    if executed.empty:
        logger.info(
            "score_market_types_by_profitability: no non-HOLD rows — returning empty result"
        )
        return pd.DataFrame(columns=_MARKET_TYPE_COLS)

    group_keys = ["city"]
    if "metric" in executed.columns:
        group_keys.append("metric")

    result = (
        executed.groupby(group_keys, sort=True)
        .agg(
            trade_count=("absolute_alpha", "count"),
            mean_absolute_alpha=("absolute_alpha", "mean"),
        )
        .reset_index()
    )

    result = _add_mean_microstructure_score(result, executed, group_keys)

    result["market_type_score"] = result["mean_absolute_alpha"]
    result = result.sort_values("market_type_score", ascending=False).reset_index(drop=True)

    for col in _MARKET_TYPE_COLS:
        if col not in result.columns:
            result[col] = None

    logger.info(
        "score_market_types_by_profitability: scored %d city/metric groups",
        len(result),
    )
    return result[_MARKET_TYPE_COLS]


def recommend_city_profile_params(
    forecast_error_df: pd.DataFrame,
) -> pd.DataFrame:
    """Recommend city profile parameter updates from historical forecast error.

    Analyses ``forecast_error_history`` records to suggest updated values for
    three city profile fields that directly affect the probability engine:

    - ``recommended_mean_bias_correction_f``:
        ``-mean_error_f``.  The negative of the observed mean signed error.
        If the model is consistently +2 °F warm, the recommended correction
        is -2 °F.

    - ``recommended_variance_multiplier``:
        ``clamp(rmse_f / 5.0, 0.5, 3.0)``.
        A high RMSE relative to the 5 °F assumed baseline suggests the
        ensemble spread should be widened.  Clamped to [0.5, 3.0].

    - ``recommended_kde_bandwidth``:
        ``max(1.0, rmse_f * 0.3)``.
        Higher RMSE warrants a wider KDE to better represent forecast
        uncertainty.  Floor of 1.0 prevents over-smoothing.

    Recommendations are produced per ``(city, model_name, metric)`` group.
    To derive a single value for a city profile YAML, average or weight
    the per-model recommendations before applying them.

    Args:
        forecast_error_df: Historical forecast error records conforming to
            the ``forecast_error_history`` schema (DATA_SCHEMA.md §9).
            Required columns: ``city``, ``model_name``, ``metric``,
            ``error_f``, ``absolute_error_f``, ``squared_error``.

    Returns:
        DataFrame with one row per ``(city, model_name, metric)`` group,
        sorted by those keys.  Columns: ``city``, ``model_name``, ``metric``,
        ``sample_count``, ``mean_error_f``, ``rmse_f``,
        ``recommended_mean_bias_correction_f``,
        ``recommended_variance_multiplier``,
        ``recommended_kde_bandwidth``.

        Returns empty DataFrame (with those columns) when input is empty.

    Raises:
        ValueError: If required columns are missing.

    Example::

        recs = recommend_city_profile_params(error_df)
        chicago = recs[recs.city == "Chicago"]
        # recommended_mean_bias_correction_f → e.g. -1.5 (was running 1.5 warm)
        # recommended_variance_multiplier    → e.g. 1.2
        # recommended_kde_bandwidth          → e.g. 1.5
    """
    if forecast_error_df.empty:
        logger.info("recommend_city_profile_params: empty input")
        return pd.DataFrame(columns=_PARAM_REC_COLS)

    _validate_columns(forecast_error_df, _ERROR_REQUIRED, "forecast_error_df")

    agg = (
        forecast_error_df.groupby(["city", "model_name", "metric"], sort=True)
        .agg(
            sample_count=("error_f", "count"),
            mean_error_f=("error_f", "mean"),
            mean_squared_error=("squared_error", "mean"),
        )
        .reset_index()
    )

    agg["rmse_f"] = agg["mean_squared_error"].apply(math.sqrt)
    agg = agg.drop(columns=["mean_squared_error"])

    agg["recommended_mean_bias_correction_f"] = -agg["mean_error_f"]

    agg["recommended_variance_multiplier"] = agg["rmse_f"].apply(
        lambda r: float(
            max(_MIN_VARIANCE_MULTIPLIER, min(_MAX_VARIANCE_MULTIPLIER, r / _BASELINE_STD_F))
        )
    )

    agg["recommended_kde_bandwidth"] = agg["rmse_f"].apply(
        lambda r: float(max(_KDE_BANDWIDTH_FLOOR, r * _KDE_BANDWIDTH_FACTOR))
    )

    logger.info(
        "recommend_city_profile_params: generated %d recommendations from %d error rows",
        len(agg),
        len(forecast_error_df),
    )
    return agg[_PARAM_REC_COLS]
