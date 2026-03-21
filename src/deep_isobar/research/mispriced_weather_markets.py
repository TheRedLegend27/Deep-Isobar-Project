"""Mispriced weather markets analyzer for Deep Isobar — Step 27.

Phase 10 targeting and optimisation.  Two levels of interface:

**Level 1 — Generic scoring functions** (used by multi-city research pipelines):
These accept pre-built feature-store DataFrames (alpha_opportunity_df,
forecast_error_df) and aggregate them into ranked tables.  They are input-
agnostic and can be wired to any upstream data source.

    score_cities_by_alpha_opportunity(alpha_opportunity_df) -> pd.DataFrame
    score_thresholds_by_profitability(alpha_opportunity_df) -> pd.DataFrame
    score_market_types_by_profitability(alpha_opportunity_df) -> pd.DataFrame
    recommend_city_profile_params(forecast_error_df) -> pd.DataFrame

**Level 2 — Real-data backtest analysis** (specific to Chicago 2023 backtest):
``analyze_mispriced_markets()`` loads the historical parquets directly, rebuilds
the backtest trade log, and answers four targeted questions:

    1. Which thresholds (≥80°F vs ≥90°F) drove the most drawdown?
    2. What is the optimal ``mean_bias_correction_f`` from real GFS errors?
    3. Which effective std (sigma) minimises the Brier calibration score?
       → implies the recommended ``variance_multiplier`` and sigma floor.
    4. What ``kde_bandwidth`` does Brier-optimal sigma imply?

    Usage::

        python -m deep_isobar.research.mispriced_weather_markets [--city Chicago]

This module is a **research tool** only.  It does not connect to live exchange
APIs, the scheduler, or the execution layer.

Generic input schemas
---------------------
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

Output schemas (generic scoring functions)
------------------------------------------
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
    Floor of 1.0 °F prevents over-smoothing."""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

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


# ===========================================================================
# Level 2 — Real-data backtest analysis (Chicago 2023)
# ===========================================================================
#
# analyze_mispriced_markets() loads historical parquets directly, rebuilds the
# backtest trade log with rich per-row diagnostics, then answers four specific
# questions about the Chicago 2023 backtest results.
#
# Paths and constants deliberately mirror run_chicago_backtest.py so that
# the analysis represents exactly what the backtest driver saw.

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Per-city parquet paths.  Keys are city names from config/cities.yaml.
# Add a new entry here when expanding to additional cities.
_CITY_PATHS: dict[str, dict[str, Path]] = {
    "Chicago": {
        "settlement": _PROJECT_ROOT / "data/historical/settlement/chicago_2023.parquet",
        "forecasts":  _PROJECT_ROOT / "data/historical/forecasts/gfs_chicago_2023.parquet",
        "snapshots":  _PROJECT_ROOT / "data/historical/markets/kalshi_kxhighchi_2023.parquet",
        "contracts":  _PROJECT_ROOT / "data/historical/markets/kalshi_kxhighchi_2023_contracts.parquet",
    },
    "New York": {
        "settlement": _PROJECT_ROOT / "data/historical/settlement/nyc_2023.parquet",
        "forecasts":  _PROJECT_ROOT / "data/historical/forecasts/gfs_nyc_2023.parquet",
        "snapshots":  _PROJECT_ROOT / "data/historical/markets/kalshi_kxhighny_2023.parquet",
        "contracts":  _PROJECT_ROOT / "data/historical/markets/kalshi_kxhighny_2023_contracts.parquet",
    },
}

# Kept for backward compatibility — Chicago is the default when paths are
# resolved via the module-level constants (not used by Level 2 any more).
_SETTLEMENT_PATH = _CITY_PATHS["Chicago"]["settlement"]
_FORECAST_PATH   = _CITY_PATHS["Chicago"]["forecasts"]
_SNAPSHOTS_PATH  = _CITY_PATHS["Chicago"]["snapshots"]
_CONTRACTS_PATH  = _CITY_PATHS["Chicago"]["contracts"]

_MIN_ENSEMBLE_STD_F       = 3.0   # hardcoded floor when single-model run → std=0
_PRIMARY_MAX_LEAD         = 48
_FALLBACK_MAX_LEAD        = 72
_DECISION_CUTOFF_HOUR_UTC = 12
_SIGNAL_THRESHOLD         = 0.08
_MAX_SPREAD               = 0.40  # wide-spread hard-stop (alpha_engine)
_QUANTITY                 = 10.0
_BACKTEST_THRESHOLDS      = [80, 90]
_BACKTEST_METRIC          = "high_temp_f"

# Sigma sweep for Brier calibration analysis
_SIGMA_MIN  = 1.0
_SIGMA_MAX  = 9.0
_SIGMA_STEP = 0.25


# ---------------------------------------------------------------------------
# Level 2 — data loaders (local, no coupling to run_chicago_backtest)
# ---------------------------------------------------------------------------


def _l2_load_settlement(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path or _SETTLEMENT_PATH)
    if pd.api.types.is_datetime64_any_dtype(df["target_date"]):
        df["target_date"] = df["target_date"].dt.date
    return df[df["quality_flag"] != "missing"].copy()


def _l2_load_forecasts(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path or _FORECAST_PATH)
    if pd.api.types.is_datetime64_any_dtype(df["target_date"]):
        df["target_date"] = df["target_date"].dt.date
    if df["run_time_utc"].dt.tz is None:
        df["run_time_utc"] = df["run_time_utc"].dt.tz_localize("UTC")
    return df


def _l2_load_market(
    snapshots_path: Path | None = None,
    contracts_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    snapshots = pd.read_parquet(snapshots_path or _SNAPSHOTS_PATH)
    if snapshots["timestamp_utc"].dt.tz is None:
        snapshots["timestamp_utc"] = snapshots["timestamp_utc"].dt.tz_localize("UTC")
    contracts = pd.read_parquet(contracts_path or _CONTRACTS_PATH)
    if pd.api.types.is_datetime64_any_dtype(contracts["target_date"]):
        contracts["target_date"] = contracts["target_date"].dt.date
    contracts["threshold_f"] = pd.to_numeric(contracts["threshold_f"], errors="coerce")
    return snapshots, contracts


# ---------------------------------------------------------------------------
# Level 2 — inner helpers
# ---------------------------------------------------------------------------


def _l2_best_gfs_row(forecasts_df: pd.DataFrame, target_date: date) -> pd.Series | None:
    """Shortest-lead GFS row for target_date (≤48 h preferred, ≤72 h fallback)."""
    day = forecasts_df[forecasts_df["target_date"] == target_date]
    if day.empty:
        return None
    for max_lead in (_PRIMARY_MAX_LEAD, _FALLBACK_MAX_LEAD):
        cands = day[day["lead_hours"] <= max_lead]
        if not cands.empty:
            min_lead = cands["lead_hours"].min()
            return cands[cands["lead_hours"] == min_lead].iloc[0]
    return None


def _l2_snapshot_prices(
    snapshots_df: pd.DataFrame,
    contract_id: str,
    cutoff: datetime,
) -> tuple[float, float] | None:
    sub = snapshots_df[snapshots_df["contract_id"] == contract_id]
    before = sub[sub["timestamp_utc"] < cutoff]
    if before.empty:
        return None
    last = before.sort_values("timestamp_utc").iloc[-1]
    return float(last["best_bid"]), float(last["best_ask"])


def _l2_max_drawdown(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    cum = pnl.cumsum()
    with_origin = pd.concat([pd.Series([0.0]), cum.reset_index(drop=True)], ignore_index=True)
    return float((with_origin.cummax() - with_origin).max())


# ---------------------------------------------------------------------------
# Level 2 — rich trade log builder
# ---------------------------------------------------------------------------


def _build_rich_trade_log(city: str = "Chicago") -> pd.DataFrame:
    """Rebuild the backtest trade log for *city* with per-row diagnostics.

    Includes ALL (date, threshold) rows that had GFS + Kalshi data,
    not just rows that generated a trade.  HOLD rows are flagged
    ``traded=False`` and contribute to calibration analysis without
    biasing it toward high-alpha rows.

    Thresholds are derived dynamically from T-format contracts in the
    contracts parquet (``contract_id`` matching ``-T\\d``), so cities with
    non-fixed thresholds (e.g. NYC) work correctly.

    Columns
    -------
    target_date, threshold_f, contract_id,
    gfs_raw_f, actual_high_f, forecast_error_f,
    bias_corrected_mean_f, effective_std_f,
    model_prob, market_prob, alpha, spread,
    entry_price_mid, realized_outcome,
    side, traded, pnl_raw, realized_pnl
    """
    # Lazy imports — only needed for Level 2 analysis
    from deep_isobar.core.types import ForecastPoint, OrderBookSnapshot
    from deep_isobar.data.city_universe import get_city_profile
    from deep_isobar.market.market_price_adapter import compute_market_probability
    from deep_isobar.models.probability_engine import probability_ge_normal
    from deep_isobar.models.temperature_ensemble import build_temperature_ensemble

    city_profile = get_city_profile(city)

    paths = _CITY_PATHS.get(city)
    if paths is None:
        raise ValueError(
            f"No data paths configured for city '{city}'. "
            f"Add an entry to _CITY_PATHS in mispriced_weather_markets.py."
        )

    settlement_df           = _l2_load_settlement(paths["settlement"])
    forecasts_df            = _l2_load_forecasts(paths["forecasts"])
    snapshots_df, contracts = _l2_load_market(paths["snapshots"], paths["contracts"])

    # Derive thresholds dynamically from T-format contracts (e.g. HIGHNY-23AUG15-T85)
    t_contracts = contracts[
        contracts["contract_id"].str.contains(r"-T\d", regex=True, na=False)
    ].copy()

    # Build (date, threshold) -> contract_id index from T-format contracts
    idx_map: dict[tuple, str] = {}
    for _, row in t_contracts.iterrows():
        thr = row["threshold_f"]
        if pd.notna(thr):
            idx_map[(row["target_date"], int(thr))] = row["contract_id"]

    # Per-date threshold lists (sorted) for the main loop
    from collections import defaultdict
    date_thresholds: dict = defaultdict(list)
    for (td, thr) in idx_map:
        date_thresholds[td].append(thr)
    date_thresholds = {td: sorted(thrs) for td, thrs in date_thresholds.items()}

    settlement_by_date: dict[date, float] = {
        row["target_date"]: float(row["high_temp_f"])
        for _, row in settlement_df.iterrows()
    }
    active_dates = sorted(
        set(settlement_df["target_date"].unique()) & set(forecasts_df["target_date"].unique())
    )

    rows: list[dict] = []
    for target_date in active_dates:
        actual_high_f = settlement_by_date[target_date]
        gfs_row       = _l2_best_gfs_row(forecasts_df, target_date)
        if gfs_row is None:
            continue

        gfs_raw_f = float(gfs_row["forecast_value_f"])
        fp = ForecastPoint(
            city=str(gfs_row["city"]),
            station_id=str(gfs_row["station_id"]),
            model_name=str(gfs_row["model_name"]),
            run_time_utc=gfs_row["run_time_utc"],
            target_date=target_date,
            metric=_BACKTEST_METRIC,
            forecast_value_f=gfs_raw_f,
            lead_hours=int(gfs_row["lead_hours"]),
            source_name=str(gfs_row["source_name"]),
        )
        ensemble = build_temperature_ensemble(
            city_profile=city_profile,
            forecasts=[fp],
            target_date=target_date,
            metric=_BACKTEST_METRIC,
            ensemble_run_time_utc=fp.run_time_utc,
        )
        effective_std       = max(ensemble.adjusted_std_f, _MIN_ENSEMBLE_STD_F)
        bias_corrected_mean = ensemble.bias_corrected_mean_f

        cutoff = datetime(
            target_date.year, target_date.month, target_date.day,
            _DECISION_CUTOFF_HOUR_UTC, 0, 0, tzinfo=timezone.utc,
        )

        thresholds_today = date_thresholds.get(target_date, [])
        for threshold_f in thresholds_today:
            contract_id = idx_map.get((target_date, threshold_f))
            if contract_id is None:
                continue

            prices = _l2_snapshot_prices(snapshots_df, contract_id, cutoff)
            if prices is None:
                continue

            best_bid, best_ask = prices
            spread          = best_ask - best_bid
            entry_price_mid = (best_bid + best_ask) / 2.0

            model_prob = probability_ge_normal(
                mean_f=bias_corrected_mean,
                std_f=effective_std,
                threshold_f=float(threshold_f),
            )
            snapshot = OrderBookSnapshot(
                timestamp_utc=cutoff,
                contract_id=contract_id,
                market_source="Kalshi",
                best_bid=best_bid,
                best_ask=best_ask,
            )
            market_prob      = compute_market_probability(snapshot)
            alpha            = model_prob - market_prob
            realized_outcome = int(actual_high_f >= threshold_f)

            # Mirror alpha_engine: wide-spread hard-stop, then threshold
            if spread > _MAX_SPREAD or abs(alpha) < _SIGNAL_THRESHOLD:
                side = "HOLD"
            elif alpha > 0:
                side = "BUY"
            else:
                side = "SELL"

            pnl_raw = (
                (realized_outcome - entry_price_mid) if side == "BUY"
                else (entry_price_mid - realized_outcome) if side == "SELL"
                else 0.0
            )

            rows.append({
                "target_date":           target_date,
                "threshold_f":           threshold_f,
                "contract_id":           contract_id,
                "gfs_raw_f":             gfs_raw_f,
                "actual_high_f":         actual_high_f,
                "forecast_error_f":      gfs_raw_f - actual_high_f,
                "bias_corrected_mean_f": bias_corrected_mean,
                "effective_std_f":       effective_std,
                "model_prob":            model_prob,
                "market_prob":           market_prob,
                "alpha":                 alpha,
                "spread":                spread,
                "entry_price_mid":       entry_price_mid,
                "realized_outcome":      realized_outcome,
                "side":                  side,
                "traded":                side != "HOLD",
                "pnl_raw":               pnl_raw,
                "realized_pnl":          pnl_raw * _QUANTITY,
            })

    df = pd.DataFrame(rows)
    n_traded = int(df["traded"].sum()) if not df.empty else 0
    logger.info(
        "Rich trade log: %d total rows | %d traded | %d HOLD",
        len(df), n_traded, len(df) - n_traded,
    )
    return df


# ---------------------------------------------------------------------------
# Level 2 — analysis functions
# ---------------------------------------------------------------------------


_DRAWDOWN_COLS: list[str] = [
    "threshold_f", "n_trades", "n_wins", "win_rate",
    "gross_pnl", "max_drawdown", "avg_loss",
]


def _analysis1_drawdown_by_threshold(df: pd.DataFrame) -> pd.DataFrame:
    """Per-threshold PnL statistics sorted by worst max-drawdown first.

    Returns an empty DataFrame (with the correct columns) when no trades
    were generated (all HOLD or df is empty).
    """
    traded = df[df["traded"]].copy().sort_values("target_date")
    if traded.empty:
        return pd.DataFrame(columns=_DRAWDOWN_COLS)

    records: list[dict] = []
    for thr in sorted(traded["threshold_f"].unique()):
        sub       = traded[traded["threshold_f"] == thr]
        n         = len(sub)
        n_wins    = int((sub["realized_pnl"] > 0).sum())
        gross_pnl = float(sub["realized_pnl"].sum())
        max_dd    = _l2_max_drawdown(sub["realized_pnl"])
        loss_rows = sub.loc[sub["realized_pnl"] < 0, "realized_pnl"]
        avg_loss  = float(loss_rows.mean()) if not loss_rows.empty else 0.0
        records.append({
            "threshold_f":  int(thr),
            "n_trades":     n,
            "n_wins":       n_wins,
            "win_rate":     round(n_wins / n, 3),
            "gross_pnl":    round(gross_pnl, 2),
            "max_drawdown": round(max_dd, 2),
            "avg_loss":     round(avg_loss, 2),
        })
    return pd.DataFrame(records).sort_values("max_drawdown", ascending=False)


def _analysis2_forecast_bias(df: pd.DataFrame, current_correction: float) -> dict:
    """GFS forecast error stats and recommended mean_bias_correction_f.

    ``forecast_error_f = gfs_raw_f - actual_high_f``
    Positive mean error → GFS runs systematically warm.
    Optimal correction = -mean_error (zeroes out mean residual).
    """
    date_df = df.drop_duplicates("target_date")
    errors  = date_df["forecast_error_f"]

    mean_error   = float(errors.mean())
    median_error = float(errors.median())
    rmse         = float(np.sqrt((errors ** 2).mean()))
    std_error    = float(errors.std(ddof=1))

    residual      = errors + current_correction   # bias_corrected - actual
    residual_mean = float(residual.mean())
    residual_rmse = float(np.sqrt((residual ** 2).mean()))

    return {
        "n_dates":                     len(date_df),
        "gfs_mean_error_f":            round(mean_error, 3),
        "gfs_median_error_f":          round(median_error, 3),
        "gfs_rmse_f":                  round(rmse, 3),
        "gfs_error_std_f":             round(std_error, 3),
        "current_bias_correction_f":   round(current_correction, 3),
        "residual_mean_error_f":       round(residual_mean, 3),
        "residual_rmse_f":             round(residual_rmse, 3),
        "recommended_bias_correction": round(-mean_error, 2),
    }


def _analysis3_sigma_brier_sweep(
    df: pd.DataFrame,
    optimal_bias_correction: float,
) -> pd.DataFrame:
    """Brier score over a range of effective-std values.

    Uses ``optimal_bias_correction`` (not the current one) so the sigma
    sweep is orthogonal to the bias recommendation.  All rows are used
    (including HOLDs) to avoid selection bias.

    Returns a DataFrame with columns ``sigma_f``, ``brier_score``,
    ``mean_calib_error``.
    """
    corrected_mean = (df["gfs_raw_f"] + optimal_bias_correction).values
    thresholds     = df["threshold_f"].values.astype(float)
    outcomes       = df["realized_outcome"].values.astype(float)

    sigmas = np.arange(_SIGMA_MIN, _SIGMA_MAX + _SIGMA_STEP / 2, _SIGMA_STEP)
    rows: list[dict] = []
    for sigma in sigmas:
        model_probs = np.clip(
            1.0 - norm.cdf(thresholds, loc=corrected_mean, scale=sigma),
            0.0, 1.0,
        )
        brier       = float(np.mean((model_probs - outcomes) ** 2))
        calib_error = float(np.mean(model_probs) - np.mean(outcomes))
        rows.append({
            "sigma_f":          round(float(sigma), 2),
            "brier_score":      round(brier, 6),
            "mean_calib_error": round(calib_error, 4),
        })
    return pd.DataFrame(rows)


def _calibration_decile_table(
    df: pd.DataFrame,
    optimal_bias_correction: float,
    optimal_sigma: float,
) -> pd.DataFrame:
    """Predicted vs actual frequency in 10 equal-width probability bins.

    Uses optimal parameters so the table reflects the recommended settings.
    A well-calibrated model shows ``actual_freq ≈ mean_pred`` in every row.
    """
    corrected_mean = (df["gfs_raw_f"] + optimal_bias_correction).values
    model_probs = np.clip(
        1.0 - norm.cdf(df["threshold_f"].values.astype(float), loc=corrected_mean, scale=optimal_sigma),
        0.0, 1.0,
    )
    outcomes  = df["realized_outcome"].values.astype(float)
    bins      = np.linspace(0.0, 1.0, 11)
    bin_labels = [f"{bins[i]:.1f}–{bins[i+1]:.1f}" for i in range(10)]
    bin_idx   = np.digitize(model_probs, bins[1:-1])   # 0..9

    records: list[dict] = []
    for b in range(10):
        mask = bin_idx == b
        n    = int(mask.sum())
        if n == 0:
            continue
        mean_pred   = float(model_probs[mask].mean())
        actual_freq = float(outcomes[mask].mean())
        records.append({
            "prob_bin":    bin_labels[b],
            "n":           n,
            "mean_pred":   round(mean_pred, 3),
            "actual_freq": round(actual_freq, 3),
            "calib_err":   round(actual_freq - mean_pred, 3),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Level 2 — print helpers
# ---------------------------------------------------------------------------


def _section(title: str) -> None:
    sep = "=" * 70
    logger.info(sep)
    logger.info("  %s", title)
    logger.info(sep)


def _log_df(df: pd.DataFrame) -> None:
    logger.info("\n%s", df.to_string(index=False))


# ---------------------------------------------------------------------------
# Level 2 — main public entry point
# ---------------------------------------------------------------------------


def analyze_mispriced_markets(city: str = "Chicago") -> dict:
    """Full mispriced-weather-markets analysis on real historical parquets.

    Loads the four historical parquet files for *city* (Chicago or New York),
    rebuilds the backtest trade log, and runs four analyses:

    1. Drawdown attribution by temperature threshold (e.g. >=80/90 for Chicago,
       dynamic thresholds for NYC).
    2. GFS forecast bias → optimal ``mean_bias_correction_f``.
    3. Sigma calibration sweep (Brier score) → optimal effective_std_f,
       implied ``variance_multiplier``, and ``_MIN_ENSEMBLE_STD_F`` floor.
    4. KDE bandwidth recommendation (Silverman's rule + Brier-optimal sigma).

    All results are logged via ``logging.INFO`` and returned as a dict.

    Args:
        city: City name matching an entry in ``config/cities.yaml``.

    Returns:
        Dict with keys::

            mean_bias_correction_f   — float, ready to paste into cities.yaml
            variance_multiplier      — float, optimal_sigma / GFS_RMSE
            kde_bandwidth            — float, Brier-optimal sigma
            optimal_sigma_f          — float, Brier-optimal effective std
            gfs_rmse_f               — float, raw GFS RMSE over the period
            brier_score_current      — float, Brier at _MIN_ENSEMBLE_STD_F=3.0
            brier_score_optimal      — float, Brier at optimal sigma
            drawdown_by_threshold    — pd.DataFrame
            brier_sweep              — pd.DataFrame
            calibration_deciles      — pd.DataFrame

    Raises:
        FileNotFoundError: If any required parquet file is missing.
        RuntimeError: If the rich trade log is empty.
    """
    from deep_isobar.data.city_universe import get_city_profile

    _section(f"Mispriced Weather Markets Analyzer — {city}")

    city_paths = _CITY_PATHS.get(city)
    if city_paths is None:
        raise ValueError(
            f"No data paths configured for city '{city}'. "
            f"Add an entry to _CITY_PATHS in mispriced_weather_markets.py."
        )
    for path in city_paths.values():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing data file: {path}\n"
                "Run the ingestion scripts first — see run_chicago_backtest.py docstring."
            )

    city_profile = get_city_profile(city)
    logger.info(
        "Current city profile:  variance_multiplier=%.2f  "
        "mean_bias_correction_f=%.2f  kde_bandwidth=%.2f",
        city_profile.variance_multiplier,
        city_profile.mean_bias_correction_f,
        city_profile.kde_bandwidth,
    )

    rich_df = _build_rich_trade_log(city)
    if rich_df.empty:
        raise RuntimeError(
            "Rich trade log is empty — check that all four parquets cover the same date range."
        )

    # ── Analysis 1: Drawdown by threshold ────────────────────────────────
    _section("Analysis 1 — Drawdown Attribution by Threshold")
    drawdown_df = _analysis1_drawdown_by_threshold(rich_df)
    _log_df(drawdown_df)
    if not drawdown_df.empty:
        worst = drawdown_df.iloc[0]
        logger.info(
            "→ Threshold ≥%d°F has the largest drawdown (%.2f). "
            "Win rate = %.1f%%.  "
            "Consider raising signal_threshold or tightening spread filter for this bucket.",
            int(worst["threshold_f"]), worst["max_drawdown"], worst["win_rate"] * 100,
        )

    # ── Analysis 2: Forecast bias ─────────────────────────────────────────
    _section("Analysis 2 — GFS Forecast Bias → mean_bias_correction_f")
    bias_stats = _analysis2_forecast_bias(rich_df, city_profile.mean_bias_correction_f)
    _log_df(pd.DataFrame(list(bias_stats.items()), columns=["metric", "value"]))

    optimal_bias = bias_stats["recommended_bias_correction"]
    gfs_rmse     = bias_stats["gfs_rmse_f"]
    residual     = bias_stats["residual_mean_error_f"]

    if abs(residual) < 0.15:
        logger.info(
            "→ Current mean_bias_correction_f=%.2f is well-calibrated "
            "(residual %.3f°F < 0.15°F).  No change needed.",
            city_profile.mean_bias_correction_f, residual,
        )
    else:
        direction = "warm" if bias_stats["gfs_mean_error_f"] > 0 else "cold"
        logger.info(
            "→ GFS mean %s bias = %.3f°F over %d dates.  "
            "Current correction=%.2f leaves residual=%.3f°F.  "
            "Recommended mean_bias_correction_f = %.2f.",
            direction, abs(bias_stats["gfs_mean_error_f"]), bias_stats["n_dates"],
            city_profile.mean_bias_correction_f, residual, optimal_bias,
        )

    # ── Analysis 3: Sigma calibration sweep ──────────────────────────────
    _section("Analysis 3 — Sigma Calibration Sweep → variance_multiplier")
    brier_df      = _analysis3_sigma_brier_sweep(rich_df, optimal_bias)
    optimal_idx   = brier_df["brier_score"].idxmin()
    optimal_sigma = float(brier_df.loc[optimal_idx, "sigma_f"])
    optimal_brier = float(brier_df.loc[optimal_idx, "brier_score"])

    floor_row     = brier_df[brier_df["sigma_f"] == round(_MIN_ENSEMBLE_STD_F, 2)]
    current_brier = (
        float(floor_row["brier_score"].iloc[0]) if not floor_row.empty else float("nan")
    )

    logger.debug("Full Brier sweep:\n%s", brier_df.to_string(index=False))

    window = brier_df[
        (brier_df["sigma_f"] >= max(optimal_sigma - 2.0, _SIGMA_MIN))
        & (brier_df["sigma_f"] <= min(optimal_sigma + 2.0, _SIGMA_MAX))
    ]
    logger.info("Brier sweep — window around optimum (±2°F):")
    _log_df(window)

    current_brier_is_valid = not math.isnan(current_brier) and current_brier > 0
    brier_improvement_pct = (
        100 * (current_brier - optimal_brier) / current_brier
        if current_brier_is_valid
        else 0.0
    )
    logger.info(
        "→ Brier-optimal sigma = %.2f°F (score=%.6f). "
        "Current floor=%.1f°F (score=%.6f). "
        "Improvement = %.4f (%.1f%%).",
        optimal_sigma, optimal_brier,
        _MIN_ENSEMBLE_STD_F, current_brier,
        current_brier - optimal_brier, brier_improvement_pct,
    )

    implied_vm = round(optimal_sigma / gfs_rmse, 2) if gfs_rmse > 0 else float("nan")
    logger.info(
        "Implied variance_multiplier = %.2f / %.2f = %.2f  (current: %.2f).",
        optimal_sigma, gfs_rmse, implied_vm, city_profile.variance_multiplier,
    )

    # ── Calibration decile table ──────────────────────────────────────────
    _section("Calibration Decile Table at Recommended Parameters")
    calib_df = _calibration_decile_table(rich_df, optimal_bias, optimal_sigma)
    _log_df(calib_df)
    if not calib_df.empty:
        max_err = calib_df["calib_err"].abs().max()
        logger.info(
            "Max |calibration error| across deciles = %.3f  "
            "(<0.05 excellent, <0.10 acceptable for n<200).",
            max_err,
        )

    # ── Analysis 4: KDE bandwidth ─────────────────────────────────────────
    _section("Analysis 4 — KDE Bandwidth Recommendation")
    n_dates      = bias_stats["n_dates"]
    silverman_bw = round(1.06 * optimal_sigma * (n_dates ** -0.2), 2)
    logger.info(
        "Brier-optimal sigma = %.2f°F → use directly as kde_bandwidth "
        "when the KDE engine is active.",
        optimal_sigma,
    )
    logger.info(
        "Silverman rule (for KDE from temperature samples): "
        "h = 1.06 × %.2f × %d^(-0.2) = %.2f°F.",
        optimal_sigma, n_dates, silverman_bw,
    )
    logger.info(
        "Note: current backtest uses probability_ge_normal (not KDE); "
        "kde_bandwidth takes effect when the KDE engine is wired in."
    )

    # ── Final recommendation table ────────────────────────────────────────
    _section(f"RECOMMENDATIONS — paste into config/cities.yaml  ({city})")
    recs_df = pd.DataFrame([
        {
            "parameter":   "mean_bias_correction_f",
            "current":     city_profile.mean_bias_correction_f,
            "recommended": optimal_bias,
            "basis":       f"GFS mean error {bias_stats['gfs_mean_error_f']:+.2f}°F "
                           f"over {n_dates} dates; residual {residual:+.3f}°F after current correction",
        },
        {
            "parameter":   "variance_multiplier",
            "current":     city_profile.variance_multiplier,
            "recommended": implied_vm,
            "basis":       f"Brier-optimal σ ({optimal_sigma:.2f}°F) ÷ GFS RMSE ({gfs_rmse:.2f}°F)",
        },
        {
            "parameter":   "kde_bandwidth",
            "current":     city_profile.kde_bandwidth,
            "recommended": optimal_sigma,
            "basis":       "Brier-optimal effective std; use as kde_bandwidth when KDE engine active",
        },
        {
            "parameter":   "_MIN_ENSEMBLE_STD_F  [run_chicago_backtest.py]",
            "current":     _MIN_ENSEMBLE_STD_F,
            "recommended": optimal_sigma,
            "basis":       (
                f"Minimises Brier from {current_brier:.6f} → {optimal_brier:.6f} "
                f"({brier_improvement_pct:.1f}% improvement)"
                if current_brier_is_valid
                else f"Brier-optimal score={optimal_brier:.6f} (floor sigma not in sweep)"
            ),
        },
    ])
    _log_df(recs_df)

    return {
        "mean_bias_correction_f": optimal_bias,
        "variance_multiplier":    implied_vm,
        "kde_bandwidth":          optimal_sigma,
        "optimal_sigma_f":        optimal_sigma,
        "gfs_rmse_f":             gfs_rmse,
        "brier_score_current":    round(current_brier, 6),
        "brier_score_optimal":    round(optimal_brier, 6),
        "drawdown_by_threshold":  drawdown_df,
        "brier_sweep":            brier_df,
        "calibration_deciles":    calib_df,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Mispriced Weather Markets Analyzer — Deep Isobar Step 27"
    )
    parser.add_argument(
        "--city", default="Chicago",
        help="City name matching an entry in config/cities.yaml (default: Chicago)",
    )
    args = parser.parse_args()

    result = analyze_mispriced_markets(city=args.city)

    print("\n--- Scalar recommendations ---")
    for k in (
        "mean_bias_correction_f", "variance_multiplier", "kde_bandwidth",
        "optimal_sigma_f", "gfs_rmse_f", "brier_score_current", "brier_score_optimal",
    ):
        print(f"  {k:<42}: {result[k]}")
