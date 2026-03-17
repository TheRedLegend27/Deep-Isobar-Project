"""Historical forecast error tracking for Deep Isobar.

Compares historical forecast values against settled actual temperatures to
produce per-row error metrics and per-city/model aggregate summaries.

Join strategy
-------------
``compute_forecast_error`` joins ``forecast_df`` (long, one row per
model/run/target/metric) to ``actual_df`` (wide, one row per city/target_date
with ``high_temp_f`` and/or ``low_temp_f`` columns) on the composite key::

    (city, target_date, metric)

``actual_df`` is melted to long format internally before the join; the caller
does not need to reshape it.

Duplicate handling
------------------
Multiple forecast runs for the same ``(city, model_name, target_date, metric)``
are expected and produce one error row each.  True duplicates — identical rows
on the natural key — trigger a logged warning; no rows are silently dropped.

Example usage::

    import pandas as pd
    from datetime import date
    from deep_isobar.models.forecast_error import (
        compute_forecast_error,
        summarize_forecast_error_by_city_model,
    )

    forecast_df = pd.DataFrame({
        "city": ["Chicago", "Chicago"],
        "model_name": ["GFS", "ECMWF"],
        "target_date": [date(2026, 1, 1), date(2026, 1, 1)],
        "metric": ["high_temp_f", "high_temp_f"],
        "forecast_value_f": [35.0, 38.0],
    })

    actual_df = pd.DataFrame({
        "city": ["Chicago"],
        "target_date": [date(2026, 1, 1)],
        "high_temp_f": [33.0],
        "low_temp_f": [20.0],
    })

    error_df = compute_forecast_error(forecast_df, actual_df)
    summary_df = summarize_forecast_error_by_city_model(error_df)
    print(summary_df)
"""

from __future__ import annotations

import logging
import math

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column constants
# ---------------------------------------------------------------------------

_FORECAST_REQUIRED: frozenset[str] = frozenset(
    {"city", "model_name", "target_date", "metric", "forecast_value_f"}
)
_ACTUAL_REQUIRED: frozenset[str] = frozenset({"city", "target_date"})
_ACTUAL_VALUE_COLS: frozenset[str] = frozenset({"high_temp_f", "low_temp_f"})
_ERROR_REQUIRED: frozenset[str] = frozenset(
    {"city", "model_name", "metric", "error_f", "absolute_error_f", "squared_error"}
)

_JOIN_KEYS: list[str] = ["city", "target_date", "metric"]

_EMPTY_ERROR_COLS: list[str] = [
    "city",
    "model_name",
    "target_date",
    "metric",
    "forecast_value_f",
    "actual_value_f",
    "error_f",
    "absolute_error_f",
    "squared_error",
]

_EMPTY_SUMMARY_COLS: list[str] = [
    "city",
    "model_name",
    "metric",
    "sample_count",
    "mean_error_f",
    "mean_absolute_error_f",
    "rmse_f",
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_columns(
    df: pd.DataFrame,
    required: frozenset[str],
    df_name: str,
) -> None:
    """Raise ``ValueError`` if *df* is missing any of the *required* columns."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{df_name} is missing required columns: {sorted(missing)}"
        )


def _melt_actual_df(actual_df: pd.DataFrame) -> pd.DataFrame:
    """Convert *actual_df* from wide to long format.

    Wide format (one row per city/target_date with ``high_temp_f`` and/or
    ``low_temp_f``) is melted into long format with columns ``metric`` and
    ``actual_value_f``.  Rows where ``actual_value_f`` is NaN are dropped.

    Duplicate rows on ``(city, target_date, metric)`` after melting are
    deduplicated (first occurrence kept) with a warning.

    Args:
        actual_df: Wide actual observations.

    Returns:
        Long DataFrame with columns from *actual_df* plus ``metric`` and
        ``actual_value_f``, guaranteed unique on ``(city, target_date, metric)``.

    Raises:
        ValueError: If *actual_df* already has a ``metric`` column (ambiguous —
            it would conflict with the melt output column of the same name).
        ValueError: If *actual_df* contains none of the expected temperature
            columns.
    """
    if "metric" in actual_df.columns:
        raise ValueError(
            "actual_df already has a 'metric' column; it must be in wide format "
            "with 'high_temp_f' / 'low_temp_f' value columns, not pre-melted"
        )

    value_cols = [c for c in actual_df.columns if c in _ACTUAL_VALUE_COLS]
    if not value_cols:
        raise ValueError(
            "actual_df must contain at least one of: "
            + ", ".join(sorted(_ACTUAL_VALUE_COLS))
        )

    id_vars = [c for c in actual_df.columns if c not in _ACTUAL_VALUE_COLS]
    melted = actual_df.melt(
        id_vars=id_vars,
        value_vars=value_cols,
        var_name="metric",
        value_name="actual_value_f",
    )
    n_before = len(melted)
    melted = melted.dropna(subset=["actual_value_f"]).reset_index(drop=True)
    n_dropped = n_before - len(melted)
    if n_dropped:
        logger.debug(
            "_melt_actual_df: dropped %d rows with null actual_value_f", n_dropped
        )

    # Guard against fan-out in the merge caused by duplicate actual rows
    dup_mask = melted.duplicated(subset=_JOIN_KEYS)
    n_dups = dup_mask.sum()
    if n_dups:
        logger.warning(
            "_melt_actual_df: actual_df has %d duplicate rows on %s after melting — "
            "keeping first occurrence to prevent fan-out in the merge",
            n_dups,
            _JOIN_KEYS,
        )
        melted = melted[~dup_mask].reset_index(drop=True)

    return melted


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def compute_forecast_error(
    forecast_df: pd.DataFrame,
    actual_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join forecasts against settled actuals and compute error metrics.

    Join key: ``(city, target_date, metric)``.

    ``actual_df`` should be in **wide** format with ``high_temp_f`` and/or
    ``low_temp_f`` columns alongside ``city`` and ``target_date``.  The
    function melts it to long format automatically before joining.

    Forecast rows with no matching actual are dropped and counted in a warning
    log.  Multiple forecast runs for the same
    ``(city, model_name, target_date, metric)`` are retained — each run
    produces its own error row.

    Args:
        forecast_df: Historical forecasts.  Required columns:
            ``city``, ``model_name``, ``target_date``, ``metric``,
            ``forecast_value_f``.  Additional columns (e.g. ``run_time_utc``,
            ``lead_hours``) are preserved in the output.
        actual_df: Settled daily observations.  Required columns:
            ``city``, ``target_date`` and at least one of
            ``high_temp_f`` / ``low_temp_f``.

    Returns:
        DataFrame with one row per matched forecast/actual pair containing all
        required output columns.

        Output columns (at minimum):

        - ``city``
        - ``model_name``
        - ``target_date``
        - ``metric``
        - ``forecast_value_f``
        - ``actual_value_f``
        - ``error_f`` — ``forecast_value_f - actual_value_f``
          (positive = warm bias, negative = cold bias)
        - ``absolute_error_f`` — ``abs(error_f)``
        - ``squared_error`` — ``error_f ** 2``

    Raises:
        ValueError: If required columns are missing from either DataFrame.
        ValueError: If ``actual_df`` contains no temperature value columns.
    """
    if forecast_df.empty:
        logger.info(
            "compute_forecast_error: forecast_df is empty — returning empty DataFrame"
        )
        return pd.DataFrame(columns=_EMPTY_ERROR_COLS)

    if actual_df.empty:
        logger.info(
            "compute_forecast_error: actual_df is empty — returning empty DataFrame"
        )
        return pd.DataFrame(columns=_EMPTY_ERROR_COLS)

    _validate_columns(forecast_df, _FORECAST_REQUIRED, "forecast_df")
    _validate_columns(actual_df, _ACTUAL_REQUIRED, "actual_df")

    # Warn on true duplicates within forecast_df
    dup_key_candidates = ["city", "model_name", "run_time_utc", "target_date", "metric"]
    dup_keys = [c for c in dup_key_candidates if c in forecast_df.columns]
    n_dups = forecast_df.duplicated(subset=dup_keys).sum()
    if n_dups:
        logger.warning(
            "compute_forecast_error: forecast_df has %d duplicate rows on %s — "
            "all rows are preserved in the output",
            n_dups,
            dup_keys,
        )

    actual_long = _melt_actual_df(actual_df)

    logger.debug(
        "compute_forecast_error: joining %d forecast rows to %d actual rows on %s",
        len(forecast_df),
        len(actual_long),
        _JOIN_KEYS,
    )

    merged = forecast_df.merge(
        actual_long[_JOIN_KEYS + ["actual_value_f"]],
        on=_JOIN_KEYS,
        how="inner",
    )

    n_unmatched = len(forecast_df) - len(merged)
    if n_unmatched > 0:
        logger.warning(
            "compute_forecast_error: %d forecast rows had no matching actual — "
            "dropped (check city / target_date / metric alignment)",
            n_unmatched,
        )

    if merged.empty:
        logger.warning(
            "compute_forecast_error: no rows matched after join — returning empty DataFrame"
        )
        return merged

    merged["error_f"] = merged["forecast_value_f"] - merged["actual_value_f"]
    merged["absolute_error_f"] = merged["error_f"].abs()
    merged["squared_error"] = merged["error_f"] ** 2

    logger.info(
        "compute_forecast_error: produced %d error rows across %d city/model/metric groups",
        len(merged),
        merged.groupby(["city", "model_name", "metric"]).ngroups,
    )

    return merged.reset_index(drop=True)


def summarize_forecast_error_by_city_model(
    error_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate forecast error metrics by ``(city, model_name, metric)``.

    Args:
        error_df: Output of :func:`compute_forecast_error`.  Required columns:
            ``city``, ``model_name``, ``metric``, ``error_f``,
            ``absolute_error_f``, ``squared_error``.

    Returns:
        DataFrame with one row per ``(city, model_name, metric)`` group,
        sorted by those keys.  Columns:

        - ``city``
        - ``model_name``
        - ``metric``
        - ``sample_count`` — number of forecast/actual pairs
        - ``mean_error_f`` — mean signed error (positive = warm bias)
        - ``mean_absolute_error_f`` — mean absolute error (MAE)
        - ``rmse_f`` — root mean squared error

    Raises:
        ValueError: If required columns are missing from *error_df*.
    """
    if error_df.empty:
        logger.info(
            "summarize_forecast_error_by_city_model: error_df is empty — "
            "returning empty summary"
        )
        return pd.DataFrame(columns=_EMPTY_SUMMARY_COLS)

    _validate_columns(error_df, _ERROR_REQUIRED, "error_df")

    summary = (
        error_df.groupby(["city", "model_name", "metric"], sort=True)
        .agg(
            sample_count=("error_f", "count"),
            mean_error_f=("error_f", "mean"),
            mean_absolute_error_f=("absolute_error_f", "mean"),
            mean_squared_error=("squared_error", "mean"),
        )
        .reset_index()
    )

    summary["rmse_f"] = summary["mean_squared_error"].apply(math.sqrt)
    summary = summary.drop(columns=["mean_squared_error"])

    logger.info(
        "summarize_forecast_error_by_city_model: summarized %d groups from %d error rows",
        len(summary),
        len(error_df),
    )

    return summary
