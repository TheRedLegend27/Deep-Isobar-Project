"""Temperature ensemble aggregator for Deep Isobar.

Combines multiple model ForecastPoints into a single EnsembleSummary.
Applies city-profile model weights, variance adjustment, and bias correction
so that downstream modules (probability_surface, alpha_engine) receive a
single bias-corrected, uncertainty-adjusted distribution parameter set.

Canonical interface (from INTERFACES.md)::

    build_temperature_ensemble(
        city_profile, forecasts, target_date, metric, ensemble_run_time_utc
    ) -> EnsembleSummary
"""

from __future__ import annotations

import logging
from datetime import date, datetime

from deep_isobar.core.types import CityProfile, EnsembleSummary, ForecastPoint
from deep_isobar.models.forecast_volatility import compute_forecast_std

logger = logging.getLogger(__name__)

_VALID_METRICS = {"high_temp_f", "low_temp_f"}


def _build_weights(
    city_profile: CityProfile,
    forecasts: list[ForecastPoint],
) -> tuple[list[float], str]:
    """Compute normalized per-forecast weights and determine methodology.

    Args:
        city_profile: City configuration with optional per-model weights.
        forecasts: Non-empty list of ForecastPoints to weight.

    Returns:
        A tuple ``(weights, methodology)`` where *weights* is a list of
        floats summing to 1.0 and *methodology* is either
        ``"equal_weight_normal"`` or ``"weighted_normal"``.
    """
    profile_weights = {
        "GFS": city_profile.model_weight_gfs,
        "ECMWF": city_profile.model_weight_ecmwf,
        "NAM": city_profile.model_weight_nam,
    }

    all_none = all(v is None for v in profile_weights.values())

    if all_none:
        n = len(forecasts)
        weights = [1.0 / n] * n
        logger.debug("_build_weights: all profile weights None → equal weighting n=%d", n)
        return weights, "equal_weight_normal"

    # Use mapped weight if present; fall back to 1.0 for unknown/unconfigured models.
    # Explicit None means "not configured" → fallback 1.0.
    # Explicit 0.0 means "exclude this model" → keep 0.0.
    raw_weights = [
        1.0 if profile_weights.get(fp.model_name) is None else profile_weights[fp.model_name]
        for fp in forecasts
    ]
    total = sum(raw_weights)
    weights = [w / total for w in raw_weights]
    logger.debug(
        "_build_weights: weighted methodology raw=%s normalized=%s",
        raw_weights,
        [round(w, 4) for w in weights],
    )
    return weights, "weighted_normal"


def build_temperature_ensemble(
    city_profile: CityProfile,
    forecasts: list[ForecastPoint],
    target_date: date,
    metric: str,
    ensemble_run_time_utc: datetime,
) -> EnsembleSummary:
    """Aggregate model ForecastPoints into a bias-adjusted EnsembleSummary.

    Args:
        city_profile: City configuration including weights and bias params.
        forecasts: One or more ForecastPoints for the same city/date/metric.
        target_date: The settlement date all forecasts must target.
        metric: Temperature metric; must be ``"high_temp_f"`` or ``"low_temp_f"``.
        ensemble_run_time_utc: Timestamp when this ensemble run was triggered.

    Returns:
        An :class:`~deep_isobar.core.types.EnsembleSummary` with weighted mean,
        raw spread, variance-adjusted std, and bias-corrected mean.

    Raises:
        ValueError: If *forecasts* is empty, *metric* is invalid, or any
            ForecastPoint has a mismatched ``target_date`` or ``metric``.

    Example::

        summary = build_temperature_ensemble(
            city_profile=profile,
            forecasts=[fp_gfs, fp_ecmwf],
            target_date=date(2026, 3, 17),
            metric="high_temp_f",
            ensemble_run_time_utc=datetime.now(timezone.utc),
        )
    """
    # ── Validation ────────────────────────────────────────────────────────
    if not forecasts:
        raise ValueError("forecasts cannot be empty")

    if metric not in _VALID_METRICS:
        raise ValueError(
            f"metric must be one of {sorted(_VALID_METRICS)!r}, got {metric!r}"
        )

    for fp in forecasts:
        if fp.target_date != target_date:
            raise ValueError(
                f"ForecastPoint target_date mismatch: expected {target_date}, "
                f"got {fp.target_date} (model={fp.model_name})"
            )
        if fp.metric != metric:
            raise ValueError(
                f"ForecastPoint metric mismatch: expected {metric!r}, "
                f"got {fp.metric!r} (model={fp.model_name})"
            )

    # ── Weights & methodology ─────────────────────────────────────────────
    weights, methodology = _build_weights(city_profile, forecasts)

    # ── Ensemble mean (weighted) ──────────────────────────────────────────
    ensemble_mean_f = sum(w * fp.forecast_value_f for w, fp in zip(weights, forecasts))

    # ── Ensemble std (raw spread, unweighted) ─────────────────────────────
    ensemble_std_f = compute_forecast_std([fp.forecast_value_f for fp in forecasts])

    # ── Variance adjustment ───────────────────────────────────────────────
    adjusted_std_f = ensemble_std_f * city_profile.variance_multiplier

    # ── Bias correction ───────────────────────────────────────────────────
    bias_corrected_mean_f = ensemble_mean_f + city_profile.mean_bias_correction_f
    if metric == "high_temp_f":
        bias_corrected_mean_f += city_profile.heat_bias_adjustment_f
    elif metric == "low_temp_f":
        bias_corrected_mean_f += city_profile.cold_bias_adjustment_f

    # ── Contributing models ───────────────────────────────────────────────
    contributing_models = sorted({fp.model_name for fp in forecasts})

    logger.debug(
        "build_temperature_ensemble: city=%s date=%s metric=%s models=%s "
        "mean=%.2f bias_mean=%.2f std=%.2f adj_std=%.2f methodology=%s",
        city_profile.city,
        target_date,
        metric,
        contributing_models,
        ensemble_mean_f,
        bias_corrected_mean_f,
        ensemble_std_f,
        adjusted_std_f,
        methodology,
    )

    return EnsembleSummary(
        city=city_profile.city,
        target_date=target_date,
        metric=metric,
        ensemble_run_time_utc=ensemble_run_time_utc,
        contributing_models=contributing_models,
        model_count=len(contributing_models),
        ensemble_mean_f=ensemble_mean_f,
        ensemble_std_f=ensemble_std_f,
        variance_multiplier=city_profile.variance_multiplier,
        adjusted_std_f=adjusted_std_f,
        bias_corrected_mean_f=bias_corrected_mean_f,
        methodology=methodology,
    )
