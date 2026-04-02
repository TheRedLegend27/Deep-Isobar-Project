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
import math
from datetime import date, datetime

from deep_isobar.core.types import CityProfile, EnsembleSummary, ForecastPoint
from deep_isobar.models.forecast_volatility import compute_forecast_std

logger = logging.getLogger(__name__)

_VALID_METRICS = {"high_temp_f", "low_temp_f"}

# Lead-time activity windows (inclusive, in hours) per model source.
_SOURCE_WINDOWS: dict[str, tuple[int, int]] = {
    "NAM":   (6, 48),
    "GFS":   (6, 240),
    "ECMWF": (72, 240),
}


def get_lead_weights(
    lead_hours: int | float,
    sources: list[str],
    halflife: float = 48.0,
) -> dict[str, float]:
    """Return normalized decay weights for *sources* at a given lead time.

    Applies exponential lead-time decay and per-source activity windows, then
    re-normalizes so weights sum to 1.0 across active sources.  Sources that
    fall outside their activity window receive a weight of 0.0.

    Args:
        lead_hours: Forecast lead time in hours.
        sources: Model names to include (e.g. ``["GFS", "ECMWF", "NAM"]``).
        halflife: Decay half-life in hours (default 48).  Matches
            ``CityProfile.lead_decay_halflife_hours``.

    Returns:
        ``{source: weight}`` dict with values summing to 1.0 for active
        sources (or all-zero if every source is masked out).

    Example::

        weights = get_lead_weights(120, ["GFS", "ECMWF", "NAM"])
        # NAM → 0.0 (beyond T+48); GFS and ECMWF share the remaining weight.
    """
    raw: dict[str, float] = {}
    for src in sources:
        window = _SOURCE_WINDOWS.get(src)
        if window is not None:
            lo, hi = window
            in_window = lo <= lead_hours <= hi
        else:
            in_window = True  # unknown model: no mask applied
        decay = math.exp(-lead_hours / halflife) if in_window else 0.0
        raw[src] = decay

    total = sum(raw.values())
    if total == 0.0:
        return {src: 0.0 for src in sources}
    return {src: w / total for src, w in raw.items()}


def _build_weights(
    city_profile: CityProfile,
    forecasts: list[ForecastPoint],
) -> tuple[list[float], str]:
    """Compute normalized per-forecast weights with lead-time decay.

    For each ForecastPoint:
    1. Compute exponential decay: ``exp(-lead_hours / halflife)``
    2. Apply source activity-window mask (zero outside the window)
    3. Multiply by the city-profile static weight (if any are configured)
    4. Re-normalize so all weights sum to 1.0

    Source windows:
    - NAM:   T+6 – T+48
    - GFS:   T+6 – T+240
    - ECMWF: T+72 – T+240

    Args:
        city_profile: City configuration with optional per-model weights and
            ``lead_decay_halflife_hours``.
        forecasts: Non-empty list of ForecastPoints to weight.

    Returns:
        A tuple ``(weights, methodology)`` where *weights* is a list of
        floats summing to 1.0 and *methodology* is one of
        ``"equal_weight_decay"`` or ``"weighted_decay"``.
    """
    halflife = city_profile.lead_decay_halflife_hours

    profile_weights = {
        "GFS":   city_profile.model_weight_gfs,
        "ECMWF": city_profile.model_weight_ecmwf,
        "NAM":   city_profile.model_weight_nam,
    }
    all_none = all(v is None for v in profile_weights.values())

    raw_weights: list[float] = []
    for fp in forecasts:
        # ── Lead-time decay ───────────────────────────────────────────────
        window = _SOURCE_WINDOWS.get(fp.model_name)
        if window is not None:
            lo, hi = window
            in_window = lo <= fp.lead_hours <= hi
        else:
            in_window = True
        decay = math.exp(-fp.lead_hours / halflife) if in_window else 0.0

        # ── Static profile weight ─────────────────────────────────────────
        # Explicit None → "not configured" → fall back to 1.0
        # Explicit 0.0  → "exclude this model" → keep 0.0
        if all_none:
            profile_w = 1.0
        else:
            configured = profile_weights.get(fp.model_name)
            profile_w = 1.0 if configured is None else configured

        raw_weights.append(profile_w * decay)

    total = sum(raw_weights)
    if total == 0.0:
        # Every source masked out at this lead time — fall back to equal.
        n = len(forecasts)
        logger.warning(
            "_build_weights: all sources masked at lead_hours=%s → equal fallback",
            [fp.lead_hours for fp in forecasts],
        )
        return [1.0 / n] * n, "equal_weight_decay"

    weights = [w / total for w in raw_weights]
    methodology = "equal_weight_decay" if all_none else "weighted_decay"
    logger.debug(
        "_build_weights: halflife=%.0f lead_hours=%s raw=%s normalized=%s methodology=%s",
        halflife,
        [fp.lead_hours for fp in forecasts],
        [round(w, 4) for w in raw_weights],
        [round(w, 4) for w in weights],
        methodology,
    )
    return weights, methodology


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
