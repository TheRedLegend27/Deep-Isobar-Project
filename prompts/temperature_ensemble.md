Project: Deep Isobar

Relevant docs:
- ARCHITECTURE.md
- INTERFACES.md
- MODULE_DEPENDENCIES.md

Follow these rules:

Iterative Development:
Build only the requested module.

Prompt Engineering:
Follow the interfaces exactly as defined in INTERFACES.md.

Error Handling:
Raise clear errors for invalid inputs.

Testing:
Include pytest tests.

Now build:

Build the module `src/deep_isobar/models/temperature_ensemble.py` for Deep Isobar.

Context:
- This module combines multiple model ForecastPoints into a single EnsembleSummary.
- It is the bridge between raw forecasts and the probability engine.
- It must use:
  - `CityProfile` from `deep_isobar.core.types`
  - `ForecastPoint` from `deep_isobar.core.types`
  - `EnsembleSummary` from `deep_isobar.core.types`
  - `compute_forecast_std` from `deep_isobar.models.forecast_volatility`
- It must expose:
  - `build_temperature_ensemble(city_profile, forecasts, target_date, metric, ensemble_run_time_utc) -> EnsembleSummary`

Interface signature (exact):
```python
from datetime import date, datetime
from deep_isobar.core.types import CityProfile, ForecastPoint, EnsembleSummary

def build_temperature_ensemble(
    city_profile: CityProfile,
    forecasts: list[ForecastPoint],
    target_date: date,
    metric: str,
    ensemble_run_time_utc: datetime,
) -> EnsembleSummary:
    ...
```

Requirements:

Validation:
- Raise `ValueError` if `forecasts` is empty
- Raise `ValueError` if `metric` is not `"high_temp_f"` or `"low_temp_f"`
- Raise `ValueError` if any ForecastPoint has a different `target_date` or `metric` than requested
- All inputs must be validated with clear error messages

Weighting logic:
- Build a weight map from city_profile: {"GFS": model_weight_gfs, "ECMWF": model_weight_ecmwf, "NAM": model_weight_nam}
- If ALL model weights on the profile are None: use equal weights (1/n per model) and set methodology = "equal_weight_normal"
- If any weights are set: use them for matching models; fall back to 1.0 for models not in the weight map
- Normalize the final weights so they sum to 1.0 before computing the weighted mean
- Set methodology = "weighted_normal" if any city weights were used

Ensemble mean:
- ensemble_mean_f = sum(weight_i * forecast_value_f_i) across all forecasts

Ensemble std:
- ensemble_std_f = compute_forecast_std([fp.forecast_value_f for fp in forecasts])
- This is the raw spread before any city adjustments

Variance adjustment:
- adjusted_std_f = ensemble_std_f * city_profile.variance_multiplier

Bias correction:
- Start: bias_corrected_mean_f = ensemble_mean_f + city_profile.mean_bias_correction_f
- If metric == "high_temp_f": also add city_profile.heat_bias_adjustment_f
- If metric == "low_temp_f": also add city_profile.cold_bias_adjustment_f

EnsembleSummary fields to populate exactly:
- city: city_profile.city
- target_date: the target_date argument
- metric: the metric argument
- ensemble_run_time_utc: the ensemble_run_time_utc argument
- contributing_models: sorted list of unique model_name values from the forecasts
- model_count: len(contributing_models)
- ensemble_mean_f: weighted average computed above
- ensemble_std_f: raw spread from compute_forecast_std
- variance_multiplier: city_profile.variance_multiplier
- adjusted_std_f: ensemble_std_f * variance_multiplier
- bias_corrected_mean_f: bias-adjusted mean computed above
- methodology: "weighted_normal" or "equal_weight_normal"

Other requirements:
- Use Python 3.11+
- Use `from __future__ import annotations`
- Add docstrings and type hints to all functions
- Add logging with `logging.getLogger(__name__)`
- Keep helpers private (prefix with `_`)

Tests must cover:
- Equal weighting when city_profile has no model weights
- Weighted average when city_profile has model weights (e.g. GFS=0.5, ECMWF=0.5)
- Unknown model not in weight map falls back to equal contribution
- bias_corrected_mean_f includes mean_bias_correction_f
- heat_bias_adjustment_f is applied for metric="high_temp_f"
- cold_bias_adjustment_f is applied for metric="low_temp_f"
- adjusted_std_f == ensemble_std_f * variance_multiplier
- contributing_models is a sorted unique list of model names
- model_count matches len(contributing_models)
- ValueError on empty forecasts list
- ValueError on invalid metric string
- ValueError on ForecastPoint with mismatched target_date
- ValueError on ForecastPoint with mismatched metric
- methodology == "equal_weight_normal" when no profile weights are set
- methodology == "weighted_normal" when profile weights are used

Output requirements:
1. Explanation
2. Python implementation
3. Example usage
4. Unit tests with pytest
5. Edge cases handled

Stop after this module.