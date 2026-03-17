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

Build the module `src/deep_isobar/models/forecast_shift.py` for Deep Isobar.

Context:
- This module compares two forecast runs and detects shift events.
- It must use:
  - `ForecastPoint`
  - `ForecastShiftEvent`
- It must expose:
  - `compute_forecast_shift(previous: ForecastPoint, current: ForecastPoint, significance_threshold_f: float = 2.0) -> ForecastShiftEvent`

Requirements:
- Validate that both forecast points refer to the same:
  - city
  - model_name
  - target_date
  - metric
- Compute:
  - `shift_f`
  - `absolute_shift_f`
  - `shift_direction`
  - `significant_shift_flag`
- Directions must be:
  - `up`
  - `down`
  - `flat`
- Raise `ValueError` on incompatible forecast points
- Add docstrings, type hints, and logging

Output requirements:
1. Explanation
2. Python implementation
3. Example usage
4. Unit tests with pytest
5. Edge cases handled

Stop after this module.