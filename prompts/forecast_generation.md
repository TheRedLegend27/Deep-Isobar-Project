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

Build the module `src/deep_isobar/models/forecast_generation.py` for Deep Isobar.

Context:
- This module creates per-model point forecasts for a city.
- It must use the `ForecastPoint` dataclass.
- It must expose:
  - `fetch_model_forecast(city: str, station_id: str, model_name: str, run_time_utc: datetime, target_date: date, metric: str) -> ForecastPoint`
  - `fetch_forecasts_for_city(city: str, target_date: date, metric: str, model_names: list[str]) -> list[ForecastPoint]`
  - `register_forecast_run(model_name: str, run_time_utc: datetime, cycle_label: str, source_name: str) -> dict`

Requirements:
- Valid metrics:
  - `high_temp_f`
  - `low_temp_f`
- Raise `ValueError` for invalid metrics
- Use `get_city_profile()` for station lookup
- First version can return deterministic stub forecast values, but structure must be production-ready
- Add docstrings, type hints, and logging
- Keep it small and testable

Output requirements:
1. Explanation
2. Python implementation
3. Example usage
4. Unit tests with pytest
5. Edge cases handled

Stop after this module.