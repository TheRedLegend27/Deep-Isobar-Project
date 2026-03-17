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

Build the module `src/deep_isobar/models/forecast_volatility.py` for Deep Isobar.

Context:
- This module measures ensemble spread and uncertainty.
- It must expose:
  - `compute_forecast_std(forecast_values_f: list[float]) -> float`
  - `compute_forecast_variance(forecast_values_f: list[float]) -> float`
  - `compute_forecast_volatility_score(forecast_values_f: list[float]) -> float`

Requirements:
- Use numpy
- Raise `ValueError` on empty input
- Return floats
- Keep implementation deterministic and simple
- Add type hints, docstrings, and logging
- The volatility score can equal standard deviation for MVP, but document that clearly

Output requirements:
1. Explanation
2. Python implementation
3. Example usage
4. Unit tests with pytest
5. Edge cases handled

Stop after this module.