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
Build the module `src/deep_isobar/models/probability_engine.py` for Deep Isobar.

Context:
- This is the normal-distribution fallback probability engine.
- It must expose:
  - `probability_ge_normal(mean_f: float, std_f: float, threshold_f: float) -> float`
  - `probability_le_normal(mean_f: float, std_f: float, threshold_f: float) -> float`

Requirements:
- Use a mathematically correct normal CDF implementation
- Return probabilities strictly clamped to `[0.0, 1.0]`
- Raise `ValueError` if `std_f <= 0`
- Add type hints, docstrings, and logging
- Keep the implementation dependency-light if possible
- Include examples for thresholds above and below the mean

Output requirements:
1. Explanation
2. Python implementation
3. Example usage
4. Unit tests with pytest
5. Edge cases handled

Stop after this module.