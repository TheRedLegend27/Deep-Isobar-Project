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

Build the module `src/deep_isobar/data/city_universe.py` for Deep Isobar.

Context:
- This module loads tradable city profiles from `config/cities.yaml`.
- It must use the shared `CityProfile` dataclass from `src/deep_isobar/core/types.py`.
- It must expose:
  - `load_city_profiles(config_dir: str | None = None) -> list[CityProfile]`
  - `get_city_profile(city: str, config_dir: str | None = None) -> CityProfile`
  - `list_active_cities(config_dir: str | None = None) -> list[str]`

Requirements:
- Use `pathlib`
- Use `PyYAML`
- Raise:
  - `FileNotFoundError` if config missing
  - `KeyError` if city not found
  - `ValueError` if YAML format is invalid
- Preserve city-specific fields like:
  - station_id
  - variance_multiplier
  - mean_bias_correction_f
  - kde_bandwidth
  - tail_multiplier
  - model weights
- Add type hints and docstrings

Output requirements:
1. Explanation
2. Python implementation
3. Example usage
4. Unit tests with pytest
5. Edge cases handled

Stop after this module.