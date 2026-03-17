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

Build the module `src/deep_isobar/config.py` for the Deep Isobar project.

Context:
- Deep Isobar is a weather prediction market trading engine.
- This module is the central configuration loader.
- It must load `config/settings.yaml`.
- It must expose:
  - `get_project_root() -> Path`
  - `load_settings(config_path: str | None = None) -> dict`
  - `get_setting(key: str, default: Any = None) -> Any`

Requirements:
- Use Python 3.11+
- Use `pathlib`
- Use `PyYAML`
- Cache default settings after first load
- Support dot-path lookup in `get_setting`, for example:
  - `get_setting("risk.alpha_threshold")`
- Raise:
  - `FileNotFoundError` for missing config
  - `ValueError` for invalid config format
- Add docstrings and type hints
- Keep implementation small and clean

Output requirements:
1. Explanation
2. Python implementation
3. Example usage
4. Unit tests with pytest
5. Edge cases handled

Stop after this module.