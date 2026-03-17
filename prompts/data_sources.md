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

Build the module `src/deep_isobar/data/data_sources.py` for Deep Isobar.

Context:
- This module loads the source registry from `config/data_sources.yaml`.
- It must expose:
  - `load_data_sources(config_path: str | None = None) -> dict[str, dict]`
  - `get_data_source(name: str, config_path: str | None = None) -> dict[str, Any]`

Requirements:
- Use `pathlib`
- Use `PyYAML`
- Validate that the YAML has a top-level `sources` mapping
- Raise:
  - `FileNotFoundError` for missing config
  - `ValueError` for malformed YAML shape
  - `KeyError` for missing source name
- Keep the module simple and testable
- Add docstrings and type hints

Output requirements:
1. Explanation
2. Python implementation
3. Example usage
4. Unit tests with pytest
5. Edge cases handled

Stop after this module.