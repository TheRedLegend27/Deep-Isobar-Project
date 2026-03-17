"""Central configuration loader for the Deep Isobar project.

Loads settings from ``config/settings.yaml`` (or a caller-supplied path),
caches them after the first load, and exposes dot-path lookup for nested keys.

Typical usage::

    from deep_isobar.config import get_project_root, load_settings, get_setting

    root = get_project_root()
    settings = load_settings()
    threshold = get_setting("risk.alpha_threshold", default=0.10)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
_SETTINGS_CACHE: dict[str, Any] | None = None

# ---------------------------------------------------------------------------
# Load .env into the process environment (no-op if file is absent or
# python-dotenv is not installed — never overwrites existing env vars)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(_PROJECT_ROOT / ".env", override=False)
except ImportError:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """Return the absolute path to the project root directory.

    The root is determined at import time as two levels above this file
    (``src/deep_isobar/config.py`` → project root).

    Returns:
        Path: Absolute path to the project root.
    """
    return _PROJECT_ROOT


def load_settings(config_path: str | None = None) -> dict[str, Any]:
    """Load and return the project settings dictionary.

    On the first call (with no explicit *config_path*), the default
    ``config/settings.yaml`` is loaded and cached for subsequent calls.
    Passing an explicit *config_path* always reads from disk and does **not**
    populate the cache.

    Args:
        config_path: Optional path to a YAML config file.  When ``None``,
            defaults to ``<project_root>/config/settings.yaml``.

    Returns:
        dict[str, Any]: Parsed settings dictionary.

    Raises:
        FileNotFoundError: If the resolved config file does not exist.
        ValueError: If the YAML content is not a mapping (dict).
    """
    global _SETTINGS_CACHE

    if _SETTINGS_CACHE is not None and config_path is None:
        return _SETTINGS_CACHE

    path = Path(config_path) if config_path else _PROJECT_ROOT / "config" / "settings.yaml"

    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid config format: expected a YAML mapping, got {type(data).__name__}"
        )

    if config_path is None:
        _SETTINGS_CACHE = data

    return data


def get_setting(key: str, default: Any = None) -> Any:
    """Retrieve a setting value using dot-separated path notation.

    Traverses the cached settings dictionary by splitting *key* on ``"."``
    and following each segment as a nested dict key.

    Examples::

        get_setting("risk.alpha_threshold")          # → 0.10
        get_setting("models.kde.default_bandwidth")  # → 1.0
        get_setting("missing.key", default=42)       # → 42

    Args:
        key: Dot-separated path into the settings dict (e.g.
            ``"risk.alpha_threshold"``).
        default: Value to return when *key* (or any intermediate segment)
            is not found.  Defaults to ``None``.

    Returns:
        The value at the resolved path, or *default* if not found.
    """
    data = load_settings()
    current: Any = data
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


# ---------------------------------------------------------------------------
# Test helper (not part of the public API)
# ---------------------------------------------------------------------------

def _clear_cache() -> None:
    """Reset the module-level settings cache.

    Intended **only** for test isolation — call this in test teardown so
    that each test starts with a fresh cache.
    """
    global _SETTINGS_CACHE
    _SETTINGS_CACHE = None