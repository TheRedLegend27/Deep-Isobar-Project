from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SETTINGS_CACHE: dict[str, Any] | None = None


def get_project_root() -> Path:
    return _PROJECT_ROOT


def load_settings(config_path: str | None = None) -> dict[str, Any]:
    global _SETTINGS_CACHE

    if _SETTINGS_CACHE is not None and config_path is None:
        return _SETTINGS_CACHE

    path = Path(config_path) if config_path else _PROJECT_ROOT / "config" / "settings.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Invalid config format: expected mapping")

    if config_path is None:
        _SETTINGS_CACHE = data

    return data


def get_setting(key: str, default: Any = None) -> Any:
    data = load_settings()
    current: Any = data
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current