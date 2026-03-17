"""Comprehensive tests for deep_isobar.config module."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from deep_isobar.config import (
    _clear_cache,
    get_project_root,
    get_setting,
    load_settings,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    """Ensure each test starts and ends with a clean settings cache."""
    _clear_cache()
    yield
    _clear_cache()


# ── get_project_root ─────────────────────────────────────────────────────

class TestGetProjectRoot:
    def test_returns_path(self):
        root = get_project_root()
        assert isinstance(root, Path)

    def test_points_to_real_root(self):
        root = get_project_root()
        assert (root / "config" / "settings.yaml").exists(), (
            f"Expected config/settings.yaml under project root: {root}"
        )

    def test_is_absolute(self):
        assert get_project_root().is_absolute()


# ── load_settings ────────────────────────────────────────────────────────

class TestLoadSettings:
    def test_default_loads_settings_yaml(self):
        settings = load_settings()
        assert isinstance(settings, dict)
        assert "project" in settings

    def test_caching_returns_same_object(self):
        first = load_settings()
        second = load_settings()
        assert first is second

    def test_custom_path_bypasses_cache(self, tmp_path: Path):
        custom = tmp_path / "custom.yaml"
        custom.write_text(yaml.dump({"custom_key": "custom_value"}), encoding="utf-8")

        result = load_settings(config_path=str(custom))
        assert result == {"custom_key": "custom_value"}

    def test_missing_file_raises(self, tmp_path: Path):
        missing = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError, match="Missing config file"):
            load_settings(config_path=str(missing))

    def test_invalid_format_raises(self, tmp_path: Path):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("- just\n- a\n- list\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid config format"):
            load_settings(config_path=str(bad_yaml))

    def test_empty_yaml_raises(self, tmp_path: Path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid config format"):
            load_settings(config_path=str(empty))

    def test_scalar_yaml_raises(self, tmp_path: Path):
        scalar = tmp_path / "scalar.yaml"
        scalar.write_text("42", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid config format"):
            load_settings(config_path=str(scalar))


# ── get_setting ──────────────────────────────────────────────────────────

class TestGetSetting:
    def test_top_level_key(self):
        result = get_setting("project")
        assert isinstance(result, dict)
        assert "name" in result

    def test_dot_path_lookup(self):
        assert get_setting("risk.alpha_threshold") == 0.10

    def test_deep_dot_path(self):
        assert get_setting("models.probability_surface.min_temp_f") == 50

    def test_missing_key_returns_none(self):
        assert get_setting("nonexistent.path") is None

    def test_missing_key_returns_custom_default(self):
        assert get_setting("nonexistent.key", default="fallback") == "fallback"

    def test_partial_path_returns_default(self):
        assert get_setting("risk.nonexistent_field", default=-1) == -1

    def test_list_value_returned(self):
        sources = get_setting("markets.enabled_sources")
        assert isinstance(sources, list)
        assert "Kalshi" in sources