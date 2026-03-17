"""Comprehensive tests for deep_isobar.data.data_sources module."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from deep_isobar.data.data_sources import get_data_source, load_data_sources


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def valid_config(tmp_path: Path) -> Path:
    """Write a minimal valid data_sources.yaml and return its path."""
    data = {
        "sources": {
            "NWS": {
                "source_type": "observation",
                "authority_level": 10,
                "active": True,
                "notes": "Official observation source",
            },
            "GFS": {
                "source_type": "forecast_model",
                "authority_level": 8,
                "active": True,
                "notes": "Global Forecast System",
            },
        }
    }
    config_file = tmp_path / "data_sources.yaml"
    config_file.write_text(yaml.dump(data), encoding="utf-8")
    return config_file


@pytest.fixture()
def single_source_config(tmp_path: Path) -> Path:
    """Config with exactly one source."""
    data = {
        "sources": {
            "Kalshi": {
                "source_type": "market",
                "authority_level": 10,
                "active": True,
            }
        }
    }
    config_file = tmp_path / "data_sources.yaml"
    config_file.write_text(yaml.dump(data), encoding="utf-8")
    return config_file


# ── load_data_sources ────────────────────────────────────────────────────


class TestLoadDataSources:
    """Tests for the load_data_sources function."""

    def test_loads_default_config(self):
        """Smoke test: default config/data_sources.yaml loads successfully."""
        sources = load_data_sources()
        assert isinstance(sources, dict)
        assert len(sources) > 0

    def test_default_config_contains_nws(self):
        """The real config should include the NWS source."""
        sources = load_data_sources()
        assert "NWS" in sources

    def test_returns_dict_of_dicts(self):
        """Every value in the returned mapping should be a dict."""
        sources = load_data_sources()
        for name, attrs in sources.items():
            assert isinstance(name, str)
            assert isinstance(attrs, dict), f"Expected dict for {name}"

    def test_custom_path(self, valid_config: Path):
        """Passing an explicit config_path reads from that file."""
        sources = load_data_sources(config_path=str(valid_config))
        assert set(sources.keys()) == {"NWS", "GFS"}

    def test_source_attributes(self, valid_config: Path):
        """Source attributes should match what was written."""
        sources = load_data_sources(config_path=str(valid_config))
        nws = sources["NWS"]
        assert nws["source_type"] == "observation"
        assert nws["authority_level"] == 10
        assert nws["active"] is True

    def test_single_source(self, single_source_config: Path):
        """A config with one source should still load cleanly."""
        sources = load_data_sources(config_path=str(single_source_config))
        assert list(sources.keys()) == ["Kalshi"]

    # ── Error cases ──────────────────────────────────────────────────────

    def test_missing_file_raises(self, tmp_path: Path):
        """FileNotFoundError when the config file does not exist."""
        missing = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError, match="Missing data-source config"):
            load_data_sources(config_path=str(missing))

    def test_yaml_list_raises(self, tmp_path: Path):
        """ValueError when YAML root is a list instead of a mapping."""
        bad = tmp_path / "bad.yaml"
        bad.write_text("- item_one\n- item_two\n", encoding="utf-8")
        with pytest.raises(ValueError, match="expected a YAML mapping"):
            load_data_sources(config_path=str(bad))

    def test_yaml_scalar_raises(self, tmp_path: Path):
        """ValueError when YAML root is a scalar."""
        bad = tmp_path / "scalar.yaml"
        bad.write_text("42", encoding="utf-8")
        with pytest.raises(ValueError, match="expected a YAML mapping"):
            load_data_sources(config_path=str(bad))

    def test_empty_yaml_raises(self, tmp_path: Path):
        """ValueError when the YAML file is empty (parsed as None)."""
        empty = tmp_path / "empty.yaml"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="expected a YAML mapping"):
            load_data_sources(config_path=str(empty))

    def test_missing_sources_key_raises(self, tmp_path: Path):
        """ValueError when YAML mapping lacks the 'sources' key."""
        bad = tmp_path / "no_sources.yaml"
        bad.write_text(yaml.dump({"other_key": "value"}), encoding="utf-8")
        with pytest.raises(ValueError, match="missing top-level 'sources' key"):
            load_data_sources(config_path=str(bad))

    def test_sources_not_a_dict_raises(self, tmp_path: Path):
        """ValueError when 'sources' value is a list, not a mapping."""
        bad = tmp_path / "list_sources.yaml"
        bad.write_text(yaml.dump({"sources": ["a", "b"]}), encoding="utf-8")
        with pytest.raises(ValueError, match="'sources' must be a mapping"):
            load_data_sources(config_path=str(bad))

    def test_sources_scalar_raises(self, tmp_path: Path):
        """ValueError when 'sources' value is a scalar string."""
        bad = tmp_path / "scalar_sources.yaml"
        bad.write_text(yaml.dump({"sources": "not_a_dict"}), encoding="utf-8")
        with pytest.raises(ValueError, match="'sources' must be a mapping"):
            load_data_sources(config_path=str(bad))

    def test_sources_null_raises(self, tmp_path: Path):
        """ValueError when 'sources' value is null / None."""
        bad = tmp_path / "null_sources.yaml"
        bad.write_text("sources:\n", encoding="utf-8")
        with pytest.raises(ValueError, match="'sources' must be a mapping"):
            load_data_sources(config_path=str(bad))


# ── get_data_source ──────────────────────────────────────────────────────


class TestGetDataSource:
    """Tests for the get_data_source function."""

    def test_returns_known_source(self, valid_config: Path):
        """Should return the attribute dict for a known source."""
        result = get_data_source("NWS", config_path=str(valid_config))
        assert result["source_type"] == "observation"
        assert result["authority_level"] == 10

    def test_returns_second_source(self, valid_config: Path):
        """Should be able to retrieve any source by name."""
        result = get_data_source("GFS", config_path=str(valid_config))
        assert result["source_type"] == "forecast_model"

    def test_default_config_nws(self):
        """Smoke test: retrieve NWS from the real config."""
        nws = get_data_source("NWS")
        assert isinstance(nws, dict)
        assert "source_type" in nws

    # ── Error cases ──────────────────────────────────────────────────────

    def test_missing_source_raises_key_error(self, valid_config: Path):
        """KeyError when the requested source name does not exist."""
        with pytest.raises(KeyError, match="Data source not found: 'MISSING'"):
            get_data_source("MISSING", config_path=str(valid_config))

    def test_key_error_lists_available(self, valid_config: Path):
        """The KeyError message should include available source names."""
        with pytest.raises(KeyError, match="Available sources:"):
            get_data_source("NOPE", config_path=str(valid_config))

    def test_case_sensitive_lookup(self, valid_config: Path):
        """Source lookup is case-sensitive — 'nws' ≠ 'NWS'."""
        with pytest.raises(KeyError):
            get_data_source("nws", config_path=str(valid_config))

    def test_missing_file_propagates(self, tmp_path: Path):
        """FileNotFoundError should propagate from load_data_sources."""
        missing = tmp_path / "gone.yaml"
        with pytest.raises(FileNotFoundError):
            get_data_source("NWS", config_path=str(missing))

    def test_invalid_yaml_propagates(self, tmp_path: Path):
        """ValueError should propagate when YAML is malformed."""
        bad = tmp_path / "bad.yaml"
        bad.write_text("- list\n", encoding="utf-8")
        with pytest.raises(ValueError):
            get_data_source("NWS", config_path=str(bad))


# ── Edge cases ───────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge-case scenarios."""

    def test_source_with_extra_fields(self, tmp_path: Path):
        """Extra fields in a source entry are preserved."""
        data = {
            "sources": {
                "Custom": {
                    "source_type": "experimental",
                    "authority_level": 1,
                    "active": False,
                    "api_endpoint": "https://example.com/api",
                    "rate_limit": 100,
                }
            }
        }
        cfg = tmp_path / "extra.yaml"
        cfg.write_text(yaml.dump(data), encoding="utf-8")

        result = get_data_source("Custom", config_path=str(cfg))
        assert result["api_endpoint"] == "https://example.com/api"
        assert result["rate_limit"] == 100

    def test_source_with_minimal_fields(self, tmp_path: Path):
        """A source with only one attribute should still load."""
        data = {"sources": {"Bare": {"active": True}}}
        cfg = tmp_path / "minimal.yaml"
        cfg.write_text(yaml.dump(data), encoding="utf-8")

        result = get_data_source("Bare", config_path=str(cfg))
        assert result == {"active": True}

    def test_empty_sources_mapping(self, tmp_path: Path):
        """An empty 'sources: {}' mapping should load but yield no sources."""
        data = {"sources": {}}
        cfg = tmp_path / "empty_sources.yaml"
        cfg.write_text(yaml.dump(data), encoding="utf-8")

        sources = load_data_sources(config_path=str(cfg))
        assert sources == {}

    def test_empty_sources_get_raises(self, tmp_path: Path):
        """get_data_source raises KeyError when sources mapping is empty."""
        data = {"sources": {}}
        cfg = tmp_path / "empty_sources.yaml"
        cfg.write_text(yaml.dump(data), encoding="utf-8")

        with pytest.raises(KeyError, match="Data source not found"):
            get_data_source("NWS", config_path=str(cfg))

    def test_sources_with_numeric_values(self, tmp_path: Path):
        """Numeric values in source attributes are preserved correctly."""
        data = {
            "sources": {
                "NumberSource": {
                    "authority_level": 7,
                    "weight": 0.85,
                    "count": 0,
                }
            }
        }
        cfg = tmp_path / "numbers.yaml"
        cfg.write_text(yaml.dump(data), encoding="utf-8")

        result = get_data_source("NumberSource", config_path=str(cfg))
        assert result["authority_level"] == 7
        assert result["weight"] == 0.85
        assert result["count"] == 0

    def test_sources_with_boolean_values(self, tmp_path: Path):
        """Boolean values are loaded as Python booleans, not strings."""
        data = {
            "sources": {
                "BoolSource": {"active": True, "deprecated": False}
            }
        }
        cfg = tmp_path / "bools.yaml"
        cfg.write_text(yaml.dump(data), encoding="utf-8")

        result = get_data_source("BoolSource", config_path=str(cfg))
        assert result["active"] is True
        assert result["deprecated"] is False
