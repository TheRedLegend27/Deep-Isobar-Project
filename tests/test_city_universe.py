"""Comprehensive tests for deep_isobar.data.city_universe module."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from deep_isobar.core.types import CityProfile
from deep_isobar.data.city_universe import (
    get_city_profile,
    list_active_cities,
    load_city_profiles,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MINIMAL_CITY: dict = {
    "city": "TestCity",
    "city_code": "TST",
    "station_id": "KTST",
    "timezone": "America/Chicago",
    "settlement_source": "NWS",
    "variance_multiplier": 1.0,
    "mean_bias_correction_f": 0.0,
    "kde_bandwidth": 1.0,
    "active": True,
}

_FULL_CITY: dict = {
    **_MINIMAL_CITY,
    "city": "FullCity",
    "city_code": "FUL",
    "station_id": "KFUL",
    "tail_multiplier": 1.15,
    "model_weight_gfs": 0.4,
    "model_weight_ecmwf": 0.4,
    "model_weight_nam": 0.2,
    "heat_bias_adjustment_f": 0.5,
    "cold_bias_adjustment_f": -0.3,
}


@pytest.fixture()
def config_dir(tmp_path: Path) -> Path:
    """Create a temporary config directory with a valid cities.yaml."""
    cities_yaml = tmp_path / "cities.yaml"
    cities_yaml.write_text(
        yaml.dump({"cities": [_MINIMAL_CITY]}, default_flow_style=False),
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture()
def full_config_dir(tmp_path: Path) -> Path:
    """Config dir with a city entry that has all optional fields populated."""
    cities_yaml = tmp_path / "cities.yaml"
    cities_yaml.write_text(
        yaml.dump({"cities": [_FULL_CITY]}, default_flow_style=False),
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture()
def multi_city_config(tmp_path: Path) -> Path:
    """Config with three cities: two active, one inactive."""
    inactive = {**_MINIMAL_CITY, "city": "InactiveCity", "city_code": "INA", "active": False}
    second = {**_MINIMAL_CITY, "city": "AlphaCity", "city_code": "ALP", "active": True}
    cities_yaml = tmp_path / "cities.yaml"
    cities_yaml.write_text(
        yaml.dump(
            {"cities": [_MINIMAL_CITY, inactive, second]},
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
    return tmp_path


# ── load_city_profiles ────────────────────────────────────────────────────


class TestLoadCityProfiles:
    """Tests for load_city_profiles()."""

    def test_loads_from_real_config(self) -> None:
        """Smoke test: load from the actual project config/cities.yaml."""
        profiles = load_city_profiles()
        assert len(profiles) >= 1
        assert all(isinstance(p, CityProfile) for p in profiles)

    def test_loads_from_custom_dir(self, config_dir: Path) -> None:
        profiles = load_city_profiles(config_dir=str(config_dir))
        assert len(profiles) == 1
        assert profiles[0].city == "TestCity"

    def test_returns_city_profile_instances(self, config_dir: Path) -> None:
        profiles = load_city_profiles(config_dir=str(config_dir))
        assert all(isinstance(p, CityProfile) for p in profiles)

    def test_required_fields_populated(self, config_dir: Path) -> None:
        p = load_city_profiles(config_dir=str(config_dir))[0]
        assert p.city == "TestCity"
        assert p.city_code == "TST"
        assert p.station_id == "KTST"
        assert p.timezone == "America/Chicago"
        assert p.settlement_source == "NWS"
        assert p.variance_multiplier == 1.0
        assert p.mean_bias_correction_f == 0.0
        assert p.kde_bandwidth == 1.0

    def test_optional_defaults_applied(self, config_dir: Path) -> None:
        p = load_city_profiles(config_dir=str(config_dir))[0]
        assert p.tail_multiplier == 1.0
        assert p.model_weight_gfs is None
        assert p.model_weight_ecmwf is None
        assert p.model_weight_nam is None
        assert p.heat_bias_adjustment_f == 0.0
        assert p.cold_bias_adjustment_f == 0.0

    def test_all_optional_fields_preserved(self, full_config_dir: Path) -> None:
        p = load_city_profiles(config_dir=str(full_config_dir))[0]
        assert p.tail_multiplier == 1.15
        assert p.model_weight_gfs == 0.4
        assert p.model_weight_ecmwf == 0.4
        assert p.model_weight_nam == 0.2
        assert p.heat_bias_adjustment_f == 0.5
        assert p.cold_bias_adjustment_f == -0.3

    def test_multiple_cities_loaded(self, multi_city_config: Path) -> None:
        profiles = load_city_profiles(config_dir=str(multi_city_config))
        assert len(profiles) == 3
        names = [p.city for p in profiles]
        assert "TestCity" in names
        assert "InactiveCity" in names
        assert "AlphaCity" in names

    def test_numeric_fields_are_floats(self, config_dir: Path) -> None:
        p = load_city_profiles(config_dir=str(config_dir))[0]
        assert isinstance(p.variance_multiplier, float)
        assert isinstance(p.mean_bias_correction_f, float)
        assert isinstance(p.kde_bandwidth, float)
        assert isinstance(p.tail_multiplier, float)
        assert isinstance(p.heat_bias_adjustment_f, float)
        assert isinstance(p.cold_bias_adjustment_f, float)


# ── load_city_profiles: error handling ────────────────────────────────────


class TestLoadCityProfilesErrors:
    """Error-path tests for load_city_profiles()."""

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Missing city config"):
            load_city_profiles(config_dir=str(tmp_path / "nope"))

    def test_yaml_not_a_mapping_raises_value_error(self, tmp_path: Path) -> None:
        (tmp_path / "cities.yaml").write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="expected a YAML mapping"):
            load_city_profiles(config_dir=str(tmp_path))

    def test_empty_yaml_raises_value_error(self, tmp_path: Path) -> None:
        (tmp_path / "cities.yaml").write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="expected a YAML mapping"):
            load_city_profiles(config_dir=str(tmp_path))

    def test_scalar_yaml_raises_value_error(self, tmp_path: Path) -> None:
        (tmp_path / "cities.yaml").write_text("42", encoding="utf-8")
        with pytest.raises(ValueError, match="expected a YAML mapping"):
            load_city_profiles(config_dir=str(tmp_path))

    def test_missing_cities_key_raises_value_error(self, tmp_path: Path) -> None:
        (tmp_path / "cities.yaml").write_text(
            yaml.dump({"other_key": 123}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="missing top-level 'cities' key"):
            load_city_profiles(config_dir=str(tmp_path))

    def test_cities_not_a_list_raises_value_error(self, tmp_path: Path) -> None:
        (tmp_path / "cities.yaml").write_text(
            yaml.dump({"cities": "not_a_list"}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="'cities' must be a list"):
            load_city_profiles(config_dir=str(tmp_path))

    def test_empty_cities_list_raises_value_error(self, tmp_path: Path) -> None:
        (tmp_path / "cities.yaml").write_text(
            yaml.dump({"cities": []}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="'cities' list is empty"):
            load_city_profiles(config_dir=str(tmp_path))

    def test_non_dict_entry_raises_value_error(self, tmp_path: Path) -> None:
        (tmp_path / "cities.yaml").write_text(
            yaml.dump({"cities": ["not_a_dict"]}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="entry at index 0.*is not a mapping"):
            load_city_profiles(config_dir=str(tmp_path))

    def test_missing_required_field_raises_value_error(self, tmp_path: Path) -> None:
        incomplete = {k: v for k, v in _MINIMAL_CITY.items() if k != "station_id"}
        (tmp_path / "cities.yaml").write_text(
            yaml.dump({"cities": [incomplete]}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="missing required field.*station_id"):
            load_city_profiles(config_dir=str(tmp_path))

    def test_multiple_missing_fields_reported(self, tmp_path: Path) -> None:
        sparse = {"city": "Sparse", "city_code": "SPA"}
        (tmp_path / "cities.yaml").write_text(
            yaml.dump({"cities": [sparse]}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="missing required field"):
            load_city_profiles(config_dir=str(tmp_path))


# ── get_city_profile ──────────────────────────────────────────────────────


class TestGetCityProfile:
    """Tests for get_city_profile()."""

    def test_returns_matching_profile(self, config_dir: Path) -> None:
        p = get_city_profile("TestCity", config_dir=str(config_dir))
        assert isinstance(p, CityProfile)
        assert p.city == "TestCity"

    def test_city_not_found_raises_key_error(self, config_dir: Path) -> None:
        with pytest.raises(KeyError, match="City not found.*Atlantis"):
            get_city_profile("Atlantis", config_dir=str(config_dir))

    def test_key_error_lists_available_cities(self, multi_city_config: Path) -> None:
        with pytest.raises(KeyError, match="Available cities"):
            get_city_profile("Atlantis", config_dir=str(multi_city_config))

    def test_case_sensitive_lookup(self, config_dir: Path) -> None:
        with pytest.raises(KeyError):
            get_city_profile("testcity", config_dir=str(config_dir))

    def test_real_config_chicago(self) -> None:
        """Smoke test against the actual project config."""
        p = get_city_profile("Chicago")
        assert p.station_id == "KORD"
        assert p.city_code == "CHI"

    def test_preserves_model_weights(self) -> None:
        """Verify model weights survived the round-trip from YAML."""
        p = get_city_profile("Chicago")
        assert p.model_weight_gfs is not None
        assert p.model_weight_ecmwf is not None
        assert p.model_weight_nam is not None

    def test_preserves_variance_multiplier(self) -> None:
        p = get_city_profile("Denver")
        assert p.variance_multiplier == 1.3

    def test_preserves_kde_bandwidth(self) -> None:
        p = get_city_profile("Denver")
        assert p.kde_bandwidth == 1.5

    def test_preserves_tail_multiplier(self) -> None:
        p = get_city_profile("Denver")
        assert p.tail_multiplier == 1.15


# ── list_active_cities ────────────────────────────────────────────────────


class TestListActiveCities:
    """Tests for list_active_cities()."""

    def test_returns_list_of_strings(self, config_dir: Path) -> None:
        result = list_active_cities(config_dir=str(config_dir))
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)

    def test_excludes_inactive_cities(self, multi_city_config: Path) -> None:
        actives = list_active_cities(config_dir=str(multi_city_config))
        assert "InactiveCity" not in actives
        assert "TestCity" in actives
        assert "AlphaCity" in actives

    def test_returns_sorted(self, multi_city_config: Path) -> None:
        actives = list_active_cities(config_dir=str(multi_city_config))
        assert actives == sorted(actives)

    def test_missing_active_flag_defaults_to_active(self, tmp_path: Path) -> None:
        no_flag = {k: v for k, v in _MINIMAL_CITY.items() if k != "active"}
        (tmp_path / "cities.yaml").write_text(
            yaml.dump({"cities": [no_flag]}, default_flow_style=False),
            encoding="utf-8",
        )
        actives = list_active_cities(config_dir=str(tmp_path))
        assert "TestCity" in actives

    def test_all_inactive_returns_empty(self, tmp_path: Path) -> None:
        inactive = {**_MINIMAL_CITY, "active": False}
        (tmp_path / "cities.yaml").write_text(
            yaml.dump({"cities": [inactive]}), encoding="utf-8"
        )
        actives = list_active_cities(config_dir=str(tmp_path))
        assert actives == []

    def test_real_config_has_active_cities(self) -> None:
        """Smoke test against the actual project config."""
        actives = list_active_cities()
        assert len(actives) >= 1
        assert "Chicago" in actives

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            list_active_cities(config_dir=str(tmp_path / "missing"))


# ── Edge cases ────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge-case and integration tests."""

    def test_city_with_zero_model_weights(self, tmp_path: Path) -> None:
        """Model weights of 0.0 should not be treated as None."""
        city = {**_MINIMAL_CITY, "model_weight_gfs": 0.0}
        (tmp_path / "cities.yaml").write_text(
            yaml.dump({"cities": [city]}), encoding="utf-8"
        )
        p = load_city_profiles(config_dir=str(tmp_path))[0]
        assert p.model_weight_gfs == 0.0

    def test_negative_bias_correction(self, tmp_path: Path) -> None:
        city = {**_MINIMAL_CITY, "mean_bias_correction_f": -1.5}
        (tmp_path / "cities.yaml").write_text(
            yaml.dump({"cities": [city]}), encoding="utf-8"
        )
        p = load_city_profiles(config_dir=str(tmp_path))[0]
        assert p.mean_bias_correction_f == -1.5

    def test_large_variance_multiplier(self, tmp_path: Path) -> None:
        city = {**_MINIMAL_CITY, "variance_multiplier": 99.99}
        (tmp_path / "cities.yaml").write_text(
            yaml.dump({"cities": [city]}), encoding="utf-8"
        )
        p = load_city_profiles(config_dir=str(tmp_path))[0]
        assert p.variance_multiplier == 99.99

    def test_string_config_dir_converted_to_path(self, config_dir: Path) -> None:
        """Ensure passing a str (not Path) works seamlessly."""
        profiles = load_city_profiles(config_dir=str(config_dir))
        assert len(profiles) == 1

    def test_extra_yaml_fields_ignored(self, tmp_path: Path) -> None:
        """Unknown keys in the YAML should not cause errors."""
        city = {**_MINIMAL_CITY, "extra_field": "ignored", "another": 42}
        (tmp_path / "cities.yaml").write_text(
            yaml.dump({"cities": [city]}), encoding="utf-8"
        )
        p = load_city_profiles(config_dir=str(tmp_path))[0]
        assert p.city == "TestCity"
        # Extra fields should not be on the dataclass
        assert not hasattr(p, "extra_field")

    def test_roundtrip_all_real_cities(self) -> None:
        """Every city from the real config should be retrievable individually."""
        all_profiles = load_city_profiles()
        for profile in all_profiles:
            looked_up = get_city_profile(profile.city)
            assert looked_up.city == profile.city
            assert looked_up.station_id == profile.station_id

    def test_active_cities_is_subset_of_all_profiles(self) -> None:
        """Active cities must be a subset of all loaded cities."""
        all_names = {p.city for p in load_city_profiles()}
        active_names = set(list_active_cities())
        assert active_names.issubset(all_names)