"""City Universe — tradable city registry for Deep Isobar.

Loads city profiles from ``config/cities.yaml``, validates the YAML
structure, and exposes lookup / listing helpers used by downstream
modules (weather ingestion, ensemble generation, contract generator, etc.).

Typical usage::

    from deep_isobar.data.city_universe import (
        load_city_profiles,
        get_city_profile,
        list_active_cities,
    )

    profiles = load_city_profiles()           # all profiles from default config
    chi      = get_city_profile("Chicago")    # single profile lookup
    actives  = list_active_cities()           # names of active cities only
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from deep_isobar.config import get_project_root
from deep_isobar.core.types import CityProfile

# ---------------------------------------------------------------------------
# Required fields — every city entry MUST contain these keys.
# ---------------------------------------------------------------------------
_REQUIRED_FIELDS: tuple[str, ...] = (
    "city",
    "city_code",
    "station_id",
    "timezone",
    "settlement_source",
    "variance_multiplier",
    "mean_bias_correction_f",
    "kde_bandwidth",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_config_path(config_dir: str | None) -> Path:
    """Build the absolute path to ``cities.yaml``.

    Args:
        config_dir: Explicit config directory. When ``None`` the project
            root's ``config/`` folder is used.

    Returns:
        Resolved ``Path`` to the cities YAML file.

    Raises:
        FileNotFoundError: If the resolved file does not exist.
    """
    base = Path(config_dir) if config_dir else get_project_root() / "config"
    path = base / "cities.yaml"

    if not path.exists():
        raise FileNotFoundError(f"Missing city config: {path}")

    return path


def _parse_yaml(path: Path) -> list[dict[str, Any]]:
    """Read and validate the top-level structure of ``cities.yaml``.

    The file must contain a YAML mapping with a ``cities`` key whose value
    is a non-empty list of mappings.

    Args:
        path: Absolute path to the YAML file.

    Returns:
        List of raw city dictionaries.

    Raises:
        ValueError: If the YAML content is not a valid mapping,
            has no ``cities`` key, or the value is not a non-empty list
            of mappings.
    """
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid cities config: expected a YAML mapping, "
            f"got {type(data).__name__}"
        )

    cities_raw = data.get("cities")

    if cities_raw is None:
        raise ValueError(
            "Invalid cities config: missing top-level 'cities' key"
        )

    if not isinstance(cities_raw, list):
        raise ValueError(
            f"Invalid cities config: 'cities' must be a list, "
            f"got {type(cities_raw).__name__}"
        )

    if len(cities_raw) == 0:
        raise ValueError("Invalid cities config: 'cities' list is empty")

    # Validate that every element is a mapping
    for idx, entry in enumerate(cities_raw):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Invalid cities config: entry at index {idx} "
                f"is not a mapping (got {type(entry).__name__})"
            )

    return cities_raw


def _validate_entry(entry: dict[str, Any], index: int) -> None:
    """Ensure a single city entry contains all required fields.

    Args:
        entry: Raw dictionary for one city.
        index: Position in the YAML list (for error messages).

    Raises:
        ValueError: If any required field is missing.
    """
    missing = [f for f in _REQUIRED_FIELDS if f not in entry]
    if missing:
        city_label = entry.get("city", f"<index {index}>")
        raise ValueError(
            f"City '{city_label}' is missing required field(s): "
            f"{', '.join(missing)}"
        )


def _build_profile(entry: dict[str, Any]) -> CityProfile:
    """Construct a ``CityProfile`` dataclass from a raw YAML dict.

    Required fields are passed positionally; optional fields use
    ``.get()`` with their dataclass defaults.

    Args:
        entry: Validated dictionary for one city.

    Returns:
        Populated ``CityProfile`` instance.
    """
    return CityProfile(
        city=entry["city"],
        city_code=entry["city_code"],
        station_id=entry["station_id"],
        timezone=entry["timezone"],
        settlement_source=entry["settlement_source"],
        variance_multiplier=float(entry["variance_multiplier"]),
        mean_bias_correction_f=float(entry["mean_bias_correction_f"]),
        kde_bandwidth=float(entry["kde_bandwidth"]),
        tail_multiplier=float(entry.get("tail_multiplier", 1.0)),
        model_weight_gfs=entry.get("model_weight_gfs"),
        model_weight_ecmwf=entry.get("model_weight_ecmwf"),
        model_weight_nam=entry.get("model_weight_nam"),
        heat_bias_adjustment_f=float(entry.get("heat_bias_adjustment_f", 0.0)),
        cold_bias_adjustment_f=float(entry.get("cold_bias_adjustment_f", 0.0)),
        sep_climate_normal_f=entry.get("sep_climate_normal_f"),
        sep_anomaly_trigger_f=entry.get("sep_anomaly_trigger_f"),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_city_profiles(config_dir: str | None = None) -> list[CityProfile]:
    """Load all city profiles from ``config/cities.yaml``.

    Each entry in the YAML ``cities`` list is validated and mapped to a
    :class:`~deep_isobar.core.types.CityProfile` dataclass.

    Args:
        config_dir: Optional path to the directory containing
            ``cities.yaml``.  Defaults to ``<project_root>/config``.

    Returns:
        List of ``CityProfile`` instances — one per city entry.

    Raises:
        FileNotFoundError: If ``cities.yaml`` does not exist at the
            resolved path.
        ValueError: If the YAML structure is invalid or a required
            field is missing from any city entry.

    Example::

        profiles = load_city_profiles()
        for p in profiles:
            print(p.city, p.station_id)
    """
    path = _resolve_config_path(config_dir)
    cities_raw = _parse_yaml(path)

    profiles: list[CityProfile] = []
    for idx, entry in enumerate(cities_raw):
        _validate_entry(entry, idx)
        profiles.append(_build_profile(entry))

    return profiles


def get_city_profile(
    city: str,
    config_dir: str | None = None,
) -> CityProfile:
    """Return the ``CityProfile`` for the given city name.

    City matching is **case-sensitive** and must exactly match the
    ``city`` field in the config.

    Args:
        city: Exact city name (e.g. ``"Chicago"``).
        config_dir: Optional config directory override.

    Returns:
        The matching ``CityProfile``.

    Raises:
        FileNotFoundError: If ``cities.yaml`` is missing.
        ValueError: If the YAML structure is invalid.
        KeyError: If no city matches the given name.

    Example::

        chi = get_city_profile("Chicago")
        print(chi.station_id)  # KORD
    """
    profiles = load_city_profiles(config_dir)
    for profile in profiles:
        if profile.city == city:
            return profile

    available = [p.city for p in profiles]
    raise KeyError(
        f"City not found: '{city}'. Available cities: {available}"
    )


def list_active_cities(config_dir: str | None = None) -> list[str]:
    """Return the names of all cities marked ``active: true`` in the config.

    Cities that are missing the ``active`` key default to **active**
    (inclusive by default).

    Args:
        config_dir: Optional config directory override.

    Returns:
        Sorted list of active city names.

    Raises:
        FileNotFoundError: If ``cities.yaml`` is missing.
        ValueError: If the YAML structure is invalid.

    Example::

        actives = list_active_cities()
        print(actives)  # ['Chicago', 'Dallas', 'Denver', ...]
    """
    path = _resolve_config_path(config_dir)
    cities_raw = _parse_yaml(path)

    active_names: list[str] = []
    for entry in cities_raw:
        # Default to active if key is absent
        if entry.get("active", True):
            active_names.append(entry["city"])

    return sorted(active_names)