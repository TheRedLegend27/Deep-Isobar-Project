"""Data Sources — source registry for Deep Isobar.

Loads the data-source registry from ``config/data_sources.yaml``, validates
the YAML structure, and exposes lookup helpers used by downstream modules
(weather ingestion, ensemble generation, market adapters, etc.).

The YAML file must contain a top-level ``sources`` mapping whose keys are
source names (e.g. ``"NWS"``, ``"GFS"``) and whose values are attribute
dictionaries.

Typical usage::

    from deep_isobar.data.data_sources import load_data_sources, get_data_source

    all_sources = load_data_sources()               # full registry
    nws         = get_data_source("NWS")            # single source lookup
    print(nws["authority_level"])                    # 10
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from deep_isobar.config import get_project_root

# ---------------------------------------------------------------------------
# Default config path (relative to project root)
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG_FILENAME: str = "data_sources.yaml"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_config_path(config_path: str | None) -> Path:
    """Build the absolute path to ``data_sources.yaml``.

    Args:
        config_path: Explicit path to a YAML file.  When ``None`` the
            project root's ``config/data_sources.yaml`` is used.

    Returns:
        Resolved ``Path`` to the data-sources YAML file.

    Raises:
        FileNotFoundError: If the resolved file does not exist.
    """
    path = (
        Path(config_path)
        if config_path
        else get_project_root() / "config" / _DEFAULT_CONFIG_FILENAME
    )

    if not path.exists():
        raise FileNotFoundError(f"Missing data-source config: {path}")

    return path


def _parse_sources(path: Path) -> dict[str, dict[str, Any]]:
    """Read and validate the top-level structure of ``data_sources.yaml``.

    The file must contain a YAML mapping with a ``sources`` key whose value
    is itself a non-empty mapping.

    Args:
        path: Absolute path to the YAML file.

    Returns:
        Dictionary of ``{source_name: source_attributes}``.

    Raises:
        ValueError: If the YAML content is not a mapping, has no
            ``sources`` key, or ``sources`` is not a mapping.
    """
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid data_sources config: expected a YAML mapping, "
            f"got {type(data).__name__}"
        )

    if "sources" not in data:
        raise ValueError(
            "Invalid data_sources config: missing top-level 'sources' key"
        )

    sources = data["sources"]

    if not isinstance(sources, dict):
        raise ValueError(
            f"Invalid data_sources config: 'sources' must be a mapping, "
            f"got {type(sources).__name__}"
        )

    return sources


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data_sources(config_path: str | None = None) -> dict[str, dict[str, Any]]:
    """Load and return the full data-source registry.

    Each key in the returned dictionary is a source name (e.g. ``"NWS"``),
    and the corresponding value is a dictionary of that source's attributes
    (``source_type``, ``authority_level``, ``active``, ``notes``, etc.).

    Args:
        config_path: Optional path to a YAML config file.  When ``None``,
            defaults to ``<project_root>/config/data_sources.yaml``.

    Returns:
        dict[str, dict[str, Any]]: Mapping of source names to their
        attribute dictionaries.

    Raises:
        FileNotFoundError: If the resolved config file does not exist.
        ValueError: If the YAML content is malformed — not a mapping
            or missing / invalid ``sources`` key.

    Example::

        sources = load_data_sources()
        for name, attrs in sources.items():
            print(f"{name}: type={attrs['source_type']}")
    """
    path = _resolve_config_path(config_path)
    return _parse_sources(path)


def get_data_source(
    name: str,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Return the attribute dictionary for a single data source.

    Source matching is **case-sensitive** and must exactly match a key
    under the ``sources`` mapping in the config file.

    Args:
        name: Exact source name (e.g. ``"NWS"``, ``"GFS"``).
        config_path: Optional config file path override.

    Returns:
        dict[str, Any]: Attribute dictionary for the requested source.

    Raises:
        FileNotFoundError: If the config file is missing.
        ValueError: If the YAML structure is invalid.
        KeyError: If no source matches the given name.  The error
            message includes the list of available source names.

    Example::

        nws = get_data_source("NWS")
        print(nws["authority_level"])  # 10
    """
    sources = load_data_sources(config_path)

    if name not in sources:
        available = sorted(sources.keys())
        raise KeyError(
            f"Data source not found: '{name}'. "
            f"Available sources: {available}"
        )

    return sources[name]