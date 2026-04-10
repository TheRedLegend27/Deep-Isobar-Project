"""METAR fetch and parse utilities for KMDW (Chicago Midway Airport).

Fetches the latest raw METAR from aviationweather.gov and extracts the
key fields used by the anomaly detector: wind direction, sky cover,
visibility, and temperature.  All functions are designed to never raise —
they return safe defaults on any parse or network failure.
"""

from __future__ import annotations

import logging
import re

import requests

logger = logging.getLogger(__name__)

_METAR_URL = "https://aviationweather.gov/api/data/metar?ids=KMDW&format=raw"

_CARDINAL = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
_SKY_PRIORITY = {"CLR": 0, "SKC": 0, "FEW": 1, "SCT": 2, "BKN": 3, "OVC": 4}


def fetch_kmdw_metar() -> str:
    """Fetch the latest raw METAR for KMDW from aviationweather.gov.

    Returns:
        Raw METAR string (e.g. ``"KMDW 101053Z 34007KT ..."``), or an empty
        string on any network or parse failure.
    """
    try:
        resp = requests.get(_METAR_URL, timeout=10)
        resp.raise_for_status()
        text = resp.text.strip()
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("KMDW"):
                return line
        # Fall back to first non-empty line if no KMDW-prefixed line found
        return text.splitlines()[0].strip() if text else ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch KMDW METAR: %s", exc)
        return ""


def _deg_to_cardinal(deg: float) -> str:
    """Convert wind direction in degrees to an 8-point cardinal string."""
    return _CARDINAL[round(deg / 45) % 8]


def parse_metar_fields(metar: str) -> dict:
    """Extract key weather fields from a raw METAR string.

    Uses simple regex — no external METAR library required.  Returns sensible
    defaults for any field that fails to parse.

    Args:
        metar: Raw METAR string (e.g. ``"KMDW 101053Z 34007KT 10SM FEW018
            OVC250 08/02 A2994 RMK AO2"``).

    Returns:
        Dict with keys:

        - ``wind_dir`` (str): Cardinal direction, e.g. ``"NE"``.  Empty string
          if wind direction cannot be parsed (e.g. variable).
        - ``sky_cover`` (str): Worst sky layer code — CLR/FEW/SCT/BKN/OVC.
        - ``visibility_mi`` (float): Prevailing visibility in statute miles.
        - ``temp_f`` (float): Current temperature in °F.
    """
    # Strip remarks — anything after RMK uses a different format
    body = metar.split("RMK")[0] if "RMK" in metar else metar

    # ── Wind direction ──────────────────────────────────────────────────────
    wind_dir = ""
    wind_match = re.search(r"\b(\d{3})\d{2,3}(?:G\d{2,3})?KT\b", body)
    if wind_match:
        wind_dir = _deg_to_cardinal(int(wind_match.group(1)))

    # ── Visibility ──────────────────────────────────────────────────────────
    visibility_mi = 10.0
    # Mixed fraction form: "1 1/4SM"
    vis_mixed = re.search(r"\b(\d+)\s+(\d+)/(\d+)SM\b", body)
    if vis_mixed:
        visibility_mi = int(vis_mixed.group(1)) + int(vis_mixed.group(2)) / int(vis_mixed.group(3))
    else:
        # Pure fraction form: "1/4SM"
        vis_frac = re.search(r"\b(\d+)/(\d+)SM\b", body)
        if vis_frac:
            visibility_mi = int(vis_frac.group(1)) / int(vis_frac.group(2))
        else:
            # Integer form: "10SM"
            vis_int = re.search(r"\b(\d+)SM\b", body)
            if vis_int:
                visibility_mi = float(vis_int.group(1))

    # ── Sky cover (worst layer) ─────────────────────────────────────────────
    sky_cover = "CLR"
    for sky_match in re.finditer(r"\b(CLR|SKC|FEW|SCT|BKN|OVC)(\d{3})?\b", body):
        layer = sky_match.group(1)
        if _SKY_PRIORITY.get(layer, 0) > _SKY_PRIORITY.get(sky_cover, 0):
            sky_cover = layer
    if sky_cover == "SKC":
        sky_cover = "CLR"

    # ── Temperature ─────────────────────────────────────────────────────────
    # Format: TT/TD where M prefix means negative (M02 = -2°C)
    temp_f = 32.0
    temp_match = re.search(r"\b(M?\d{2})/(M?\d{2})\b", body)
    if temp_match:
        t_str = temp_match.group(1).replace("M", "-")
        temp_c = float(t_str)
        temp_f = temp_c * 9.0 / 5.0 + 32.0

    return {
        "wind_dir": wind_dir,
        "sky_cover": sky_cover,
        "visibility_mi": visibility_mi,
        "temp_f": round(temp_f, 1),
    }
