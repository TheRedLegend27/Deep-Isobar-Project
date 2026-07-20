"""METAR fetch and parse utilities for arbitrary ICAO stations.

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

_METAR_URL_TEMPLATE = "https://aviationweather.gov/api/data/metar?ids={station_id}&format=raw"
_METAR_HOURS_URL_TEMPLATE = (
    "https://aviationweather.gov/api/data/metar?ids={station_id}&format=raw&hours={hours}"
)

_CARDINAL = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
_SKY_PRIORITY = {"CLR": 0, "SKC": 0, "FEW": 1, "SCT": 2, "BKN": 3, "OVC": 4}

# Present-weather group: optional intensity/vicinity prefix, then a token
# composed entirely of 2-letter descriptor/phenomenon codes (FU=smoke,
# HZ=haze, BR=mist, TS=thunderstorm, ...).  Whole-token anchoring keeps
# station ids, cloud groups, and wind groups from false-matching.
_WX_CODES = (
    "MI|PR|BC|DR|BL|SH|TS|FZ"
    "|DZ|RA|SN|SG|IC|PL|GR|GS|UP"
    "|BR|FG|FU|VA|DU|SA|HZ|PY"
    "|PO|SQ|FC|SS|DS"
)
_WX_GROUP_RE = re.compile(rf"^(?:[+-]|VC)?((?:{_WX_CODES})+)$")
_WX_PHENOMENON_RE = re.compile(_WX_CODES)


def fetch_metar(station_id: str) -> str:
    """Fetch the latest raw METAR for *station_id* from aviationweather.gov.

    Args:
        station_id: ICAO station identifier (e.g. ``"KMDW"``, ``"KDFW"``).

    Returns:
        Raw METAR string (e.g. ``"KMDW 101053Z 34007KT ..."``), or an empty
        string on any network or parse failure.
    """
    url = _METAR_URL_TEMPLATE.format(station_id=station_id.upper())
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        text = resp.text.strip()
        for line in text.splitlines():
            line = line.strip()
            if line.startswith(station_id.upper()):
                return line
        # Fall back to first non-empty line if no station-prefixed line found
        return text.splitlines()[0].strip() if text else ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch METAR for %s: %s", station_id, exc)
        return ""


# Backward-compatible alias — existing imports of fetch_kmdw_metar continue to work.
fetch_kmdw_metar = lambda: fetch_metar("KMDW")  # noqa: E731


def fetch_metars(station_id: str, hours: int = 6) -> list[str]:
    """Fetch all raw METARs for *station_id* from the past *hours* hours.

    Returns:
        List of raw METAR strings, most recent first (aviationweather.gov
        order), or an empty list on any network failure.
    """
    url = _METAR_HOURS_URL_TEMPLATE.format(station_id=station_id.upper(), hours=hours)
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        out = []
        for line in resp.text.splitlines():
            line = line.strip()
            # With the hours param the API prefixes the report type
            # ("METAR KNYC ..." / "SPECI KNYC ..."); strip it so parsing
            # sees the same shape as the single-report endpoint.
            for prefix in ("METAR ", "SPECI "):
                if line.startswith(prefix):
                    line = line[len(prefix):]
                    break
            if line.startswith(station_id.upper()):
                out.append(line)
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch METAR history for %s: %s", station_id, exc)
        return []


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
        - ``wx_codes`` (list[str]): Present-weather phenomena in the body,
          e.g. ``["FU"]`` for smoke, ``["HZ"]`` for haze, ``["TS", "RA"]``
          for a thunderstorm with rain.  Empty when no weather group.
    """
    # Strip remarks — anything after RMK uses a different format
    body = metar.split("RMK")[0] if "RMK" in metar else metar

    # ── Wind direction ──────────────────────────────────────────────────────
    wind_dir = ""
    wind_match = re.search(r"\b(\d{3})\d{2,3}(?:G\d{2,3})?KT\b", body)
    if wind_match:
        wind_dir = _deg_to_cardinal(int(wind_match.group(1)))

    # ── Visibility ──────────────────────────────────────────────────────────
    # M prefix means "less than" (e.g. M1/4SM = < 0.25 mi); strip it and use
    # the value as-is — it still represents near-zero visibility for our purposes.
    visibility_mi = 10.0
    # Mixed fraction form: "1 1/4SM" or "M1 1/4SM"
    vis_mixed = re.search(r"\bM?(\d+)\s+(\d+)/(\d+)SM\b", body)
    if vis_mixed:
        visibility_mi = int(vis_mixed.group(1)) + int(vis_mixed.group(2)) / int(vis_mixed.group(3))
    else:
        # Pure fraction form: "1/4SM" or "M1/4SM"
        vis_frac = re.search(r"\bM?(\d+)/(\d+)SM\b", body)
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

    # ── Temperature ─────────────────────────────────────────────────────────
    # Format: TT/TD where M prefix means negative (M02 = -2°C)
    temp_f = 32.0
    temp_match = re.search(r"\b(M?\d{2})/(M?\d{2})\b", body)
    if temp_match:
        t_str = temp_match.group(1).replace("M", "-")
        temp_c = float(t_str)
        temp_f = temp_c * 9.0 / 5.0 + 32.0

    # ── Present weather ─────────────────────────────────────────────────────
    wx_codes: list[str] = []
    for token in body.split():
        group = _WX_GROUP_RE.match(token)
        if group:
            wx_codes.extend(_WX_PHENOMENON_RE.findall(group.group(1)))

    return {
        "wind_dir": wind_dir,
        "sky_cover": sky_cover,
        "visibility_mi": visibility_mi,
        "temp_f": round(temp_f, 1),
        "wx_codes": wx_codes,
    }
