"""NWS human forecast fetcher for the Deep Isobar anomaly detector.

Fetches the NWS official high-temperature forecast for a lat/lon point using
the public api.weather.gov REST API (no auth required).  Used to feed the
NWS_MODEL_DIVERGENCE flag in the anomaly detector.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

_NWS_POINTS_URL = "https://api.weather.gov/points/{lat},{lon}"
_REQUEST_TIMEOUT_S = 10
_NWS_USER_AGENT = "deep-isobar/1.0 (weather-trading-research; contact via github)"


def fetch_nws_high_forecast_f(lat: float, lon: float) -> Optional[float]:
    """Fetch the NWS human high temperature forecast for the given lat/lon.

    Returns the forecast high in °F for today if available before noon local,
    tomorrow if called after noon. Returns None on any failure — never raises.

    Uses NWS public API (no auth required):
    Step 1: GET https://api.weather.gov/points/{lat},{lon}
            → extract properties.forecast URL
    Step 2: GET {forecast_url}
            → find first period where isDaytime=True
            → return temperature (already in °F per NWS default)

    Timeout: 10 seconds per request.
    On any exception (network, parse, unexpected schema): log at DEBUG, return None.

    Args:
        lat: Latitude in decimal degrees (standard -90..+90).
        lon: Longitude in decimal degrees (standard -180..+180; west is negative).

    Returns:
        Forecast high temperature in °F, or ``None`` if unavailable.
    """
    try:
        points_url = _NWS_POINTS_URL.format(lat=round(lat, 4), lon=round(lon, 4))
        req = urllib.request.Request(
            points_url,
            headers={"User-Agent": _NWS_USER_AGENT, "Accept": "application/geo+json"},
        )
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
            points_data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.debug("NWS points lookup failed for (%.4f, %.4f): %s", lat, lon, exc)
        return None

    try:
        forecast_url: str = points_data["properties"]["forecast"]
    except (KeyError, TypeError) as exc:
        logger.debug("NWS points response missing properties.forecast: %s", exc)
        return None

    try:
        req2 = urllib.request.Request(
            forecast_url,
            headers={"User-Agent": _NWS_USER_AGENT, "Accept": "application/geo+json"},
        )
        with urllib.request.urlopen(req2, timeout=_REQUEST_TIMEOUT_S) as resp2:
            forecast_data = json.loads(resp2.read().decode("utf-8"))
    except Exception as exc:
        logger.debug("NWS forecast fetch failed (%s): %s", forecast_url, exc)
        return None

    try:
        periods: list = forecast_data["properties"]["periods"]
    except (KeyError, TypeError) as exc:
        logger.debug("NWS forecast response missing properties.periods: %s", exc)
        return None

    for period in periods:
        if period.get("isDaytime"):
            temp = period.get("temperature")
            if temp is not None:
                return float(temp)
            logger.debug("NWS daytime period found but temperature is None")
            return None

    logger.debug("NWS forecast returned no daytime periods")
    return None
