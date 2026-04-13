"""Anomaly detection sub-package for Deep Isobar.

Provides weather anomaly detection using the Anthropic API alongside
METAR fetch and parse utilities for KMDW.  All results are additive —
they are logged for review and never gate or modify trade execution.
"""

from .detector import AnomalyFlag, AnomalyReport, check_anomalies
from .metar_fetcher import fetch_kmdw_metar, parse_metar_fields
from .nws_fetcher import fetch_nws_high_forecast_f

__all__ = [
    "AnomalyFlag",
    "AnomalyReport",
    "check_anomalies",
    "fetch_kmdw_metar",
    "parse_metar_fields",
    "fetch_nws_high_forecast_f",
]
