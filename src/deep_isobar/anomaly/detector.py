"""Anomaly detection module for Deep Isobar.

Uses the Anthropic API to identify synoptic weather anomalies that may cause
GFS model forecasts to be systematically wrong for today's Chicago high.
Results are logged alongside trade signals for later review — they do not
affect trade execution.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

logger = logging.getLogger(__name__)

_MODEL = "claude-opus-4-6"

_SYSTEM_PROMPT = """\
You are an anomaly detection module for a quantitative weather trading system targeting Chicago high temperature contracts (KXHIGHCHI) on Kalshi. Kalshi settles on KMDW (Chicago Midway Airport).

Your job is to analyze morning weather conditions and identify synoptic anomalies that could cause GFS model forecasts to be systematically wrong for today's high temperature. The trading system already applies a monthly mean bias correction. Your job is to find ADDITIONAL situational factors on top of that.

Key anomalies to detect:
- FOG_MORNING: Low visibility + low cloud base at 9am. Fog/low stratus suppresses daytime high by 3-8°F. Trigger if sky is OVC/BKN below 1500ft and visibility < 5mi before 10am.
- LAKE_BREEZE: NE or E wind off Lake Michigan. Can hold Chicago 5-15°F cooler than inland. Stronger effect in spring/early summer.
- COLD_START: Current temp more than 12°F below climatological normal for the hour. Less runway to warm up.
- FRONTAL_SUPPRESSION: NE quadrant winds with overcast suggest cold air advection suppressing daytime warming.
- NWS_MODEL_DIVERGENCE: If NWS human forecast is 3°F+ below the GFS model mean, flag it — NWS forecasters post-process GFS and their disagreement is meaningful.

Respond ONLY with a JSON object, no markdown, no preamble. Schema:
{"flags": [{"code": "FOG_MORNING", "description": "...", "temp_impact_f": -5}], "total_temp_penalty_f": -7, "adjusted_signal": "BUY"|"REDUCE"|"PASS", "confidence": "HIGH"|"MEDIUM"|"LOW", "reasoning": "2-3 sentence plain English summary."}"""


@dataclass
class AnomalyFlag:
    code: str
    description: str
    temp_impact_f: float


@dataclass
class AnomalyReport:
    flags: list[AnomalyFlag]
    total_temp_penalty_f: float
    adjusted_signal: str
    confidence: str
    reasoning: str


def check_anomalies(
    metar: str,
    nws_forecast_f: Optional[float],
    model_mean_f: float,
    wind_dir: str,
    sky_cover: str,
) -> AnomalyReport:
    """Call the Anthropic API to detect weather anomalies for today's KMDW session.

    Results are informational only and must not affect trade execution.

    Args:
        metar: Raw METAR string for KMDW.
        nws_forecast_f: NWS human forecast high in °F. Pass ``None`` until
            NWS fetch is wired.
        model_mean_f: GFS ensemble mean forecast in °F (bias-corrected).
        wind_dir: Cardinal wind direction string (e.g. ``"NE"``).
        sky_cover: Worst sky layer code (CLR/FEW/SCT/BKN/OVC).

    Returns:
        :class:`AnomalyReport` parsed from the API response.  On any failure
        returns a safe fallback report with empty flags and LOW confidence.
    """
    try:
        import anthropic  # imported here so missing SDK gives a clean error message

        nws_str = f"{nws_forecast_f:.1f}°F" if nws_forecast_f is not None else "not provided"
        user_message = (
            f"Current KMDW conditions:\n"
            f"METAR: {metar or 'unavailable'}\n"
            f"GFS model mean (bias-corrected): {model_mean_f:.1f}°F\n"
            f"NWS human forecast: {nws_str}\n"
            f"Wind direction: {wind_dir or 'unknown'}\n"
            f"Sky cover: {sky_cover or 'unknown'}\n"
            f"Identify any anomalies and return JSON."
        )

        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model=_MODEL,
            max_tokens=512,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        raw = message.content[0].text.strip()
        data = json.loads(raw)

        flags = [
            AnomalyFlag(
                code=str(f["code"]),
                description=str(f["description"]),
                temp_impact_f=float(f["temp_impact_f"]),
            )
            for f in (data.get("flags") or [])
        ]

        return AnomalyReport(
            flags=flags,
            total_temp_penalty_f=float(data.get("total_temp_penalty_f", 0.0)),
            adjusted_signal=str(data.get("adjusted_signal", "BUY")),
            confidence=str(data.get("confidence", "LOW")),
            reasoning=str(data.get("reasoning", "")),
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning("Anomaly check failed: %s", exc)
        return AnomalyReport(
            flags=[],
            total_temp_penalty_f=0.0,
            adjusted_signal="BUY",
            confidence="LOW",
            reasoning=f"Anomaly check failed: {exc}",
        )
