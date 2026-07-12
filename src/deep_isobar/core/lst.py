"""Local-Standard-Time settlement windows.

NWS Daily Climate Reports (CLI) — Kalshi's only settlement source — record
daily max/min over midnight-to-midnight **Local Standard Time, year-round**
(NWSI 10-1004). During Daylight Saving Time that window is 1 AM-1 AM on the
local clock, so any aggregation over the DST-aware local calendar day is
shifted one hour from the settlement window for ~7 months a year. A minimum
observed at 12:30 AM CDT belongs to the *previous* CLI day (11:30 PM LST);
verified real cases: KMDW 2026-06-10 (min 68°F at 11:59 PM LST) and
2025-07-06 (min 69°F at 11:59 PM LST). Minima are routinely set near the
boundary; maxima rarely — but the same window governs both.

The fix used throughout the pipeline: request Open-Meteo data in the
station's **fixed standard-time offset zone** (``Etc/GMT+N``) instead of its
DST-aware IANA zone. Hourly timestamps and native daily aggregations then
fall on the CLI settlement day with no further conversion.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo


def standard_offset_hours(tz_name: str) -> int:
    """The zone's standard-time UTC offset in whole hours (DST removed).

    Uses a mid-January instant (standard time in all US zones) and
    subtracts any DST component for robustness.
    """
    tz = ZoneInfo(tz_name)
    jan = datetime(date.today().year, 1, 15, 12, tzinfo=tz)
    off = jan.utcoffset() - (jan.dst() or timedelta(0))
    hours = off.total_seconds() / 3600.0
    if hours != int(hours):
        raise ValueError(
            f"{tz_name}: standard offset {hours}h is not a whole hour — "
            "Etc/GMT zones cannot represent it"
        )
    return int(hours)


def lst_timezone(tz_name: str) -> str:
    """IANA fixed-offset zone matching the station's Local Standard Time.

    Note the Etc/GMT sign convention is INVERTED: UTC-6 (Chicago standard
    time) is ``Etc/GMT+6``. Pass the result as Open-Meteo's ``timezone``
    parameter so hourly times and daily aggregations use the CLI
    settlement day year-round.
    """
    return f"Etc/GMT{-standard_offset_hours(tz_name):+d}"
