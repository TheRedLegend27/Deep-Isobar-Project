"""Fixed-LST settlement-window tests (deep_isobar.core.lst).

The NWS CLI — Kalshi's settlement source — records daily extremes over
midnight-to-midnight Local Standard Time year-round. During DST a naive
local-calendar-day aggregation is shifted one hour and mis-assigns any
extreme observed at 12:00-12:59 AM local daylight time to the wrong day
(verified real cases: KMDW minima at 11:59 PM LST on 2026-06-10 and
2025-07-06).
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from deep_isobar.core.lst import lst_timezone, standard_offset_hours


class TestStandardOffsets:
    @pytest.mark.parametrize(
        "tz_name, hours, etc_zone",
        [
            ("America/New_York", -5, "Etc/GMT+5"),
            ("America/Chicago", -6, "Etc/GMT+6"),
            ("America/Denver", -7, "Etc/GMT+7"),
            ("America/Phoenix", -7, "Etc/GMT+7"),  # no DST; identical zone
            ("America/Los_Angeles", -8, "Etc/GMT+8"),
        ],
    )
    def test_us_zones(self, tz_name, hours, etc_zone):
        assert standard_offset_hours(tz_name) == hours
        assert lst_timezone(tz_name) == etc_zone

    def test_every_configured_city_resolves(self):
        """Each cities.yaml timezone must map to a valid Etc/GMT zone whose
        offset equals the station's standard offset."""
        from deep_isobar.data.city_universe import get_city_universe

        for profile in get_city_universe():
            zone = lst_timezone(profile.timezone)
            fixed = ZoneInfo(zone)
            jan = datetime(2026, 1, 15, 12, tzinfo=fixed)
            assert jan.utcoffset().total_seconds() / 3600 == standard_offset_hours(
                profile.timezone
            ), f"{profile.city}: {zone} offset mismatch"


class TestDstBoundaryAssignment:
    def test_trough_after_midnight_daylight_belongs_to_previous_cli_day(self):
        """KMDW worked example: a minimum at 12:30 AM CDT on June 11 is
        11:30 PM LST June 10 — it settles the June 10 contract."""
        trough_local = datetime(2026, 6, 11, 0, 30, tzinfo=ZoneInfo("America/Chicago"))
        in_lst = trough_local.astimezone(ZoneInfo(lst_timezone("America/Chicago")))
        assert in_lst.date().isoformat() == "2026-06-10"
        # The naive DST-clock calendar day disagrees — that's the trap.
        assert trough_local.date().isoformat() == "2026-06-11"

    def test_standard_time_agrees_with_local_day(self):
        """In winter CST == LST, so both conventions assign the same day."""
        trough_local = datetime(2026, 1, 8, 23, 59, tzinfo=ZoneInfo("America/Chicago"))
        in_lst = trough_local.astimezone(ZoneInfo(lst_timezone("America/Chicago")))
        assert in_lst.date() == trough_local.date()
