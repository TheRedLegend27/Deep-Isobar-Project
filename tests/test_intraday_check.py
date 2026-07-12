"""Boundary semantics for the 2pm intraday advisory.

The YES-outcome classification must mirror settle_paper_trades grading
exactly: ``less`` = actual < cap (strict), ``greater`` = actual > floor
(strict), ``between`` = floor <= actual <= cap (cap INCLUSIVE — Kalshi
B82.5 "82-83" pays on 82 AND 83). A running high can only rise, so a
bracket is DEAD only when the high is strictly above the cap; sitting
exactly on the cap is still a winning position at risk of overshoot.
"""

import pytest

from deep_isobar.research.intraday_check import _classify_yes, _position_state


class TestBetweenCapInclusive:
    def test_high_equal_to_cap_is_still_in_range(self):
        # 2026-07-11 regression: KXHIGHCHI-26JUL11-B81.5 (floor 81, cap 82)
        # was reported BUSTED at a running high of exactly 82.0.
        state, _ = _classify_yes("between", 81.0, 82.0, 82.0)
        assert state == "IN_RANGE"

    def test_high_above_cap_is_dead(self):
        state, _ = _classify_yes("between", 86.0, 87.0, 87.1)
        assert state == "DEAD"

    def test_high_below_floor_is_pending(self):
        state, _ = _classify_yes("between", 81.0, 82.0, 79.0)
        assert state == "PENDING"

    def test_high_at_floor_is_in_range(self):
        state, _ = _classify_yes("between", 81.0, 82.0, 81.0)
        assert state == "IN_RANGE"


class TestTailStrikes:
    def test_less_dead_once_high_reaches_cap(self):
        # less grades actual < cap, so high == cap already kills YES.
        state, _ = _classify_yes("less", None, 79.0, 79.0)
        assert state == "DEAD"

    def test_less_alive_below_cap(self):
        state, _ = _classify_yes("less", None, 79.0, 77.0)
        assert state == "IN_RANGE"

    def test_greater_locked_only_strictly_above_floor(self):
        # greater grades actual > floor: high == floor is not yet a win.
        state, _ = _classify_yes("greater", 83.0, None, 83.0)
        assert state == "PENDING"
        state, _ = _classify_yes("greater", 83.0, None, 84.0)
        assert state == "LOCKED"


class TestPositionState:
    @pytest.mark.parametrize(
        "direction, yes_state, expected",
        [
            ("BUY", "LOCKED", "LOCKED_WIN"),
            ("BUY", "DEAD", "BUSTED"),
            ("SELL", "LOCKED", "BUSTED"),
            ("SELL", "DEAD", "LOCKED_WIN"),
            ("BUY", "IN_RANGE", "IN_RANGE"),
            ("SELL", "PENDING", "PENDING"),
        ],
    )
    def test_direction_translation(self, direction, yes_state, expected):
        assert _position_state(direction, yes_state) == expected
