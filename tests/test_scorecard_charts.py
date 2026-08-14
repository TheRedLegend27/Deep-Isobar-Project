"""Tests for deep_isobar.research.scorecard_charts (pure computations + markup)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import date
from html.parser import HTMLParser

import numpy as np
import pandas as pd
import pytest

from deep_isobar.ops.health import ALARM, OK, SKIP, HealthCheck
from deep_isobar.research.daily_scorecard import realized_outcome
from deep_isobar.research.scorecard_charts import (
    _attr_json,
    _nice_ticks,
    daily_equity_curve,
    render_bar_chart,
    render_html_dashboard,
    render_line_chart,
    rolling_daily_metrics,
)


def _trades_frame() -> pd.DataFrame:
    """Mirrors tests/test_daily_scorecard.py's fixture shape, with a gap day
    (Jul 4) that has zero settlements — the exact shape a real outage leaves."""
    rows = [
        (date(2026, 7, 1), "BUY",  "WIN",  0.70, 0.55, 4.5),
        (date(2026, 7, 1), "SELL", "LOSS", 0.30, 0.40, -6.0),
        (date(2026, 7, 2), "BUY",  "WIN",  0.65, 0.50, 5.0),
        (date(2026, 7, 3), "SELL", "WIN",  0.20, 0.35, 3.5),
        # Jul 4: nothing settles (outage day) — deliberately no rows.
        (date(2026, 7, 5), "BUY",  "LOSS", 0.60, 0.45, -4.5),
        (date(2026, 7, 5), "BUY",  "WIN",  0.75, 0.60, 4.0),
    ]
    df = pd.DataFrame(
        rows, columns=["date", "direction", "status", "model_prob", "market_prob", "realized_pnl"],
    )
    df["outcome"] = [realized_outcome(d, s) for d, s in zip(df["direction"], df["status"])]
    return df


# ── daily_equity_curve ───────────────────────────────────────────────────────


def test_daily_equity_curve_cumulative_and_ordered():
    curve = daily_equity_curve(_trades_frame())
    assert list(curve["date"]) == [date(2026, 7, 1), date(2026, 7, 2), date(2026, 7, 3), date(2026, 7, 5)]
    assert curve["daily_pnl"].tolist() == pytest.approx([-1.5, 5.0, 3.5, -0.5])
    assert curve["cumulative_pnl"].tolist() == pytest.approx([-1.5, 3.5, 7.0, 6.5])


def test_daily_equity_curve_empty():
    curve = daily_equity_curve(pd.DataFrame(columns=["date", "realized_pnl"]))
    assert curve.empty
    assert list(curve.columns) == ["date", "daily_pnl", "cumulative_pnl"]


# ── rolling_daily_metrics ────────────────────────────────────────────────────


def test_rolling_daily_metrics_window_respects_calendar_gap():
    df = _trades_frame()
    out = rolling_daily_metrics(df, windows=(2, 30))
    out = out.set_index("date")

    # Jul 4 is a zero-trade calendar day but must still exist as a row
    # (the outage-day gap the rolling window has to see, not skip over) —
    # its trailing 2-day window [Jul 3, Jul 4] still carries Jul 3's 1 trade.
    assert date(2026, 7, 4) in out.index
    assert out.loc[date(2026, 7, 4), "n_2d"] == 1

    # 2-day window on Jul 3 sees only Jul 2 + Jul 3's trades (2 rows), not Jul 1's.
    assert out.loc[date(2026, 7, 3), "n_2d"] == 2

    # 30-day window on the last day sees every trade.
    assert out.loc[date(2026, 7, 5), "n_30d"] == 6
    assert out.loc[date(2026, 7, 5), "win_rate_30d"] == pytest.approx(4 / 6)


def test_rolling_daily_metrics_nan_before_any_window_data():
    df = _trades_frame()
    out = rolling_daily_metrics(df, windows=(1,))
    out = out.set_index("date")
    # A 1-day window on the outage day (Jul 4, zero trades that day) has no
    # data at all — must be NaN, never silently zero.
    assert np.isnan(out.loc[date(2026, 7, 4), "brier_edge_1d"])
    assert np.isnan(out.loc[date(2026, 7, 4), "win_rate_1d"])


def test_rolling_daily_metrics_empty():
    out = rolling_daily_metrics(pd.DataFrame(columns=["date", "model_prob", "market_prob", "status", "outcome"]))
    assert out.empty


# ── _nice_ticks ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("vmin,vmax", [(0, 1), (-0.09, 0.06), (0, 600), (5, 5), (-3.5, -1.2)])
def test_nice_ticks_evenly_spaced_and_covers_vmin(vmin, vmax):
    """Ticks are round numbers at a fixed step, starting at/below vmin.

    They are NOT guaranteed to reach vmax — callers (render_line_chart) pad
    the plotted domain themselves; a "nice" step landing just short of vmax
    is normal chart behavior (matplotlib/d3 do the same).
    """
    ticks = _nice_ticks(vmin, vmax, count=4)
    assert ticks == sorted(ticks)
    assert len(ticks) >= 2
    assert ticks[0] <= vmin
    steps = [round(b - a, 10) for a, b in zip(ticks, ticks[1:])]
    assert len(set(steps)) == 1  # constant step throughout
    # The last tick must be within one step of vmax — not short by more than that.
    assert ticks[-1] >= vmax - steps[0]


# ── _attr_json — regression test for the P&L-breaks-the-attribute bug ───────


def test_attr_json_escapes_ampersand_lt_and_quote():
    payload = {"d": "Cumulative P&L", "rows": [["a < b & c's", "1", "x"]]}
    encoded = _attr_json(payload)
    assert "&" not in encoded.replace("&amp;", "").replace("&lt;", "").replace("&#39;", "")
    # Simulate the browser's attribute-value decoding, then JSON.parse.
    decoded = (
        encoded.replace("&#39;", "'").replace("&lt;", "<").replace("&amp;", "&")
    )
    import json
    assert json.loads(decoded) == payload


def test_attr_json_embeds_safely_in_single_quoted_html_attribute():
    payload = {"d": "Cumulative P&L", "rows": []}
    html = f"<rect data-tip='{_attr_json(payload)}'/>"
    # Well-formed as XML — this is exactly what broke before the fix
    # (raw '&' in "P&L" made the attribute value invalid).
    ET.fromstring(html)


# ── render_line_chart / render_bar_chart — markup smoke tests ───────────────


def _assert_svgs_well_formed(html: str) -> None:
    svgs = re.findall(r"<svg.*?</svg>", html, re.S)
    assert svgs, "expected at least one <svg> block"
    for s in svgs:
        ET.fromstring(s)  # raises on malformed markup


def test_render_line_chart_single_series_is_well_formed():
    dates = [date(2026, 7, 1), date(2026, 7, 2), date(2026, 7, 3)]
    series = [{"name": "Cumulative P&L", "color_var": "--series-1", "values": [1.0, -2.0, 3.5]}]
    html = render_line_chart("equity", "Equity curve", "Cumulative realized P&L", dates, series,
                              y_zero_line=True, area_fill_first=True)
    _assert_svgs_well_formed(html)
    assert "NaN" not in html


def test_render_line_chart_two_series_with_gaps_is_well_formed():
    dates = [date(2026, 7, 1), date(2026, 7, 2), date(2026, 7, 3)]
    series = [
        {"name": "7d edge", "color_var": "--series-1", "values": [0.01, None, -0.02]},
        {"name": "30d edge", "color_var": "--series-2", "values": [None, None, 0.005]},
    ]
    html = render_line_chart("edge", "Brier edge", "Rolling", dates, series, y_zero_line=True)
    _assert_svgs_well_formed(html)
    assert "chart-legend" in html  # 2 series → legend required


def test_render_line_chart_empty_does_not_crash():
    html = render_line_chart("empty", "Equity curve", "Cumulative realized P&L", [], [])
    assert "<svg" not in html
    assert "Not enough settled trades" in html


def test_render_bar_chart_well_formed_and_sorted_best_first():
    calibrations = [
        {"city": "Boston", "crps": 1.5, "n_days": 30},
        {"city": "Miami", "crps": 0.6, "n_days": 30},
        {"city": "Chicago", "crps": 1.0, "n_days": 30},
    ]
    html = render_bar_chart("Calibration", "CRPS by station", calibrations)
    _assert_svgs_well_formed(html)
    # Best (lowest CRPS) must be drawn first (top row).
    assert html.index("Miami") < html.index("Chicago") < html.index("Boston")


def test_render_bar_chart_empty_does_not_crash():
    html = render_bar_chart("Calibration", "CRPS by station", [])
    assert "<svg" not in html
    assert "No calibrated stations" in html


# ── render_html_dashboard — full integration ─────────────────────────────────


class _TagBalanceChecker(HTMLParser):
    """Fails the test if any start/end tag pair mismatches anywhere in the doc."""

    _VOID = {"br", "img", "input", "hr", "meta", "link", "line", "circle", "rect", "path"}

    def __init__(self):
        super().__init__()
        self.stack: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag not in self._VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self._VOID:
            return
        assert self.stack and self.stack[-1] == tag, f"mismatched </{tag}>, stack={self.stack[-5:]}"
        self.stack.pop()


def test_render_html_dashboard_end_to_end_is_balanced_and_escaped():
    health_checks = [
        HealthCheck("session_activity", OK, "10 signals placed"),
        HealthCheck("stub_books", ALARM, "3 snapshot(s) look stub-shaped"),
        HealthCheck("scorecard_gaps", SKIP, "before 11:00"),
    ]
    lifetime = {"n": 6, "win_rate": 0.6, "pnl": 6.5, "brier_edge": 0.01}
    verification = {"checked": 5, "mismatches": 0}
    equity_df = daily_equity_curve(_trades_frame())
    rolling_df = rolling_daily_metrics(_trades_frame())
    calibrations = [{"city": "Chicago", "crps": 1.0, "n_days": 30}]
    # Deliberately includes '&', '<', and a quote — the exact shapes that broke before.
    incident_md = '### 2026-08-09 — Outage\n**Impact:** P&L reporting froze; `a < b` in one log line.\n'

    html = render_html_dashboard(
        date(2026, 8, 14), health_checks, lifetime, verification,
        equity_df, rolling_df, calibrations, incident_md,
    )

    checker = _TagBalanceChecker()
    checker.feed(html)  # raises via assert on any mismatch
    assert checker.stack == []

    _assert_svgs_well_formed(html)
    assert "NaN" not in html and "undefined" not in html
    assert "P&amp;L" in html  # incident log's raw '&' got escaped, not dropped
    assert "&lt;" in html  # incident log's raw '<' got escaped
