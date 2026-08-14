"""Charts for the scorecard — rolling stats, an inline-SVG HTML dashboard,
and two headline PNGs embedded in ``SUMMARY.md``.

Two outputs, one data layer:

- :func:`render_html_dashboard` — writes ``data/reports/SUMMARY.html``, a
  self-contained (no CDN, no build step) page with four interactive charts:
  equity curve, rolling Brier edge (7d/30d), rolling win rate (7d/30d), and
  per-station CRPS. Pure inline SVG + vanilla JS for the crosshair/tooltip
  layer — open it directly in a browser, nothing to install.
- :func:`save_equity_curve_png` / :func:`save_brier_edge_png` — the two
  headline charts as PNGs (matplotlib) for the two spots that can't run JS:
  embedded directly in ``SUMMARY.md`` and (if wired up later) a Discord
  embed attachment.

Palette, mark specs, and interaction rules follow the project's dataviz
skill (categorical slots 1 blue / 2 orange, validated colorblind-safe in
both light and dark; 2px lines, 4px rounded bar ends, hairline gridlines,
crosshair + tooltip on every line chart, per-bar hover on the bar chart).
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Palette — see dataviz skill references/palette.md. Only the slots this
# module uses; light/dark pairs validated with scripts/validate_palette.js.
# ---------------------------------------------------------------------------

_SERIES_1 = {"light": "#2a78d6", "dark": "#3987e5"}  # blue  — primary series
_SERIES_2 = {"light": "#eb6834", "dark": "#d95926"}  # orange — secondary series
_STATUS_GOOD = {"light": "#0ca30c", "dark": "#0ca30c"}
_STATUS_CRITICAL = {"light": "#d03b3b", "dark": "#e66767"}

_MPL_BLUE = _SERIES_1["light"]
_MPL_ORANGE = _SERIES_2["light"]
_MPL_INK = "#0b0b0b"
_MPL_MUTED = "#898781"
_MPL_GRID = "#e1e0d9"
_MPL_SURFACE = "#fcfcfb"


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------


def daily_equity_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative realized P&L, one row per settlement date.

    ``df`` is the output of ``daily_scorecard.load_settled_trades`` —
    already filtered to WIN/LOSS rows with parsed ``date``.
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "daily_pnl", "cumulative_pnl"])
    daily = df.groupby("date")["realized_pnl"].sum().sort_index()
    return pd.DataFrame({
        "date": daily.index,
        "daily_pnl": daily.to_numpy(),
        "cumulative_pnl": daily.cumsum().to_numpy(),
    })


def rolling_daily_metrics(df: pd.DataFrame, windows: tuple[int, ...] = (7, 30)) -> pd.DataFrame:
    """Rolling Brier edge + win rate, one row per calendar day, for each window.

    Calendar-day rolling (not row-count), matching ``daily_scorecard.window_stats``
    exactly — a day with zero settlements (e.g. mid-outage) still counts as a
    day in the window, it just contributes zero trades. Reindexes to a full
    daily calendar first so ``.rolling('Nd')`` sees the true gap length.
    Columns: ``date``, ``win_rate_{w}d``, ``brier_edge_{w}d``, ``n_{w}d`` per window
    (NaN wherever the window has zero trades — a chart should skip, not zero, those).
    """
    cols = ["date"] + [f"{m}_{w}d" for w in windows for m in ("win_rate", "brier_edge", "n")]
    if df.empty:
        return pd.DataFrame(columns=cols)

    daily = df.copy()
    outcome = daily["outcome"].to_numpy(dtype=float)
    daily["model_sq_err"] = (daily["model_prob"].to_numpy(float) - outcome) ** 2
    daily["market_sq_err"] = (daily["market_prob"].to_numpy(float) - outcome) ** 2
    daily["win"] = (daily["status"] == "WIN").astype(float)

    agg = daily.groupby("date").agg(
        n=("outcome", "size"),
        wins=("win", "sum"),
        model_sq_err_sum=("model_sq_err", "sum"),
        market_sq_err_sum=("market_sq_err", "sum"),
    ).sort_index()

    full_idx = pd.date_range(agg.index.min(), agg.index.max(), freq="D")
    agg = agg.reindex(full_idx.date, fill_value=0)
    agg.index = pd.DatetimeIndex(full_idx)

    out = pd.DataFrame(index=agg.index)
    for w in windows:
        roll_n = agg["n"].rolling(f"{w}D").sum()
        roll_wins = agg["wins"].rolling(f"{w}D").sum()
        roll_model = agg["model_sq_err_sum"].rolling(f"{w}D").sum()
        roll_market = agg["market_sq_err_sum"].rolling(f"{w}D").sum()
        has_data = roll_n > 0
        out[f"win_rate_{w}d"] = np.where(has_data, roll_wins / roll_n, np.nan)
        model_brier = np.where(has_data, roll_model / roll_n, np.nan)
        market_brier = np.where(has_data, roll_market / roll_n, np.nan)
        out[f"brier_edge_{w}d"] = market_brier - model_brier
        out[f"n_{w}d"] = roll_n

    out = out.reset_index(names="date")
    out["date"] = out["date"].dt.date
    return out


# ---------------------------------------------------------------------------
# Small shared numeric helpers
# ---------------------------------------------------------------------------


def _nice_ticks(vmin: float, vmax: float, count: int = 4) -> list[float]:
    """Evenly-spaced, human-round tick values spanning [vmin, vmax]."""
    if vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0
    span = vmax - vmin
    raw_step = span / max(count, 1)
    mag = 10 ** np.floor(np.log10(raw_step)) if raw_step > 0 else 1.0
    residual = raw_step / mag
    step = (10 if residual > 5 else 5 if residual > 2 else 2 if residual > 1 else 1) * mag
    start = np.floor(vmin / step) * step
    ticks: list[float] = []
    v = start
    while v <= vmax + step * 0.5:
        ticks.append(round(float(v), 10))
        v += step
    return ticks


def _fmt_date_short(d: date) -> str:
    return d.strftime("%b %d")


def _esc(s: str) -> str:
    """Minimal HTML-attribute escaping for our own generated strings."""
    return s.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


def _attr_json(payload: dict) -> str:
    """JSON-encode *payload* for embedding inside a single-quoted HTML attribute.

    ``json.dumps`` alone is not enough — a series name like "Cumulative P&L"
    produces a raw ``&`` that makes the surrounding markup not well-formed
    (and a stray ``'`` would close the attribute early). Escape after
    encoding so the JSON's own double quotes pass through untouched.
    """
    return (
        json.dumps(payload)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace("'", "&#39;")
    )


# ---------------------------------------------------------------------------
# SVG line chart (equity curve / rolling Brier edge / rolling win rate)
# ---------------------------------------------------------------------------

_LINE_W, _LINE_H = 780, 300
_LINE_MARGIN = {"top": 16, "right": 20, "bottom": 30, "left": 56}


def render_line_chart(
    chart_id: str,
    title: str,
    subtitle: str,
    dates: list[date],
    series: list[dict],
    value_fmt: str = "{:+.2f}",
    y_is_percent: bool = False,
    y_zero_line: bool = False,
    area_fill_first: bool = False,
) -> str:
    """One line-chart card as an HTML fragment (title + legend + SVG + hit layer).

    ``series``: list of ``{"name": str, "color_var": "--series-1",
    "hex": {"light":..,"dark":..}, "values": list[float|None]}`` — each
    ``values`` list is the same length as ``dates``; ``None`` marks a gap.
    """
    m = _LINE_MARGIN
    plot_w = _LINE_W - m["left"] - m["right"]
    plot_h = _LINE_H - m["top"] - m["bottom"]

    n = len(dates)
    if n == 0 or all(all(v is None for v in s["values"]) for s in series):
        return (
            f'<section class="chart-card"><div class="chart-head">'
            f"<h3>{_esc(title)}</h3><p class='chart-sub'>{_esc(subtitle)}</p></div>"
            f'<p class="chart-empty">Not enough settled trades yet.</p></section>'
        )

    all_vals = [v for s in series for v in s["values"] if v is not None]
    vmin, vmax = min(all_vals), max(all_vals)
    if y_zero_line:
        vmin, vmax = min(vmin, 0.0), max(vmax, 0.0)
    pad = (vmax - vmin) * 0.12 or 1.0
    vmin, vmax = vmin - pad, vmax + pad
    y_ticks = _nice_ticks(vmin, vmax, 4)
    vmin, vmax = min(vmin, y_ticks[0]), max(vmax, y_ticks[-1])

    def xpx(i: int) -> float:
        return m["left"] + (i / max(n - 1, 1)) * plot_w

    def ypx(v: float) -> float:
        return m["top"] + (1 - (v - vmin) / (vmax - vmin)) * plot_h

    def fmt_y(v: float) -> str:
        return f"{v:.0%}" if y_is_percent else value_fmt.format(v)

    svg: list[str] = [
        f'<svg viewBox="0 0 {_LINE_W} {_LINE_H}" class="chart-svg" '
        f'role="img" aria-label="{_esc(title)}">'
    ]

    # Gridlines + y ticks
    for t in y_ticks:
        y = ypx(t)
        svg.append(
            f'<line x1="{m["left"]}" y1="{y:.1f}" x2="{_LINE_W - m["right"]}" y2="{y:.1f}" class="grid"/>'
        )
        svg.append(
            f'<text x="{m["left"] - 8}" y="{y + 3:.1f}" class="tick tick-y">{_esc(fmt_y(t))}</text>'
        )

    # Zero baseline (drawn heavier than gridlines when in range)
    if y_zero_line and vmin < 0 < vmax:
        y0 = ypx(0)
        svg.append(f'<line x1="{m["left"]}" y1="{y0:.1f}" x2="{_LINE_W - m["right"]}" y2="{y0:.1f}" class="baseline"/>')

    # X ticks — up to 6, evenly spaced by index
    n_ticks = min(6, n)
    tick_idxs = sorted({round(i * (n - 1) / max(n_ticks - 1, 1)) for i in range(n_ticks)})
    for i in tick_idxs:
        svg.append(
            f'<text x="{xpx(i):.1f}" y="{_LINE_H - m["bottom"] + 18}" class="tick tick-x">'
            f"{_esc(_fmt_date_short(dates[i]))}</text>"
        )

    # Series paths + area fill + end marker/label
    end_labels: list[tuple[float, str, str]] = []  # (y_px, text, color_var)
    for si, s in enumerate(series):
        vals = s["values"]
        segments: list[list[tuple[float, float]]] = []
        cur: list[tuple[float, float]] = []
        for i, v in enumerate(vals):
            if v is None:
                if cur:
                    segments.append(cur)
                    cur = []
                continue
            cur.append((xpx(i), ypx(v)))
        if cur:
            segments.append(cur)

        color = f"var({s['color_var']})"
        if area_fill_first and si == 0:
            base_y = ypx(max(min(vmin, 0.0), vmin))
            for seg in segments:
                if len(seg) < 2:
                    continue
                d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in seg)
                d += f" L {seg[-1][0]:.1f},{base_y:.1f} L {seg[0][0]:.1f},{base_y:.1f} Z"
                svg.append(f'<path d="{d}" fill="{color}" opacity="0.10" stroke="none"/>')

        for seg in segments:
            if len(seg) < 2:
                continue
            d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in seg)
            svg.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" '
                       f'stroke-linejoin="round" stroke-linecap="round"/>')

        last = next(((i, v) for i, v in reversed(list(enumerate(vals))) if v is not None), None)
        if last is not None:
            li, lv = last
            ex, ey = xpx(li), ypx(lv)
            svg.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="4" fill="{color}" '
                       f'stroke="var(--surface-1)" stroke-width="2"/>')
            end_labels.append((ey, fmt_y(lv), s["color_var"]))

    # Separate colliding end-labels (min 14px apart), draw after paths so they sit on top
    end_labels.sort(key=lambda t: t[0])
    for i in range(1, len(end_labels)):
        y, txt, cv = end_labels[i]
        prev_y = end_labels[i - 1][0]
        if y - prev_y < 14:
            end_labels[i] = (prev_y + 14, txt, cv)
    for y, txt, _cv in end_labels:
        svg.append(f'<text x="{_LINE_W - m["right"] + 6}" y="{y + 3:.1f}" class="end-label">{_esc(txt)}</text>')

    # Invisible per-index hit strips, carrying a precomputed multi-series tooltip payload
    strip_w = max(plot_w / max(n - 1, 1), 6)
    for i, d in enumerate(dates):
        rows = []
        for s in series:
            v = s["values"][i]
            if v is None:
                continue
            rows.append([s["name"], fmt_y(v), f"var({s['color_var']})"])
        if not rows:
            continue
        payload = {"d": _fmt_date_short(d), "rows": rows}
        svg.append(
            f'<rect class="hit" x="{xpx(i) - strip_w / 2:.1f}" y="{m["top"]}" '
            f'width="{strip_w:.1f}" height="{plot_h}" data-x="{xpx(i):.1f}" '
            f"data-tip='{_attr_json(payload)}' tabindex=\"0\"/>"
        )

    svg.append(f'<line class="crosshair" x1="0" y1="{m["top"]}" x2="0" y2="{_LINE_H - m["bottom"]}" hidden="hidden"/>')
    svg.append("</svg>")

    legend = ""
    if len(series) > 1:
        items = "".join(
            f'<span class="legend-item"><span class="legend-key" style="background:var({s["color_var"]})"></span>{_esc(s["name"])}</span>'
            for s in series
        )
        legend = f'<div class="chart-legend">{items}</div>'

    return (
        f'<section class="chart-card">'
        f'<div class="chart-head"><h3>{_esc(title)}</h3><p class="chart-sub">{_esc(subtitle)}</p></div>'
        f"{legend}"
        f'<div class="chart-wrap">{"".join(svg)}<div class="tooltip" hidden="hidden"></div></div>'
        f"</section>"
    )


# ---------------------------------------------------------------------------
# SVG horizontal bar chart (per-station CRPS)
# ---------------------------------------------------------------------------

_BAR_W = 780
_BAR_ROW_H = 22
_BAR_GAP = 4
_BAR_MARGIN = {"top": 8, "right": 56, "bottom": 8, "left": 190}


def render_bar_chart(title: str, subtitle: str, calibrations: list[dict]) -> str:
    """Per-station CRPS, ascending (best/lowest first)."""
    if not calibrations:
        return (
            f'<section class="chart-card"><div class="chart-head">'
            f"<h3>{_esc(title)}</h3><p class='chart-sub'>{_esc(subtitle)}</p></div>"
            f'<p class="chart-empty">No calibrated stations yet.</p></section>'
        )

    rows = sorted(calibrations, key=lambda c: c["crps"])
    m = _BAR_MARGIN
    plot_w = _BAR_W - m["left"] - m["right"]
    h = m["top"] + m["bottom"] + len(rows) * (_BAR_ROW_H + _BAR_GAP)
    vmax = max(c["crps"] for c in rows) * 1.08 or 1.0

    def xpx(v: float) -> float:
        return m["left"] + (v / vmax) * plot_w

    svg = [f'<svg viewBox="0 0 {_BAR_W} {h}" class="chart-svg" role="img" aria-label="{_esc(title)}">']
    for t in _nice_ticks(0, vmax, 4):
        x = xpx(t)
        svg.append(f'<line x1="{x:.1f}" y1="{m["top"]}" x2="{x:.1f}" y2="{h - m["bottom"]}" class="grid"/>')
        svg.append(f'<text x="{x:.1f}" y="{h - m["bottom"] + 16}" class="tick tick-x" text-anchor="middle">{t:.1f}</text>')

    for i, c in enumerate(rows):
        y = m["top"] + i * (_BAR_ROW_H + _BAR_GAP)
        bar_w = max(xpx(c["crps"]) - m["left"], 3)
        cy = y + _BAR_ROW_H / 2
        svg.append(f'<text x="{m["left"] - 10}" y="{cy + 4:.1f}" text-anchor="end" class="tick tick-y">{_esc(c["city"])}</text>')
        tip = {"d": c["city"], "rows": [[f"CRPS ({c['n_days']}d)", f"{c['crps']:.3f}°F", "var(--series-1)"]]}
        svg.append(
            f'<rect class="bar" x="{m["left"]}" y="{y}" width="{bar_w:.1f}" height="{_BAR_ROW_H}" rx="4" '
            f'fill="var(--series-1)" data-tip=\'{_attr_json(tip)}\'/>'
        )
        svg.append(f'<text x="{m["left"] + bar_w + 6:.1f}" y="{cy + 4:.1f}" class="end-label">{c["crps"]:.2f}</text>')

    svg.append("</svg>")
    return (
        f'<section class="chart-card">'
        f'<div class="chart-head"><h3>{_esc(title)}</h3><p class="chart-sub">{_esc(subtitle)}</p></div>'
        f'<div class="chart-wrap chart-wrap-bar">{"".join(svg)}<div class="tooltip" hidden="hidden"></div></div>'
        f"</section>"
    )


# ---------------------------------------------------------------------------
# HTML dashboard assembly
# ---------------------------------------------------------------------------

_CSS = """
.viz-root {
  color-scheme: light;
  --surface-1:      #fcfcfb;
  --page:           #f9f9f7;
  --text-primary:   #0b0b0b;
  --text-secondary: #52514e;
  --text-muted:     #898781;
  --grid:           #e1e0d9;
  --baseline:       #c3c2b7;
  --border:         rgba(11,11,11,0.10);
  --series-1:       #2a78d6;
  --series-2:       #eb6834;
  --good:           #0ca30c;
  --critical:       #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-1: #1a1a19; --page: #0d0d0d; --text-primary: #ffffff;
    --text-secondary: #c3c2b7; --text-muted: #898781; --grid: #2c2c2a;
    --baseline: #383835; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-2: #d95926; --good: #0ca30c; --critical: #e66767;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-1: #1a1a19; --page: #0d0d0d; --text-primary: #ffffff;
  --text-secondary: #c3c2b7; --text-muted: #898781; --grid: #2c2c2a;
  --baseline: #383835; --border: rgba(255,255,255,0.10);
  --series-1: #3987e5; --series-2: #d95926; --good: #0ca30c; --critical: #e66767;
}
* { box-sizing: border-box; }
.viz-root {
  background: var(--page);
  color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  padding: 24px 20px 60px;
  max-width: 860px;
  margin: 0 auto;
}
.viz-root h1 { font-size: 20px; margin: 0 0 2px; }
.viz-root .updated { color: var(--text-muted); font-size: 13px; margin: 0 0 24px; }
.health-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 8px; margin-bottom: 28px;
}
.health-row {
  display: flex; align-items: center; gap: 8px;
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 8px; padding: 8px 10px; font-size: 12.5px;
}
.health-dot { width: 8px; height: 8px; border-radius: 50%; flex: none; }
.health-dot.ok { background: var(--good); }
.health-dot.alarm { background: var(--critical); }
.health-dot.skip { background: var(--text-muted); }
.health-name { font-weight: 600; color: var(--text-primary); }
.health-detail { color: var(--text-secondary); }
.lifetime-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px; margin-bottom: 32px;
}
.stat-tile {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 10px; padding: 12px 14px;
}
.stat-tile .label { font-size: 12px; color: var(--text-muted); margin: 0 0 4px; }
.stat-tile .value { font-size: 20px; font-weight: 600; margin: 0; }
.chart-card {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 12px; padding: 16px 18px 8px; margin-bottom: 20px;
}
.chart-head h3 { font-size: 14px; margin: 0; }
.chart-sub { font-size: 12px; color: var(--text-muted); margin: 2px 0 10px; }
.chart-empty { color: var(--text-muted); font-size: 13px; padding: 20px 0; }
.chart-legend { display: flex; gap: 16px; font-size: 12px; color: var(--text-secondary); margin-bottom: 6px; }
.legend-item { display: inline-flex; align-items: center; gap: 6px; }
.legend-key { width: 12px; height: 2px; border-radius: 1px; display: inline-block; }
.chart-wrap { position: relative; }
.chart-svg { width: 100%; height: auto; display: block; }
.chart-svg .grid { stroke: var(--grid); stroke-width: 1; shape-rendering: crispEdges; }
.chart-svg .baseline { stroke: var(--baseline); stroke-width: 1; }
.chart-svg .tick { fill: var(--text-muted); font-size: 10.5px; }
.chart-svg .tick-x { text-anchor: middle; }
.chart-svg .end-label { fill: var(--text-primary); font-size: 11.5px; font-weight: 600; dominant-baseline: middle; }
.chart-svg .hit { fill: transparent; cursor: crosshair; }
.chart-svg .bar { cursor: pointer; }
.chart-svg .bar:hover { opacity: 0.85; }
.chart-svg .crosshair { stroke: var(--text-muted); stroke-width: 1; pointer-events: none; }
.tooltip {
  position: absolute; pointer-events: none; z-index: 5;
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 8px; padding: 8px 10px; font-size: 12px;
  box-shadow: 0 4px 14px rgba(0,0,0,0.18); min-width: 120px;
}
.tooltip .tt-date { font-weight: 600; color: var(--text-primary); margin-bottom: 4px; }
.tooltip .tt-row { display: flex; align-items: center; gap: 6px; justify-content: space-between; color: var(--text-secondary); }
.tooltip .tt-key { width: 10px; height: 2px; border-radius: 1px; display: inline-block; flex: none; }
.tooltip .tt-val { color: var(--text-primary); font-weight: 600; margin-left: auto; }
.incident-log { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 4px 20px; font-size: 13.5px; line-height: 1.55; }
.incident-log h3 { font-size: 13px; }
.incident-log code { background: var(--grid); padding: 1px 4px; border-radius: 4px; font-size: 12px; }
"""

_JS = """
(function () {
  function showTooltip(evt, wrap, payload) {
    var tip = wrap.querySelector('.tooltip');
    tip.replaceChildren();
    var dateEl = document.createElement('div');
    dateEl.className = 'tt-date';
    dateEl.textContent = payload.d;
    tip.appendChild(dateEl);
    (payload.rows || []).forEach(function (row) {
      var rowEl = document.createElement('div');
      rowEl.className = 'tt-row';
      var key = document.createElement('span');
      key.className = 'tt-key';
      key.style.background = row[2];
      var name = document.createElement('span');
      name.textContent = row[0];
      var val = document.createElement('span');
      val.className = 'tt-val';
      val.textContent = row[1];
      rowEl.appendChild(key);
      rowEl.appendChild(name);
      rowEl.appendChild(val);
      tip.appendChild(rowEl);
    });
    var wrapRect = wrap.getBoundingClientRect();
    var x = evt.clientX - wrapRect.left + 14;
    var y = evt.clientY - wrapRect.top + 14;
    if (x + 180 > wrapRect.width) x = evt.clientX - wrapRect.left - 194;
    tip.style.left = x + 'px';
    tip.style.top = y + 'px';
    tip.hidden = false;
  }

  function hideTooltip(wrap) {
    var tip = wrap.querySelector('.tooltip');
    tip.hidden = true;
    var cross = wrap.querySelector('.crosshair');
    if (cross) cross.hidden = true;
  }

  document.querySelectorAll('.chart-wrap').forEach(function (wrap) {
    var svg = wrap.querySelector('svg');
    var cross = wrap.querySelector('.crosshair');
    svg.querySelectorAll('.hit, .bar').forEach(function (el) {
      var handler = function (evt) {
        var payload;
        try { payload = JSON.parse(el.dataset.tip); } catch (e) { return; }
        if (cross && el.dataset.x) {
          cross.setAttribute('x1', el.dataset.x);
          cross.setAttribute('x2', el.dataset.x);
          cross.hidden = false;
        }
        showTooltip(evt, wrap, payload);
      };
      el.addEventListener('pointermove', handler);
      el.addEventListener('pointerenter', handler);
      el.addEventListener('focus', function (evt) { handler(evt); });
      el.addEventListener('pointerleave', function () { hideTooltip(wrap); });
      el.addEventListener('blur', function () { hideTooltip(wrap); });
    });
  });
})();
"""


def render_html_dashboard(
    asof: date,
    health_checks: list,  # list[deep_isobar.ops.health.HealthCheck]
    lifetime: dict,
    verification: dict | None,
    equity_df: pd.DataFrame,
    rolling_df: pd.DataFrame,
    calibrations: list[dict],
    incident_log_md: str,
) -> str:
    """The full SUMMARY.html page — self-contained, no external requests."""
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M")

    health_rows = []
    for c in health_checks:
        cls = "ok" if c.status == "OK" else "alarm" if c.status == "ALARM" else "skip"
        health_rows.append(
            f'<div class="health-row"><span class="health-dot {cls}"></span>'
            f'<span class="health-name">{_esc(c.name)}</span>'
            f'<span class="health-detail">{_esc(c.detail)}</span></div>'
        )

    lifetime_tiles = []
    if lifetime.get("n", 0):
        verify_txt = (
            f"{verification['checked']} checked / {verification['mismatches']} mismatch"
            if verification is not None else "never run"
        )
        lifetime_tiles = [
            ("Settled trades", f"{lifetime['n']}"),
            ("Win rate", f"{lifetime['win_rate']:.1%}"),
            ("Realized P&L", f"{lifetime['pnl']:+.2f}"),
            ("Brier edge", f"{lifetime['brier_edge']:+.4f}"),
            ("Exchange verify", verify_txt),
        ]
    tiles_html = "".join(
        f'<div class="stat-tile"><p class="label">{_esc(label)}</p><p class="value">{_esc(val)}</p></div>'
        for label, val in lifetime_tiles
    )

    equity_chart = render_line_chart(
        "equity", "Equity curve", "Cumulative realized P&L, every settled trade",
        list(equity_df["date"]),
        [{"name": "Cumulative P&L", "color_var": "--series-1", "values": list(equity_df["cumulative_pnl"])}],
        value_fmt="{:+.2f}", y_zero_line=True, area_fill_first=True,
    ) if not equity_df.empty else render_line_chart("equity", "Equity curve", "Cumulative realized P&L", [], [])

    if not rolling_df.empty:
        dates = list(rolling_df["date"])
        edge_chart = render_line_chart(
            "edge", "Brier edge vs market", "Rolling — positive means our probabilities beat the market's",
            dates,
            [
                {"name": "7d edge", "color_var": "--series-1", "values": [None if pd.isna(v) else float(v) for v in rolling_df["brier_edge_7d"]]},
                {"name": "30d edge", "color_var": "--series-2", "values": [None if pd.isna(v) else float(v) for v in rolling_df["brier_edge_30d"]]},
            ],
            value_fmt="{:+.4f}", y_zero_line=True,
        )
        winrate_chart = render_line_chart(
            "winrate", "Win rate", "Rolling share of settled trades that won",
            dates,
            [
                {"name": "7d win rate", "color_var": "--series-1", "values": [None if pd.isna(v) else float(v) for v in rolling_df["win_rate_7d"]]},
                {"name": "30d win rate", "color_var": "--series-2", "values": [None if pd.isna(v) else float(v) for v in rolling_df["win_rate_30d"]]},
            ],
            y_is_percent=True,
        )
    else:
        edge_chart = render_line_chart("edge", "Brier edge vs market", "Rolling model-vs-market Brier edge", [], [])
        winrate_chart = render_line_chart("winrate", "Win rate", "Rolling win rate", [], [])

    crps_chart = render_bar_chart(
        "Calibration — CRPS by station", "Lower is better; current params scored on stored forecasts", calibrations,
    )

    incident_html = _render_incident_log_html(incident_log_md)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Deep Isobar — summary scorecard</title>
<style>{_CSS}</style>
</head>
<body>
<div class="viz-root">
  <h1>Deep Isobar — summary scorecard</h1>
  <p class="updated">Last updated {_esc(now_iso)} (scorecard run for {asof})</p>

  <div class="health-grid">{"".join(health_rows)}</div>

  <div class="lifetime-grid">{tiles_html}</div>

  {equity_chart}
  {edge_chart}
  {winrate_chart}
  {crps_chart}

  <div class="incident-log">{incident_html}</div>
</div>
<script>{_JS}</script>
</body>
</html>
"""


def _render_incident_log_html(md_text: str) -> str:
    """Minimal markdown → HTML for the incident log (### headers, **bold**, `code`, paragraphs).

    Deliberately not a full markdown parser — the log's own convention
    (see ``daily_scorecard.load_incident_log``) only ever uses these three
    constructs, so a tiny converter avoids a new dependency.
    """
    import re

    def inline(text: str) -> str:
        text = _esc(text)
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
        return text

    html_lines: list[str] = []
    para: list[str] = []

    def flush_para() -> None:
        if para:
            html_lines.append("<p>" + " ".join(para) + "</p>")
            para.clear()

    for raw in md_text.splitlines():
        line = raw.strip()
        if not line:
            flush_para()
            continue
        if line.startswith("### "):
            flush_para()
            html_lines.append(f"<h3>{inline(line[4:])}</h3>")
            continue
        para.append(inline(line))
    flush_para()
    return "\n".join(html_lines) if html_lines else "<p>No incidents logged.</p>"


# ---------------------------------------------------------------------------
# Headline PNGs (matplotlib) — for SUMMARY.md, which can't run JS
# ---------------------------------------------------------------------------


def _mpl_style(ax) -> None:
    """Shared minimal chrome: no top/right spine, hairline grid, muted ticks."""
    ax.set_facecolor(_MPL_SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_MPL_MUTED)
    ax.spines["bottom"].set_color(_MPL_MUTED)
    ax.tick_params(colors=_MPL_MUTED, labelsize=9)
    ax.grid(axis="y", color=_MPL_GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def save_equity_curve_png(equity_df: pd.DataFrame, out_path: Path) -> None:
    """Cumulative realized P&L line, filled area wash, end-value label."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=150, facecolor=_MPL_SURFACE)
    _mpl_style(ax)

    if equity_df.empty:
        ax.text(0.5, 0.5, "No settled trades yet", ha="center", va="center",
                 color=_MPL_MUTED, transform=ax.transAxes)
    else:
        dates = pd.to_datetime(equity_df["date"])
        vals = equity_df["cumulative_pnl"].to_numpy()
        ax.axhline(0, color="#c3c2b7", linewidth=1, zorder=1)
        ax.plot(dates, vals, color=_MPL_BLUE, linewidth=2, zorder=3, solid_capstyle="round")
        ax.fill_between(dates, vals, 0, color=_MPL_BLUE, alpha=0.10, zorder=2)
        ax.scatter([dates.iloc[-1]], [vals[-1]], color=_MPL_BLUE, s=28, zorder=4,
                   edgecolors=_MPL_SURFACE, linewidths=1.5)
        ax.annotate(f"{vals[-1]:+.2f}", (dates.iloc[-1], vals[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=10, fontweight="bold", color=_MPL_INK)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.margins(x=0.03)

    ax.set_title("Equity curve — cumulative realized P&L", fontsize=11, color=_MPL_INK,
                 loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=_MPL_SURFACE)
    plt.close(fig)


def save_brier_edge_png(rolling_df: pd.DataFrame, out_path: Path) -> None:
    """7d/30d rolling Brier edge, two lines, zero baseline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=150, facecolor=_MPL_SURFACE)
    _mpl_style(ax)

    if rolling_df.empty:
        ax.text(0.5, 0.5, "No settled trades yet", ha="center", va="center",
                 color=_MPL_MUTED, transform=ax.transAxes)
    else:
        dates = pd.to_datetime(rolling_df["date"])
        ax.axhline(0, color="#c3c2b7", linewidth=1, zorder=1)
        for col, label, color in (
            ("brier_edge_7d", "7d edge", _MPL_BLUE),
            ("brier_edge_30d", "30d edge", _MPL_ORANGE),
        ):
            vals = rolling_df[col].to_numpy(dtype=float)
            ax.plot(dates, vals, color=color, linewidth=2, label=label, zorder=3,
                    solid_capstyle="round")
        ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=_MPL_INK)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.margins(x=0.03)

    ax.set_title("Brier edge vs market — rolling", fontsize=11, color=_MPL_INK,
                 loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=_MPL_SURFACE)
    plt.close(fig)
