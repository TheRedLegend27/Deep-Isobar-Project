"""Generate a self-contained HTML dashboard from paper trade data.

Reads ``data/paper_trades/paper_trades.csv``, computes summary statistics,
and writes ``data/paper_trades/dashboard.html`` — a fully self-contained
file that uses Chart.js (loaded from cdnjs CDN) for charts.

Usage::

    python -m deep_isobar.research.generate_dashboard
    python -m deep_isobar.research.generate_dashboard --open

The ``--open`` flag calls :func:`webbrowser.open` on the output file after
writing it so it launches immediately in the default browser.

Scheduled daily at 19:15 CDT (after settlement) via Windows Task Scheduler
task ``DeepIsobar_Dashboard``.
"""

from __future__ import annotations

import argparse
import json
import logging
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PAPER_TRADES_CSV = _PROJECT_ROOT / "data" / "paper_trades" / "paper_trades.csv"
_DASHBOARD_HTML = _PROJECT_ROOT / "data" / "paper_trades" / "dashboard.html"


# ---------------------------------------------------------------------------
# Data loading & stats
# ---------------------------------------------------------------------------


def _load_trades() -> pd.DataFrame:
    df = pd.read_csv(
        _PAPER_TRADES_CSV,
        dtype={"threshold_f": float, "position_size": float},
    )
    df["realized_pnl"] = pd.to_numeric(df["realized_pnl"], errors="coerce")
    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
    df["model_prob"] = pd.to_numeric(df["model_prob"], errors="coerce")
    df["market_prob"] = pd.to_numeric(df["market_prob"], errors="coerce")
    df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
    return df


def _compute_stats(df: pd.DataFrame) -> dict:
    settled = df[df["status"].isin(["WIN", "LOSS"])]
    n_settled = len(settled)
    wins = int((settled["status"] == "WIN").sum())
    win_rate = wins / n_settled if n_settled > 0 else 0.0
    net_pnl = float(settled["realized_pnl"].sum()) if not settled.empty else 0.0
    avg_alpha = float(df["alpha"].mean()) if not df.empty else 0.0
    open_count = int((df["status"] == "OPEN").sum())

    return {
        "total_trades": len(df),
        "settled_trades": n_settled,
        "wins": wins,
        "losses": n_settled - wins,
        "win_rate": win_rate,
        "net_pnl": net_pnl,
        "avg_alpha": avg_alpha,
        "open_count": open_count,
    }


def _build_chart_data(df: pd.DataFrame) -> dict:
    """Build JSON-serialisable chart data from the trades DataFrame."""
    settled = (
        df[df["status"].isin(["WIN", "LOSS"])]
        .copy()
        .sort_values("date")
    )

    # Cumulative PnL over time
    cumulative: list[dict] = []
    running = 0.0
    for _, row in settled.iterrows():
        running += float(row["realized_pnl"]) if pd.notna(row["realized_pnl"]) else 0.0
        cumulative.append({"date": str(row["date"]), "pnl": round(running, 4)})

    # Win/Loss by date
    by_date: dict[str, dict] = {}
    for _, row in settled.iterrows():
        d = str(row["date"])
        if d not in by_date:
            by_date[d] = {"wins": 0, "losses": 0}
        if row["status"] == "WIN":
            by_date[d]["wins"] += 1
        else:
            by_date[d]["losses"] += 1

    # Alpha distribution buckets
    alphas = df["alpha"].dropna().tolist()

    # Full trade rows for the table
    table_rows = []
    for _, row in df.sort_values("date", ascending=False).iterrows():
        table_rows.append({
            "date": str(row["date"]),
            "ticker": str(row["contract_ticker"]),
            "direction": str(row["direction"]),
            "alpha": round(float(row["alpha"]), 4) if pd.notna(row["alpha"]) else None,
            "model_prob": round(float(row["model_prob"]), 4) if pd.notna(row["model_prob"]) else None,
            "market_prob": round(float(row["market_prob"]), 4) if pd.notna(row["market_prob"]) else None,
            "entry_price": round(float(row["entry_price"]), 4) if pd.notna(row["entry_price"]) else None,
            "threshold_f": int(row["threshold_f"]) if pd.notna(row["threshold_f"]) else None,
            "position_size": float(row["position_size"]) if pd.notna(row["position_size"]) else None,
            "status": str(row["status"]),
            "realized_pnl": round(float(row["realized_pnl"]), 4) if pd.notna(row["realized_pnl"]) else None,
            "settled_temp": float(row["settled_temp"]) if pd.notna(row.get("settled_temp")) else None,
        })

    return {
        "cumulative_pnl": cumulative,
        "by_date": [
            {"date": d, "wins": v["wins"], "losses": v["losses"]}
            for d, v in sorted(by_date.items())
        ],
        "alphas": alphas,
        "table": table_rows,
    }


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Deep Isobar — Paper Trade Dashboard</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"
          integrity="sha512-ZwR1/gSZM3ai6vCdI+LVF1zSq/5HznD3oD+sCoJrzXJ+yKiBSAa/A9F5EzDOBFDByGKUBKr5y6H7G3TgLgMw=="
          crossorigin="anonymous" referrerpolicy="no-referrer"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:        #0d1117;
      --surface:   #161b22;
      --border:    #30363d;
      --text:      #e6edf3;
      --muted:     #8b949e;
      --accent:    #58a6ff;
      --green:     #3fb950;
      --red:       #f85149;
      --yellow:    #d29922;
      --radius:    8px;
      --gap:       1.25rem;
    }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      font-size: 14px;
      line-height: 1.5;
      padding: var(--gap);
    }

    /* ── Header ─────────────────────────────────────────────────── */
    header {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-bottom: var(--gap);
      padding-bottom: var(--gap);
      border-bottom: 1px solid var(--border);
    }
    header h1 { font-size: 1.35rem; font-weight: 600; color: var(--accent); }
    header .subtitle { font-size: 0.8rem; color: var(--muted); }

    /* ── Summary cards ───────────────────────────────────────────── */
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
      gap: var(--gap);
      margin-bottom: var(--gap);
    }
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1rem 1.1rem;
    }
    .card .label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
    .card .value { font-size: 1.6rem; font-weight: 700; margin-top: 0.2rem; }
    .card .value.green  { color: var(--green); }
    .card .value.red    { color: var(--red); }
    .card .value.yellow { color: var(--yellow); }
    .card .value.accent { color: var(--accent); }

    /* ── Charts grid ─────────────────────────────────────────────── */
    .charts {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
      gap: var(--gap);
      margin-bottom: var(--gap);
    }
    .chart-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1rem;
    }
    .chart-card h2 {
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .05em;
      margin-bottom: 0.75rem;
    }
    .chart-card canvas { max-height: 220px; }

    /* ── Trade table ─────────────────────────────────────────────── */
    .table-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      overflow: hidden;
    }
    .table-card h2 {
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .05em;
      padding: 0.85rem 1rem;
      border-bottom: 1px solid var(--border);
    }
    .table-wrap { overflow-x: auto; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.78rem;
      white-space: nowrap;
    }
    thead th {
      padding: 0.55rem 0.75rem;
      text-align: left;
      font-weight: 600;
      color: var(--muted);
      background: var(--bg);
      border-bottom: 1px solid var(--border);
    }
    tbody tr { border-bottom: 1px solid var(--border); transition: background 0.1s; }
    tbody tr:last-child { border-bottom: none; }
    tbody tr:hover { background: rgba(88,166,255,0.05); }
    tbody td { padding: 0.5rem 0.75rem; color: var(--text); }

    .badge {
      display: inline-block;
      padding: 1px 8px;
      border-radius: 12px;
      font-size: 0.72rem;
      font-weight: 600;
    }
    .badge-WIN  { background: rgba(63,185,80,0.15);  color: var(--green); }
    .badge-LOSS { background: rgba(248,81,73,0.15);  color: var(--red); }
    .badge-OPEN { background: rgba(210,153,34,0.15); color: var(--yellow); }
    .badge-NO_SIGNAL { background: rgba(139,148,158,0.15); color: var(--muted); }
    .badge-BUY  { background: rgba(63,185,80,0.1);  color: var(--green); }
    .badge-SELL { background: rgba(248,81,73,0.1);  color: var(--red); }

    .pnl-pos { color: var(--green); }
    .pnl-neg { color: var(--red); }

    footer {
      margin-top: var(--gap);
      text-align: center;
      font-size: 0.72rem;
      color: var(--muted);
    }
  </style>
</head>
<body>

<header>
  <div>
    <h1>Deep Isobar &mdash; Paper Trade Dashboard</h1>
    <div class="subtitle">Chicago High Temperature &middot; KXHIGHCHI</div>
  </div>
  <div class="subtitle" id="generated-at"></div>
</header>

<!-- Summary cards -->
<div class="cards" id="cards"></div>

<!-- Charts -->
<div class="charts">
  <div class="chart-card">
    <h2>Cumulative P&amp;L</h2>
    <canvas id="chartPnl"></canvas>
  </div>
  <div class="chart-card">
    <h2>Win / Loss by Date</h2>
    <canvas id="chartWinLoss"></canvas>
  </div>
  <div class="chart-card">
    <h2>Alpha Distribution</h2>
    <canvas id="chartAlpha"></canvas>
  </div>
</div>

<!-- Trade table -->
<div class="table-card">
  <h2>Trade Log</h2>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Date</th>
          <th>Ticker</th>
          <th>Dir</th>
          <th>Thr&nbsp;(°F)</th>
          <th>Alpha</th>
          <th>Model&nbsp;P</th>
          <th>Mkt&nbsp;P</th>
          <th>Entry</th>
          <th>Qty</th>
          <th>Actual&nbsp;(°F)</th>
          <th>Status</th>
          <th>P&amp;L</th>
        </tr>
      </thead>
      <tbody id="trade-tbody"></tbody>
    </table>
  </div>
</div>

<footer id="footer"></footer>

<script>
/* ── Injected data ───────────────────────────────────────────── */
const STATS = __STATS_JSON__;
const CHART_DATA = __CHART_DATA_JSON__;
const GENERATED_AT = "__GENERATED_AT__";

/* ── Helpers ─────────────────────────────────────────────────── */
function fmt(v, decimals=4) {
  if (v === null || v === undefined) return "—";
  return Number(v).toFixed(decimals);
}
function pct(v) { return (v * 100).toFixed(1) + "%"; }
function pnlClass(v) {
  if (v === null || v === undefined) return "";
  return v >= 0 ? "pnl-pos" : "pnl-neg";
}

/* ── Summary cards ───────────────────────────────────────────── */
document.getElementById("generated-at").textContent =
  "Generated " + GENERATED_AT;

const cardDefs = [
  { label: "Total Trades",    value: STATS.total_trades,   cls: "accent" },
  { label: "Settled",         value: STATS.settled_trades, cls: "accent" },
  { label: "Open Positions",  value: STATS.open_count,     cls: "yellow" },
  { label: "Wins / Losses",   value: STATS.wins + " / " + STATS.losses, cls: "green" },
  { label: "Win Rate",        value: pct(STATS.win_rate),  cls: STATS.win_rate >= 0.5 ? "green" : "red" },
  { label: "Net P&L",         value: (STATS.net_pnl >= 0 ? "+" : "") + STATS.net_pnl.toFixed(4),
                                                            cls: STATS.net_pnl >= 0 ? "green" : "red" },
  { label: "Avg Alpha",       value: (STATS.avg_alpha >= 0 ? "+" : "") + STATS.avg_alpha.toFixed(4),
                                                            cls: STATS.avg_alpha >= 0 ? "green" : "red" },
];

const cardsEl = document.getElementById("cards");
cardDefs.forEach(cd => {
  cardsEl.insertAdjacentHTML("beforeend", `
    <div class="card">
      <div class="label">${cd.label}</div>
      <div class="value ${cd.cls}">${cd.value}</div>
    </div>`);
});

/* ── Trade table (rendered before Chart.js so a CDN failure
       cannot block table population) ──────────────────────────── */
(function() {
  const tbody = document.getElementById("trade-tbody");
  CHART_DATA.table.forEach(function(r) {
    var pnlFmt = r.realized_pnl !== null
      ? '<span class="' + pnlClass(r.realized_pnl) + '">'
          + (r.realized_pnl >= 0 ? "+" : "") + fmt(r.realized_pnl)
          + "</span>"
      : "—";
    var alphaStr = r.alpha !== null
      ? (r.alpha >= 0 ? "+" : "") + fmt(r.alpha)
      : "—";
    var tempStr = r.settled_temp !== null ? r.settled_temp + "\u00b0F" : "—";
    var thrStr  = r.threshold_f  !== null ? r.threshold_f  : "—";
    var row = "<tr>"
      + "<td>" + r.date + "</td>"
      + '<td style="font-family:monospace;font-size:0.75rem">' + r.ticker + "</td>"
      + '<td><span class="badge badge-' + r.direction + '">' + r.direction + "</span></td>"
      + "<td>" + thrStr + "</td>"
      + "<td>" + alphaStr + "</td>"
      + "<td>" + fmt(r.model_prob) + "</td>"
      + "<td>" + fmt(r.market_prob) + "</td>"
      + "<td>" + fmt(r.entry_price) + "</td>"
      + "<td>" + (r.position_size !== null ? r.position_size : "—") + "</td>"
      + "<td>" + tempStr + "</td>"
      + '<td><span class="badge badge-' + r.status + '">' + r.status + "</span></td>"
      + "<td>" + pnlFmt + "</td>"
      + "</tr>";
    tbody.insertAdjacentHTML("beforeend", row);
  });
})();

/* ── Chart defaults ──────────────────────────────────────────── */
if (typeof Chart !== "undefined") {
  Chart.defaults.color = "#8b949e";
  Chart.defaults.borderColor = "#30363d";
  Chart.defaults.font.family =
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif';
  Chart.defaults.font.size = 11;

/* ── Cumulative P&L line chart ───────────────────────────────── */
  (function() {
    const pts = CHART_DATA.cumulative_pnl;
    if (!pts.length) return;
    const labels = pts.map(p => p.date);
    const values = pts.map(p => p.pnl);
    const lastVal = values[values.length - 1];
    const lineColor = lastVal >= 0 ? "#3fb950" : "#f85149";

    new Chart(document.getElementById("chartPnl"), {
      type: "line",
      data: {
        labels,
        datasets: [{
          data: values,
          borderColor: lineColor,
          backgroundColor: lineColor + "1a",
          fill: true,
          tension: 0.3,
          pointRadius: pts.length > 30 ? 2 : 4,
          pointHoverRadius: 5,
        }]
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { maxRotation: 45, maxTicksLimit: 8 } },
          y: { ticks: { callback: v => (v >= 0 ? "+" : "") + v.toFixed(2) } },
        }
      }
    });
  })();

/* ── Win / Loss bar chart ────────────────────────────────────── */
  (function() {
    const pts = CHART_DATA.by_date;
    if (!pts.length) return;
    new Chart(document.getElementById("chartWinLoss"), {
      type: "bar",
      data: {
        labels: pts.map(p => p.date),
        datasets: [
          { label: "Win",  data: pts.map(p => p.wins),   backgroundColor: "#3fb95099" },
          { label: "Loss", data: pts.map(p => p.losses), backgroundColor: "#f8514999" },
        ]
      },
      options: {
        responsive: true,
        plugins: { legend: { position: "top" } },
        scales: {
          x: { stacked: true, ticks: { maxRotation: 45, maxTicksLimit: 8 } },
          y: { stacked: true, ticks: { stepSize: 1 } },
        }
      }
    });
  })();

/* ── Alpha histogram ─────────────────────────────────────────── */
  (function() {
    const alphas = CHART_DATA.alphas;
    if (!alphas.length) return;

    const mn = Math.min(...alphas);
    const mx = Math.max(...alphas);
    const N  = 12;
    const w  = (mx - mn) / N || 0.1;
    const buckets = Array.from({length: N}, (_, i) => ({
      lo: mn + i * w, hi: mn + (i + 1) * w, count: 0
    }));
    alphas.forEach(a => {
      let i = Math.floor((a - mn) / w);
      if (i >= N) i = N - 1;
      buckets[i].count++;
    });

    new Chart(document.getElementById("chartAlpha"), {
      type: "bar",
      data: {
        labels: buckets.map(b => (b.lo >= 0 ? "+" : "") + b.lo.toFixed(2)),
        datasets: [{
          data: buckets.map(b => b.count),
          backgroundColor: buckets.map(b => b.lo >= 0 ? "#58a6ff99" : "#f8514999"),
        }]
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { maxRotation: 45 } },
          y: { ticks: { stepSize: 1 }, title: { display: true, text: "Count" } },
        }
      }
    });
  })();
} // end typeof Chart !== "undefined"

document.getElementById("footer").textContent =
  "Deep Isobar Paper Trading · Chicago High Temperature · " + GENERATED_AT;
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Dashboard generation
# ---------------------------------------------------------------------------


def generate(output_path: Path = _DASHBOARD_HTML) -> Path:
    """Read paper trades CSV and write a self-contained HTML dashboard.

    Args:
        output_path: Destination for the HTML file.

    Returns:
        The resolved path of the written HTML file.
    """
    if not _PAPER_TRADES_CSV.exists():
        raise FileNotFoundError(
            f"paper_trades.csv not found at {_PAPER_TRADES_CSV}\n"
            "Run paper_trade_session.py first."
        )

    df = _load_trades()
    stats = _compute_stats(df)
    chart_data = _build_chart_data(df)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    html = (
        _HTML_TEMPLATE
        .replace("__STATS_JSON__", json.dumps(stats))
        .replace("__CHART_DATA_JSON__", json.dumps(chart_data))
        .replace("__GENERATED_AT__", generated_at)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Dashboard written to %s", output_path)

    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Generate a self-contained HTML paper trade dashboard.\n\n"
            "Reads data/paper_trades/paper_trades.csv and writes\n"
            "data/paper_trades/dashboard.html."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the dashboard in the default browser after generating it.",
    )
    args = parser.parse_args()

    out = generate()
    print(f"Dashboard written to: {out}")

    if args.open:
        webbrowser.open(out.as_uri())
        print("Opened in browser.")
