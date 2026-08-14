"""Daily calibration & trading scorecard — the day-to-day watch.

Runs after settlement (supervisor job, ~18:45 local) and answers three
questions in one page:

1. **What settled today?**  Per-trade table for the latest settlement day.
2. **Which way are we moving?**  Rolling 7-day vs 30-day realized P&L, win
   rate, and model-vs-market Brier score (the sign of real edge: positive
   edge means our probabilities beat the market's on settled trades).
3. **Is the distribution still calibrated?**  Per-station CRPS / MAE
   (EMOS vs the NBM benchmark), PIT histogram, modal-bucket probability —
   recomputed from the training parquets with the currently saved params —
   plus ops health: params age, ensemble-spread coverage toward the 60%
   variance-source gate, and recent EMOS-vs-NBM mean divergence.

Output goes to stdout, ``data/reports/scorecard_YYYY-MM-DD.md``, and a
Discord embed (silent no-op without a webhook). Also (re)writes
``data/reports/SUMMARY.md`` — a single evergreen file, overwritten (not
dated) on every run, holding the same-day ops health + trend + calibration
picture plus lifetime cumulative stats, settlement-verification status, and
the running incident log (``data/reports/incident_log.md``, hand-maintained
— append a dated ``### YYYY-MM-DD — Title`` section there for anything
future-you needs to remember). SUMMARY.md is the one file to open for "what
is the current state of everything"; the dated files are the historical
record.

Caveats baked into the numbers:

- Trading stats cover **traded** contracts only (|alpha| >= threshold), so
  Brier comparisons are conditioned on us having disagreed with the market.
- Same-day trades across cities are correlated; a 7-day window is ~7
  independent samples, not 7 x cities.  Read trends, not significance.

CLI::

    python -m deep_isobar.research.daily_scorecard
    python -m deep_isobar.research.daily_scorecard --days 60 --no-discord
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from deep_isobar.calibration.emos import crps_gaussian, load_params
from deep_isobar.calibration.emos_training import (
    MIN_SPREAD_COVERAGE,
    _training_path,
)
from deep_isobar.core.types import CityProfile
from deep_isobar.data.city_universe import get_city_universe
from deep_isobar.data.ensemble_ingest import load_t1_spread_history
from deep_isobar.notifications.discord_notifier import (
    COLOR_GREEN,
    COLOR_RED,
    post_embed,
)
from deep_isobar.ops.health import (
    post_alarm_embed,
    render_health_section,
    run_health_checks,
)
from deep_isobar.research.forecast_evaluation import (
    modal_bucket_prob,
    pit_histogram,
    pit_values,
)
from deep_isobar.research.scorecard_charts import (
    daily_equity_curve,
    render_html_dashboard,
    rolling_daily_metrics,
    save_brier_edge_png,
    save_equity_curve_png,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PAPER_TRADES_CSV = _PROJECT_ROOT / "data" / "paper_trades" / "paper_trades.csv"
_SETTLEMENT_VERIFIED_CSV = _PROJECT_ROOT / "data" / "paper_trades" / "settlement_verified.csv"
_REPORTS_DIR = _PROJECT_ROOT / "data" / "reports"
_SUMMARY_PATH = _REPORTS_DIR / "SUMMARY.md"
_SUMMARY_HTML_PATH = _REPORTS_DIR / "SUMMARY.html"
_INCIDENT_LOG_PATH = _REPORTS_DIR / "incident_log.md"
_CHARTS_DIR = _REPORTS_DIR / "charts"
_EQUITY_PNG_PATH = _CHARTS_DIR / "equity_curve.png"
_BRIER_PNG_PATH = _CHARTS_DIR / "brier_edge.png"

_SETTLED = ("WIN", "LOSS")
_PIT_BLOCKS = " ▁▂▃▄▅▆▇█"


# ---------------------------------------------------------------------------
# Trading review
# ---------------------------------------------------------------------------


def realized_outcome(direction: str, status: str) -> int | None:
    """Map (direction, settle status) to the contract's YES outcome (0/1).

    BUY wins when YES settled 1; SELL wins when YES settled 0.  Returns
    ``None`` for unsettled rows.
    """
    if status not in _SETTLED:
        return None
    won = status == "WIN"
    if direction == "BUY":
        return 1 if won else 0
    if direction == "SELL":
        return 0 if won else 1
    return None


def load_settled_trades(csv_path: Path | None = None) -> pd.DataFrame:
    """Load settled (WIN/LOSS) rows with parsed dates and YES outcomes.

    Returns an empty frame (with the needed columns) when the CSV does not
    exist yet or nothing has settled.
    """
    path = csv_path or _PAPER_TRADES_CSV
    cols = [
        "date", "city", "contract_ticker", "direction", "alpha", "model_prob",
        "market_prob", "entry_price", "position_size", "status", "realized_pnl",
        "settled_temp", "outcome",
    ]
    if not path.exists():
        return pd.DataFrame(columns=cols)

    df = pd.read_csv(path)
    df = df[df["status"].isin(_SETTLED)].copy()
    if df.empty:
        return pd.DataFrame(columns=cols)

    df["date"] = pd.to_datetime(df["date"]).dt.date
    for col in ("alpha", "model_prob", "market_prob", "entry_price", "position_size", "realized_pnl"):
        if col not in df.columns:
            df[col] = float("nan")  # legacy CSVs written before this column existed
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["outcome"] = [
        realized_outcome(d, s) for d, s in zip(df["direction"], df["status"])
    ]
    return df


def _stats_from_frame(w: pd.DataFrame) -> dict:
    """Shared stat computation for any already-filtered slice of settled trades."""
    n = len(w)
    if n == 0:
        return {"n": 0}

    outcome = w["outcome"].to_numpy(dtype=float)
    model_brier = float(np.mean((w["model_prob"].to_numpy(float) - outcome) ** 2))
    market_brier = float(np.mean((w["market_prob"].to_numpy(float) - outcome) ** 2))
    return {
        "n": n,
        "days": int(w["date"].nunique()),
        "pnl": float(w["realized_pnl"].sum()),
        "win_rate": float((w["status"] == "WIN").mean()),
        "mean_abs_alpha": float(w["alpha"].abs().mean()),
        "model_brier": model_brier,
        "market_brier": market_brier,
        # Positive = our probabilities were closer to reality than the market's.
        "brier_edge": market_brier - model_brier,
    }


def window_stats(df: pd.DataFrame, days: int, asof: date) -> dict:
    """Trading stats over the last *days* settlement days ending at *asof*."""
    cutoff = asof - timedelta(days=days - 1)
    return _stats_from_frame(df[(df["date"] >= cutoff) & (df["date"] <= asof)])


def all_time_stats(df: pd.DataFrame) -> dict:
    """Trading stats over every settled trade ever recorded."""
    stats = _stats_from_frame(df)
    if stats["n"]:
        stats["first_date"] = df["date"].min()
        stats["last_date"] = df["date"].max()
    return stats


def verification_summary(csv_path: Path | None = None) -> dict | None:
    """Lifetime exchange-verification tally, or ``None`` if never run.

    Reads ``verify_settlements.py``'s cache — every WIN/LOSS grade it has
    cross-checked against Kalshi's official result. A nonzero mismatch
    count means our settlement grading disagreed with the exchange at
    least once and needs investigating; it is the strongest available
    check that the trade-record data pool itself is trustworthy.
    """
    path = csv_path or _SETTLEMENT_VERIFIED_CSV
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return {"checked": 0, "mismatches": 0}
    return {
        "checked": int(len(df)),
        "mismatches": int((~df["match"].astype(bool)).sum()),
    }


def load_incident_log(path: Path | None = None) -> str:
    """Raw markdown of the hand-maintained incident log, or a placeholder.

    Not parsed — just embedded verbatim under the summary's incident-log
    heading. Convention: newest entry first, each a ``### YYYY-MM-DD —
    Title`` section.
    """
    p = path or _INCIDENT_LOG_PATH
    if not p.exists():
        return "_No incidents logged._"
    text = p.read_text(encoding="utf-8").strip()
    return text or "_No incidents logged._"


def _trend_arrow(recent: float, longer: float, higher_is_better: bool = True) -> str:
    if np.isnan(recent) or np.isnan(longer):
        return "→"
    delta = recent - longer
    if abs(delta) < 1e-9:
        return "→"
    improving = (delta > 0) == higher_is_better
    return "↑" if improving else "↓"


# ---------------------------------------------------------------------------
# Calibration review
# ---------------------------------------------------------------------------


def station_calibration(
    city: CityProfile,
    window_days: int = 30,
    training_dir: Path | None = None,
    params_dir: Path | None = None,
    spread_dir: Path | None = None,
    asof: date | None = None,
    metric: str = "high",
) -> dict | None:
    """Recompute calibration metrics for one station from stored artifacts.

    Scores the *currently saved* params over the last *window_days* rows of
    the training parquet — i.e. "if today's model had made the last month's
    forecasts, how calibrated were they".  Returns ``None`` when the station
    has no params or no usable training rows.
    """
    key = city.station_id if metric == "high" else f"{city.station_id}_low"
    params = load_params(key, params_dir=params_dir)
    path = _training_path(key, training_dir)
    if params is None or not path.exists():
        return None

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    model_names = [m for m in params.model_names if m in df.columns]
    if len(model_names) != len(params.model_names):
        logger.warning(
            "[%s] training parquet lacks %s — skipping calibration section",
            city.city, set(params.model_names) - set(model_names),
        )
        return None
    df = df.dropna(subset=model_names + ["actual_f"]).sort_values("date")
    df = df.tail(window_days)
    if len(df) < 5:
        return None

    members = df[model_names].to_numpy(float)
    y = df["actual_f"].to_numpy(float)
    mu = params.a0 + members @ np.asarray(params.a)

    member_var = members.var(axis=1, ddof=1)
    if params.spread_source == "ensemble" and "ens_spread_var" in df.columns:
        ens = df["ens_spread_var"].to_numpy(float)
        spread_var = np.where(np.isnan(ens), member_var, ens)
    else:
        spread_var = member_var
    sigma = np.sqrt(np.maximum(params.c + params.d * spread_var, params.sigma_floor_f**2))

    out = {
        "city": city.city if metric == "high" else f"{city.city} LOW",
        "station_id": key,
        "n_days": len(df),
        "crps": float(np.mean(crps_gaussian(mu, sigma, y))),
        "mae": float(np.mean(np.abs(mu - y))),
        "mean_sigma": float(np.mean(sigma)),
        "modal_prob": float(np.mean([modal_bucket_prob(m, s) for m, s in zip(mu, sigma)])),
        "pit_hist": pit_histogram(pit_values(mu, sigma, y), bins=5),
        "spread_source": params.spread_source,
        "nbm_mae": float("nan"),
        "nbm_divergence_f": float("nan"),
    }

    if "NBM" in df.columns and df["NBM"].notna().any():
        nbm = df["NBM"].to_numpy(float)
        ok = ~np.isnan(nbm)
        out["nbm_mae"] = float(np.mean(np.abs(nbm[ok] - y[ok])))
        out["nbm_divergence_f"] = float(np.mean(np.abs(mu[ok] - nbm[ok])))

    # Ops health: params age + spread-history progress toward the 60% gate.
    try:
        fitted = datetime.fromisoformat(params.fitted_at_utc)
        now = datetime.now(timezone.utc)
        if fitted.tzinfo is None:
            fitted = fitted.replace(tzinfo=timezone.utc)
        out["params_age_days"] = (now - fitted).days
    except ValueError:
        out["params_age_days"] = -1

    spread_hist = load_t1_spread_history(key, spread_dir=spread_dir)
    today = asof or date.today()
    recent = spread_hist[spread_hist["date"] >= today - timedelta(days=window_days)]
    out["spread_days"] = len(recent)
    out["spread_coverage"] = len(recent) / window_days
    return out


def _pit_sparkline(freq: np.ndarray) -> str:
    """Render a PIT histogram as unicode blocks (flat ≈ calibrated)."""
    if len(freq) == 0 or float(np.max(freq)) <= 0:
        return "n/a"
    scaled = freq / float(np.max(freq)) * (len(_PIT_BLOCKS) - 1)
    return "".join(_PIT_BLOCKS[int(round(v))] for v in scaled)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _trend_table_lines(stats7: dict, stats30: dict) -> list[str]:
    """The '7d vs 30d' trend table body, shared by the daily and summary reports."""
    if stats30.get("n", 0) == 0:
        return ["_No settled trades in the last 30 days._"]

    s7, s30 = stats7, stats30

    def fmt(key: str, spec: str, higher_better: bool) -> str:
        v7 = s7.get(key, float("nan")) if s7.get("n") else float("nan")
        v30 = s30.get(key, float("nan"))
        arrow = _trend_arrow(v7, v30, higher_better)
        v7s = format(v7, spec) if not np.isnan(v7) else "—"
        return f"{v7s} {arrow} (30d {format(v30, spec)})"

    lines = [
        "| Metric | 7d vs 30d |",
        "|---|---|",
        f"| Trades settled | {s7.get('n', 0)} / {s30['n']} |",
        f"| Realized P&L | {fmt('pnl', '+.2f', True)} |",
        f"| Win rate | {fmt('win_rate', '.0%', True)} |",
        f"| Model Brier (lower better) | {fmt('model_brier', '.4f', False)} |",
        f"| Market Brier | {fmt('market_brier', '.4f', False)} |",
        f"| **Brier edge vs market** | {fmt('brier_edge', '+.4f', True)} |",
        f"| Mean \\|alpha\\| | {fmt('mean_abs_alpha', '.3f', False)} |",
    ]
    lines.append("")
    edge = s7["brier_edge"] if s7.get("n") else s30["brier_edge"]
    lines.append(
        "**Reading:** "
        + (
            "our probabilities are beating the market on settled trades."
            if edge > 0
            else "the market's probabilities are beating ours — treat new "
                 "signals with suspicion and check calibration below."
        )
    )
    return lines


def _calibration_table_lines(calibrations: list[dict]) -> list[str]:
    """The per-station calibration table body, shared by the daily and summary reports."""
    if not calibrations:
        return ["_No stations with fitted params + training data yet._"]

    lines = [
        "| Station | n | CRPS °F | MAE °F | NBM MAE | σ̄ °F | Modal P | PIT (flat=good) | Spread src | Coverage → 60% | Params age |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in calibrations:
        nbm_mae = f"{c['nbm_mae']:.2f}" if not np.isnan(c["nbm_mae"]) else "—"
        lines.append(
            f"| {c['city']} ({c['station_id']}) | {c['n_days']} "
            f"| {c['crps']:.3f} | {c['mae']:.2f} | {nbm_mae} "
            f"| {c['mean_sigma']:.2f} | {c['modal_prob']:.2f} "
            f"| `{_pit_sparkline(c['pit_hist'])}` "
            f"| {c['spread_source']} "
            f"| {c['spread_coverage']:.0%} ({c['spread_days']}d) "
            f"| {c['params_age_days']}d |"
        )
    lines.append("")
    divergent = [
        c for c in calibrations
        if not np.isnan(c["nbm_divergence_f"]) and c["nbm_divergence_f"] > 2.0
    ]
    if divergent:
        lines.append(
            "**NBM divergence flags:** "
            + ", ".join(
                f"{c['city']} (|μ−NBM| avg {c['nbm_divergence_f']:.1f}°F)"
                for c in divergent
            )
        )
    stale = [c for c in calibrations if c["params_age_days"] > 7]
    if stale:
        lines.append(
            "**Stale params (>7d):** "
            + ", ".join(f"{c['city']} ({c['params_age_days']}d)" for c in stale)
            + " — is the 06:15 emos_training job running?"
        )
    return lines


def render_markdown(
    asof: date,
    latest: pd.DataFrame,
    latest_day: date | None,
    stats7: dict,
    stats30: dict,
    calibrations: list[dict],
    health_section: str = "",
) -> str:
    lines: list[str] = [f"# Deep Isobar scorecard — {asof}", ""]

    # ── 0. Ops health — alarms live above everything they protect ────────
    if health_section:
        lines.append(health_section)

    # ── 1. Latest settlements ────────────────────────────────────────────
    lines.append(f"## Settled trades ({latest_day or 'none yet'})")
    if latest.empty:
        lines.append("")
        lines.append("_No settled trades on the latest settlement day._")
    else:
        lines += [
            "",
            "| City | Contract | Dir | Alpha | Entry | Settled °F | Result | P&L |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for _, r in latest.sort_values(["city", "contract_ticker"]).iterrows():
            lines.append(
                f"| {r['city']} | {r['contract_ticker']} | {r['direction']} "
                f"| {r['alpha']:+.2f} | {r['entry_price']:.2f} "
                f"| {r['settled_temp']} | {r['status']} "
                f"| {r['realized_pnl']:+.2f} |"
            )
        day_pnl = float(latest["realized_pnl"].sum())
        lines.append("")
        lines.append(
            f"**Day P&L: {day_pnl:+.2f}**  "
            f"({int((latest['status'] == 'WIN').sum())}W / "
            f"{int((latest['status'] == 'LOSS').sum())}L)"
        )
    lines.append("")

    # ── 2. Direction of travel ───────────────────────────────────────────
    lines.append("## Which way are we moving? (7d vs 30d)")
    lines.append("")
    lines += _trend_table_lines(stats7, stats30)
    lines.append("")

    # ── 3. Calibration per station ───────────────────────────────────────
    lines.append("## Calibration (rolling window, current params on stored forecasts)")
    lines.append("")
    lines += _calibration_table_lines(calibrations)
    lines.append("")
    return "\n".join(lines)


def render_summary_markdown(
    asof: date,
    health_section: str,
    stats7: dict,
    stats30: dict,
    lifetime: dict,
    verification: dict | None,
    calibrations: list[dict],
    incident_log_md: str,
) -> str:
    """The evergreen single-file summary — 'what is the state of everything, right now'.

    Overwritten (not dated) on every scorecard run. Distinct from the dated
    per-day scorecard in three ways: lifetime (not just 7d/30d) stats,
    settlement-verification status, and the accumulated incident log —
    context that would otherwise only exist scattered across old dated
    files or in someone's memory.
    """
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [
        "# Deep Isobar — summary scorecard",
        "",
        f"_Last updated: {now_iso} (scorecard run for {asof})_",
        "",
        "📊 [Full interactive dashboard](SUMMARY.html) — same data, with rolling-edge, "
        "win-rate, and per-station CRPS charts.",
        "",
    ]

    if health_section:
        lines.append(health_section)

    lines.append("## Lifetime")
    lines.append("")
    if lifetime.get("n", 0) == 0:
        lines.append("_No settled trades yet._")
    else:
        verify_line = (
            f"{verification['checked']} checked, {verification['mismatches']} mismatch(es)"
            if verification is not None
            else "never run"
        )
        lines += [
            "| Metric | Value |",
            "|---|---|",
            f"| Settled trades | {lifetime['n']} ({lifetime['first_date']} → {lifetime['last_date']}) |",
            f"| Win rate | {lifetime['win_rate']:.1%} |",
            f"| Realized P&L | {lifetime['pnl']:+.2f} |",
            f"| Model Brier (lower better) | {lifetime['model_brier']:.4f} |",
            f"| Market Brier | {lifetime['market_brier']:.4f} |",
            f"| **Brier edge vs market** | {lifetime['brier_edge']:+.4f} |",
            f"| Exchange-settlement verification | {verify_line} |",
        ]
    lines.append("")

    lines.append("## Which way are we moving? (7d vs 30d)")
    lines.append("")
    lines += _trend_table_lines(stats7, stats30)
    lines.append("")
    lines.append("![Equity curve](charts/equity_curve.png)")
    lines.append("")
    lines.append("![Brier edge — rolling 7d/30d](charts/brier_edge.png)")
    lines.append("")

    lines.append("## Calibration (rolling window, current params on stored forecasts)")
    lines.append("")
    lines += _calibration_table_lines(calibrations)
    lines.append("")

    lines.append("## Incident log")
    lines.append("")
    lines.append(incident_log_md)
    lines.append("")

    return "\n".join(lines)


def build_discord_summary(
    latest: pd.DataFrame,
    latest_day: date | None,
    stats7: dict,
    stats30: dict,
    calibrations: list[dict],
) -> tuple[str, str, int, list[dict]]:
    """Compose (title, description, color, fields) for the Discord embed."""
    day_pnl = float(latest["realized_pnl"].sum()) if not latest.empty else 0.0
    wins = int((latest["status"] == "WIN").sum()) if not latest.empty else 0
    losses = int((latest["status"] == "LOSS").sum()) if not latest.empty else 0

    title = f"📊 Scorecard {latest_day or date.today()}: {day_pnl:+.2f} ({wins}W/{losses}L)"
    color = COLOR_GREEN if day_pnl >= 0 else COLOR_RED

    fields: list[dict] = []
    if stats30.get("n"):
        fields += [
            {"name": "7d P&L", "value": f"{stats7.get('pnl', 0.0):+.2f} ({stats7.get('n', 0)} trades)"},
            {"name": "30d P&L", "value": f"{stats30['pnl']:+.2f} ({stats30['n']} trades)"},
            {"name": "Win rate 7d/30d",
             "value": f"{stats7.get('win_rate', float('nan')):.0%} / {stats30['win_rate']:.0%}"
             if stats7.get("n") else f"— / {stats30['win_rate']:.0%}"},
            {"name": "Brier edge 30d", "value": f"{stats30['brier_edge']:+.4f}"},
        ]
    if calibrations:
        worst = max(calibrations, key=lambda c: c["crps"])
        best = min(calibrations, key=lambda c: c["crps"])
        fields.append({
            "name": "CRPS best/worst",
            "value": f"{best['city']} {best['crps']:.2f} / {worst['city']} {worst['crps']:.2f}",
        })
        gated = [c for c in calibrations if c["spread_source"] == "members"]
        if gated:
            closest = max(gated, key=lambda c: c["spread_coverage"])
            fields.append({
                "name": "Spread gate progress",
                "value": f"{closest['city']} leads at "
                         f"{closest['spread_coverage']:.0%} of {MIN_SPREAD_COVERAGE:.0%}",
            })
    description = "Full report: `data/reports/`"
    return title, description, color, fields


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Daily calibration & trading scorecard")
    parser.add_argument("--days", type=int, default=30, help="Calibration window (default 30)")
    parser.add_argument("--no-discord", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=_REPORTS_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asof = date.today()

    trades = load_settled_trades()
    if trades.empty:
        latest, latest_day = trades, None
    else:
        latest_day = trades["date"].max()
        latest = trades[trades["date"] == latest_day]
    stats7 = window_stats(trades, 7, asof)
    stats30 = window_stats(trades, 30, asof)

    calibrations = []
    for city in (c for c in get_city_universe() if c.active):
        metrics = ["high"] + (["low"] if city.kalshi_low_series else [])
        for metric in metrics:
            try:
                cal = station_calibration(
                    city, window_days=args.days, asof=asof, metric=metric
                )
            except Exception:  # noqa: BLE001
                logger.exception("[%s/%s] calibration section failed", city.city, metric)
                cal = None
            if cal is not None:
                calibrations.append(cal)

    health_checks = run_health_checks()
    report = render_markdown(
        asof, latest, latest_day, stats7, stats30, calibrations,
        health_section=render_health_section(health_checks),
    )
    print(report)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"scorecard_{asof}.md"
    out_path.write_text(report, encoding="utf-8")
    logger.info("Scorecard written to %s", out_path)

    equity_df = daily_equity_curve(trades)
    rolling_df = rolling_daily_metrics(trades)
    lifetime = all_time_stats(trades)
    verification = verification_summary()

    try:
        save_equity_curve_png(equity_df, _EQUITY_PNG_PATH)
        save_brier_edge_png(rolling_df, _BRIER_PNG_PATH)
    except Exception:  # noqa: BLE001 — charts are a bonus, never block the text report
        logger.exception("Chart PNG generation failed — SUMMARY.md will keep its old images")

    summary = render_summary_markdown(
        asof,
        health_section=render_health_section(health_checks),
        stats7=stats7,
        stats30=stats30,
        lifetime=lifetime,
        verification=verification,
        calibrations=calibrations,
        incident_log_md=load_incident_log(),
    )
    _SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SUMMARY_PATH.write_text(summary, encoding="utf-8")
    logger.info("Summary scorecard written to %s", _SUMMARY_PATH)

    try:
        dashboard = render_html_dashboard(
            asof, health_checks, lifetime, verification, equity_df, rolling_df,
            calibrations, load_incident_log(),
        )
        _SUMMARY_HTML_PATH.write_text(dashboard, encoding="utf-8")
        logger.info("HTML dashboard written to %s", _SUMMARY_HTML_PATH)
    except Exception:  # noqa: BLE001 — the HTML dashboard is a bonus, never block the text report
        logger.exception("HTML dashboard generation failed — SUMMARY.html will keep its old content")

    if not args.no_discord:
        title, description, color, fields = build_discord_summary(
            latest, latest_day, stats7, stats30, calibrations
        )
        post_embed(title, description, color, fields)
        post_alarm_embed(health_checks)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
