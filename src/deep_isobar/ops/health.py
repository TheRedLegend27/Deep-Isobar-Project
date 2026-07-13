"""Ops health invariants — make silent failure impossible.

Every incident in the project log shares one shape: a component stopped
doing its job and *nothing noticed* — the Jul 8-10 sessions that placed
zero trades for three days, the five days of stub 48/52 orderbooks.  This
module checks the invariants that break in those failure modes and screams
(red Discord embed + nonzero exit) the day they break:

1. **session_activity** — the 10:30 session logged evaluations for
   tomorrow's markets (target date = today+1) in ``daily_log.csv``.
2. **settlement_currency** — no position is still OPEN more than 48h past
   its target date (the settle job is running and grading).
3. **orderbook_freshness** — the collector wrote a snapshot recently
   (within its 06:00-21:00 window, "recently" means the last 45 min).
4. **stub_books** — no recent snapshot is stub-shaped (all books at the
   deterministic 0.48/0.52 stub prices).
5. **params_age** — every active station's EMOS params were refit within
   30h (the 06:15 refit is running).
6. **scorecard_gaps** — a scorecard markdown exists for each recent day
   (the 18:45 job is running).

Checks that cannot be judged yet at the current time of day (e.g. session
activity before 11:00) report SKIP, not OK — silence is never green.

Wired in two places so alarms sit where the eyes already are:

- supervisor job ``ops_health`` at 11:15 (catches a dead session the same
  morning; exit 1 also triggers the supervisor's job-FAILED embed), and
- a section at the top of the daily scorecard (18:45).

CLI::

    python -m deep_isobar.ops.health
    python -m deep_isobar.ops.health --no-discord
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd

from deep_isobar.calibration.emos import load_params
from deep_isobar.data.city_universe import get_city_universe
from deep_isobar.notifications.discord_notifier import COLOR_RED, post_embed

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PAPER_TRADES_DIR = _PROJECT_ROOT / "data" / "paper_trades"
_HISTORY_DIR = _PROJECT_ROOT / "data" / "market_history"
_REPORTS_DIR = _PROJECT_ROOT / "data" / "reports"

OK = "OK"
ALARM = "ALARM"
SKIP = "SKIP"

# Session runs at 10:30; judge its output only after this local time.
_SESSION_JUDGE_AFTER = time(11, 0)
# Collector cadence is 10 min — 45 min of silence is >3 missed cycles.
_COLLECTOR_WINDOW = (time(6, 0), time(21, 0))
_COLLECTOR_MAX_GAP = timedelta(minutes=45)
# Stub client always quotes 0.48/0.52 (the fake books of the Jul incident).
_STUB_BID, _STUB_ASK = 0.48, 0.52
# Refit is daily at 06:15 — params older than 30h mean a missed refit.
_PARAMS_MAX_AGE = timedelta(hours=30)
_SCORECARD_LOOKBACK_DAYS = 3


@dataclass(frozen=True)
class HealthCheck:
    name: str
    status: str  # OK | ALARM | SKIP
    detail: str


def _read_trades_csv(path: Path) -> pd.DataFrame | None:
    """Load a trades CSV with parsed dates, or None when absent/empty."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:  # noqa: BLE001 — unreadable is the caller's alarm
        return None
    if df.empty or "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    return df


# ---------------------------------------------------------------------------
# Checks — each takes explicit inputs so tests can drive them directly.
# ---------------------------------------------------------------------------


def check_session_activity(now: datetime, daily_log_csv: Path) -> HealthCheck:
    """After 11:00, today's session must have logged evaluations.

    The session trades *tomorrow's* markets, so its rows carry target date
    today+1.  Zero such rows is exactly the Jul 8-10 silent failure.
    """
    name = "session_activity"
    if now.time() < _SESSION_JUDGE_AFTER:
        return HealthCheck(name, SKIP, "before 11:00 — session not judged yet")

    target = now.date() + timedelta(days=1)
    df = _read_trades_csv(daily_log_csv)
    if df is None:
        return HealthCheck(name, ALARM, f"{daily_log_csv.name} missing or unreadable")

    rows = df[df["date"] == target]
    if rows.empty:
        return HealthCheck(
            name, ALARM,
            f"0 contracts evaluated for {target} — session did not run or "
            "found no markets (Jul 8-10 failure mode)",
        )
    n_open = int((rows.get("status") == "OPEN").sum())
    return HealthCheck(
        name, OK, f"{len(rows)} contracts evaluated for {target}, {n_open} signals placed"
    )


def check_settlement_currency(now: datetime, paper_trades_csv: Path) -> HealthCheck:
    """No position may sit OPEN more than 48h past its target date."""
    name = "settlement_currency"
    df = _read_trades_csv(paper_trades_csv)
    if df is None:
        return HealthCheck(name, SKIP, "no trades recorded yet")

    stale_cutoff = now.date() - timedelta(days=2)
    stale = df[(df["status"] == "OPEN") & (df["date"] <= stale_cutoff)]
    if not stale.empty:
        oldest = stale["date"].min()
        return HealthCheck(
            name, ALARM,
            f"{len(stale)} position(s) OPEN >48h past target date (oldest "
            f"{oldest}) — is the 18:00 settle job running?",
        )
    n_open = int((df["status"] == "OPEN").sum())
    return HealthCheck(name, OK, f"no stale OPEN positions ({n_open} currently open)")


def _recent_book_files(history_dir: Path, days: int = 2) -> list[Path]:
    """Parquet files from the *days* most recent date= partitions."""
    if not history_dir.exists():
        return []
    day_dirs = sorted(
        (d for d in history_dir.iterdir() if d.is_dir() and d.name.startswith("date=")),
        key=lambda d: d.name,
    )[-days:]
    return [f for d in day_dirs for f in sorted(d.glob("books_*.parquet"))]


def _window_active_seconds(a: datetime, b: datetime) -> float:
    """Seconds between *a* and *b* that fall inside the collector window.

    The overnight pause must not count toward the silence gap, but every
    in-window minute must — a collector that died mid-window yesterday is
    still an alarm this morning.
    """
    start, end = _COLLECTOR_WINDOW
    total = 0.0
    day = a.date()
    while day <= b.date():
        lo = max(a, datetime.combine(day, start))
        hi = min(b, datetime.combine(day, end))
        if hi > lo:
            total += (hi - lo).total_seconds()
        day += timedelta(days=1)
    return total


def check_orderbook_freshness(now: datetime, history_dir: Path) -> HealthCheck:
    """The collector wrote a snapshot within 45 min of window-active time.

    Uses file mtimes rather than the UTC-named partitions so the check is
    immune to the local-vs-UTC date rollover in the evening.
    """
    name = "orderbook_freshness"
    files = _recent_book_files(history_dir)
    if not files:
        return HealthCheck(name, ALARM, "no orderbook snapshots in the last 2 days")

    latest = max(datetime.fromtimestamp(f.stat().st_mtime) for f in files)
    gap_min = _window_active_seconds(latest, now) / 60
    if gap_min > _COLLECTOR_MAX_GAP.total_seconds() / 60:
        return HealthCheck(
            name, ALARM,
            f"last snapshot {gap_min:.0f} window-min ago — collector is "
            "silent (stub mode refusal, credentials, or supervisor down?)",
        )
    return HealthCheck(
        name, OK, f"last snapshot {gap_min:.0f} window-min ago"
    )


def check_stub_books(history_dir: Path) -> HealthCheck:
    """No recent snapshot may be stub-shaped (all books at 0.48/0.52).

    The collector refuses stub mode, but this is the backstop for the exact
    failure that once archived 5 days of fakes — defense in depth.
    """
    name = "stub_books"
    files = _recent_book_files(history_dir)
    if not files:
        return HealthCheck(name, SKIP, "no snapshots to inspect")

    suspect: list[str] = []
    for f in files:
        try:
            df = pd.read_parquet(f, columns=["best_bid", "best_ask"])
        except Exception:  # noqa: BLE001
            suspect.append(f"{f.parent.name}/{f.name} (unreadable)")
            continue
        if len(df) < 5:
            continue
        stubby = ((df["best_bid"] - _STUB_BID).abs() < 1e-9) & (
            (df["best_ask"] - _STUB_ASK).abs() < 1e-9
        )
        if stubby.mean() >= 0.8:
            suspect.append(f"{f.parent.name}/{f.name}")
    if suspect:
        shown = ", ".join(suspect[:3]) + ("…" if len(suspect) > 3 else "")
        return HealthCheck(
            name, ALARM,
            f"{len(suspect)} snapshot(s) look stub-shaped (0.48/0.52 books): {shown}",
        )
    return HealthCheck(name, OK, f"{len(files)} recent snapshots, none stub-shaped")


def check_params_age(
    now_utc: datetime,
    params_dir: Path | None = None,
    cities: list | None = None,
) -> HealthCheck:
    """Every active station's EMOS params must be <30h old (daily refit)."""
    name = "params_age"
    cities = cities if cities is not None else [
        c for c in get_city_universe() if c.active
    ]
    stale: list[str] = []
    checked = 0
    for city in cities:
        keys = [city.station_id]
        if getattr(city, "kalshi_low_series", None):
            keys.append(f"{city.station_id}_low")
        for key in keys:
            params = load_params(key, params_dir=params_dir)
            if params is None or not params.fitted_at_utc:
                continue  # onboarding gaps are preflight's problem, not a refit failure
            try:
                fitted = datetime.fromisoformat(params.fitted_at_utc)
            except ValueError:
                stale.append(f"{key} (unparseable fitted_at)")
                continue
            if fitted.tzinfo is None:
                fitted = fitted.replace(tzinfo=timezone.utc)
            checked += 1
            age = now_utc - fitted
            if age > _PARAMS_MAX_AGE:
                stale.append(f"{key} ({age.total_seconds() / 3600:.0f}h)")
    if not checked and not stale:
        return HealthCheck(name, SKIP, "no fitted params found")
    if stale:
        shown = ", ".join(stale[:5]) + ("…" if len(stale) > 5 else "")
        return HealthCheck(
            name, ALARM,
            f"{len(stale)} station(s) with params older than 30h — is the "
            f"06:15 emos_training job running? {shown}",
        )
    return HealthCheck(name, OK, f"{checked} stations refit within 30h")


def check_scorecard_gaps(now: datetime, reports_dir: Path) -> HealthCheck:
    """A scorecard file must exist for each of the last few days."""
    name = "scorecard_gaps"
    missing = [
        str(d)
        for d in (
            now.date() - timedelta(days=i)
            for i in range(1, _SCORECARD_LOOKBACK_DAYS + 1)
        )
        if not (reports_dir / f"scorecard_{d}.md").exists()
    ]
    if missing:
        return HealthCheck(
            name, ALARM,
            f"no scorecard for {', '.join(missing)} — the 18:45 job failed "
            "or the supervisor was down",
        )
    return HealthCheck(
        name, OK, f"scorecards present for the last {_SCORECARD_LOOKBACK_DAYS} days"
    )


# ---------------------------------------------------------------------------
# Aggregation / rendering / notification
# ---------------------------------------------------------------------------


def run_health_checks(
    now: datetime | None = None,
    project_root: Path | None = None,
) -> list[HealthCheck]:
    """Run every invariant check; a crashing check reports ALARM, not silence."""
    now = now or datetime.now()
    root = project_root or _PROJECT_ROOT
    trades_dir = root / "data" / "paper_trades"

    specs = [
        lambda: check_session_activity(now, trades_dir / "daily_log.csv"),
        lambda: check_settlement_currency(now, trades_dir / "paper_trades.csv"),
        lambda: check_orderbook_freshness(now, root / "data" / "market_history"),
        lambda: check_stub_books(root / "data" / "market_history"),
        lambda: check_params_age(datetime.now(timezone.utc)),
        lambda: check_scorecard_gaps(now, root / "data" / "reports"),
    ]
    names = [
        "session_activity", "settlement_currency", "orderbook_freshness",
        "stub_books", "params_age", "scorecard_gaps",
    ]
    checks: list[HealthCheck] = []
    for name, spec in zip(names, specs):
        try:
            checks.append(spec())
        except Exception as exc:  # noqa: BLE001
            logger.exception("health check %s crashed", name)
            checks.append(HealthCheck(name, ALARM, f"check crashed: {exc}"))
    return checks


def alarms(checks: list[HealthCheck]) -> list[HealthCheck]:
    return [c for c in checks if c.status == ALARM]


def render_health_section(checks: list[HealthCheck]) -> str:
    """Markdown section for the top of the daily scorecard."""
    lines = ["## Ops health", ""]
    broken = alarms(checks)
    if broken:
        lines.append(
            f"**🚨 {len(broken)} invariant(s) broken — do not trust the numbers "
            "below until these are explained.**"
        )
        lines.append("")
    icons = {OK: "✅", ALARM: "🚨", SKIP: "⏸"}
    for c in checks:
        lines.append(f"- {icons[c.status]} `{c.name}` — {c.detail}")
    lines.append("")
    return "\n".join(lines)


def post_alarm_embed(checks: list[HealthCheck]) -> None:
    """Scream on Discord — silent no-op when nothing is broken."""
    broken = alarms(checks)
    if not broken:
        return
    post_embed(
        title=f"🚨 Ops health: {len(broken)} invariant(s) broken",
        description="The system may be failing silently — investigate today.",
        color=COLOR_RED,
        fields=[{"name": c.name, "value": c.detail} for c in broken],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deep Isobar ops health invariants")
    parser.add_argument("--no-discord", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    checks = run_health_checks()
    print(render_health_section(checks))

    broken = alarms(checks)
    if broken and not args.no_discord:
        post_alarm_embed(checks)
    # Nonzero exit also fires the supervisor's job-FAILED embed.
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
