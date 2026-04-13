"""Deep Isobar — Watchdog process.

Runs every 15 minutes via Windows Task Scheduler.  Checks whether each
scheduled task produced its expected output for today.  Posts Discord alerts
for any failures.  Never modifies data, never retries failed tasks, never
raises uncaught exceptions.

Usage::

    python -m deep_isobar.monitoring.watchdog

Schedule::

    Task Scheduler → every 15 minutes, all day.

Log format (one line per run, appended to data/logs/watchdog.log)::

    2026-04-13 07:30:00 CDT | PASS | 4/4 checks ok
    2026-04-13 08:15:00 CDT | WARN | 2 alert(s): morning_session, kalshi_api

Check summary
-------------
1. Morning session (after 07:30 CDT)  — daily_log.csv has a row for today.
2. Settlement     (after 18:30 CDT)   — no OPEN rows for today in paper_trades.csv.
3. Dashboard      (after 19:30 CDT)   — dashboard.html mtime is today.
4. Kalshi API     (always)            — GET /exchange/status returns 200.
5. Bias profile   (always)            — KMDW profile updated within 7 days.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DAILY_LOG_CSV     = _PROJECT_ROOT / "data" / "paper_trades" / "daily_log.csv"
_PAPER_TRADES_CSV  = _PROJECT_ROOT / "data" / "paper_trades" / "paper_trades.csv"
_DASHBOARD_HTML    = _PROJECT_ROOT / "data" / "paper_trades" / "dashboard.html"
_BIAS_PROFILE      = _PROJECT_ROOT / "data" / "bias_profiles" / "KMDW_monthly_profile.parquet"
_WATCHDOG_LOG      = _PROJECT_ROOT / "data" / "logs" / "watchdog.log"

_KALSHI_STATUS_URL = "https://api.elections.kalshi.com/trade-api/v2/exchange/status"

# ---------------------------------------------------------------------------
# Timezone helpers
# ---------------------------------------------------------------------------

# CDT = UTC-5, CST = UTC-6.  We hard-code CDT (UTC-5) year-round for
# simplicity — the watchdog thresholds are loose enough that a one-hour
# shift around DST transitions does not matter.
_CDT = timezone(timedelta(hours=-5), name="CDT")

_BIAS_STALE_DAYS = 7


def _now_cdt() -> datetime:
    return datetime.now(_CDT)


def _today_cdt() -> date:
    return _now_cdt().date()


# ---------------------------------------------------------------------------
# Discord alert helpers
# ---------------------------------------------------------------------------

# Watchdog-specific colours (distinct from the trading-signal colours).
_COLOR_AMBER = 0xF59E0B   # warning
_COLOR_RED   = 0xEF4444   # hard failure

_ALERT_TITLE = "\u26a0\ufe0f Deep Isobar \u2014 Watchdog Alert"
_FOOTER_TEXT = "Watchdog \u2014 check logs for detail"


def _post_alert(
    check_name: str,
    expected: str,
    found: str,
    webhook_url: str,
    color: int = _COLOR_AMBER,
) -> None:
    """Post one watchdog alert embed to Discord.

    Never raises — a webhook failure is logged and swallowed.
    """
    now_str = _now_cdt().strftime("%Y-%m-%d %H:%M:%S CDT")

    payload = json.dumps(
        {
            "embeds": [
                {
                    "title": _ALERT_TITLE,
                    "color": color,
                    "fields": [
                        {"name": "Check",      "value": check_name, "inline": False},
                        {"name": "Expected",   "value": expected,   "inline": True},
                        {"name": "Found",      "value": found,      "inline": True},
                        {"name": "Time (CDT)", "value": now_str,    "inline": False},
                    ],
                    "footer": {"text": _FOOTER_TEXT},
                }
            ]
        }
    ).encode()

    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent":   "DiscordBot (deep-isobar-watchdog, 1.0)",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status not in (200, 204):
                logger.warning(
                    "Discord webhook returned unexpected status %d", resp.status
                )
    except urllib.error.HTTPError as exc:
        logger.warning("Discord webhook HTTP error %d: %s", exc.code, exc.reason)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Discord webhook failed: %s", exc)


# ---------------------------------------------------------------------------
# Log helper
# ---------------------------------------------------------------------------


def _append_watchdog_log(status: str, detail: str) -> None:
    """Append one line to data/logs/watchdog.log.

    Format::

        2026-04-13 07:30:00 CDT | PASS | 4/4 checks ok
    """
    _WATCHDOG_LOG.parent.mkdir(parents=True, exist_ok=True)
    now_str = _now_cdt().strftime("%Y-%m-%d %H:%M:%S CDT")
    line = f"{now_str} | {status:<4} | {detail}\n"
    try:
        with _WATCHDOG_LOG.open("a", encoding="utf-8") as fh:
            fh.write(line)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not write to watchdog.log: %s", exc)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


@dataclass
class _Alert:
    check_name: str
    expected: str
    found: str
    color: int = _COLOR_AMBER


def _check_morning_session(today: date) -> _Alert | None:
    """Check 1 — Morning session ran (active after 07:30 CDT).

    The morning session runs each day targeting the *next* calendar day,
    so rows written by yesterday's session carry date=today.  We look for
    at least one row with date matching today.
    """
    now = _now_cdt()
    if now.hour < 7 or (now.hour == 7 and now.minute < 30):
        logger.debug("Check 1 (morning_session): before 07:30 CDT — skipping.")
        return None

    if not _DAILY_LOG_CSV.exists():
        return _Alert(
            check_name="morning_session",
            expected=f"daily_log.csv exists with a row for {today}",
            found="daily_log.csv does not exist",
            color=_COLOR_RED,
        )

    today_str = str(today)
    try:
        with _DAILY_LOG_CSV.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("date", "").strip() == today_str:
                    logger.debug("Check 1 (morning_session): PASS — row found for %s.", today)
                    return None
    except Exception as exc:  # noqa: BLE001
        return _Alert(
            check_name="morning_session",
            expected=f"daily_log.csv readable with a row for {today}",
            found=f"Error reading file: {exc}",
            color=_COLOR_RED,
        )

    return _Alert(
        check_name="morning_session",
        expected=f"at least one row with date={today} in daily_log.csv",
        found=f"no rows for {today} found (file exists but no matching date)",
    )


def _check_settlement(today: date) -> _Alert | None:
    """Check 2 — Settlement ran (active after 18:30 CDT).

    After 18:30, any OPEN row in paper_trades.csv dated today means the
    settlement script did not run (it should have closed them by ~18:00).
    """
    now = _now_cdt()
    if now.hour < 18 or (now.hour == 18 and now.minute < 30):
        logger.debug("Check 2 (settlement): before 18:30 CDT — skipping.")
        return None

    if not _PAPER_TRADES_CSV.exists():
        # No trades file at all — no OPEN rows, so settlement is vacuously fine.
        logger.debug("Check 2 (settlement): paper_trades.csv absent — vacuously PASS.")
        return None

    today_str = str(today)
    open_tickers: list[str] = []
    try:
        with _PAPER_TRADES_CSV.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if (
                    row.get("date", "").strip() == today_str
                    and row.get("status", "").strip() == "OPEN"
                ):
                    open_tickers.append(row.get("contract_ticker", "?"))
    except Exception as exc:  # noqa: BLE001
        return _Alert(
            check_name="settlement",
            expected=f"paper_trades.csv readable with no OPEN rows for {today}",
            found=f"Error reading file: {exc}",
            color=_COLOR_RED,
        )

    if open_tickers:
        tickers_str = ", ".join(open_tickers[:5])
        if len(open_tickers) > 5:
            tickers_str += f" … (+{len(open_tickers) - 5} more)"
        return _Alert(
            check_name="settlement",
            expected=f"no OPEN rows for {today} after 18:30 CDT",
            found=f"{len(open_tickers)} OPEN row(s) still present: {tickers_str}",
        )

    logger.debug("Check 2 (settlement): PASS — no OPEN rows for %s.", today)
    return None


def _check_dashboard(today: date) -> _Alert | None:
    """Check 3 — Dashboard generated (active after 19:30 CDT).

    Verifies that data/paper_trades/dashboard.html was modified today
    (local CDT date) by comparing the file's mtime.
    """
    now = _now_cdt()
    if now.hour < 19 or (now.hour == 19 and now.minute < 30):
        logger.debug("Check 3 (dashboard): before 19:30 CDT — skipping.")
        return None

    if not _DASHBOARD_HTML.exists():
        return _Alert(
            check_name="dashboard",
            expected=f"dashboard.html exists and mtime is {today}",
            found="dashboard.html does not exist",
            color=_COLOR_RED,
        )

    try:
        mtime_utc = datetime.fromtimestamp(_DASHBOARD_HTML.stat().st_mtime, tz=timezone.utc)
        mtime_cdt = mtime_utc.astimezone(_CDT).date()
    except Exception as exc:  # noqa: BLE001
        return _Alert(
            check_name="dashboard",
            expected="dashboard.html mtime readable",
            found=f"Error reading mtime: {exc}",
            color=_COLOR_RED,
        )

    if mtime_cdt != today:
        return _Alert(
            check_name="dashboard",
            expected=f"dashboard.html modified on {today}",
            found=f"last modified on {mtime_cdt}",
        )

    logger.debug("Check 3 (dashboard): PASS — mtime=%s.", mtime_cdt)
    return None


def _check_kalshi_api() -> _Alert | None:
    """Check 4 — Kalshi API reachable (always active).

    GETs the public /exchange/status endpoint with a 5-second timeout.
    Alerts on any non-200 or connection failure.
    """
    req = urllib.request.Request(
        _KALSHI_STATUS_URL,
        headers={"User-Agent": "deep-isobar-watchdog/1.0"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            status_code = resp.status
    except urllib.error.HTTPError as exc:
        return _Alert(
            check_name="kalshi_api",
            expected="HTTP 200 from /exchange/status",
            found=f"HTTP {exc.code}: {exc.reason}",
            color=_COLOR_RED,
        )
    except Exception as exc:  # noqa: BLE001
        return _Alert(
            check_name="kalshi_api",
            expected="successful GET to /exchange/status within 5 s",
            found=f"Connection failed: {exc}",
            color=_COLOR_RED,
        )

    if status_code != 200:
        return _Alert(
            check_name="kalshi_api",
            expected="HTTP 200 from /exchange/status",
            found=f"HTTP {status_code}",
            color=_COLOR_RED,
        )

    logger.debug("Check 4 (kalshi_api): PASS — HTTP 200.")
    return None


def _check_bias_profile(today: date) -> _Alert | None:
    """Check 5 — Bias profile fresh (always active).

    Reads KMDW_monthly_profile.parquet and verifies that the current
    month's row has a last_updated date within the past 7 days.
    """
    if not _BIAS_PROFILE.exists():
        return _Alert(
            check_name="bias_profile",
            expected=f"{_BIAS_PROFILE.name} exists",
            found="file does not exist",
            color=_COLOR_RED,
        )

    try:
        import pandas as pd  # deferred — not available in every env

        df = pd.read_parquet(_BIAS_PROFILE)
    except Exception as exc:  # noqa: BLE001
        return _Alert(
            check_name="bias_profile",
            expected=f"{_BIAS_PROFILE.name} readable",
            found=f"Error reading parquet: {exc}",
            color=_COLOR_RED,
        )

    current_month = today.month
    month_rows = df[df["month"] == current_month] if "month" in df.columns else df.iloc[:0]

    if month_rows.empty:
        return _Alert(
            check_name="bias_profile",
            expected=f"row for month={current_month} in {_BIAS_PROFILE.name}",
            found=f"no row for month={current_month}",
            color=_COLOR_RED,
        )

    try:
        # pd.Timestamp(...).date() handles Timestamp, datetime, date, and string.
        last_updated: date = pd.Timestamp(month_rows.iloc[0]["last_updated"]).date()
    except Exception as exc:  # noqa: BLE001
        return _Alert(
            check_name="bias_profile",
            expected="parseable last_updated date for current month",
            found=f"Could not parse last_updated: {exc}",
            color=_COLOR_RED,
        )

    age_days = (today - last_updated).days
    if age_days > _BIAS_STALE_DAYS:
        return _Alert(
            check_name="bias_profile",
            expected=f"last_updated within {_BIAS_STALE_DAYS} days",
            found=f"last_updated={last_updated} ({age_days} days ago)",
        )

    logger.debug(
        "Check 5 (bias_profile): PASS — last_updated=%s (%d days ago).",
        last_updated, age_days,
    )
    return None


# ---------------------------------------------------------------------------
# Main watchdog run
# ---------------------------------------------------------------------------


def run() -> None:
    """Execute all watchdog checks and post Discord alerts for any failures.

    Always appends one line to data/logs/watchdog.log regardless of outcome.
    Never raises.
    """
    today = _today_cdt()
    alerts: list[_Alert] = []

    webhook_url: str | None = os.environ.get("DISCORD_WEBHOOK_URL") or None
    if not webhook_url:
        logger.warning("DISCORD_WEBHOOK_URL not set — Discord alerts suppressed for this run.")

    checks = [
        ("morning_session", lambda: _check_morning_session(today)),
        ("settlement",      lambda: _check_settlement(today)),
        ("dashboard",       lambda: _check_dashboard(today)),
        ("kalshi_api",      _check_kalshi_api),
        ("bias_profile",    lambda: _check_bias_profile(today)),
    ]

    for check_name, check_fn in checks:
        try:
            alert = check_fn()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error in check %r", check_name)
            alert = _Alert(
                check_name=check_name,
                expected="check completed without error",
                found=f"Unhandled exception: {type(exc).__name__}: {exc}",
                color=_COLOR_RED,
            )

        if alert is not None:
            alerts.append(alert)

    # Post Discord alerts for each failure (no alert = healthy = silence).
    if webhook_url:
        for alert in alerts:
            _post_alert(
                check_name=alert.check_name,
                expected=alert.expected,
                found=alert.found,
                webhook_url=webhook_url,
                color=alert.color,
            )

    # Log one summary line.
    n_checks = len(checks)
    n_failed = len(alerts)
    n_passed = n_checks - n_failed

    if n_failed == 0:
        status = "PASS"
        detail = f"{n_passed}/{n_checks} checks ok"
    else:
        failed_names = ", ".join(a.check_name for a in alerts)
        status = "WARN"
        detail = f"{n_failed} alert(s): {failed_names}"

    _append_watchdog_log(status, detail)
    logger.info("Watchdog run complete: %s — %s", status, detail)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run()
