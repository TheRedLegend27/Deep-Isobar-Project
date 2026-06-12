"""Deep Isobar daily-job supervisor.

A single long-running process that replaces the three separate Windows
Task Scheduler entries (session / settle / dashboard).  Register **once**
via ``start_supervisor.bat`` at logon and it handles all daily jobs:

- runs each job at its configured local time via subprocess isolation
  (a crashing job never takes the supervisor down),
- catches up on missed jobs (PC asleep / rebooted at the scheduled time:
  the job runs as soon as the supervisor is back, instead of skipping
  the whole day),
- posts a daily Discord heartbeat so silence becomes a signal,
- persists last-run state to ``data/scheduler_state.json`` across restarts.

Jobs are configured in ``config/settings.yaml`` under ``scheduler.jobs``::

    scheduler:
      poll_seconds: 60
      jobs:
        - name: paper_trade_session
          module: deep_isobar.research.paper_trade_session
          at: "07:00"

Times are **local machine time** — check them after a timezone move.

Usage::

    python -m deep_isobar.supervisor             # run the loop
    python -m deep_isobar.supervisor --status    # print schedule + state, exit
    python -m deep_isobar.supervisor --run-now paper_trade_session
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")

from deep_isobar.config import get_setting
from deep_isobar.notifications.discord_notifier import (
    COLOR_GREEN,
    COLOR_RED,
    post_embed,
)

logger = logging.getLogger(__name__)

_STATE_PATH = _PROJECT_ROOT / "data" / "scheduler_state.json"
_JOB_LOG_DIR = _PROJECT_ROOT / "data" / "logs" / "jobs"
_SUPERVISOR_LOG = _PROJECT_ROOT / "data" / "logs" / "supervisor.log"
_JOB_TIMEOUT_SECONDS = 3600

_DEFAULT_JOBS = [
    {"name": "paper_trade_session", "module": "deep_isobar.research.paper_trade_session", "at": "07:00"},
    {"name": "intraday_check",      "module": "deep_isobar.research.intraday_check",      "at": "14:00"},
    {"name": "settle_paper_trades", "module": "deep_isobar.research.settle_paper_trades", "at": "18:00"},
    {"name": "generate_dashboard",  "module": "deep_isobar.research.generate_dashboard",  "at": "19:15"},
]


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


def _load_state() -> dict:
    if _STATE_PATH.exists():
        try:
            return json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("State file unreadable (%s) — starting fresh", exc)
    return {"last_run": {}, "last_heartbeat": ""}


def _save_state(state: dict) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------


def _parse_at(at: str) -> tuple[int, int]:
    """Parse an ``"HH:MM"`` string into ``(hour, minute)``."""
    hh, mm = at.split(":")
    return int(hh), int(mm)


def _run_job(job: dict) -> bool:
    """Run one job as ``python -m <module>`` and return success.

    stdout/stderr go to ``data/logs/jobs/<name>_<date>.log`` so a job
    failure is always diagnosable after the fact.
    """
    name = job["name"]
    module = job["module"]
    _JOB_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _JOB_LOG_DIR / f"{name}_{date.today():%Y%m%d}.log"

    logger.info("Running job %s (%s) — log: %s", name, module, log_path)
    started = datetime.now()

    env = dict(os.environ)
    env["PYTHONPATH"] = str(_PROJECT_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

    try:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"\n=== {name} started {started:%Y-%m-%d %H:%M:%S} ===\n")
            fh.flush()
            proc = subprocess.run(
                [sys.executable, "-m", module],
                cwd=str(_PROJECT_ROOT),
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                timeout=_JOB_TIMEOUT_SECONDS,
            )
        ok = proc.returncode == 0
    except subprocess.TimeoutExpired:
        logger.error("Job %s timed out after %ds", name, _JOB_TIMEOUT_SECONDS)
        ok = False
    except Exception as exc:  # noqa: BLE001
        logger.exception("Job %s failed to launch: %s", name, exc)
        ok = False

    elapsed = (datetime.now() - started).total_seconds()
    logger.info("Job %s %s in %.0fs", name, "succeeded" if ok else "FAILED", elapsed)

    if not ok:
        try:
            post_embed(
                title=f"Supervisor — job FAILED: {name}",
                description=f"See {log_path.name} for output.",
                color=COLOR_RED,
            )
        except Exception:  # noqa: BLE001
            logger.warning("Could not post failure embed for %s", name)
    return ok


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------


def _post_heartbeat(jobs: list[dict], state: dict) -> None:
    today = str(date.today())
    fields = []
    for job in jobs:
        last = state["last_run"].get(job["name"], "never")
        if last == today:
            value = f"✅ ran (last: {last})"
        else:
            value = f"⏳ scheduled {job['at']} (last: {last})"
        fields.append({"name": job["name"], "value": value})
    post_embed(
        title=f"Deep Isobar supervisor — alive ({today})",
        color=COLOR_GREEN,
        fields=fields,
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_loop() -> None:
    jobs: list[dict] = get_setting("scheduler.jobs", default=_DEFAULT_JOBS)
    poll_seconds: int = int(get_setting("scheduler.poll_seconds", default=60))
    state = _load_state()

    logger.info(
        "Supervisor started — %d job(s): %s",
        len(jobs),
        {j["name"]: j["at"] for j in jobs},
    )

    while True:
        now = datetime.now()
        today = str(now.date())

        # ── Daily heartbeat (first pass of each day, and on first start) ──
        if state.get("last_heartbeat") != today:
            try:
                _post_heartbeat(jobs, state)
            except Exception:  # noqa: BLE001
                logger.warning("Heartbeat embed failed", exc_info=True)
            state["last_heartbeat"] = today
            _save_state(state)

        # ── Due jobs (includes catch-up for missed times) ─────────────────
        for job in jobs:
            hh, mm = _parse_at(job["at"])
            sched = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if now < sched:
                continue
            if state["last_run"].get(job["name"]) == today:
                continue

            late_minutes = (now - sched).total_seconds() / 60.0
            if late_minutes > 5:
                logger.warning(
                    "Job %s is %.0f min late (missed %s) — running catch-up",
                    job["name"], late_minutes, job["at"],
                )
            _run_job(job)
            # Mark as run regardless of success: a deterministic failure
            # must not retry every poll and spam Discord all day.
            state["last_run"][job["name"]] = today
            _save_state(state)

        time.sleep(poll_seconds)


def print_status() -> None:
    jobs: list[dict] = get_setting("scheduler.jobs", default=_DEFAULT_JOBS)
    state = _load_state()
    print(f"Supervisor schedule ({len(jobs)} jobs, local machine time):")
    for job in jobs:
        last = state["last_run"].get(job["name"], "never")
        print(f"  {job['at']}  {job['name']:<22} module={job['module']}  last_run={last}")
    print(f"State file     : {_STATE_PATH}")
    print(f"Job logs       : {_JOB_LOG_DIR}")
    print(f"Last heartbeat : {state.get('last_heartbeat') or 'never'}")


def main() -> None:
    _SUPERVISOR_LOG.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(_SUPERVISOR_LOG, encoding="utf-8"),
        ],
    )

    parser = argparse.ArgumentParser(description="Deep Isobar daily-job supervisor.")
    parser.add_argument("--status", action="store_true", help="Print schedule and state, then exit.")
    parser.add_argument("--run-now", metavar="JOB", help="Run a single named job immediately, then exit.")
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    if args.run_now:
        jobs: list[dict] = get_setting("scheduler.jobs", default=_DEFAULT_JOBS)
        match = next((j for j in jobs if j["name"] == args.run_now), None)
        if match is None:
            print(f"Unknown job: {args.run_now}. Available: {[j['name'] for j in jobs]}")
            sys.exit(1)
        ok = _run_job(match)
        sys.exit(0 if ok else 1)

    run_loop()


if __name__ == "__main__":
    main()
