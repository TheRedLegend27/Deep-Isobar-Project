"""Intraday position check — METAR running high vs. open paper trades.

Daily-high markets have a one-way clock: the running high only ratchets
upward.  That makes some outcomes *irreversible* intraday:

- ``less`` (YES if T < cap): once the running high reaches the cap, YES is
  permanently dead — no afternoon cooling can undo it.
- ``greater`` (YES if T > floor): once the running high clears the floor,
  YES is permanently locked.
- ``between``: dead above the cap; "in range" below it (can still be
  overshot until the diurnal peak passes).

This script runs at ~14:00 local (supervisor job), pulls today's METAR
history for each city with OPEN trades settling today, classifies each
position, logs to ``data/paper_trades/intraday_log.csv``, and posts a
Discord alert for anything locked or busted.

It is advisory only — it never modifies ``paper_trades.csv``.  Note the
settlement high (NWS CLI / 5-min ASOS) can run ~1°F above the hourly METAR
max, so "still alive" calls near a boundary deserve skepticism.

Usage::

    python -m deep_isobar.research.intraday_check
"""

from __future__ import annotations

import csv
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_PROJECT_ROOT / ".env")

from deep_isobar.data.city_universe import get_city_universe
from deep_isobar.notifications.discord_notifier import (
    COLOR_AMBER,
    COLOR_GRAY,
    COLOR_GREEN,
    COLOR_RED,
    post_embed,
)

logger = logging.getLogger(__name__)

_PAPER_TRADES_CSV = _PROJECT_ROOT / "data" / "paper_trades" / "paper_trades.csv"
_INTRADAY_LOG_CSV = _PROJECT_ROOT / "data" / "paper_trades" / "intraday_log.csv"
_METAR_URL = "https://aviationweather.gov/api/data/metar"
_TIMEOUT_SECONDS = 20

_LOG_COLUMNS = [
    "check_time_utc", "trade_date", "city", "contract_ticker", "direction",
    "strike_type", "floor_strike", "cap_strike",
    "running_high_f", "obs_count", "yes_state", "position_state", "note",
]


def _fetch_running_high_f(station_id: str, tz_name: str) -> tuple[float | None, int]:
    """Return (max temp °F observed today local time, observation count)."""
    try:
        resp = requests.get(
            _METAR_URL,
            params={"ids": station_id, "format": "json", "hours": 20},
            timeout=_TIMEOUT_SECONDS,
            headers={"User-Agent": "deep-isobar/1.0"},
        )
        resp.raise_for_status()
        obs = resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] METAR history fetch failed: %s", station_id, exc)
        return None, 0

    tz = ZoneInfo(tz_name)
    today_local = datetime.now(tz).date()
    temps_f: list[float] = []
    for o in obs:
        t_c = o.get("temp")
        epoch = o.get("obsTime")
        if t_c is None or epoch is None:
            continue
        obs_local = datetime.fromtimestamp(epoch, tz=timezone.utc).astimezone(tz)
        if obs_local.date() == today_local:
            temps_f.append(float(t_c) * 9.0 / 5.0 + 32.0)

    if not temps_f:
        return None, 0
    return max(temps_f), len(temps_f)


def _classify_yes(strike_type: str, floor_strike: float | None,
                  cap_strike: float | None, high_f: float) -> tuple[str, str]:
    """Classify the YES outcome given the running high.

    Returns ``(yes_state, note)`` where yes_state is one of
    ``DEAD`` / ``LOCKED`` / ``IN_RANGE`` / ``PENDING``.
    """
    if strike_type == "less":
        if cap_strike is not None and high_f >= cap_strike:
            return "DEAD", f"running high {high_f:.0f}F already >= cap {cap_strike:.0f}F"
        return "IN_RANGE", f"alive while high stays under {cap_strike:.0f}F"
    if strike_type == "greater":
        if floor_strike is not None and high_f > floor_strike:
            return "LOCKED", f"running high {high_f:.0f}F already > floor {floor_strike:.0f}F"
        return "PENDING", f"needs high above {floor_strike:.0f}F"
    if strike_type == "between":
        if cap_strike is not None and high_f >= cap_strike:
            return "DEAD", f"running high {high_f:.0f}F already >= cap {cap_strike:.0f}F"
        if floor_strike is not None and high_f >= floor_strike:
            return "IN_RANGE", "in range now; overshoot risk until diurnal peak"
        return "PENDING", f"needs high to reach {floor_strike:.0f}F without hitting {cap_strike:.0f}F"
    return "PENDING", f"unknown strike_type {strike_type!r}"


def _position_state(direction: str, yes_state: str) -> str:
    """Translate the YES outcome into this position's state."""
    if yes_state in ("IN_RANGE", "PENDING"):
        return yes_state
    if direction == "BUY":
        return "LOCKED_WIN" if yes_state == "LOCKED" else "BUSTED"
    # SELL profits when YES dies
    return "BUSTED" if yes_state == "LOCKED" else "LOCKED_WIN"


def _f_or_none(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    today = str(date.today())

    if not _PAPER_TRADES_CSV.exists():
        logger.info("No paper_trades.csv yet — nothing to check.")
        return
    with _PAPER_TRADES_CSV.open(newline="", encoding="utf-8") as fh:
        open_today = [
            r for r in csv.DictReader(fh)
            if r.get("status") == "OPEN" and r.get("date") == today
        ]

    if not open_today:
        logger.info("No OPEN trades settling today (%s) — nothing to check.", today)
        print(f"No OPEN trades settling today ({today}).")
        return

    profiles = {p.city: p for p in get_city_universe()}
    rows: list[dict] = []
    alerts: list[dict] = []
    check_time = datetime.now(timezone.utc).isoformat(timespec="seconds")

    by_city: dict[str, list[dict]] = {}
    for r in open_today:
        by_city.setdefault(r.get("city", ""), []).append(r)

    for city_name, trades in by_city.items():
        profile = profiles.get(city_name)
        if profile is None:
            logger.warning("City %r not in cities.yaml — skipping %d trade(s)", city_name, len(trades))
            continue

        high_f, n_obs = _fetch_running_high_f(profile.station_id, profile.timezone)
        if high_f is None:
            print(f"[{city_name}] No METAR data for {profile.station_id} — skipping.")
            continue

        print(f"\n[{city_name}] {profile.station_id} running high: {high_f:.1f}F ({n_obs} obs today)")

        for r in trades:
            yes_state, note = _classify_yes(
                r.get("strike_type", ""),
                _f_or_none(r.get("floor_strike", "")),
                _f_or_none(r.get("cap_strike", "")),
                high_f,
            )
            pos_state = _position_state(r.get("direction", "BUY"), yes_state)
            print(f"  {r['contract_ticker']:<32} {r.get('direction', ''):<4} -> {pos_state:<10} ({note})")

            row = {
                "check_time_utc": check_time,
                "trade_date": r.get("date", ""),
                "city": city_name,
                "contract_ticker": r.get("contract_ticker", ""),
                "direction": r.get("direction", ""),
                "strike_type": r.get("strike_type", ""),
                "floor_strike": r.get("floor_strike", ""),
                "cap_strike": r.get("cap_strike", ""),
                "running_high_f": round(high_f, 1),
                "obs_count": n_obs,
                "yes_state": yes_state,
                "position_state": pos_state,
                "note": note,
            }
            rows.append(row)
            if pos_state in ("LOCKED_WIN", "BUSTED"):
                alerts.append(row)

    if rows:
        _INTRADAY_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
        new_file = not _INTRADAY_LOG_CSV.exists()
        with _INTRADAY_LOG_CSV.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_LOG_COLUMNS)
            if new_file:
                writer.writeheader()
            writer.writerows(rows)

    if alerts:
        post_embed(
            title=f"Intraday check — {len(alerts)} position(s) decided early",
            color=COLOR_RED if any(a["position_state"] == "BUSTED" for a in alerts) else COLOR_GREEN,
            fields=[
                {
                    "name": f"[{a['city']}] {a['contract_ticker']}",
                    "value": f"{a['direction']} -> **{a['position_state']}** — {a['note']}",
                }
                for a in alerts[:10]
            ],
        )
    elif rows:
        post_embed(
            title="Intraday check — all open positions still live",
            description=f"{len(rows)} position(s) checked; none locked or busted yet.",
            color=COLOR_GRAY,
        )

    print(f"\n{len(rows)} position(s) checked, {len(alerts)} decided early.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
