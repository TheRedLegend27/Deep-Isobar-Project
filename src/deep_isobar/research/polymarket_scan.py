"""Polymarket scan — model vs. Polymarket prices, with Kalshi alongside.

Read-only research tool (no trades are logged).  For each active city that
Polymarket lists (Chicago, New York, Dallas), it:

1. builds tomorrow's multi-model ensemble (same pipeline as the paper
   trade session),
2. prices every Polymarket bucket with the probability engine,
3. fetches the live CLOB book and computes alpha vs. the executable price,
4. prints the matching Kalshi board for the same date so cross-venue
   divergence is visible at a glance,
5. appends rows to ``data/paper_trades/polymarket_scan.csv``.

STATION CAVEAT: Polymarket settles on different stations than Kalshi
(KORD vs KMDW, KLGA vs Central Park, KDAL vs KDFW) and the bias profiles
were calibrated on the Kalshi-side stations.  Treat Polymarket alphas as
indicative until per-station calibration exists — and treat cross-venue
divergence as a correlated-station spread, never a pure arb.

Usage::

    python -m deep_isobar.research.polymarket_scan
    python -m deep_isobar.research.polymarket_scan --date 2026-06-13
"""

from __future__ import annotations

import argparse
import csv
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_PROJECT_ROOT / ".env")

from deep_isobar.data.city_universe import get_city_universe
from deep_isobar.market import polymarket_client
from deep_isobar.market.kalshi_client import fetch_live_contracts as fetch_kalshi_contracts
from deep_isobar.market.kalshi_client import fetch_orderbook_for_contract as fetch_kalshi_book
from deep_isobar.models.probability_engine import probability_for_contract
from deep_isobar.models.temperature_ensemble import build_temperature_ensemble
from deep_isobar.research.paper_trade_session import (
    METRIC,
    _MIN_ENSEMBLE_STD_F,
    _fetch_live_forecasts_t24,
)

logger = logging.getLogger(__name__)

_SCAN_CSV = _PROJECT_ROOT / "data" / "paper_trades" / "polymarket_scan.csv"
_CSV_COLUMNS = [
    "scan_time_utc", "target_date", "city", "venue", "contract_ticker",
    "bucket", "strike_type", "floor_strike", "cap_strike",
    "model_prob", "best_bid", "best_ask", "mid", "alpha_vs_mid",
]


def _fmt(x, width=6):
    return f"{x:>{width}.3f}" if isinstance(x, float) else f"{'—':>{width}}"


def _bucket_label(c) -> str:
    # ASCII only — Windows consoles default to cp1252 and choke on en-dash/≥.
    if c.strike_type == "between":
        return f"{c.floor_strike}-{c.cap_strike - 1}F"
    if c.strike_type == "greater":
        return f">={c.floor_strike + 1}F"
    if c.strike_type == "less":
        return f"<={c.cap_strike - 1}F"
    return "?"


def _scan_venue(city_profile, contracts, fetch_book, venue, mean_f, std_f, rows):
    print(f"\n  {venue} — {len(contracts)} contract(s)")
    print(f"    {'bucket':<12} {'model':>6} {'bid':>6} {'ask':>6} {'mid':>6} {'alpha':>7}")
    for c in sorted(contracts, key=lambda x: (x.floor_strike if x.floor_strike is not None else -999)):
        try:
            model_p = probability_for_contract(
                strike_type=c.strike_type,
                floor_strike=c.floor_strike,
                cap_strike=c.cap_strike,
                mean_f=mean_f,
                std_f=std_f,
            )
        except ValueError as exc:
            logger.debug("Skipping %s: %s", c.contract_id, exc)
            continue
        book = fetch_book(venue, c.contract_id)
        bid, ask = book.best_bid, book.best_ask
        mid = (bid + ask) / 2.0 if bid is not None and ask is not None else None
        alpha = (model_p - mid) if mid is not None else None

        flag = ""
        if alpha is not None and abs(alpha) >= 0.25:
            flag = "  <<<"
        print(
            f"    {_bucket_label(c):<12} {_fmt(model_p)} {_fmt(bid)} {_fmt(ask)} "
            f"{_fmt(mid)} {_fmt(alpha, 7)}{flag}"
        )
        rows.append({
            "scan_time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "target_date": str(c.target_date),
            "city": city_profile.city,
            "venue": venue,
            "contract_ticker": c.contract_id,
            "bucket": _bucket_label(c),
            "strike_type": c.strike_type,
            "floor_strike": c.floor_strike if c.floor_strike is not None else "",
            "cap_strike": c.cap_strike if c.cap_strike is not None else "",
            "model_prob": round(model_p, 6),
            "best_bid": bid if bid is not None else "",
            "best_ask": ask if ask is not None else "",
            "mid": round(mid, 6) if mid is not None else "",
            "alpha_vs_mid": round(alpha, 6) if alpha is not None else "",
        })


def main(target_date: date | None = None) -> None:
    if target_date is None:
        target_date = date.today() + timedelta(days=1)
    run_date = target_date - timedelta(days=1)

    actives = [p for p in get_city_universe() if p.active and p.city in polymarket_client.PM_CITIES]
    if not actives:
        print("No active cities overlap with the Polymarket registry.")
        return

    rows: list[dict] = []
    for city in actives:
        print(f"\n=== {city.city}  target={target_date} ===")
        points = _fetch_live_forecasts_t24(city, run_date)
        if not points:
            print("  No forecasts available — skipping.")
            continue
        ens = build_temperature_ensemble(
            city_profile=city,
            forecasts=points,
            target_date=target_date,
            metric=METRIC,
            ensemble_run_time_utc=points[0].run_time_utc,
        )
        std = max(ens.adjusted_std_f, _MIN_ENSEMBLE_STD_F)
        pm_station = polymarket_client.PM_CITIES[city.city]["station_id"]
        print(
            f"  Model: mean={ens.bias_corrected_mean_f:.1f}°F  std={std:.1f}°F  "
            f"models={ens.contributing_models}"
        )
        print(
            f"  NOTE: bias profile is for {city.station_id}; Polymarket settles on "
            f"{pm_station} — alphas there are indicative only."
        )

        pm_contracts = polymarket_client.fetch_live_contracts(
            "Polymarket", city=city.city, target_date=target_date, metric=METRIC,
        )
        if pm_contracts:
            _scan_venue(
                city, pm_contracts, polymarket_client.fetch_orderbook_for_contract,
                "Polymarket", ens.bias_corrected_mean_f, std, rows,
            )
        else:
            print("  Polymarket: no event found for this date.")

        try:
            k_contracts = [
                c for c in fetch_kalshi_contracts("Kalshi", series_ticker=city.kalshi_series or None)
                if c.target_date == target_date and c.metric == METRIC
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] Kalshi fetch failed: %s", city.city, exc)
            k_contracts = []
        if k_contracts:
            _scan_venue(
                city, k_contracts, fetch_kalshi_book,
                "Kalshi", ens.bias_corrected_mean_f, std, rows,
            )
        else:
            print("  Kalshi: no contracts found for this date.")

    if rows:
        _SCAN_CSV.parent.mkdir(parents=True, exist_ok=True)
        new_file = not _SCAN_CSV.exists()
        with _SCAN_CSV.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
            if new_file:
                writer.writeheader()
            writer.writerows(rows)
        print(f"\n{len(rows)} row(s) appended to {_SCAN_CSV.relative_to(_PROJECT_ROOT)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s %(message)s")
    parser = argparse.ArgumentParser(description="Model vs Polymarket/Kalshi price scan.")
    parser.add_argument("--date", help="Target date (YYYY-MM-DD). Default: tomorrow.")
    args = parser.parse_args()
    td = date.fromisoformat(args.date) if args.date else None
    main(td)
