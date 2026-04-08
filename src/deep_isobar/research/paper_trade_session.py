"""Daily paper trade session for Deep Isobar.

Fetches the current GFS T+24 forecast for Chicago's tomorrow high,
pulls live Kalshi bid/ask for the matching contracts, runs the
ensemble → probability → alpha pipeline, and logs signals to CSV.

Run once per day before the decision cutoff (~12:00 UTC / 7 am CDT)::

    python -m deep_isobar.research.paper_trade_session
    python -m deep_isobar.research.paper_trade_session --dry-run

Output files
------------
``data/paper_trades/paper_trades.csv``
    One row per trade logged (``abs(alpha) >= SIGNAL_THRESHOLD``).
    Status is initialised to ``OPEN``; settled by ``settle_paper_trades.py``.

``data/paper_trades/daily_log.csv``
    One row per contract evaluated regardless of signal strength.
    Useful for monitoring the pipeline on no-signal days.

CSV schema
----------
date, contract_ticker, direction, alpha, model_prob, market_prob,
entry_price, position_size, status, realized_pnl, settled_temp, threshold_f

``threshold_f`` is an extra column (beyond the spec minimum) stored so
``settle_paper_trades.py`` can settle without parsing the ticker string.

GFS forecast note
-----------------
The script tries the 12z run (f030, 30 h lead) first, then falls back
to the 00z run (f042, 42 h lead).  Both require cfgrib / xarray / eccodes::

    pip install eccodes cfgrib xarray
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")  # load KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PATH, etc.

from deep_isobar.core.types import ForecastPoint
from deep_isobar.data.city_universe import get_city_profile
from deep_isobar.data.historical_forecast_ingest import (
    _LEAD_FHOUR,
    _STATION_COORDS,
    _cache_path,
    _check_cfgrib,
    _download_snippet,
    _extract_fahrenheit,
    _gfs_urls,
)
from deep_isobar.market.kalshi_client import (
    fetch_live_contracts,
    fetch_orderbook_for_contract,
)
from deep_isobar.market.market_scanner import evaluate_contract_opportunity
from deep_isobar.market.microstructure_scanner import compute_microstructure_score
from deep_isobar.models.probability_engine import probability_for_contract
from deep_isobar.models.temperature_ensemble import build_temperature_ensemble
from deep_isobar.trading.distribution_tail_alpha import is_tail_threshold
from deep_isobar.notifications.discord_notifier import (
    COLOR_AMBER,
    COLOR_BLUE,
    COLOR_GRAY,
    COLOR_RED,
    post_embed,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PAPER_TRADES_DIR = _PROJECT_ROOT / "data" / "paper_trades"
_PAPER_TRADES_CSV = _PAPER_TRADES_DIR / "paper_trades.csv"
_DAILY_LOG_CSV = _PAPER_TRADES_DIR / "daily_log.csv"
_GFS_CACHE_DIR = _PROJECT_ROOT / "data" / "historical" / "forecasts" / ".grib_cache"

CITY = "Chicago"
SIGNAL_THRESHOLD = 0.25     # minimum |alpha| to generate a trade
POSITION_SIZE = 10.0        # contracts per trade (paper)
METRIC = "high_temp_f"
# Minimum ensemble std applied when only one GFS run is available.
# Matches the floor used in the Chicago backtest.
_MIN_ENSEMBLE_STD_F = 5.5

_CSV_COLUMNS = [
    "date",
    "contract_ticker",
    "direction",
    "alpha",
    "model_prob",
    "market_prob",
    "entry_price",
    "position_size",
    "status",
    "realized_pnl",
    "settled_temp",
    "threshold_f",
    "strike_type",
    "floor_strike",
    "cap_strike",
]


# ---------------------------------------------------------------------------
# GFS live fetch
# ---------------------------------------------------------------------------


def _fetch_live_gfs_t24(city_profile, run_date: date) -> list[ForecastPoint]:
    """Download GFS T+24 forecasts for *run_date*, targeting tomorrow's high.

    Tries 12z cycle (f030, 30 h lead) first, then 00z (f042, 42 h lead).
    Returns all successfully downloaded points so the ensemble builder
    can weight them.  An empty list means neither cycle was available.

    Args:
        city_profile: City profile for coordinate and station lookup.
        run_date: Today's date — the GFS model run date.

    Returns:
        List of :class:`~deep_isobar.core.types.ForecastPoint` objects.

    Raises:
        ImportError: If cfgrib / xarray / eccodes are not installed.
    """
    _check_cfgrib()

    station_id = city_profile.station_id
    city_lat, city_lon_360 = _STATION_COORDS.get(
        station_id, (41.7868, 272.2478)  # KMDW fallback
    )
    tomorrow = run_date + timedelta(days=1)
    points: list[ForecastPoint] = []

    for cycle in ("12", "00"):
        fhour = _LEAD_FHOUR[cycle][1]   # lead_day=1 → 18z UTC on target date
        run_time_utc = datetime(
            run_date.year, run_date.month, run_date.day,
            int(cycle), 0, 0, tzinfo=timezone.utc,
        )
        grib2_url, idx_url = _gfs_urls(run_date, cycle, fhour)
        dest = _cache_path(_GFS_CACHE_DIR, run_date, cycle, fhour)

        try:
            _download_snippet(grib2_url, idx_url, dest)
            t_f = _extract_fahrenheit(dest, city_lat, city_lon_360)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "GFS %sz f%03d unavailable (%s) — skipping", cycle, fhour, exc
            )
            continue

        logger.info(
            "GFS %sz f%03d: run=%s → %.1f°F  (target %s)",
            cycle, fhour, run_date, t_f, tomorrow,
        )
        points.append(ForecastPoint(
            city=city_profile.city,
            station_id=station_id,
            model_name="GFS",
            run_time_utc=run_time_utc,
            target_date=tomorrow,
            metric=METRIC,
            forecast_value_f=t_f,
            lead_hours=fhour,
            source_name="NOAA",
        ))

    return points


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def _ensure_csv(path: Path) -> None:
    """Create the CSV file with a header row if it does not already exist."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=_CSV_COLUMNS).writeheader()


def _append_csv_row(path: Path, row: dict) -> None:
    """Append one row dict to the CSV (header assumed to exist)."""
    with path.open("a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=_CSV_COLUMNS).writerow(row)


# ---------------------------------------------------------------------------
# Signal printer (dry-run)
# ---------------------------------------------------------------------------


def _print_signal(row: dict, signal, orderbook) -> None:
    """Print a single evaluated contract to stdout (dry-run mode)."""
    is_trade = row["status"] == "OPEN"
    tag = ">>> TRADE SIGNAL (would be logged)" if is_trade else "    no trade"
    bid = orderbook.best_bid
    ask = orderbook.best_ask

    st = row.get("strike_type", "")
    cap = row.get("cap_strike", "")
    flr = row.get("floor_strike", "")
    if st == "less":
        condition = f"YES if actual < {cap}\u00b0F"
    elif st == "greater":
        condition = f"YES if actual > {flr}\u00b0F"
    else:
        condition = "unknown"

    print(
        f"\n  Contract    : {row['contract_ticker']}\n"
        f"  Condition   : {condition}\n"
        f"  Direction   : {row['direction']}\n"
        f"  Alpha       : {float(row['alpha']):+.4f}\n"
        f"  Model prob  : {float(row['model_prob']):.4f}\n"
        f"  Market prob : {float(row['market_prob']):.4f}\n"
        f"  Bid / Ask   : {bid} / {ask}\n"
        f"  Entry price : {row['entry_price']}\n"
        f"  Conf. score : {signal.confidence_score:.3f}\n"
        f"  {tag}"
    )


# ---------------------------------------------------------------------------
# Main session
# ---------------------------------------------------------------------------


def run_session(dry_run: bool = False) -> int:
    """Run one paper trade session for tomorrow's Chicago high.

    Args:
        dry_run: When ``True``, print signals to stdout without writing to
            any CSV file.

    Returns:
        Number of trade signals found (logged when ``dry_run=False``).
    """
    today = date.today()
    tomorrow = today + timedelta(days=1)

    logger.info("=== Paper trade session  target_date=%s ===", tomorrow)

    # ── City profile ───────────────────────────────────────────────────────
    city_profile = get_city_profile(CITY)

    # ── GFS forecast ───────────────────────────────────────────────────────
    try:
        forecast_points = _fetch_live_gfs_t24(city_profile, today)
    except ImportError as exc:
        logger.error("cfgrib/xarray/eccodes not installed: %s", exc)
        sys.exit(1)

    if not forecast_points:
        logger.error(
            "No GFS T+24 forecast available for run_date=%s. "
            "Check that the 00z or 12z run has been posted to AWS "
            "(00z available ~05:00 UTC, 12z ~17:00 UTC).",
            today,
        )
        sys.exit(1)

    # ── Temperature ensemble ───────────────────────────────────────────────
    ensemble = build_temperature_ensemble(
        city_profile=city_profile,
        forecasts=forecast_points,
        target_date=tomorrow,
        metric=METRIC,
        ensemble_run_time_utc=forecast_points[0].run_time_utc,
    )
    effective_std = max(ensemble.adjusted_std_f, _MIN_ENSEMBLE_STD_F)

    logger.info(
        "Ensemble: raw_mean=%.1f°F  bias_corrected=%.1f°F  "
        "adjusted_std=%.1f°F  (effective=%.1f°F)",
        ensemble.ensemble_mean_f,
        ensemble.bias_corrected_mean_f,
        ensemble.adjusted_std_f,
        effective_std,
    )

    if not dry_run:
        post_embed(
            title="Deep Isobar \u2014 Morning run started",
            color=COLOR_BLUE,
            fields=[
                {"name": "Target date",        "value": str(tomorrow)},
                {"name": "GFS raw mean",       "value": f"{ensemble.ensemble_mean_f:.1f}\u00b0F"},
                {"name": "Bias-corrected mean","value": f"{ensemble.bias_corrected_mean_f:.1f}\u00b0F"},
                {"name": "Effective std",      "value": f"{effective_std:.1f}\u00b0F"},
            ],
        )

    # ── Live Kalshi contracts for tomorrow ────────────────────────────────
    try:
        all_contracts = fetch_live_contracts("Kalshi")
        tomorrow_contracts = [
            c for c in all_contracts
            if c.target_date == tomorrow
            and c.metric == METRIC
            and c.strike_type in ("less", "greater")
        ]

        if not tomorrow_contracts:
            logger.warning(
                "No active Kalshi high_temp_f contracts found for %s — "
                "market may not be listed yet.",
                tomorrow,
            )
            # Case 1: fetch returned nothing, or nothing matched the
            # tomorrow / high_temp_f / less|greater filters.  The "No signals
            # today" embed further below covers the distinct case where
            # contracts exist but none clear the alpha threshold.
            if not dry_run:
                post_embed(
                    title="No contracts found",
                    description=(
                        f"No active Kalshi high_temp_f contracts for {tomorrow}. "
                        "The market may not be listed yet."
                    ),
                    color=COLOR_GRAY,
                )
            return 0

        logger.info(
            "Found %d contracts for %s: thresholds=%s",
            len(tomorrow_contracts),
            tomorrow,
            sorted(c.threshold_f for c in tomorrow_contracts),
        )

        # ── Probability surface ────────────────────────────────────────────────
        # Compute one model probability per contract using the correct formula
        # for each strike_type.  The dict is keyed by threshold_f (unique after
        # "between" contracts are filtered) for consumption by market_scanner.
        probability_surface: dict[int, float] = {}
        for _c in tomorrow_contracts:
            probability_surface[_c.threshold_f] = probability_for_contract(
                strike_type=_c.strike_type,
                floor_strike=_c.floor_strike,
                cap_strike=_c.cap_strike,
                mean_f=ensemble.bias_corrected_mean_f,
                std_f=effective_std,
            )

        def _surface_label(c) -> str:
            if c.strike_type == "less":
                return f"P(T<{c.cap_strike})"
            if c.strike_type == "greater":
                return f"P(T>{c.floor_strike})"
            return f"P(?{c.threshold_f})"

        logger.info(
            "Probability surface: %s",
            {_surface_label(_c): f"{probability_surface[_c.threshold_f]:.3f}"
             for _c in sorted(tomorrow_contracts, key=lambda x: x.threshold_f)},
        )

        # ── Evaluate each contract ─────────────────────────────────────────────
        # Collect all rows first so we can deduplicate before writing.
        all_rows: list[dict] = []
        dry_run_signals: list[tuple[dict, object, object]] = []  # (row, signal, orderbook)

        for contract in sorted(tomorrow_contracts, key=lambda c: c.threshold_f):
            orderbook = fetch_orderbook_for_contract("Kalshi", contract.contract_id)
            now_utc = datetime.now(timezone.utc)  # refresh after network call

            micro_score = compute_microstructure_score(orderbook, now_utc)
            tail_flag = is_tail_threshold(
                ensemble_mean_f=ensemble.bias_corrected_mean_f,
                threshold_f=contract.threshold_f,
                adjusted_std_f=effective_std,
            )

            signal = evaluate_contract_opportunity(
                contract=contract,
                probability_surface=probability_surface,
                orderbook=orderbook,
                signal_threshold=SIGNAL_THRESHOLD,
                timestamp_utc=now_utc,
                microstructure_score=micro_score,
                tail_opportunity_flag=tail_flag,
                tail_multiplier=city_profile.tail_multiplier,
            )

            # Entry price: cross the spread at execution side
            if signal.signal_side == "BUY":
                entry_price = orderbook.best_ask
            elif signal.signal_side == "SELL":
                entry_price = orderbook.best_bid
            else:
                # HOLD — record mid for the daily log
                if orderbook.best_bid is not None and orderbook.best_ask is not None:
                    entry_price = (orderbook.best_bid + orderbook.best_ask) / 2.0
                else:
                    entry_price = orderbook.best_bid or orderbook.best_ask

            is_trade = (
                signal.signal_side != "HOLD"
                and abs(signal.alpha) >= SIGNAL_THRESHOLD
            )

            row: dict = {
                "date": str(tomorrow),
                "contract_ticker": contract.contract_id,
                "direction": signal.signal_side,
                "alpha": round(signal.alpha, 6),
                "model_prob": round(signal.model_probability, 6),
                "market_prob": round(signal.market_probability, 6),
                "entry_price": round(entry_price, 6) if entry_price is not None else "",
                "position_size": POSITION_SIZE,
                "status": "OPEN" if is_trade else "NO_SIGNAL",
                "realized_pnl": "",
                "settled_temp": "",
                "threshold_f": contract.threshold_f,
                "strike_type": contract.strike_type,
                "floor_strike": contract.floor_strike if contract.floor_strike is not None else "",
                "cap_strike":   contract.cap_strike   if contract.cap_strike   is not None else "",
            }

            all_rows.append(row)
            if dry_run:
                dry_run_signals.append((row, signal, orderbook))

        # ── Deduplicate trade signals ──────────────────────────────────────────
        # For each (threshold_f, direction) pair keep only the highest-alpha signal.
        # key → best row seen so far
        best_by_key: dict[tuple, dict] = {}
        for row in all_rows:
            if row["status"] != "OPEN":
                continue
            key = (row["threshold_f"], row["direction"])
            existing = best_by_key.get(key)
            if existing is None or abs(row["alpha"]) > abs(existing["alpha"]):
                if existing is not None:
                    logger.warning(
                        "Dropping duplicate signal: %s (threshold=%.0f°F already covered"
                        " by %s with higher alpha)",
                        existing["contract_ticker"],
                        existing["threshold_f"],
                        row["contract_ticker"],
                    )
                best_by_key[key] = row
            else:
                logger.warning(
                    "Dropping duplicate signal: %s (threshold=%.0f°F already covered"
                    " by %s with higher alpha)",
                    row["contract_ticker"],
                    row["threshold_f"],
                    existing["contract_ticker"],
                )

        # Mark dropped duplicates so the daily log reflects the dedup decision
        winning_tickers = {r["contract_ticker"] for r in best_by_key.values()}
        for row in all_rows:
            if row["status"] == "OPEN" and row["contract_ticker"] not in winning_tickers:
                row["status"] = "DEDUP_DROP"

        # ── Dry-run output ─────────────────────────────────────────────────────
        if dry_run:
            signals_logged = 0
            for row, signal, orderbook in dry_run_signals:
                _print_signal(row, signal, orderbook)
                if row["status"] == "OPEN":
                    signals_logged += 1
            return signals_logged

        # ── Discord notifications ──────────────────────────────────────────────
        open_rows = [r for r in all_rows if r["status"] == "OPEN"]
        if open_rows:
            for row in open_rows:
                entry_str = f"{row['entry_price']}" if row["entry_price"] != "" else "—"
                post_embed(
                    title=f"Signal: {row['contract_ticker']}",
                    color=COLOR_AMBER,
                    fields=[
                        {"name": "Direction",    "value": row["direction"]},
                        {"name": "Alpha",        "value": f"{row['alpha']:+.4f}"},
                        {"name": "Model prob",   "value": f"{row['model_prob']:.4f}"},
                        {"name": "Market prob",  "value": f"{row['market_prob']:.4f}"},
                        {"name": "Entry price",  "value": entry_str},
                        {"name": "Threshold",    "value": f"{row['threshold_f']:.0f}\u00b0F"},
                    ],
                )
        else:
            post_embed(
                title="No signals today",
                description=f"No contracts met the |alpha| \u2265 {SIGNAL_THRESHOLD} threshold for {tomorrow}.",
                color=COLOR_GRAY,
            )

        # ── Write to CSV files ─────────────────────────────────────────────────
        signals_logged = 0
        _ensure_csv(_PAPER_TRADES_CSV)
        _ensure_csv(_DAILY_LOG_CSV)

        for row in all_rows:
            # Always record every evaluated contract in the daily log
            _append_csv_row(_DAILY_LOG_CSV, row)

            if row["status"] == "OPEN":
                _append_csv_row(_PAPER_TRADES_CSV, row)
                signals_logged += 1
                logger.info(
                    "TRADE LOGGED: %s %s  alpha=%+.3f  entry=%s  threshold=%.0f°F",
                    row["direction"],
                    row["contract_ticker"],
                    row["alpha"],
                    row["entry_price"] or 0.0,
                    row["threshold_f"],
                )
            else:
                logger.info(
                    "No trade: %-35s  alpha=%+.3f  side=%s  status=%s",
                    row["contract_ticker"],
                    row["alpha"],
                    row["direction"],
                    row["status"],
                )

        logger.info(
            "Session complete: %d trade(s) logged to %s",
            signals_logged, _PAPER_TRADES_CSV,
        )

        return signals_logged

    except Exception as exc:  # noqa: BLE001
        # Discord embed field values are capped at 1024 chars; 8 reserved for
        # the ``` fences added below.  Shed outermost lines until it fits so
        # we never cut mid-line (innermost frames are most useful).
        tb_lines = traceback.format_exc().splitlines()[-20:]
        while tb_lines and len("\n".join(tb_lines)) > 1016:
            tb_lines.pop(0)
        tb_truncated = "\n".join(tb_lines)
        exc_type = type(exc).__name__

        logger.exception("Unhandled exception in contract evaluation loop")

        try:
            post_embed(
                title="Session crashed \u2014 unhandled exception",
                color=COLOR_RED,
                fields=[
                    {"name": "Exception type", "value": exc_type},
                    {"name": "Message",        "value": str(exc) or "(no message)"},
                    {"name": "Traceback (tail)", "value": f"```\n{tb_truncated}\n```"},
                ],
            )
        except Exception:  # noqa: BLE001
            logger.warning("Failed to send error embed to Discord", exc_info=True)

        sys.exit(1)


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
            "Daily paper trade session for Chicago high temperature contracts.\n\n"
            "Fetches GFS T+24 forecast, evaluates live Kalshi contracts, and\n"
            "logs any trade signals (|alpha| >= 0.25) to data/paper_trades/."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print signals to stdout without writing to any CSV file.",
    )
    args = parser.parse_args()

    today = date.today()
    tomorrow = today + timedelta(days=1)

    if args.dry_run:
        print(f"\n[DRY RUN] Paper trade session — {today}")
        print(f"  Target date      : {tomorrow}")
        print(f"  Signal threshold : |alpha| >= {SIGNAL_THRESHOLD}")
        print(f"  Position size    : {POSITION_SIZE} contracts")
        print("-" * 55)

    n = run_session(dry_run=args.dry_run)

    if args.dry_run:
        print(f"\n[DRY RUN] {n} trade signal(s) identified — nothing written to disk.")
    else:
        print(f"\nDone. {n} trade signal(s) logged.")
