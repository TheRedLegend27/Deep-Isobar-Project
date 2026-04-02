"""Settle paper trades against NOAA actual daily high temperature.

Run after ~18:00 local time (6 PM CDT) once NOAA/ACIS has posted the
official daily high for Chicago::

    python -m deep_isobar.research.settle_paper_trades
    python -m deep_isobar.research.settle_paper_trades --date 2026-04-01

Pipeline
--------
1. Read ``data/paper_trades/paper_trades.csv``.
2. Find OPEN rows whose ``date`` matches the settle date (default: today).
3. Fetch the actual high_temp_f from NOAA/ACIS for that date.
4. For each open trade determine WIN or LOSS:

   - **BUY** (long YES):  WIN if ``actual_high >= threshold_f``
   - **SELL** (short YES): WIN if ``actual_high < threshold_f``

5. Compute realized P&L, applying the 7% Kalshi fee to winning trades:

   - BUY WIN:  ``(1 - entry_price) × qty × 0.93``
   - BUY LOSS: ``-entry_price × qty``
   - SELL WIN: ``entry_price × qty × 0.93``
   - SELL LOSS: ``-(1 - entry_price) × qty``

6. Write ``status``, ``realized_pnl``, and ``settled_temp`` back to the CSV.
7. Print a running P&L summary to stdout.

ACIS availability note
----------------------
NOAA/ACIS typically posts the official daily high by 20:00–23:00 local
time, though Chicago (KORD) is usually available by ~18:00 CDT.  If the
script reports missing data, wait 30–60 minutes and retry.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")  # load KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PATH, etc.

from deep_isobar.data.historical_noaa_ingest import fetch_settlement_observations

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PAPER_TRADES_CSV = _PROJECT_ROOT / "data" / "paper_trades" / "paper_trades.csv"

CITY = "Chicago"
KALSHI_FEE_RATE = 0.07   # 7% fee deducted from gross profit on winning trades
_DEFAULT_POSITION_SIZE = 10.0


# ---------------------------------------------------------------------------
# NOAA settlement fetch
# ---------------------------------------------------------------------------


def _fetch_noaa_actual(settle_date: date) -> float | None:
    """Return the official NOAA high_temp_f for *settle_date* from ACIS.

    Uses :func:`~deep_isobar.data.historical_noaa_ingest.fetch_settlement_observations`
    which calls the ACIS StnData endpoint — no API key required.

    Args:
        settle_date: The date to fetch the observed high for.

    Returns:
        High temperature in °F, or ``None`` if data is unavailable.
    """
    try:
        df = fetch_settlement_observations(
            city=CITY,
            start_date=settle_date,
            end_date=settle_date,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("ACIS fetch failed for %s: %s", settle_date, exc)
        return None

    if df.empty:
        logger.warning("ACIS returned no rows for %s", settle_date)
        return None

    df = df[df["quality_flag"] != "missing"]
    if df.empty:
        logger.warning("ACIS data for %s has quality_flag=missing", settle_date)
        return None

    return float(df.iloc[0]["high_temp_f"])


# ---------------------------------------------------------------------------
# P&L calculation
# ---------------------------------------------------------------------------


def _compute_pnl(
    direction: str,
    entry_price: float,
    realized_outcome: int,
    position_size: float,
    fee_rate: float = KALSHI_FEE_RATE,
) -> float:
    """Compute net realized P&L for one paper trade after Kalshi fee.

    Settlement convention (Kalshi YES binary):
    - ``realized_outcome = 1``  → YES settled (high >= threshold)
    - ``realized_outcome = 0``  → NO settled  (high <  threshold)

    Kalshi fee applies only to winning trades (gross P&L > 0).

    Args:
        direction: ``"BUY"`` (long YES) or ``"SELL"`` (short YES).
        entry_price: Price paid / received per contract, in [0, 1].
        realized_outcome: 1 if YES settled, 0 if NO settled.
        position_size: Number of contracts.
        fee_rate: Kalshi fee on winning trades (default 0.07).

    Returns:
        Net realized P&L in probability-point × contracts units.
    """
    if direction == "BUY":
        pnl_per_unit = realized_outcome - entry_price
    elif direction == "SELL":
        pnl_per_unit = entry_price - realized_outcome
    else:
        return 0.0

    gross_pnl = pnl_per_unit * position_size
    if gross_pnl > 0:
        return gross_pnl * (1.0 - fee_rate)
    return gross_pnl


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------


def _settle_open_trades(
    df: pd.DataFrame,
    settle_date: date,
    settled_temp: float,
) -> tuple[pd.DataFrame, int]:
    """Mark OPEN trades for *settle_date* as WIN/LOSS and populate PnL fields.

    Args:
        df: Full paper_trades DataFrame.
        settle_date: Date being settled.
        settled_temp: Observed high temperature in °F.

    Returns:
        ``(updated_df, n_settled)`` where *n_settled* is the count of rows
        that were updated.
    """
    df = df.copy()

    open_mask = (
        (df["date"].astype(str) == str(settle_date))
        & (df["status"] == "OPEN")
    )

    n_settled = 0
    for idx in df[open_mask].index:
        row = df.loc[idx]

        threshold_f = float(row["threshold_f"])
        entry_price = float(row["entry_price"])
        direction = str(row["direction"])
        position_size = (
            float(row["position_size"])
            if pd.notna(row.get("position_size"))
            else _DEFAULT_POSITION_SIZE
        )

        # Kalshi KXHIGHCHI: YES settles at 1 if actual_high >= threshold_f
        realized_outcome = 1 if settled_temp >= threshold_f else 0

        pnl = _compute_pnl(direction, entry_price, realized_outcome, position_size)

        # Label outcome from the trade's perspective
        if direction == "BUY":
            status = "WIN" if realized_outcome == 1 else "LOSS"
        else:  # SELL
            status = "WIN" if realized_outcome == 0 else "LOSS"

        df.loc[idx, "status"] = status
        df.loc[idx, "realized_pnl"] = round(pnl, 6)
        df.loc[idx, "settled_temp"] = settled_temp
        n_settled += 1

    return df, n_settled


# ---------------------------------------------------------------------------
# P&L summary
# ---------------------------------------------------------------------------


def _print_pnl_summary(df: pd.DataFrame) -> None:
    """Print a running P&L summary over all settled trades to stdout."""
    settled = df[df["status"].isin(["WIN", "LOSS"])].copy()

    if settled.empty:
        print("\n  No settled trades in paper_trades.csv.")
        return

    settled["realized_pnl"] = pd.to_numeric(
        settled["realized_pnl"], errors="coerce"
    ).fillna(0.0)

    total = len(settled)
    wins = int((settled["status"] == "WIN").sum())
    losses = int((settled["status"] == "LOSS").sum())
    win_rate = wins / total if total > 0 else 0.0
    running_pnl = settled["realized_pnl"].sum()

    print()
    print("=" * 58)
    print("  PAPER TRADE P&L SUMMARY  (all settled trades)")
    print("=" * 58)
    print(f"  Total trades   : {total}")
    print(f"  Wins / Losses  : {wins} / {losses}")
    print(f"  Win rate       : {win_rate * 100:.1f}%")
    print(
        f"  Running P&L    : {running_pnl:+.4f}"
        f"  (prob-pts × contracts, after {KALSHI_FEE_RATE:.0%} fee)"
    )

    if "date" in settled.columns:
        print("\n  By date:")
        by_date = (
            settled.groupby("date")["realized_pnl"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "pnl", "count": "trades"})
            .sort_index()
        )
        for dt, agg_row in by_date.iterrows():
            print(
                f"    {dt}  trades={int(agg_row['trades'])}  "
                f"pnl={agg_row['pnl']:+.4f}"
            )

    print("=" * 58)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def settle(settle_date: date) -> None:
    """Settle all OPEN paper trades for *settle_date*.

    Args:
        settle_date: The date to settle (typically today, after ~18:00 local).

    Exits with code 1 if ``paper_trades.csv`` is missing or the NOAA actual
    cannot be fetched.
    """
    if not _PAPER_TRADES_CSV.exists():
        logger.error(
            "paper_trades.csv not found at %s\n"
            "Run paper_trade_session.py first.",
            _PAPER_TRADES_CSV,
        )
        sys.exit(1)

    df = pd.read_csv(
        _PAPER_TRADES_CSV,
        dtype={"threshold_f": float, "position_size": float},
    )

    open_today = df[
        (df["date"].astype(str) == str(settle_date)) & (df["status"] == "OPEN")
    ]

    if open_today.empty:
        logger.info("No OPEN trades for %s — nothing to settle.", settle_date)
        _print_pnl_summary(df)
        return

    logger.info(
        "Fetching NOAA actual high_temp_f for %s (Chicago / KORD)…",
        settle_date,
    )
    settled_temp = _fetch_noaa_actual(settle_date)

    if settled_temp is None:
        logger.error(
            "NOAA actual high_temp_f not yet available for %s.\n"
            "ACIS typically posts by 18:00–23:00 CDT. Try again later.",
            settle_date,
        )
        sys.exit(1)

    logger.info("NOAA actual high_temp_f for %s: %.1f°F", settle_date, settled_temp)

    df, n_settled = _settle_open_trades(df, settle_date, settled_temp)

    # Write updated CSV
    df.to_csv(_PAPER_TRADES_CSV, index=False)
    logger.info("Wrote %d settled row(s) to %s", n_settled, _PAPER_TRADES_CSV)

    # Print newly settled trades
    newly_settled = df[
        (df["date"].astype(str) == str(settle_date))
        & df["status"].isin(["WIN", "LOSS"])
    ]
    if not newly_settled.empty:
        print(f"\n  Settled {len(newly_settled)} trade(s) for {settle_date}:")
        for _, trade_row in newly_settled.iterrows():
            print(
                f"    {trade_row['direction']:<4}  "
                f"{trade_row['contract_ticker']:<40}  "
                f"thr={trade_row['threshold_f']:.0f}°F  "
                f"actual={settled_temp:.1f}°F  "
                f"{trade_row['status']:<4}  "
                f"pnl={float(trade_row['realized_pnl']):+.4f}"
            )

    _print_pnl_summary(df)


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
            "Settle paper trades against NOAA actual daily high temperature.\n\n"
            "Run after ~18:00 local time when ACIS has posted the official\n"
            "daily high for Chicago (KORD)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--date",
        default=str(date.today()),
        metavar="YYYY-MM-DD",
        help="Date to settle (default: today).",
    )
    args = parser.parse_args()

    try:
        settle_date = date.fromisoformat(args.date)
    except ValueError:
        print(
            f"ERROR: invalid date {args.date!r}. Use YYYY-MM-DD format.",
            file=sys.stderr,
        )
        sys.exit(1)

    settle(settle_date)
