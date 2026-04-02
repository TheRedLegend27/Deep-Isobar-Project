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
from deep_isobar.notifications.discord_notifier import (
    COLOR_GREEN,
    COLOR_RED,
    COLOR_BLUE,
    post_embed,
)

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

    def _parse_optional_int(val) -> int | None:
        """Parse an integer from a CSV cell that may be empty or NaN."""
        try:
            s = str(val).strip()
            if s in ("", "nan", "None"):
                return None
            return int(float(s))
        except (ValueError, TypeError):
            return None

    n_settled = 0
    for idx in df[open_mask].index:
        row = df.loc[idx]

        entry_price = float(row["entry_price"])
        direction = str(row["direction"])
        position_size = (
            float(row["position_size"])
            if pd.notna(row.get("position_size"))
            else _DEFAULT_POSITION_SIZE
        )

        # Determine the YES-settlement outcome from strike_type.
        # Rows written before this fix may lack strike_type; fall back to
        # parsing the ticker for the bracket letter.
        strike_type = str(row.get("strike_type", "") or "").lower().strip()
        if not strike_type:
            # Legacy fallback: infer from ticker name
            ticker = str(row.get("contract_ticker", ""))
            import re as _re
            m = _re.search(r"-([BT])[\d.]+$", ticker, _re.IGNORECASE)
            if m and m.group(1).upper() == "T":
                # T-contracts: lower T = "less" edge, upper T = "greater" edge.
                # Without cap/floor data we can't distinguish; default to "less"
                # (T66 was a "less" contract) and log a warning.
                strike_type = "less"
                logger.warning(
                    "settle: missing strike_type for %r — inferred 'less'; "
                    "verify settlement is correct.",
                    ticker,
                )
            else:
                logger.warning(
                    "settle: cannot determine strike_type for %r — skipping.",
                    row.get("contract_ticker"),
                )
                continue

        floor_strike = _parse_optional_int(row.get("floor_strike"))
        cap_strike   = _parse_optional_int(row.get("cap_strike"))

        # Kalshi settlement rules by strike_type:
        #   less    → YES if actual < cap_strike    (e.g. T66: actual < 66)
        #   greater → YES if actual > floor_strike  (e.g. T67: actual > 67)
        if strike_type == "less":
            if cap_strike is None:
                # Legacy fallback: use threshold_f as cap_strike
                cap_strike = int(float(row["threshold_f"]))
                logger.warning(
                    "settle: missing cap_strike for %r — using threshold_f=%s",
                    row.get("contract_ticker"), cap_strike,
                )
            realized_outcome = 1 if settled_temp < cap_strike else 0
        elif strike_type == "greater":
            if floor_strike is None:
                floor_strike = int(float(row["threshold_f"]))
                logger.warning(
                    "settle: missing floor_strike for %r — using threshold_f=%s",
                    row.get("contract_ticker"), floor_strike,
                )
            realized_outcome = 1 if settled_temp > floor_strike else 0
        else:
            logger.warning(
                "settle: unexpected strike_type %r for %r — skipping.",
                strike_type, row.get("contract_ticker"),
            )
            continue

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

    # Safety net: warn about any B-type contracts that slipped into the CSV
    # before the _parse_contract fix.  Do not attempt to settle them.
    b_contracts = df[df["contract_ticker"].str.contains(r"-B", na=False)]
    if not b_contracts.empty:
        for _, b_row in b_contracts.iterrows():
            logger.warning(
                "settle: skipping bracket contract %r (status=%s) — "
                "B-type settlement logic is not implemented; mark manually.",
                b_row["contract_ticker"], b_row["status"],
            )

    open_today = df[
        (df["date"].astype(str) == str(settle_date))
        & (df["status"] == "OPEN")
        & ~df["contract_ticker"].str.contains(r"-B", na=False)
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

        # ── Per-trade Discord embeds ───────────────────────────────────────
        for _, trade_row in newly_settled.iterrows():
            is_win = trade_row["status"] == "WIN"
            pnl = float(trade_row["realized_pnl"])
            post_embed(
                title=f"{'WIN' if is_win else 'LOSS'} \u2014 {trade_row['contract_ticker']}",
                color=COLOR_GREEN if is_win else COLOR_RED,
                fields=[
                    {"name": "Settled temp",  "value": f"{settled_temp:.1f}\u00b0F"},
                    {"name": "Threshold",     "value": f"{trade_row['threshold_f']:.0f}\u00b0F"},
                    {"name": "Realized P&L",  "value": f"{pnl:+.4f}"},
                    {"name": "Entry price",   "value": f"{float(trade_row['entry_price']):.4f}"},
                ],
            )

        # ── Summary embed ─────────────────────────────────────────────────
        all_settled = df[df["status"].isin(["WIN", "LOSS"])].copy()
        all_settled["realized_pnl"] = pd.to_numeric(
            all_settled["realized_pnl"], errors="coerce"
        ).fillna(0.0)
        n_total = len(all_settled)
        n_wins  = int((all_settled["status"] == "WIN").sum())
        session_pnl    = float(newly_settled["realized_pnl"].astype(float).sum())
        cumulative_pnl = float(all_settled["realized_pnl"].sum())
        post_embed(
            title=f"Settlement complete \u2014 {settle_date}",
            color=COLOR_BLUE,
            fields=[
                {"name": "Trades settled",  "value": str(n_settled)},
                {"name": "Wins",            "value": str(n_wins)},
                {"name": "Losses",          "value": str(n_total - n_wins)},
                {"name": "Session P&L",     "value": f"{session_pnl:+.4f}"},
                {"name": "Cumulative P&L",  "value": f"{cumulative_pnl:+.4f}"},
            ],
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
