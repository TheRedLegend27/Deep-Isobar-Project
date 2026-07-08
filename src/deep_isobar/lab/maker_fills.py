"""Maker-fill simulator — would resting limit orders have filled?

Every simulated fill today is a TAKER fill (cross the spread, pay Kalshi's
0.07·P·(1−P) per contract).  Makers pay nothing.  This module replays the
collector's orderbook snapshots to estimate what maker execution would have
done to our actual trades.

**Conservative fill rule (trade-through):** with periodic snapshots and no
trade tape, queue position is unknowable.  A resting order therefore counts
as filled ONLY when the opposite side later moves STRICTLY PAST the limit:

- resting BUY at P fills iff some later ``best_ask < P``
- resting SELL at P fills iff some later ``best_bid > P``

A touch (ask == P) is never a fill.  This under-counts fills — reported
maker savings are a floor, not an estimate.

Policies compared per trade:

- ``taker`` — what production does: buy at ask / sell at bid, pay the fee.
- ``join``  — rest at the same-side touch (bid for buys, ask for sells).
- ``improve`` — rest one tick inside the spread (bid+0.01 / ask−0.01);
  fills more often, gives back one tick of price.

CLI::

    python -m deep_isobar.lab.maker_fills            # report on real trades
"""

from __future__ import annotations

import logging
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_HISTORY_DIR = _PROJECT_ROOT / "data" / "market_history"
_PAPER_TRADES_CSV = _PROJECT_ROOT / "data" / "paper_trades" / "paper_trades.csv"

TICK = 0.01
FEE_RATE = 0.07  # taker fee coefficient: fee = 0.07 * P * (1-P) per contract


def load_books(history_dir: Path | None = None) -> pd.DataFrame:
    """All collector snapshots, sorted by time (may span many day dirs)."""
    base = history_dir or _HISTORY_DIR
    frames = [pd.read_parquet(f) for d in sorted(base.glob("date=*"))
              for f in sorted(d.glob("books_*.parquet"))]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["snapshot_utc"] = pd.to_datetime(df["snapshot_utc"], utc=True)
    return df.sort_values("snapshot_utc").reset_index(drop=True)


def simulate_resting_order(
    contract_books: pd.DataFrame,
    placed_at: datetime,
    side: str,
    limit_price: float,
) -> tuple[bool, datetime | None]:
    """Apply the trade-through rule to one contract's snapshot series.

    Args:
        contract_books: Snapshots for ONE contract (any order).
        placed_at: UTC time the resting order goes live.
        side: ``"BUY"`` or ``"SELL"`` (of YES).
        limit_price: Resting limit in [0, 1].

    Returns:
        ``(filled, fill_time)`` — fill_time is the first snapshot showing
        trade-through, None when never filled.
    """
    after = contract_books[contract_books["snapshot_utc"] > placed_at]
    for _, row in after.sort_values("snapshot_utc").iterrows():
        if side == "BUY":
            ask = row["best_ask"]
            if pd.notna(ask) and ask < limit_price - 1e-9:
                return True, row["snapshot_utc"]
        else:
            bid = row["best_bid"]
            if pd.notna(bid) and bid > limit_price + 1e-9:
                return True, row["snapshot_utc"]
    return False, None


def _maker_limit(side: str, bid: float, ask: float, policy: str) -> float | None:
    if pd.isna(bid) or pd.isna(ask):
        return None
    if policy == "join":
        return bid if side == "BUY" else ask
    if policy == "improve":
        px = bid + TICK if side == "BUY" else ask - TICK
        # never cross: an "improve" that meets the far touch is a taker order
        if side == "BUY" and px >= ask:
            return None
        if side == "SELL" and px <= bid:
            return None
        return round(px, 4)
    raise ValueError(f"unknown policy {policy!r}")


def _placement_time(target_date: date) -> datetime:
    """Session places orders ~07:00 ET on D-1 for target date D."""
    et = ZoneInfo("America/New_York")
    d = target_date - timedelta(days=1)
    return datetime.combine(d, time(7, 0), tzinfo=et).astimezone(timezone.utc)


def evaluate_trades(
    trades: pd.DataFrame,
    books: pd.DataFrame,
    policy: str,
) -> pd.DataFrame:
    """Simulate one maker policy across real (non-VOID) trade rows.

    Per trade: the maker limit derived from the first snapshot at/after
    placement, the trade-through fill verdict, price improvement vs the
    taker entry, and the fee avoided.  Trades with no book coverage are
    skipped (collector started 2026-07-03).
    """
    out: list[dict] = []
    for _, t in trades.iterrows():
        cb = books[books["contract_id"] == t["contract_ticker"]]
        if cb.empty:
            continue
        placed = _placement_time(date.fromisoformat(str(t["date"])))
        at_or_after = cb[cb["snapshot_utc"] >= placed]
        if at_or_after.empty:
            continue
        first = at_or_after.iloc[0]
        # Order goes live at the first snapshot we can actually see.
        placed_eff = first["snapshot_utc"]
        limit = _maker_limit(t["direction"], first["best_bid"], first["best_ask"], policy)
        if limit is None or not (0.0 < limit < 1.0):
            continue

        filled, fill_time = simulate_resting_order(cb, placed_eff, t["direction"], limit)
        # Taker reference from the SAME snapshot the limit was derived from —
        # comparing against the CSV's session-time entry mixes in market
        # drift between session and snapshot and corrupts the economics.
        taker_ref = first["best_ask"] if t["direction"] == "BUY" else first["best_bid"]
        if pd.isna(taker_ref):
            continue
        taker_entry = float(taker_ref)
        qty = float(t["position_size"])
        # Price improvement: buys pay less, sells receive more.
        improvement = (taker_entry - limit) if t["direction"] == "BUY" else (limit - taker_entry)
        fee_saved = FEE_RATE * taker_entry * (1 - taker_entry)
        out.append({
            "contract_ticker": t["contract_ticker"],
            "direction": t["direction"],
            "status": t["status"],
            "qty": qty,
            "taker_entry": taker_entry,
            "maker_limit": limit,
            "filled": filled,
            "hours_to_fill": (
                (fill_time - placed_eff).total_seconds() / 3600.0 if filled else None
            ),
            "improvement_per_contract": round(improvement, 4),
            "fee_saved_per_contract": round(fee_saved, 4),
            "maker_gain_usd": round((improvement + fee_saved) * qty, 2) if filled else 0.0,
            "realized_pnl": t.get("realized_pnl"),
        })
    return pd.DataFrame(out)


def report(history_dir: Path | None = None, trades_csv: Path | None = None) -> str:
    books = load_books(history_dir)
    trades = pd.read_csv(trades_csv or _PAPER_TRADES_CSV)
    trades = trades[trades["status"] != "VOID"]
    if books.empty or trades.empty:
        return "No books or no trades to evaluate."

    lines = ["Maker-fill simulation (conservative trade-through rule — savings are a FLOOR)", ""]
    for policy in ("join", "improve"):
        ev = evaluate_trades(trades, books, policy)
        if ev.empty:
            lines.append(f"policy={policy}: no evaluable trades")
            continue
        filled = ev[ev["filled"]]
        missed = ev[~ev["filled"]]
        missed_pnl = pd.to_numeric(missed["realized_pnl"], errors="coerce").fillna(0.0).sum()
        lines += [
            f"policy={policy}: {len(filled)}/{len(ev)} filled "
            f"({len(filled)/len(ev):.0%}); median time-to-fill "
            f"{filled['hours_to_fill'].median():.1f}h" if len(filled) else
            f"policy={policy}: 0/{len(ev)} filled",
            f"  maker gain on filled (price improvement + fee): "
            f"${filled['maker_gain_usd'].sum():+.2f}",
            f"  missed trades: {len(missed)}  (their realized taker P&L: "
            f"${missed_pnl:+.2f} — what a maker-only book would have skipped)",
            "",
        ]
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
