"""Cross-verify our settlement grading against Kalshi's official results.

Three contract-semantics bugs in three days (2026-07-03..05: probability-
surface collision, halved bracket probabilities, backwards-capable strike
fallback) all shared one root cause: conventions we inferred instead of
verified.  This job closes the loop permanently — every WIN/LOSS row we
grade is checked against the exchange's own ``result`` field, and any
mismatch is loud.

A mismatch means our strike conventions, settlement window, or observation
source disagree with the exchange — exactly the class of silent corruption
that poisons the track record and, later, real money.

Runs daily after settlement (supervisor job).  Verified tickers are cached
in ``data/paper_trades/settlement_verified.csv`` so each is checked once.

CLI::

    python -m deep_isobar.research.verify_settlements
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_PROJECT_ROOT / ".env")

from deep_isobar.market.kalshi_client import fetch_market_result, is_live_mode
from deep_isobar.notifications.discord_notifier import COLOR_RED, post_embed

logger = logging.getLogger(__name__)

_PAPER_TRADES_CSV = _PROJECT_ROOT / "data" / "paper_trades" / "paper_trades.csv"
_VERIFIED_CSV = _PROJECT_ROOT / "data" / "paper_trades" / "settlement_verified.csv"


def our_yes_outcome(direction: str, status: str) -> int | None:
    """Our graded YES outcome implied by (direction, WIN/LOSS)."""
    if status == "WIN":
        return 1 if direction == "BUY" else 0
    if status == "LOSS":
        return 0 if direction == "BUY" else 1
    return None


def _load_verified() -> set[str]:
    if not _VERIFIED_CSV.exists():
        return set()
    try:
        return set(pd.read_csv(_VERIFIED_CSV)["contract_ticker"])
    except Exception:  # noqa: BLE001
        logger.exception("could not read %s — reverifying all", _VERIFIED_CSV)
        return set()


def _append_verified(rows: list[dict]) -> None:
    if not rows:
        return
    new = pd.DataFrame(rows)
    if _VERIFIED_CSV.exists():
        new = pd.concat([pd.read_csv(_VERIFIED_CSV), new], ignore_index=True)
    _VERIFIED_CSV.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(_VERIFIED_CSV, index=False)


def verify_settlements() -> int:
    """Check every un-verified WIN/LOSS row; returns the mismatch count."""
    if not is_live_mode():
        logger.warning("verify: Kalshi client in stub mode — cannot verify")
        return 0
    if not _PAPER_TRADES_CSV.exists():
        logger.info("verify: no trades file yet")
        return 0

    df = pd.read_csv(_PAPER_TRADES_CSV)
    settled = df[df["status"].isin(["WIN", "LOSS"])]
    already = _load_verified()
    pending = settled[~settled["contract_ticker"].isin(already)]
    if pending.empty:
        logger.info("verify: nothing new to verify")
        return 0

    mismatches = 0
    verified_rows: list[dict] = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for _, row in pending.iterrows():
        ticker = row["contract_ticker"]
        exchange = fetch_market_result(ticker)
        if exchange is None:
            logger.info("verify: %s not yet settled on exchange — will retry", ticker)
            continue

        ours = our_yes_outcome(str(row["direction"]), str(row["status"]))
        exchange_outcome = 1 if exchange == "yes" else 0
        ok = ours == exchange_outcome
        verified_rows.append({
            "contract_ticker": ticker,
            "our_outcome": ours,
            "exchange_result": exchange,
            "match": ok,
            "verified_at_utc": now,
        })

        if ok:
            logger.info("verify: %s OK (exchange=%s)", ticker, exchange)
        else:
            mismatches += 1
            logger.critical(
                "verify: MISMATCH on %s — we graded YES=%s but the exchange "
                "settled %s.  Our strike conventions or observation source "
                "disagree with Kalshi; the track record is suspect until "
                "resolved.", ticker, ours, exchange,
            )
            post_embed(
                title=f"🚨 SETTLEMENT MISMATCH — {ticker}",
                description=(
                    f"We graded YES={ours}; exchange settled '{exchange}'. "
                    "Investigate strike conventions / observation source."
                ),
                color=COLOR_RED,
            )

    _append_verified(verified_rows)
    logger.info(
        "verify: %d checked, %d mismatch(es)", len(verified_rows), mismatches
    )
    return mismatches


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return 1 if verify_settlements() > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
