"""Kalshi orderbook snapshot collector — the market-history recorder.

Historical Kalshi orderbook depth cannot be bought or fetched retroactively
(the API serves candles/trades only), so every day the books aren't recorded
is fill-realism lost forever.  This job snapshots the book for every
contract in every active city's series and appends one parquet file per run:

    data/market_history/date=YYYY-MM-DD/books_HHMMSS.parquet

One small immutable file per cycle keeps writes atomic under the
supervisor's scheduling (no read-modify-write races); a later compaction
pass can merge days into single files when the Forecast Lab ingests them.

Columns: snapshot_utc, city, series, contract_id, strike_type,
threshold_f, floor_strike, cap_strike, best_bid, best_ask,
last_trade_price, bid_size, ask_size, volume_24h, open_interest.

The collector runs every few minutes via the supervisor's interval jobs
(``scheduler.jobs`` → ``every_minutes`` + daytime ``window``).  It records
**live data only** — in stub mode it exits without writing rather than
polluting the archive with deterministic fakes.

CLI (one cycle per invocation)::

    python -m deep_isobar.data.orderbook_collector
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

from deep_isobar.data.city_universe import get_city_universe
from deep_isobar.market.kalshi_client import (
    fetch_live_contracts,
    fetch_orderbook_for_contract,
    is_live_mode,
)

logger = logging.getLogger(__name__)

_HISTORY_DIR = _PROJECT_ROOT / "data" / "market_history"


def collect_snapshot(history_dir: Path | None = None) -> Path | None:
    """Record one snapshot cycle across all active cities.

    Returns the written parquet path, or ``None`` when nothing was recorded
    (stub mode, or no contracts/books available).
    """
    if not is_live_mode():
        logger.warning(
            "orderbook_collector: Kalshi client is in stub mode — refusing "
            "to record fake books. Fix credentials to resume collection."
        )
        return None

    snapshot_utc = datetime.now(timezone.utc)
    rows: list[dict] = []

    for city in (c for c in get_city_universe() if c.active):
      for metric, series in (("high", city.kalshi_series),
                             ("low", city.kalshi_low_series)):
        if not series:
            continue
        try:
            # Never record stub fabrications: raise instead of falling back.
            contracts = fetch_live_contracts(
                "Kalshi", series_ticker=series, allow_stub_fallback=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s/%s] contract fetch failed: %s", city.city, metric, exc)
            continue

        for contract in contracts:
            try:
                book = fetch_orderbook_for_contract("Kalshi", contract.contract_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[%s] orderbook fetch failed for %s: %s",
                    city.city, contract.contract_id, exc,
                )
                continue
            rows.append({
                "snapshot_utc": snapshot_utc,
                "city": city.city,
                "series": series,
                "metric": metric,
                "contract_id": contract.contract_id,
                "strike_type": contract.strike_type,
                "threshold_f": contract.threshold_f,
                "floor_strike": contract.floor_strike,
                "cap_strike": contract.cap_strike,
                "best_bid": book.best_bid,
                "best_ask": book.best_ask,
                "last_trade_price": book.last_trade_price,
                "bid_size": book.bid_size,
                "ask_size": book.ask_size,
                # The orderbook endpoint doesn't return liquidity fields —
                # they come from the /markets payload on the contract.
                "volume_24h": book.volume_24h if book.volume_24h is not None else contract.volume_24h,
                "open_interest": book.open_interest if book.open_interest is not None else contract.open_interest,
            })

    if not rows:
        logger.warning("orderbook_collector: no books recorded this cycle")
        return None

    base = history_dir or _HISTORY_DIR
    day_dir = base / f"date={snapshot_utc:%Y-%m-%d}"
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f"books_{snapshot_utc:%H%M%S}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    logger.info(
        "orderbook_collector: %d books across %d cities → %s",
        len(rows), len({r['city'] for r in rows}), path,
    )
    return path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return 0 if collect_snapshot() is not None else 1


if __name__ == "__main__":
    sys.exit(main())
