"""Record real Kalshi API payloads as golden test fixtures.

The five contract-semantics bugs (bracket math, settlement direction, key
collisions, series metadata, stub books) all came from *assuming* what the
API returns instead of testing against it.  This script snapshots real
responses so ``tests/test_kalshi_golden_fixtures.py`` can assert the full
parse → probability → settlement pipeline against exchange truth, offline.

Recorded into ``tests/fixtures/kalshi/``:

- ``markets_<SERIES>.json``   — raw ``GET /markets?series_ticker=...&status=open``
  (first page) for one series of each ticker-prefix family:
  KXHIGHCHI (KXHIGH*), KXHIGHTPHX (KXHIGHT*), KXLOWTSFO (low series).
- ``orderbook_<TICKER>.json`` — raw ``GET /markets/{ticker}/orderbook`` for a
  few contracts spanning strike types (less / between / greater).
- ``settled_markets.json``    — for recently settled tickers already
  cross-verified in ``data/paper_trades/settlement_verified.csv``: the raw
  market payload (contains the exchange's official ``result``) plus our
  recorded ``settled_temp`` — ground truth for the settlement-direction test.

Requires live credentials (.env).  Re-run whenever Kalshi changes response
formats or new contract shapes appear; commit the refreshed fixtures.

Usage::

    python -m scripts.record_kalshi_fixtures
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from deep_isobar.market import kalshi_client as kc  # noqa: E402

FIXTURE_DIR = _PROJECT_ROOT / "tests" / "fixtures" / "kalshi"

# One series per ticker-prefix family — the Jul 12 stub-book incident was
# caused by series outside the metadata map, so coverage across prefixes
# is the point.
MARKET_SERIES = ["KXHIGHCHI", "KXHIGHTPHX", "KXLOWTSFO"]

_MAX_SETTLED = 12


def _record_markets(credentials) -> dict[str, dict]:
    responses: dict[str, dict] = {}
    for series in MARKET_SERIES:
        data = kc._kalshi_get(
            "/markets", {"series_ticker": series, "status": "open", "limit": 100},
            credentials,
        )
        n = len(data.get("markets", []))
        print(f"  {series}: {n} open markets")
        if n == 0:
            print(f"  WARNING: {series} returned no open markets — fixture skipped")
            continue
        (FIXTURE_DIR / f"markets_{series}.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
        responses[series] = data
    return responses


def _record_orderbooks(credentials, markets_by_series: dict[str, dict]) -> None:
    """One orderbook per strike type from the first series with all three."""
    chosen: dict[str, str] = {}
    for series, data in markets_by_series.items():
        for m in data.get("markets", []):
            st = (m.get("strike_type") or "").lower()
            if st in ("less", "greater", "between") and st not in chosen:
                chosen[st] = m["ticker"]
        if len(chosen) == 3:
            break
    for st, ticker in sorted(chosen.items()):
        data = kc._kalshi_get(f"/markets/{ticker}/orderbook", None, credentials)
        (FIXTURE_DIR / f"orderbook_{ticker}.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
        print(f"  orderbook {ticker} ({st})")


def _record_settled(credentials) -> None:
    verified_csv = _PROJECT_ROOT / "data" / "paper_trades" / "settlement_verified.csv"
    trades_csv = _PROJECT_ROOT / "data" / "paper_trades" / "paper_trades.csv"
    if not (verified_csv.exists() and trades_csv.exists()):
        print("  no verified settlements — skipping settled fixture")
        return

    verified = pd.read_csv(verified_csv)
    trades = pd.read_csv(trades_csv)
    temps = (
        trades[trades["status"].isin(["WIN", "LOSS"])]
        .dropna(subset=["settled_temp"])
        .set_index("contract_ticker")["settled_temp"]
        .to_dict()
    )

    entries: list[dict] = []
    # Newest first — recent tickers still resolve on the API.
    for ticker in reversed(list(verified["contract_ticker"])):
        if ticker not in temps:
            continue
        data = kc._kalshi_get(f"/markets/{ticker}", None, credentials)
        market = data.get("market") or {}
        if str(market.get("result") or "").lower() not in ("yes", "no"):
            continue
        entries.append({"settled_temp": float(temps[ticker]), "market": market})
        print(f"  settled {ticker}: result={market['result']} temp={temps[ticker]}")
        if len(entries) >= _MAX_SETTLED:
            break

    if entries:
        (FIXTURE_DIR / "settled_markets.json").write_text(
            json.dumps({"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                        "markets": entries}, indent=2),
            encoding="utf-8",
        )


def main() -> int:
    if not kc.is_live_mode():
        print("ERROR: Kalshi client is not in live mode — fixtures must be real.")
        return 1
    credentials = kc._load_credentials()
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Recording /markets responses…")
    markets = _record_markets(credentials)
    print("Recording orderbooks…")
    _record_orderbooks(credentials, markets)
    print("Recording settled markets with official results…")
    _record_settled(credentials)
    print(f"Fixtures written to {FIXTURE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
