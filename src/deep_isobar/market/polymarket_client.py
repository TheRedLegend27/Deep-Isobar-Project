"""Read-only Polymarket client for daily temperature markets.

Mirrors the ``kalshi_client`` interface so downstream code can treat both
venues uniformly::

    fetch_live_contracts("Polymarket", city="New York", target_date=...)
        -> list[MarketContract]
    fetch_orderbook_for_contract("Polymarket", contract_id)
        -> OrderBookSnapshot

No credentials required — uses the public Gamma API (market discovery,
top-of-book) and the public CLOB API (full order book).

STATION WARNING
---------------
Polymarket settles on **different stations** than Kalshi, via Wunderground
(not the NWS CLI product):

==========  ===============  ==================  =====================
City        Polymarket       Kalshi settlement   Deep Isobar model
==========  ===============  ==================  =====================
Chicago     KORD (O'Hare)    KMDW (Midway)       KMDW bias profile
New York    KLGA (LaGuardia) Central Park        KJFK METAR / CLINYC
Dallas      KDAL (Love Fld)  KDFW                KDFW bias profile
==========  ===============  ==================  =====================

A Kalshi-vs-Polymarket price divergence is therefore a *correlated-station
spread*, not a pure arbitrage.  ``MarketContract.settlement_source`` is set
to ``"Wunderground:<station>"`` so downstream code can never confuse them.

Bucket semantics
----------------
Polymarket events are negRisk bucket groups.  ``groupItemTitle`` formats map
to ``probability_for_contract`` strike types (whole-degree settlement, same
±0.5°F continuity correction as Kalshi):

- ``"94-95°F"``       → between: floor=94, cap=96   (94 ≤ T < 96 ↔ T ∈ {94,95})
- ``"100°F or higher"`` → greater: floor=99          (T > 99 ↔ T ≥ 100)
- ``"59°F or lower"``   → less: cap=60               (T < 60 ↔ T ≤ 59)
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timezone
from typing import Any, Optional

import requests

from deep_isobar.core.types import MarketContract, OrderBookSnapshot

logger = logging.getLogger(__name__)

_GAMMA_BASE = "https://gamma-api.polymarket.com"
_CLOB_BASE = "https://clob.polymarket.com"
_MARKET_SOURCE = "Polymarket"
_TIMEOUT_SECONDS = 15

# Polymarket city registry: Deep Isobar city name → event-slug fragment and
# the station Wunderground resolution actually uses.
PM_CITIES: dict[str, dict[str, str]] = {
    "Chicago":  {"slug": "chicago", "station_id": "KORD"},
    "New York": {"slug": "nyc",     "station_id": "KLGA"},
    "Dallas":   {"slug": "dallas",  "station_id": "KDAL"},
}

# groupItemTitle parsers (whole °F, see module docstring)
_RE_BETWEEN = re.compile(r"^(\d+)\s*-\s*(\d+)\s*°?F?$")
_RE_HIGHER = re.compile(r"^(\d+)\s*°?F?\s+or\s+(?:higher|above)$", re.IGNORECASE)
_RE_LOWER = re.compile(r"^(\d+)\s*°?F?\s+or\s+(?:lower|below)$", re.IGNORECASE)

# contract_id → YES clob token id, populated by fetch_live_contracts so
# fetch_orderbook_for_contract can resolve books without re-fetching gamma.
_TOKEN_CACHE: dict[str, str] = {}


def _event_slug(city: str, target_date: date, metric: str = "high_temp_f") -> str:
    """Build the Polymarket event slug for a city/date temperature market."""
    kind = "highest" if metric == "high_temp_f" else "lowest"
    city_slug = PM_CITIES[city]["slug"]
    month = target_date.strftime("%B").lower()
    return f"{kind}-temperature-in-{city_slug}-on-{month}-{target_date.day}-{target_date.year}"


def _parse_bucket(title: str) -> Optional[dict[str, Any]]:
    """Map a ``groupItemTitle`` to strike fields, or None if unrecognised."""
    title = title.replace("°", "°").strip()
    m = _RE_BETWEEN.match(title)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return {"strike_type": "between", "floor_strike": lo, "cap_strike": hi + 1,
                "threshold_f": lo}
    m = _RE_HIGHER.match(title)
    if m:
        bound = int(m.group(1))
        return {"strike_type": "greater", "floor_strike": bound - 1, "cap_strike": None,
                "threshold_f": bound - 1}
    m = _RE_LOWER.match(title)
    if m:
        bound = int(m.group(1))
        return {"strike_type": "less", "floor_strike": None, "cap_strike": bound + 1,
                "threshold_f": bound + 1}
    return None


def fetch_live_contracts(
    market_source: str = _MARKET_SOURCE,
    city: str = "New York",
    target_date: Optional[date] = None,
    metric: str = "high_temp_f",
) -> list[MarketContract]:
    """Fetch tradable bucket contracts for one city/date from Polymarket.

    Args:
        market_source: Must be ``"Polymarket"`` (interface symmetry with
            ``kalshi_client``).
        city: Deep Isobar city name; must be a key of :data:`PM_CITIES`.
        target_date: Settlement date.  Defaults to tomorrow.
        metric: ``"high_temp_f"`` or ``"low_temp_f"``.

    Returns:
        One ``MarketContract`` per recognisable bucket; empty list when the
        event does not exist (not listed yet) or the API is unreachable.
    """
    if market_source != _MARKET_SOURCE:
        raise ValueError(f"polymarket_client only serves Polymarket, got {market_source!r}")
    if city not in PM_CITIES:
        logger.info("[%s] Not in Polymarket city registry — skipping", city)
        return []
    if target_date is None:
        from datetime import timedelta
        target_date = date.today() + timedelta(days=1)

    slug = _event_slug(city, target_date, metric)
    try:
        resp = requests.get(
            f"{_GAMMA_BASE}/events", params={"slug": slug}, timeout=_TIMEOUT_SECONDS
        )
        resp.raise_for_status()
        events = resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] Gamma API request failed for %s: %s", city, slug, exc)
        return []

    if not events:
        logger.info("[%s] No Polymarket event found for slug %s", city, slug)
        return []

    event = events[0]
    station = PM_CITIES[city]["station_id"]
    now_utc = datetime.now(timezone.utc)
    contracts: list[MarketContract] = []

    for market in event.get("markets", []):
        bucket = _parse_bucket(market.get("groupItemTitle") or "")
        if bucket is None:
            logger.debug("[%s] Unrecognised bucket title: %r", city, market.get("groupItemTitle"))
            continue

        try:
            token_ids = json.loads(market.get("clobTokenIds") or "[]")
        except json.JSONDecodeError:
            token_ids = []
        yes_token = token_ids[0] if token_ids else ""

        contract_id = f"PM-{market.get('id', '')}"
        if yes_token:
            _TOKEN_CACHE[contract_id] = yes_token

        contracts.append(MarketContract(
            contract_id=contract_id,
            market_source=_MARKET_SOURCE,
            city=city,
            metric=metric,
            comparison_operator="ge" if bucket["strike_type"] == "greater" else "lt",
            threshold_f=bucket["threshold_f"],
            target_date=target_date,
            settlement_source=f"Wunderground:{station}",
            strike_type=bucket["strike_type"],
            floor_strike=bucket["floor_strike"],
            cap_strike=bucket["cap_strike"],
            raw_title=market.get("question"),
            active=bool(market.get("active", True)),
        ))

    logger.info(
        "[%s] Polymarket: %d bucket contract(s) for %s (station %s)",
        city, len(contracts), target_date, station,
    )
    return contracts


def fetch_orderbook_for_contract(
    market_source: str,
    contract_id: str,
) -> OrderBookSnapshot:
    """Fetch the YES order book for a contract returned by this module.

    Uses the public CLOB ``/book`` endpoint.  Falls back to an empty
    snapshot (None bid/ask) when the book is unavailable, mirroring
    ``kalshi_client`` behaviour on error.
    """
    if market_source != _MARKET_SOURCE:
        raise ValueError(f"polymarket_client only serves Polymarket, got {market_source!r}")

    now_utc = datetime.now(timezone.utc)
    token_id = _TOKEN_CACHE.get(contract_id)
    if not token_id:
        logger.warning("No cached CLOB token for %s — call fetch_live_contracts first", contract_id)
        return OrderBookSnapshot(
            timestamp_utc=now_utc, contract_id=contract_id,
            market_source=_MARKET_SOURCE, best_bid=None, best_ask=None,
        )

    try:
        resp = requests.get(
            f"{_CLOB_BASE}/book", params={"token_id": token_id}, timeout=_TIMEOUT_SECONDS
        )
        resp.raise_for_status()
        book = resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("CLOB book request failed for %s: %s", contract_id, exc)
        return OrderBookSnapshot(
            timestamp_utc=now_utc, contract_id=contract_id,
            market_source=_MARKET_SOURCE, best_bid=None, best_ask=None,
        )

    bids = [(float(b["price"]), float(b["size"])) for b in book.get("bids", [])]
    asks = [(float(a["price"]), float(a["size"])) for a in book.get("asks", [])]
    best_bid = max(bids, key=lambda x: x[0]) if bids else None
    best_ask = min(asks, key=lambda x: x[0]) if asks else None

    return OrderBookSnapshot(
        timestamp_utc=now_utc,
        contract_id=contract_id,
        market_source=_MARKET_SOURCE,
        best_bid=best_bid[0] if best_bid else None,
        best_ask=best_ask[0] if best_ask else None,
        bid_size=best_bid[1] if best_bid else None,
        ask_size=best_ask[1] if best_ask else None,
    )
