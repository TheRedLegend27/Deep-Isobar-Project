"""Historical Kalshi market data ingestion for Deep Isobar.

Pulls archived KXHIGHCHI (Chicago High Temperature) market data from the
Kalshi v2 REST API, covering contracts active between two dates.

Pipeline
--------
1. Discover all KXHIGHCHI contracts settled in the date range
   ``GET /markets?series_ticker=KXHIGHCHI&status=settled``
   (paginated; filtered locally by ``target_date``).

2. For each contract, fetch hourly candlestick data
   ``GET /markets/{ticker}/candlesticks?period_interval=60``

3. Map candlesticks to ``order_book_snapshot`` schema (DATA_SCHEMA.md §15):

   - ``best_bid``        ← ``low_price / 100``   (hourly-low as bid proxy)
   - ``best_ask``        ← ``high_price / 100``  (hourly-high as ask proxy)
   - ``mid_price``       ← ``close_price / 100``
   - ``last_trade_price``← ``close_price / 100``
   - ``volume_24h``      ← rolling 24-candle sum of ``volume``
   - ``open_interest``   ← candle ``open_interest``

4. Map contract metadata to ``live_market_contract`` schema (DATA_SCHEMA.md §14).

5. Save to ``data/historical/markets/kalshi_kxhighchi_2023.parquet``
   (order-book snapshots) and
   ``data/historical/markets/kalshi_kxhighchi_2023_contracts.parquet``
   (contract registry).

Bid/ask note
------------
Kalshi does not expose historical order-book depth.  The hourly candle
``low_price`` and ``high_price`` are used as bid/ask proxies: within any
trading hour the market will have traded at its lowest ask (low) and
highest bid (high).  When a candle has zero spread (no trading), a nominal
1-cent half-spread is applied symmetrically around the close price.

Credentials (.env)
------------------
``KALSHI_API_KEY_ID``       — Key ID UUID from kalshi.com/account/profile
``KALSHI_PRIVATE_KEY_PATH`` — path to RSA-4096 private key PEM file
``KALSHI_PRIVATE_KEY``      — inline PEM string (\\\\n-escaped newlines ok)

Rate limiting
-------------
200 ms between requests by default (≤ 5 req/s).  On HTTP 429 the module
respects the ``Retry-After`` header or waits 30 seconds, then retries up
to :data:`_MAX_RETRIES` times.

Usage::

    python -m deep_isobar.market.historical_kalshi_ingest

or::

    python src/deep_isobar/market/historical_kalshi_ingest.py \\
        --series KXHIGHCHI \\
        --start 2023-05-01 --end 2023-09-30 \\
        --out data/historical/markets/kalshi_kxhighchi_2023.parquet
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests

# Reuse auth infrastructure already proven in the live client
from deep_isobar.market.kalshi_client import (
    _load_credentials,
    _make_auth_headers,
    _parse_ticker,
    _series_metadata,
    _KALSHI_BASE_URL,
)
from deep_isobar.config import get_setting

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MARKET_SOURCE = "Kalshi"
_REQUEST_DELAY_S = 0.2          # 200 ms between requests (5 req/s)
_RATELIMIT_DEFAULT_WAIT_S = 30  # fallback when Retry-After absent
_MAX_RETRIES = 6
_PAGE_LIMIT = 200               # max results per /markets page
_CANDLE_INTERVAL_MIN = 60       # 1-hour candles

# order_book_snapshot output columns (DATA_SCHEMA.md §15)
_OBS_COLUMNS = [
    "timestamp_utc",
    "contract_id",
    "market_source",
    "best_bid",
    "best_ask",
    "mid_price",
    "last_trade_price",
    "bid_size",
    "ask_size",
    "volume_24h",
    "open_interest",
    "snapshot_latency_ms",
    "raw_payload_hash",
]

# live_market_contract output columns (DATA_SCHEMA.md §14)
_LMC_COLUMNS = [
    "contract_id",
    "market_source",
    "contract_template_id",
    "exchange_symbol",
    "city",
    "metric",
    "comparison_operator",
    "threshold_f",
    "target_date",
    "settlement_source",
    "listed_at_utc",
    "expires_at_utc",
    "active",
    "raw_title",
]

# ---------------------------------------------------------------------------
# Rate-limit-aware authenticated GET
# ---------------------------------------------------------------------------


def _kalshi_get(
    path: str,
    params: dict[str, Any] | None,
    credentials: tuple[str, Any],
    request_delay_s: float = _REQUEST_DELAY_S,
) -> dict[str, Any]:
    """Make an authenticated GET to the Kalshi v2 API with rate-limit handling.

    Differs from :func:`~deep_isobar.market.kalshi_client._kalshi_get` in that
    it retries transparently on HTTP 429 (respecting ``Retry-After``) and
    applies a polite inter-request delay to avoid hitting rate limits.

    Args:
        path: Endpoint path without base URL (e.g. ``"/markets"``).
        params: Optional query parameters.
        credentials: ``(key_id, private_key)`` from :func:`~deep_isobar.market.kalshi_client._load_credentials`.
        request_delay_s: Seconds to sleep after a successful response.

    Returns:
        Parsed JSON response dict.

    Raises:
        RuntimeError: After :data:`_MAX_RETRIES` failed attempts or a
            non-recoverable HTTP error.
    """
    base_url: str = get_setting("markets.kalshi.base_url", _KALSHI_BASE_URL)
    url = f"{base_url}{path}"
    base_path = urlparse(base_url).path.rstrip("/")
    sign_path = base_path + path
    key_id, private_key = credentials

    for attempt in range(_MAX_RETRIES):
        headers = _make_auth_headers(key_id, private_key, "GET", sign_path)
        try:
            resp = requests.get(
                url,
                headers=headers,
                params=params or {},
                timeout=20,
            )
        except requests.RequestException as exc:
            if attempt < _MAX_RETRIES - 1:
                wait = min(2 ** attempt, 30)
                logger.warning(
                    "Network error on GET %s (%s) — retrying in %.0fs", path, exc, wait
                )
                time.sleep(wait)
                continue
            raise RuntimeError(f"Kalshi API network error on GET {path}: {exc}") from exc

        if resp.status_code == 200:
            time.sleep(request_delay_s)
            return resp.json()  # type: ignore[no-any-return]

        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", _RATELIMIT_DEFAULT_WAIT_S))
            logger.warning(
                "Rate limited on GET %s — waiting %.0fs (attempt %d/%d)",
                path, retry_after, attempt + 1, _MAX_RETRIES,
            )
            time.sleep(retry_after)
            continue

        # Non-retryable errors
        raise RuntimeError(
            f"Kalshi API error: GET {path} → HTTP {resp.status_code}: "
            f"{resp.text[:300]}"
        )

    raise RuntimeError(
        f"Kalshi GET {path} failed after {_MAX_RETRIES} attempts (rate limit exhausted)"
    )


# ---------------------------------------------------------------------------
# Contract discovery
# ---------------------------------------------------------------------------


def _fetch_settled_contracts(
    series_ticker: str,
    start_date: date,
    end_date: date,
    credentials: tuple[str, Any],
    request_delay_s: float = _REQUEST_DELAY_S,
) -> list[dict[str, Any]]:
    """Fetch all settled Kalshi markets for *series_ticker* in the date range.

    Paginates ``GET /markets`` with ``status=settled`` and filters locally
    to keep only markets whose parsed ``target_date`` falls in
    ``[start_date, end_date]``.

    Args:
        series_ticker: Kalshi series code (e.g. ``"KXHIGHCHI"``).
        start_date: Inclusive lower bound on contract target date.
        end_date: Inclusive upper bound on contract target date.
        credentials: ``(key_id, private_key)`` pair.

    Returns:
        List of raw market dicts from the Kalshi API, filtered to the
        requested date range.
    """
    logger.info(
        "Fetching settled %s contracts: %s → %s",
        series_ticker, start_date.isoformat(), end_date.isoformat(),
    )

    # Convert date bounds to unix timestamps for optional server-side filter.
    # We always filter locally too since the API may not support these params.
    min_ts = int(datetime(start_date.year, start_date.month, start_date.day,
                          tzinfo=timezone.utc).timestamp())
    max_ts = int(datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59,
                          tzinfo=timezone.utc).timestamp())

    contracts: list[dict[str, Any]] = []
    cursor: str | None = None
    page = 0

    while True:
        params: dict[str, Any] = {
            "series_ticker": series_ticker,
            "status": "settled",
            "limit": _PAGE_LIMIT,
            "min_close_ts": min_ts,
            "max_close_ts": max_ts,
        }
        if cursor:
            params["cursor"] = cursor

        page += 1
        logger.debug("GET /markets page=%d cursor=%s", page, cursor)
        data = _kalshi_get("/markets", params, credentials, request_delay_s=request_delay_s)

        for market in data.get("markets", []):
            ticker: str = market.get("ticker", "")
            parsed = _parse_ticker(ticker)
            if parsed is None:
                continue
            if parsed["series"] not in _series_metadata():
                continue
            td: date = parsed["target_date"]
            if start_date <= td <= end_date:
                contracts.append(market)

        cursor = data.get("cursor") or None
        if not cursor:
            break

    logger.info(
        "Discovered %d %s contracts in [%s, %s]",
        len(contracts), series_ticker, start_date, end_date,
    )
    return contracts


# ---------------------------------------------------------------------------
# Candlestick fetching
# ---------------------------------------------------------------------------


def _fetch_candlesticks(
    ticker: str,
    start_ts: int,
    end_ts: int,
    credentials: tuple[str, Any],
    series_ticker: str = "KXHIGHCHI",
    period_interval_min: int = _CANDLE_INTERVAL_MIN,
    request_delay_s: float = _REQUEST_DELAY_S,
) -> list[dict[str, Any]]:
    """Fetch hourly candlestick data for one contract.

    ``GET /series/{series_ticker}/markets/{ticker}/candlesticks``

    Args:
        ticker: Kalshi exchange ticker.
        start_ts: Unix timestamp (seconds) for window start.
        end_ts: Unix timestamp (seconds) for window end.
        credentials: ``(key_id, private_key)`` pair.
        series_ticker: Parent series ticker (e.g. ``"KXHIGHCHI"``).
        period_interval_min: Candle width in minutes (default 60 = hourly).

    Returns:
        List of candlestick dicts.  Empty list if the API returns no data
        or if the endpoint responds with a non-fatal error.
    """
    # Derive series prefix from the contract ticker itself (e.g. HIGHCHI- or KXHIGHCHI-)
    # so old and new ticker formats both resolve to the right API path.
    import re as _re
    _m = _re.match(r"^([A-Z]+)-\d{2}[A-Z]{3}\d{2}-", ticker)
    effective_series = _m.group(1) if _m else series_ticker
    path = f"/series/{effective_series}/markets/{ticker}/candlesticks"
    params: dict[str, Any] = {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "period_interval": period_interval_min,
    }
    try:
        data = _kalshi_get(path, params, credentials, request_delay_s=request_delay_s)
    except RuntimeError as exc:
        logger.warning(
            "Candlestick fetch failed for %s: %s — skipping", ticker, exc
        )
        return []

    return data.get("candlesticks", [])


# ---------------------------------------------------------------------------
# Schema builders
# ---------------------------------------------------------------------------


def _build_order_book_snapshots(
    ticker: str,
    candles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert raw candlestick records to ``order_book_snapshot`` rows.

    Bid/ask mapping
    ~~~~~~~~~~~~~~~
    Kalshi candlestick prices are returned in the ``yes_bid`` / ``yes_ask``
    sub-dicts with ``close_dollars``, ``high_dollars``, ``low_dollars`` fields
    already in the 0–1 probability (dollar) range.

    - ``best_bid``         ← ``yes_bid.close_dollars``
    - ``best_ask``         ← ``yes_ask.close_dollars``
    - ``mid_price``        ← ``(bid + ask) / 2``
    - ``last_trade_price`` ← ``mid_price``

    When the candle has zero spread (thin market), a nominal half-spread of
    0.01 is applied symmetrically around mid.

    ``volume_24h`` is computed as a rolling 24-candle cumulative sum and
    is applied after all rows are built.

    Args:
        ticker: Contract ticker for populating ``contract_id``.
        candles: Raw candlestick list from the Kalshi API.

    Returns:
        List of row dicts conforming to ``order_book_snapshot`` schema.
    """
    rows: list[dict[str, Any]] = []

    for candle in candles:
        end_ts: int | None = candle.get("end_period_ts")
        if end_ts is None:
            continue

        timestamp_utc = datetime.fromtimestamp(end_ts, tz=timezone.utc)

        # Prices are in the yes_bid / yes_ask sub-dicts, already in 0–1 scale.
        yes_bid: dict[str, Any] = candle.get("yes_bid") or {}
        yes_ask: dict[str, Any] = candle.get("yes_ask") or {}

        bid_raw = yes_bid.get("close_dollars")
        ask_raw = yes_ask.get("close_dollars")

        bid: float = round(float(bid_raw), 6) if bid_raw is not None else 0.0
        ask: float = round(float(ask_raw), 6) if ask_raw is not None else bid

        # Apply nominal spread when the candle has no spread
        if bid == ask:
            half_spread = 0.01
            bid = max(0.01, round(bid - half_spread, 6))
            ask = min(0.99, round(ask + half_spread, 6))

        # Guard: ensure bid ≤ ask
        if bid > ask:
            bid, ask = ask, bid

        mid = round((bid + ask) / 2.0, 6)

        volume_raw = candle.get("volume_fp")
        oi_raw = candle.get("open_interest_fp")

        rows.append({
            "timestamp_utc":     timestamp_utc,
            "contract_id":       ticker,
            "market_source":     _MARKET_SOURCE,
            "best_bid":          bid,
            "best_ask":          ask,
            "mid_price":         mid,
            "last_trade_price":  mid,
            "bid_size":          None,
            "ask_size":          None,
            "_volume":           float(volume_raw) if volume_raw is not None else 0.0,
            "open_interest":     float(oi_raw) if oi_raw is not None else None,
            "snapshot_latency_ms": None,
            "raw_payload_hash":  None,
        })

    return rows


def _add_volume_24h(rows: list[dict[str, Any]]) -> None:
    """Compute rolling 24-candle sum of ``_volume`` and store as ``volume_24h``.

    Mutates *rows* in-place.  The helper column ``_volume`` is removed
    after aggregation.  Uses a two-pass approach so back-references to
    ``_volume`` in already-processed rows remain valid during the first pass.
    """
    # Pass 1: compute rolling sums (all _volume values still present)
    for i, row in enumerate(rows):
        start = max(0, i - 23)
        row["volume_24h"] = sum(r["_volume"] for r in rows[start : i + 1])
    # Pass 2: remove the helper column
    for row in rows:
        del row["_volume"]


def _build_live_market_contract(market: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a raw Kalshi market dict to a ``live_market_contract`` row.

    Args:
        market: One element from the Kalshi ``/markets`` response.

    Returns:
        Row dict conforming to ``live_market_contract`` schema, or ``None``
        if the ticker cannot be parsed.
    """
    ticker: str = market.get("ticker", "")
    parsed = _parse_ticker(ticker)
    if parsed is None:
        return None

    meta = _series_metadata().get(parsed["series"])
    if meta is None:
        return None

    listed_at_utc: datetime | None = None
    expires_at_utc: datetime | None = None
    for api_field, target in (
        ("created_time", "listed"),
        ("latest_expiration_time", "expires"),
    ):
        raw = market.get(api_field)
        if raw:
            try:
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                if target == "listed":
                    listed_at_utc = dt
                else:
                    expires_at_utc = dt
            except ValueError:
                pass

    return {
        "contract_id":          ticker,
        "market_source":        _MARKET_SOURCE,
        "contract_template_id": None,
        "exchange_symbol":      ticker,
        "city":                 meta["city"],
        "metric":               meta["metric"],
        "comparison_operator":  meta["comparison_operator"],
        "threshold_f":          parsed["threshold_f"],
        "target_date":          parsed["target_date"],
        "settlement_source":    meta["settlement_source"],
        "listed_at_utc":        listed_at_utc,
        "expires_at_utc":       expires_at_utc,
        "active":               False,   # all historical contracts are settled
        "raw_title":            market.get("rules_primary") or market.get("title"),
    }


# ---------------------------------------------------------------------------
# Series discovery  (--discover mode)
# ---------------------------------------------------------------------------


_DISCOVER_CANDIDATES = [
    # Current and historical Kalshi Chicago high-temperature series names.
    # The KX prefix was dropped for some series around 2023–2024.
    "KXHIGHCHI",   # original 2023 name
    "HIGHCHI",     # KX-dropped variant (already in _SERIES_METADATA)
    "KXCHI",       # possible alternate prefix
    "CHIHIGH",
    "HIGHTEMP",
    "KXHIGH",
    "CHIHI",
    "KXWX",        # possible weather umbrella series
    "WXCHI",
    "TEMPCHI",
    "HIGHORD",     # O'Hare (KORD) based naming
    "KXHIGHORD",
]


def discover_series(
    start_date: date,
    end_date: date,
    keywords: list[str],
    credentials: tuple[str, Any],
    request_delay_s: float = _REQUEST_DELAY_S,
) -> list[dict[str, Any]]:
    """Probe candidate Chicago series tickers and return those with settled
    contracts in the date range.

    Strategy: query ``GET /markets?series_ticker=CANDIDATE&status=settled``
    for each ticker in :data:`_DISCOVER_CANDIDATES` and for any extra
    candidates derived from *keywords*.  This is O(N candidates) API calls
    rather than a full market scan, so it completes in seconds.

    Additionally performs one broad page scan (first page only) without
    ``series_ticker`` to catch any series not in the candidate list, with
    keyword filtering applied to ticker and title.

    Args:
        start_date: Inclusive lower bound on contract target date.
        end_date:   Inclusive upper bound on contract target date.
        keywords:   Case-insensitive substrings to match against ticker
                    and title when scanning the broad first page.
        credentials: ``(key_id, private_key)`` pair.
        request_delay_s: Seconds between API calls.

    Returns:
        List of dicts, one per matching series, sorted by contract count
        descending.  Keys: ``series``, ``sample_ticker``, ``title``,
        ``contract_count``.
    """
    kw_lower = [k.lower() for k in keywords]
    series_map: dict[str, dict[str, Any]] = {}

    min_ts = int(datetime(start_date.year, start_date.month, start_date.day,
                          tzinfo=timezone.utc).timestamp())
    max_ts = int(datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59,
                          tzinfo=timezone.utc).timestamp())

    logger.info(
        "Discovery probe: %d candidate series  %s to %s",
        len(_DISCOVER_CANDIDATES), start_date, end_date,
    )

    # ── Phase A: probe each candidate series_ticker ────────────────────────
    for candidate in _DISCOVER_CANDIDATES:
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {
                "series_ticker": candidate,
                "status": "settled",
                "limit": _PAGE_LIMIT,
                "min_close_ts": min_ts,
                "max_close_ts": max_ts,
            }
            if cursor:
                params["cursor"] = cursor

            try:
                data = _kalshi_get(
                    "/markets", params, credentials,
                    request_delay_s=request_delay_s,
                )
            except RuntimeError as exc:
                logger.debug("Candidate %s probe failed: %s", candidate, exc)
                break

            for market in data.get("markets", []):
                ticker: str = market.get("ticker", "")
                title: str = (
                    market.get("title")
                    or market.get("rules_primary")
                    or ""
                )
                series_prefix = ticker.split("-")[0] if "-" in ticker else ticker
                if series_prefix not in series_map:
                    series_map[series_prefix] = {
                        "series":         series_prefix,
                        "sample_ticker":  ticker,
                        "title":          title,
                        "contract_count": 0,
                    }
                series_map[series_prefix]["contract_count"] += 1

            cursor = data.get("cursor") or None
            if not cursor:
                break

        if candidate in series_map or any(
            s.startswith(candidate[:4]) for s in series_map
        ):
            logger.info("  [hit]  %s -> %d contracts", candidate,
                        series_map.get(candidate, {}).get("contract_count", 0))
        else:
            logger.debug("  [miss] %s", candidate)

    # ── Phase B: one broad first-page scan (catches unlisted candidates) ───
    logger.info("Discovery Phase B: broad first-page scan (no series_ticker filter)")
    try:
        data = _kalshi_get(
            "/markets",
            {"status": "settled", "limit": _PAGE_LIMIT,
             "min_close_ts": min_ts, "max_close_ts": max_ts},
            credentials,
            request_delay_s=request_delay_s,
        )
        for market in data.get("markets", []):
            ticker = market.get("ticker", "")
            title  = market.get("title") or market.get("rules_primary") or ""
            if not any(kw in (ticker + " " + title).lower() for kw in kw_lower):
                continue
            series_prefix = ticker.split("-")[0] if "-" in ticker else ticker
            if series_prefix not in series_map:
                series_map[series_prefix] = {
                    "series":         series_prefix,
                    "sample_ticker":  ticker,
                    "title":          title,
                    "contract_count": 0,
                }
            series_map[series_prefix]["contract_count"] += 1
    except RuntimeError as exc:
        logger.warning("Phase B broad scan failed: %s", exc)

    results = sorted(
        series_map.values(),
        key=lambda r: r["contract_count"],
        reverse=True,
    )
    logger.info(
        "Discovery complete: %d candidates probed, %d matching series found",
        len(_DISCOVER_CANDIDATES), len(results),
    )
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_historical_market_data(
    series_ticker: str = "KXHIGHCHI",
    start_date: date | None = None,
    end_date: date | None = None,
    request_delay_s: float = _REQUEST_DELAY_S,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch historical Kalshi market snapshots and contract metadata.

    Discovers all settled contracts for *series_ticker* in the given date
    range, then fetches hourly candlestick data for each contract.

    Returns two DataFrames:

    - **snapshots** — one row per (contract × hourly timestamp), conforming
      to ``order_book_snapshot`` (DATA_SCHEMA.md §15).
    - **contracts** — one row per contract, conforming to
      ``live_market_contract`` (DATA_SCHEMA.md §14).

    Args:
        series_ticker: Kalshi series code.  Defaults to ``"KXHIGHCHI"``
            (Chicago High Temperature).
        start_date: Inclusive lower bound on contract target date.
            Defaults to ``date(2023, 5, 1)``.
        end_date: Inclusive upper bound on contract target date.
            Defaults to ``date(2023, 9, 30)``.
        request_delay_s: Seconds between successive API calls.  Increase
            if you see frequent 429 responses.

    Returns:
        ``(snapshots_df, contracts_df)``

    Raises:
        RuntimeError: If credentials are not available, or if no contracts
            are found for the given series/date range.

    Example::

        from datetime import date
        from deep_isobar.market.historical_kalshi_ingest import fetch_historical_market_data
        snapshots, contracts = fetch_historical_market_data(
            start_date=date(2023, 5, 1),
            end_date=date(2023, 9, 30),
        )
        print(snapshots.shape)   # (rows, 13)
        print(contracts.shape)   # (contracts, 14)
    """
    if start_date is None:
        start_date = date(2023, 5, 1)
    if end_date is None:
        end_date = date(2023, 9, 30)

    credentials = _load_credentials()
    if credentials is None:
        raise RuntimeError(
            "Kalshi credentials not found.  Set KALSHI_API_KEY_ID and "
            "KALSHI_PRIVATE_KEY_PATH (or KALSHI_PRIVATE_KEY) in your .env file."
        )

    logger.info(
        "=== Historical Kalshi ingest: series=%s  %s → %s ===",
        series_ticker, start_date, end_date,
    )

    # ── 1. Discover contracts ─────────────────────────────────────────────
    raw_markets = _fetch_settled_contracts(
        series_ticker, start_date, end_date, credentials, request_delay_s=request_delay_s
    )
    if not raw_markets:
        raise RuntimeError(
            f"No settled {series_ticker} contracts found between "
            f"{start_date} and {end_date}.  "
            "Check that the series ticker and date range are correct."
        )

    # ── 2. Build live_market_contract rows ────────────────────────────────
    contract_rows: list[dict[str, Any]] = []
    for market in raw_markets:
        row = _build_live_market_contract(market)
        if row is not None:
            contract_rows.append(row)

    # ── 3. Fetch candlesticks for each contract ───────────────────────────
    snapshot_rows: list[dict[str, Any]] = []
    total = len(raw_markets)

    # Determine the full fetch window: listed_at → expires_at per contract
    # Fall back to start_date/end_date if metadata is missing
    start_ts_default = int(
        datetime(start_date.year, start_date.month, start_date.day,
                 tzinfo=timezone.utc).timestamp()
    )
    end_ts_default = int(
        datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59,
                 tzinfo=timezone.utc).timestamp()
    )

    for idx, market in enumerate(raw_markets, 1):
        ticker: str = market.get("ticker", "")
        logger.info(
            "[%d/%d] Fetching candlesticks for %s", idx, total, ticker
        )

        # Use contract's own timestamps when available (tighter window = fewer candles)
        listed_raw = market.get("created_time")
        expires_raw = market.get("latest_expiration_time")

        if listed_raw:
            try:
                cand_start_ts = int(
                    datetime.fromisoformat(listed_raw.replace("Z", "+00:00")).timestamp()
                )
            except ValueError:
                cand_start_ts = start_ts_default
        else:
            cand_start_ts = start_ts_default

        if expires_raw:
            try:
                cand_end_ts = int(
                    datetime.fromisoformat(expires_raw.replace("Z", "+00:00")).timestamp()
                )
            except ValueError:
                cand_end_ts = end_ts_default
        else:
            cand_end_ts = end_ts_default

        candles = _fetch_candlesticks(
            ticker, cand_start_ts, cand_end_ts, credentials,
            series_ticker=series_ticker,
            request_delay_s=request_delay_s,
        )
        if not candles:
            logger.warning("No candlestick data returned for %s — skipping", ticker)
            continue

        rows = _build_order_book_snapshots(ticker, candles)
        _add_volume_24h(rows)
        snapshot_rows.extend(rows)

    logger.info(
        "Ingest complete: %d snapshot rows, %d contracts",
        len(snapshot_rows), len(contract_rows),
    )

    snapshots_df = pd.DataFrame(snapshot_rows, columns=_OBS_COLUMNS) if snapshot_rows else pd.DataFrame(columns=_OBS_COLUMNS)
    contracts_df = pd.DataFrame(contract_rows, columns=_LMC_COLUMNS) if contract_rows else pd.DataFrame(columns=_LMC_COLUMNS)

    return snapshots_df, contracts_df


def save_to_parquet(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Write *df* to Parquet at *output_path*, creating parent dirs as needed.

    Args:
        df: DataFrame to write.
        output_path: Destination file path.

    Returns:
        Resolved :class:`~pathlib.Path` of the written file.

    Raises:
        ValueError: If *df* is empty.
    """
    if df.empty:
        raise ValueError("Cannot save an empty DataFrame to Parquet.")
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved %d rows -> %s", len(df), path)
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse
    import sys

    try:
        from dotenv import load_dotenv as _load_dotenv
        _load_dotenv()
    except ImportError:
        pass  # python-dotenv not installed; rely on shell env

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    p = argparse.ArgumentParser(
        description="Pull historical Kalshi market data (bid/ask snapshots).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--series", default="KXHIGHCHI",
                   help="Kalshi series ticker")
    p.add_argument("--start",  default="2023-05-01",
                   help="Start date (YYYY-MM-DD) for contract target_date filter")
    p.add_argument("--end",    default="2023-09-30",
                   help="End date (YYYY-MM-DD) for contract target_date filter")
    p.add_argument("--delay",  type=float, default=_REQUEST_DELAY_S,
                   help="Seconds between API requests (increase to avoid 429s)")
    p.add_argument("--out",
                   default="data/historical/markets/kalshi_kxhighchi_2023.parquet",
                   help="Output Parquet path for order_book_snapshot data")
    p.add_argument("--contracts-out",
                   default="data/historical/markets/kalshi_kxhighchi_2023_contracts.parquet",
                   help="Output Parquet path for live_market_contract data")
    p.add_argument("--discover", action="store_true",
                   help=(
                       "Discovery mode: scan all settled markets in --start/--end "
                       "and list series whose ticker or title contains 'CHI', 'HIGH', "
                       "or 'TEMP'.  Prints ticker, title, and contract count.  "
                       "Does not write any parquet file."
                   ))
    p.add_argument("--keywords", default="CHI,HIGH,TEMP",
                   help="Comma-separated keywords for --discover filtering (case-insensitive)")
    args = p.parse_args()

    try:
        start = date.fromisoformat(args.start)
        end   = date.fromisoformat(args.end)
    except ValueError as exc:
        print(f"ERROR: Invalid date — {exc}", file=sys.stderr)
        sys.exit(1)

    credentials = _load_credentials()
    if credentials is None:
        print(
            "ERROR: Kalshi credentials not found.  Set KALSHI_API_KEY_ID and "
            "KALSHI_PRIVATE_KEY_PATH (or KALSHI_PRIVATE_KEY) in your .env file.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Discovery mode ─────────────────────────────────────────────────────
    if args.discover:
        keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
        print(f"\nDiscovery scan: {start} to {end}  keywords={keywords}")
        print("(no parquet files will be written)\n")

        try:
            results = discover_series(
                start_date=start,
                end_date=end,
                keywords=keywords,
                credentials=credentials,
                request_delay_s=args.delay,
            )
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

        if not results:
            print("No matching series found.")
            sys.exit(0)

        # Print table
        col_w = (20, 45, 8)
        header = (
            f"{'SERIES':<{col_w[0]}}"
            f"{'TITLE':<{col_w[1]}}"
            f"{'CONTRACTS':>{col_w[2]}}"
        )
        print(header)
        print("-" * sum(col_w))
        for r in results:
            title_trunc = r["title"][:col_w[1] - 1] if r["title"] else "(no title)"
            print(
                f"{r['series']:<{col_w[0]}}"
                f"{title_trunc:<{col_w[1]}}"
                f"{r['contract_count']:>{col_w[2]}}"
            )
        print("-" * sum(col_w))
        print(f"{'Total series found:':<{col_w[0] + col_w[1]}}"
              f"{len(results):>{col_w[2]}}")
        print(
            "\nTo ingest a series, re-run without --discover:\n"
            f"  python -m deep_isobar.market.historical_kalshi_ingest "
            f"--series <TICKER> --start {start} --end {end} "
            f"--out data/historical/markets/kalshi_<ticker>_{start.year}.parquet"
        )
        sys.exit(0)

    # ── Normal ingest mode ─────────────────────────────────────────────────
    try:
        snapshots, contracts = fetch_historical_market_data(
            series_ticker=args.series,
            start_date=start,
            end_date=end,
            request_delay_s=args.delay,
        )
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    snap_path = save_to_parquet(snapshots, args.out)
    cont_path = save_to_parquet(contracts, args.contracts_out)

    print(f"\nSnapshots  : {len(snapshots):>6,} rows -> {snap_path}")
    print(f"Contracts  : {len(contracts):>6,} rows -> {cont_path}")
    print(f"\nSnapshot columns : {list(snapshots.columns)}")
    print(f"\nFirst 5 snapshot rows:")
    print(snapshots.head(5).to_string(index=False))
    print(f"\nFirst 5 contract rows:")
    print(contracts[["contract_id", "threshold_f", "target_date",
                      "listed_at_utc", "expires_at_utc"]].head(5).to_string(index=False))
