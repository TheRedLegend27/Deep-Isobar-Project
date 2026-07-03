"""Kalshi prediction-market client for Deep Isobar.

Supports two modes selected automatically at call time:

**Live mode** — authenticated HTTP requests to the Kalshi v2 REST API.
Active when all of the following are true:

  1. ``KALSHI_API_KEY_ID`` environment variable is set.
  2. ``KALSHI_PRIVATE_KEY_PATH`` (path to RSA PEM file) **or**
     ``KALSHI_PRIVATE_KEY`` (inline PEM string) is set.
  3. The ``cryptography`` package is installed.
  4. ``markets.kalshi.stub_mode`` is not ``true`` in ``config/settings.yaml``.

**Stub mode** — deterministic mock data, no network calls.
Used when live-mode prerequisites are not met, or as an automatic
fallback when a live API call fails.

Canonical interface (from INTERFACES.md)::

    fetch_live_contracts(market_source) -> list[MarketContract]
    fetch_orderbook_for_contract(market_source, contract_id) -> OrderBookSnapshot

Kalshi API reference
--------------------
Base URL:  https://api.elections.kalshi.com/trade-api/v2
Auth:      RSA-PSS (SHA-256) — sign ``timestamp_ms + METHOD + path``
           where *path* is the full API path **without query parameters**
           e.g. ``/trade-api/v2/markets``, not ``/markets?status=open``.
Headers:   KALSHI-ACCESS-KEY, KALSHI-ACCESS-TIMESTAMP, KALSHI-ACCESS-SIGNATURE
Orderbook: ``GET /markets/{ticker}/orderbook`` returns ``orderbook_fp``
           with ``yes_dollars`` and ``no_dollars`` arrays of
           ``[price_str, count_str]`` pairs sorted **ascending**.
           Best YES bid  = last element of ``yes_dollars``.
           Best YES ask  = 1.0 − last element of ``no_dollars``.
Prices:    Dollar strings with up to 6 decimal places (e.g. ``"0.4200"``).

Weather market tickers
----------------------
Series: ``KXHIGHCHI`` (Chicago high temp), ``KXLOWCHI`` (Chicago low temp).
Ticker: ``KXHIGHCHI-26MAR18-B51.5`` where ``B51.5`` is the bracket floor (°F).
Markets are bracket-based (Kalshi scalar format).  Edge brackets map cleanly
to the ">= T" / "<= T" binary model used by this system.

Credential environment variables
---------------------------------
``KALSHI_API_KEY_ID``       — Key ID from kalshi.com/account/profile
``KALSHI_PRIVATE_KEY_PATH`` — path to RSA private key PEM file
``KALSHI_PRIVATE_KEY``      — inline PEM string (\\n-escaped newlines ok)
"""

from __future__ import annotations

import base64
import logging
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

import requests

from deep_isobar.config import get_setting
from deep_isobar.core.types import MarketContract, OrderBookSnapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Despite the "elections" subdomain this endpoint serves ALL Kalshi markets.
_KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
_MARKET_SOURCE = "Kalshi"
_TIMEOUT_SECONDS = 10
_MAX_PAGES = 5  # guard against runaway pagination

# Env var names for credentials
_ENV_KEY_ID = "KALSHI_API_KEY_ID"
_ENV_KEY_PATH = "KALSHI_PRIVATE_KEY_PATH"   # path to PEM file
_ENV_KEY_INLINE = "KALSHI_PRIVATE_KEY"      # inline PEM (\\n-escaped ok)

# ---------------------------------------------------------------------------
# Series metadata: parsed series name → domain model fields
# ---------------------------------------------------------------------------

# Add new cities / metrics / operators here when expanding beyond Chicago.
# Both KX-prefixed (current Kalshi naming) and bare names are included.
_SERIES_METADATA: dict[str, dict[str, str]] = {
    "KXHIGHCHI": {"city": "Chicago",      "metric": "high_temp_f", "comparison_operator": "ge", "settlement_source": "NWS"},
    "KXLOWCHI":  {"city": "Chicago",      "metric": "low_temp_f",  "comparison_operator": "le", "settlement_source": "NWS"},
    "HIGHCHI":   {"city": "Chicago",      "metric": "high_temp_f", "comparison_operator": "ge", "settlement_source": "NWS"},
    "LOWCHI":    {"city": "Chicago",      "metric": "low_temp_f",  "comparison_operator": "le", "settlement_source": "NWS"},
    "KXHIGHTDAL": {"city": "Dallas",      "metric": "high_temp_f", "comparison_operator": "ge", "settlement_source": "NWS"},
    "HIGHDAL":    {"city": "Dallas",      "metric": "high_temp_f", "comparison_operator": "ge", "settlement_source": "NWS"},
    "KXHIGHNY":   {"city": "New York",    "metric": "high_temp_f", "comparison_operator": "ge", "settlement_source": "NWS"},
    "HIGHNY":     {"city": "New York",    "metric": "high_temp_f", "comparison_operator": "ge", "settlement_source": "NWS"},
    "KXHIGHTBOS": {"city": "Boston",      "metric": "high_temp_f", "comparison_operator": "ge", "settlement_source": "NWS"},
    "KXHIGHBOS":  {"city": "Boston",      "metric": "high_temp_f", "comparison_operator": "ge", "settlement_source": "NWS"},
    "HIGHBOS":    {"city": "Boston",      "metric": "high_temp_f", "comparison_operator": "ge", "settlement_source": "NWS"},
    "KXHIGHPHIL": {"city": "Philadelphia","metric": "high_temp_f", "comparison_operator": "ge", "settlement_source": "NWS"},
    "HIGHPHIL":   {"city": "Philadelphia","metric": "high_temp_f", "comparison_operator": "ge", "settlement_source": "NWS"},
}

# Month abbreviation → month number (Kalshi uses 3-letter uppercase abbrevs)
_MONTH_MAP: dict[str, int] = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# Reverse map: month number → abbreviation (for normalization)
_MONTH_NUM_TO_ABV: dict[int, str] = {v: k for k, v in _MONTH_MAP.items()}

# Ticker regex — matches bracket (B) and threshold-edge (T) formats:
#   KXHIGHCHI-26MAR18-B51.5   — range bracket, floor in °F
#   KXHIGHCHI-26MAR17-T30     — edge bracket (above/below threshold)
# Groups: (series)(YY)(MMM)(DD)(bracket_type: B|T)(threshold — may be decimal)
_TICKER_RE = re.compile(
    r"^([A-Z0-9]+)-(\d{2})([A-Z]{3})(\d{2})-([BT])([\d.]+)$",
    re.IGNORECASE,
)

# Alternate ticker format returned by some cities (e.g. Boston):
#   KXHIGHBOS_HIGH_TEMP_F_GT_30_20260415
# Groups: (series)(threshold)(YYYY)(MM)(DD)
_ALT_TICKER_RE = re.compile(
    r"^([A-Z0-9]+)_HIGH_TEMP_F_GT_(\d+)_(\d{4})(\d{2})(\d{2})$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Stub constants
# ---------------------------------------------------------------------------

# Season-appropriate threshold ranges for Chicago high_temp_f stubs.
# Each season has 7 thresholds in 5°F increments centred on the typical
# seasonal high so that at least some contracts are near the money when
# running the pipeline locally in winter/spring/summer/fall.
_STUB_THRESHOLDS_BY_SEASON: dict[str, list[int]] = {
    "winter": [10, 15, 20, 25, 30, 35, 40],   # Dec / Jan / Feb
    "spring": [30, 35, 40, 45, 50, 55, 60],   # Mar / Apr / May
    "summer": [65, 70, 75, 80, 85, 90, 95],   # Jun / Jul / Aug
    "fall":   [35, 40, 45, 50, 55, 60, 65],   # Sep / Oct / Nov
}

_STUB_BID = 0.48   # decimal probability (matches live Kalshi dollar-string prices)
_STUB_ASK = 0.52   # decimal probability


def _stub_thresholds_for_date(target_date: date) -> list[int]:
    """Return season-appropriate stub thresholds for *target_date*."""
    month = target_date.month
    if month in (12, 1, 2):
        return _STUB_THRESHOLDS_BY_SEASON["winter"]
    if month in (3, 4, 5):
        return _STUB_THRESHOLDS_BY_SEASON["spring"]
    if month in (6, 7, 8):
        return _STUB_THRESHOLDS_BY_SEASON["summer"]
    return _STUB_THRESHOLDS_BY_SEASON["fall"]

# ---------------------------------------------------------------------------
# Optional dependency: cryptography (required for RSA-PSS auth)
# ---------------------------------------------------------------------------

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding as _asym_padding
    _CRYPTO_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CRYPTO_AVAILABLE = False
    logger.debug(
        "kalshi_client: 'cryptography' not installed — live mode unavailable"
    )

# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def _load_credentials() -> tuple[str, Any] | None:
    """Load Kalshi RSA credentials from environment variables.

    Returns:
        ``(key_id, private_key)`` when credentials are present and valid,
        ``None`` otherwise.
    """
    if not _CRYPTO_AVAILABLE:
        return None

    key_id = os.getenv(_ENV_KEY_ID, "").strip()
    if not key_id:
        return None

    pem: str | None = None
    key_path = os.getenv(_ENV_KEY_PATH, "").strip()
    if key_path:
        try:
            with open(key_path, encoding="utf-8") as fh:
                pem = fh.read()
        except OSError as exc:
            logger.warning(
                "kalshi_client: cannot read private key from %r — %s", key_path, exc
            )
            return None
    else:
        inline = os.getenv(_ENV_KEY_INLINE, "").strip()
        if inline:
            pem = inline.replace("\\n", "\n")

    if not pem:
        logger.warning(
            "kalshi_client: %s is set but no private key found; "
            "set %s or %s", _ENV_KEY_ID, _ENV_KEY_PATH, _ENV_KEY_INLINE,
        )
        return None

    try:
        private_key = serialization.load_pem_private_key(pem.encode(), password=None)
    except Exception as exc:
        logger.warning("kalshi_client: failed to parse private key — %s", exc)
        return None

    logger.debug("kalshi_client: credentials loaded for key_id=%r", key_id)
    return key_id, private_key


def _use_stub_mode() -> bool:
    """Return True if stub mode is forced via config."""
    return bool(get_setting("markets.kalshi.stub_mode", False))


def is_live_mode() -> bool:
    """True when this client would hit the real Kalshi API (not the stub).

    Public check for callers that must never act on stub data — the
    pre-trade circuit breakers and the orderbook collector.  Mirrors the
    decision the fetch functions make internally: stub is used when forced
    via config or when no usable credentials load.
    """
    return not _use_stub_mode() and _load_credentials() is not None


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------


def _make_auth_headers(
    key_id: str,
    private_key: Any,
    method: str,
    sign_path: str,
) -> dict[str, str]:
    """Build Kalshi v2 RSA-PSS authentication headers.

    Signs ``timestamp_ms + METHOD + sign_path`` using RSA-PSS / SHA-256.

    Args:
        key_id: Kalshi API key ID.
        private_key: RSA private key object from ``cryptography``.
        method: HTTP method (``"GET"``, ``"POST"``, …).
        sign_path: Full API path **without** query parameters, e.g.
            ``"/trade-api/v2/markets"`` — not ``"/markets?status=open"``.
    """
    timestamp_ms = str(int(time.time() * 1000))
    message = (timestamp_ms + method.upper() + sign_path).encode()
    signature = private_key.sign(
        message,
        _asym_padding.PSS(
            mgf=_asym_padding.MGF1(hashes.SHA256()),
            salt_length=_asym_padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
        "Content-Type": "application/json",
    }


def _kalshi_get(
    path: str,
    params: dict[str, Any] | None,
    credentials: tuple[str, Any],
) -> dict[str, Any]:
    """Make an authenticated GET request to the Kalshi v2 REST API.

    Constructs the signing path as ``/trade-api/v2{path}`` (the full API
    path without query parameters) per Kalshi's signature requirements.

    Args:
        path: Endpoint path without base URL, e.g. ``"/markets"``.
        params: Optional query parameters.
        credentials: ``(key_id, private_key)`` from :func:`_load_credentials`.

    Returns:
        Parsed JSON response dict.

    Raises:
        RuntimeError: On network error or non-200 HTTP status.
    """
    key_id, private_key = credentials
    base_url = get_setting("markets.kalshi.base_url", _KALSHI_BASE_URL)
    url = f"{base_url}{path}"

    # Signature uses the FULL path prefix (e.g. /trade-api/v2) without query string.
    base_path = urlparse(base_url).path.rstrip("/")  # e.g. "/trade-api/v2"
    sign_path = base_path + path                     # e.g. "/trade-api/v2/markets"

    headers = _make_auth_headers(key_id, private_key, "GET", sign_path)

    try:
        response = requests.get(
            url,
            headers=headers,
            params=params or {},
            timeout=_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Kalshi API network error on GET {path}: {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"Kalshi API error: GET {path} → HTTP {response.status_code}: "
            f"{response.text[:200]}"
        )

    return response.json()


# ---------------------------------------------------------------------------
# Ticker and contract parsing
# ---------------------------------------------------------------------------


def _normalize_ticker(ticker: str) -> str:
    """Normalize alternate Kalshi ticker formats to the standard format.

    Some cities (e.g. Boston) return tickers in the form::

        KXHIGHBOS_HIGH_TEMP_F_GT_30_20260415

    This function converts those to the canonical dash-separated format::

        KXHIGHBOS-26APR15-T30

    Standard-format tickers are returned unchanged.

    Args:
        ticker: Raw ticker string from the API or internal source.

    Returns:
        Normalized ticker string in ``SERIES-YYMMMDD-T{n}`` format,
        or the original string if it doesn't match the alternate pattern.
    """
    t = ticker.upper().strip()
    match = _ALT_TICKER_RE.match(t)
    if not match:
        return t  # already standard format or unknown — pass through

    series = match.group(1)
    threshold = match.group(2)
    year = int(match.group(3))
    month = int(match.group(4))
    day = int(match.group(5))

    month_abbv = _MONTH_NUM_TO_ABV.get(month)
    if month_abbv is None:
        logger.debug("_normalize_ticker: unknown month %d in ticker %r", month, ticker)
        return t  # can't normalize, return as-is

    yy = str(year)[-2:]
    normalized = f"{series}-{yy}{month_abbv}{day:02d}-T{threshold}"
    logger.debug("_normalize_ticker: %r → %r", ticker, normalized)
    return normalized


def _parse_ticker(ticker: str) -> dict[str, Any] | None:
    """Extract structured fields from a Kalshi weather contract ticker.

    Handles the current Kalshi format: ``KXHIGHCHI-26MAR18-B51.5``

    The bracket suffix ``B51.5`` is a temperature floor in °F.  It is
    rounded to the nearest integer to populate ``threshold_f`` so it fits
    the existing ``MarketContract`` integer field.

    Args:
        ticker: Raw Kalshi exchange ticker string.

    Returns:
        Dict with ``series`` (str), ``target_date`` (date),
        ``threshold_f`` (int) on success, or ``None`` on parse failure.

    Example::

        _parse_ticker("KXHIGHCHI-26MAR18-B51.5")
        # → {"series": "KXHIGHCHI", "target_date": date(2026, 3, 18), "threshold_f": 52}
    """
    ticker = _normalize_ticker(ticker)
    match = _TICKER_RE.match(ticker.upper().strip())
    if not match:
        logger.debug("_parse_ticker: no match for ticker %r", ticker)
        return None

    series = match.group(1)
    year = 2000 + int(match.group(2))
    month_str = match.group(3).upper()
    day = int(match.group(4))

    month = _MONTH_MAP.get(month_str)
    if month is None:
        logger.debug("_parse_ticker: unknown month %r in ticker %r", month_str, ticker)
        return None

    try:
        target_date = date(year, month, day)
    except ValueError as exc:
        logger.debug("_parse_ticker: invalid date in ticker %r — %s", ticker, exc)
        return None

    try:
        threshold_f = round(float(match.group(6)))
    except ValueError:
        logger.debug("_parse_ticker: unparseable threshold in ticker %r", ticker)
        return None

    return {"series": series, "target_date": target_date, "threshold_f": threshold_f}


def _parse_contract(market: dict[str, Any], now_utc: datetime) -> MarketContract | None:
    """Parse a single Kalshi market API dict into a :class:`MarketContract`.

    Skips markets whose ticker cannot be parsed or whose series is not
    in :data:`_SERIES_METADATA`.

    Args:
        market: One element from the ``markets`` list in a Kalshi API response.
        now_utc: Reference timestamp (unused currently, reserved for future use).

    Returns:
        A populated :class:`~deep_isobar.core.types.MarketContract`,
        or ``None`` if the market cannot be mapped to the Deep Isobar schema.
    """
    raw_ticker: str = market.get("ticker", "")
    ticker = _normalize_ticker(raw_ticker)  # normalize alternate formats (e.g. Boston)
    parsed = _parse_ticker(ticker)
    if parsed is None:
        logger.debug("_parse_contract: skipping unparseable ticker %r", raw_ticker)
        return None

    # Read strike_type from the API response.  Kalshi returns "less", "greater",
    # or "between" (bracket).  All three are now supported.
    strike_type: str = (market.get("strike_type") or "").lower()
    if not strike_type:
        logger.debug("_parse_contract: missing strike_type for %r — skipping", ticker)
        return None
    if strike_type not in ("less", "greater", "between"):
        logger.debug(
            "_parse_contract: unknown strike_type %r for %r — skipping",
            strike_type, ticker,
        )
        return None

    # Parse floor/cap strike boundaries from the API response (integers or null).
    floor_strike_raw = market.get("floor_strike")
    cap_strike_raw   = market.get("cap_strike")
    floor_strike: int | None = int(floor_strike_raw) if floor_strike_raw is not None else None
    cap_strike:   int | None = int(cap_strike_raw)   if cap_strike_raw   is not None else None

    meta = _SERIES_METADATA.get(parsed["series"])
    if meta is None:
        logger.debug(
            "_parse_contract: unknown series %r — skipping %r", parsed["series"], ticker
        )
        return None

    status: str = market.get("status", "active")
    active = status.lower() in ("active", "open", "")

    # Parse optional ISO-8601 timestamps.
    # Kalshi v2 field names: created_time → listed_at, latest_expiration_time → expires_at
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

    # Map strike_type → legacy comparison_operator kept for TradeSignal compat.
    _OP_MAP: dict[str, str] = {"less": "lt", "greater": "gt"}
    comparison_operator = _OP_MAP.get(strike_type, strike_type)

    return MarketContract(
        contract_id=ticker,
        market_source=_MARKET_SOURCE,
        city=meta["city"],
        metric=meta["metric"],
        comparison_operator=comparison_operator,
        threshold_f=parsed["threshold_f"],
        target_date=parsed["target_date"],
        settlement_source=meta["settlement_source"],
        strike_type=strike_type,
        floor_strike=floor_strike,
        cap_strike=cap_strike,
        raw_title=market.get("rules_primary") or market.get("title"),
        listed_at_utc=listed_at_utc,
        expires_at_utc=expires_at_utc,
        active=active,
    )


# ---------------------------------------------------------------------------
# Orderbook parsing
# ---------------------------------------------------------------------------


def _best_bid_from_yes(yes_levels: list) -> float | None:
    """Return the best YES bid (dollars, 0–1) from Kalshi YES orderbook levels.

    Kalshi returns ``yes_dollars`` as ``[price_str, count_str]`` pairs
    sorted **ascending** — the highest (best) bid is the **last** element.

    Args:
        yes_levels: List of ``[price_str, count_str]`` pairs from the API.

    Returns:
        Best bid as a float in [0, 1], or ``None`` if no levels exist.
    """
    if not yes_levels:
        return None
    try:
        return float(max(float(level[0]) for level in yes_levels))
    except (IndexError, TypeError, ValueError):
        logger.debug("_best_bid_from_yes: malformed yes_levels: %r", yes_levels)
        return None


def _best_ask_from_no(no_levels: list) -> float | None:
    """Derive best YES ask (dollars) from Kalshi NO orderbook levels.

    In Kalshi's complementary-pricing model:
    ``YES_ask = $1.00 − best_NO_bid``

    Kalshi returns ``no_dollars`` sorted **ascending** — best NO bid is last.

    Args:
        no_levels: List of ``[price_str, count_str]`` pairs for the NO side.

    Returns:
        Implied YES ask as a float in [0, 1], or ``None`` if no levels exist.
    """
    if not no_levels:
        return None
    try:
        best_no_bid = max(float(level[0]) for level in no_levels)
        return round(1.0 - best_no_bid, 10)  # avoid float drift
    except (IndexError, TypeError, ValueError):
        logger.debug("_best_ask_from_no: malformed no_levels: %r", no_levels)
        return None


def _parse_orderbook_response(
    contract_id: str,
    data: dict[str, Any],
    now_utc: datetime,
) -> OrderBookSnapshot:
    """Parse a Kalshi orderbook API response into an :class:`OrderBookSnapshot`.

    Response key is ``orderbook_fp`` with sub-fields ``yes_dollars`` and
    ``no_dollars``.  Prices are dollar strings in [0, 1] — they are stored
    as-is so :func:`~deep_isobar.market.market_price_adapter.compute_market_probability`
    receives values ≤ 1.0 and uses them directly.

    Arrays are sorted ascending; best bid is the last element.

    Args:
        contract_id: Exchange ticker used to populate ``contract_id``.
        data: Parsed JSON from ``GET /markets/{ticker}/orderbook``.
        now_utc: Snapshot timestamp.

    Returns:
        An :class:`~deep_isobar.core.types.OrderBookSnapshot`.
    """
    ob = data.get("orderbook_fp", {})
    yes_levels: list = ob.get("yes_dollars") or []
    no_levels: list = ob.get("no_dollars") or []

    best_bid = _best_bid_from_yes(yes_levels)
    best_ask = _best_ask_from_no(no_levels)

    # Guard: crossed market (bad data) — drop ask rather than raise
    if best_bid is not None and best_ask is not None and best_bid > best_ask:
        logger.warning(
            "_parse_orderbook_response: crossed market for %r "
            "(bid=%.4f > ask=%.4f) — dropping ask",
            contract_id, best_bid, best_ask,
        )
        best_ask = None

    # Sizes: arrays sorted ascending so best bid/ask size is at [-1]
    bid_size: float | None = None
    ask_size: float | None = None
    try:
        if yes_levels:
            bid_size = float(yes_levels[-1][1])
    except (IndexError, TypeError, ValueError):
        pass
    try:
        if no_levels:
            ask_size = float(no_levels[-1][1])
    except (IndexError, TypeError, ValueError):
        pass

    return OrderBookSnapshot(
        timestamp_utc=now_utc,
        contract_id=contract_id,
        market_source=_MARKET_SOURCE,
        best_bid=best_bid,
        best_ask=best_ask,
        bid_size=bid_size,
        ask_size=ask_size,
    )


# ---------------------------------------------------------------------------
# Live API calls
# ---------------------------------------------------------------------------


def _fetch_live_contracts_from_api(
    credentials: tuple[str, Any],
    series_ticker: str | None = None,
) -> list[MarketContract]:
    """Fetch and parse open weather contracts for one series from the Kalshi API.

    Paginates automatically (up to :data:`_MAX_PAGES` pages).

    Args:
        credentials: ``(key_id, private_key)`` from :func:`_load_credentials`.
        series_ticker: Kalshi series identifier (e.g. ``"KXHIGHCHI"``).
            When ``None``, reads ``markets.kalshi.series_ticker`` from
            ``config/settings.yaml`` (default: ``"KXHIGHCHI"``).

    Returns:
        List of parsed :class:`~deep_isobar.core.types.MarketContract` objects.
    """
    series_ticker = series_ticker or get_setting("markets.kalshi.series_ticker", "KXHIGHCHI")
    now_utc = datetime.now(timezone.utc)
    contracts: list[MarketContract] = []
    cursor: str | None = None

    for page in range(_MAX_PAGES):
        params: dict[str, Any] = {
            "series_ticker": series_ticker,
            "status": "open",
            "limit": 100,
        }
        if cursor:
            params["cursor"] = cursor

        logger.debug(
            "_fetch_live_contracts_from_api: GET /markets series=%s page=%d",
            series_ticker, page + 1,
        )

        data = _kalshi_get("/markets", params, credentials)

        for market in data.get("markets", []):
            contract = _parse_contract(market, now_utc)
            if contract is not None:
                contracts.append(contract)

        cursor = data.get("cursor") or None
        if not cursor:
            break

    logger.info(
        "_fetch_live_contracts_from_api: parsed %d contracts", len(contracts)
    )
    return contracts


def _fetch_orderbook_from_api(
    contract_id: str,
    credentials: tuple[str, Any],
) -> OrderBookSnapshot:
    """Fetch the live order book for a single contract from the Kalshi API.

    Args:
        contract_id: Kalshi exchange ticker, e.g. ``"KXHIGHCHI-26MAR18-B51.5"``.
        credentials: ``(key_id, private_key)`` from :func:`_load_credentials`.

    Returns:
        Parsed :class:`~deep_isobar.core.types.OrderBookSnapshot`.

    Raises:
        RuntimeError: If the API call fails.
    """
    now_utc = datetime.now(timezone.utc)
    path = f"/markets/{contract_id}/orderbook"
    logger.debug("_fetch_orderbook_from_api: GET %s", path)
    data = _kalshi_get(path, {"depth": 1}, credentials)
    return _parse_orderbook_response(contract_id, data, now_utc)


# ---------------------------------------------------------------------------
# Stub implementations
# ---------------------------------------------------------------------------


def _stub_fetch_live_contracts(series_ticker: str | None = None) -> list[MarketContract]:
    """Return deterministic mock high_temp_f contracts (stub mode).

    Returns 7 contracts in a season-appropriate 5°F threshold range for
    tomorrow's date.  Thresholds are chosen so that at least some contracts
    are near the money when running the pipeline locally in any season.
    The city is derived from *series_ticker* via :data:`_SERIES_METADATA`
    when provided, defaulting to Chicago.
    """
    target_date = date.today() + timedelta(days=1)
    thresholds = _stub_thresholds_for_date(target_date)

    meta = _SERIES_METADATA.get(series_ticker.upper() if series_ticker else "") or {}
    city = meta.get("city", "Chicago")
    city_tag = (series_ticker or "CHI").upper()

    # Build canonical ticker: KXHIGHBOS-26APR15-T30 (not the legacy underscore format)
    month_abbv = _MONTH_NUM_TO_ABV[target_date.month]
    yy = str(target_date.year)[-2:]
    dd = f"{target_date.day:02d}"

    logger.info(
        "fetch_live_contracts [STUB]: target_date=%s thresholds=%s",
        target_date, thresholds,
    )
    return [
        MarketContract(
            contract_id=f"{city_tag}-{yy}{month_abbv}{dd}-T{t}",
            market_source=_MARKET_SOURCE,
            city=city,
            metric="high_temp_f",
            comparison_operator="gt",
            threshold_f=t,
            target_date=target_date,
            settlement_source="NWS",
            strike_type="greater",
            floor_strike=t,
            cap_strike=None,
            raw_title=f"{city} High > {t}°F on {target_date.isoformat()}",
            active=True,
        )
        for t in thresholds
    ]


def _stub_fetch_orderbook(contract_id: str) -> OrderBookSnapshot:
    """Return a synthetic order-book snapshot (stub mode).

    Prices are in decimal probability form (0.48/0.52) matching the live
    Kalshi dollar-string format, so mid=0.50 → market probability 0.50.
    """
    logger.debug("fetch_orderbook_for_contract [STUB]: contract=%s", contract_id)
    return OrderBookSnapshot(
        timestamp_utc=datetime.now(timezone.utc),
        contract_id=contract_id,
        market_source=_MARKET_SOURCE,
        best_bid=_STUB_BID,
        best_ask=_STUB_ASK,
        last_trade_price=(_STUB_BID + _STUB_ASK) / 2,
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def fetch_live_contracts(
    market_source: str,
    series_ticker: str | None = None,
) -> list[MarketContract]:
    """Return live temperature contracts for the given market source and series.

    In **live mode** (credentials present and valid), calls
    ``GET /markets`` filtered to *series_ticker*.
    Falls back to stub data automatically on any API error.

    In **stub mode**, returns 7 deterministic ``high_temp_f >= T`` contracts
    for tomorrow derived from *series_ticker* without any network call.

    Args:
        market_source: Exchange identifier.  Only ``"Kalshi"`` (case-
            insensitive) is accepted; any other value raises immediately.
        series_ticker: Kalshi series (e.g. ``"KXHIGHCHI"``, ``"KXHIGHTDAL"``).
            When ``None``, reads ``markets.kalshi.series_ticker`` from
            ``config/settings.yaml``.

    Returns:
        List of :class:`~deep_isobar.core.types.MarketContract` objects.

    Raises:
        RuntimeError: If ``market_source`` is not ``"Kalshi"``.

    Example::

        contracts = fetch_live_contracts("Kalshi", series_ticker="KXHIGHTDAL")
        for c in contracts:
            print(c.contract_id, c.threshold_f)
    """
    if market_source.lower() != "kalshi":
        raise RuntimeError(
            f"market_source {market_source!r} is not supported by this client. "
            "Only 'Kalshi' is accepted."
        )

    if _use_stub_mode():
        logger.info("fetch_live_contracts: stub_mode=true in config — using stub")
        return _stub_fetch_live_contracts(series_ticker)

    credentials = _load_credentials()
    if credentials is None:
        logger.info(
            "fetch_live_contracts: no usable credentials found — using stub mode"
        )
        return _stub_fetch_live_contracts(series_ticker)

    logger.info("fetch_live_contracts: live mode — calling Kalshi API")
    try:
        contracts = _fetch_live_contracts_from_api(credentials, series_ticker=series_ticker)
        if not contracts:
            logger.warning(
                "fetch_live_contracts: API returned 0 parseable contracts — "
                "falling back to stub"
            )
            return _stub_fetch_live_contracts(series_ticker)
        return contracts
    except Exception as exc:
        logger.warning(
            "fetch_live_contracts: live API call failed (%s: %s) — "
            "falling back to stub",
            type(exc).__name__, exc,
        )
        return _stub_fetch_live_contracts(series_ticker)


def fetch_orderbook_for_contract(
    market_source: str,
    contract_id: str,
) -> OrderBookSnapshot:
    """Return the current order-book snapshot for a contract.

    In **live mode**, calls ``GET /markets/{ticker}/orderbook`` and parses
    the Kalshi v2 ``orderbook_fp`` response (dollar-denominated prices).
    Falls back to stub data on any API failure.

    In **stub mode**, returns bid=48¢, ask=52¢ → mid=50¢ → market
    probability 0.50.

    Args:
        market_source: Exchange identifier.  Only ``"Kalshi"`` accepted.
        contract_id: Kalshi ticker (live) or internal ID (stub).

    Returns:
        An :class:`~deep_isobar.core.types.OrderBookSnapshot`.

    Raises:
        RuntimeError: If ``market_source`` is not ``"Kalshi"``.

    Example::

        ob = fetch_orderbook_for_contract("Kalshi", "KXHIGHCHI-26MAR18-B51.5")
        print(ob.best_bid, ob.best_ask)   # → 0.48, 0.52  (live example)
    """
    if market_source.lower() != "kalshi":
        raise RuntimeError(
            f"market_source {market_source!r} is not supported by this client."
        )

    if _use_stub_mode():
        return _stub_fetch_orderbook(contract_id)

    credentials = _load_credentials()
    if credentials is None:
        return _stub_fetch_orderbook(contract_id)

    try:
        snapshot = _fetch_orderbook_from_api(contract_id, credentials)
        if (
            snapshot.best_bid is None
            and snapshot.best_ask is None
            and snapshot.last_trade_price is None
        ):
            logger.warning(
                "fetch_orderbook_for_contract: live API returned empty orderbook "
                "for %r — falling back to stub",
                contract_id,
            )
            return _stub_fetch_orderbook(contract_id)
        return snapshot
    except Exception as exc:
        logger.warning(
            "fetch_orderbook_for_contract: live API call failed for %r "
            "(%s: %s) — falling back to stub",
            contract_id, type(exc).__name__, exc,
        )
        return _stub_fetch_orderbook(contract_id)


def get_balance() -> dict | None:
    """Return Kalshi account balance and open-position portfolio value.

    Calls ``GET /trade-api/v2/portfolio/balance`` (authenticated, RSA-PSS).
    Kalshi returns amounts in cents; this function converts to dollars.

    Returns:
        Dict with keys ``balance_usd`` (available cash), ``portfolio_value_usd``
        (open position value), and ``total_usd`` (sum of both).
        Returns the same zero-valued dict in stub mode.
        Returns ``None`` on any live API error — never raises.

    Example::

        bal = get_balance()
        # → {"balance_usd": 125.50, "portfolio_value_usd": 45.00, "total_usd": 170.50}
    """
    _stub_response: dict = {"balance_usd": 0.0, "portfolio_value_usd": 0.0, "total_usd": 0.0}

    if _use_stub_mode():
        logger.debug("get_balance: stub_mode=true — returning zero balances")
        return _stub_response

    credentials = _load_credentials()
    if credentials is None:
        logger.debug("get_balance: no credentials — returning zero balances")
        return _stub_response

    try:
        data = _kalshi_get("/portfolio/balance", None, credentials)
        balance_cents = data.get("balance", 0) or 0
        portfolio_cents = data.get("portfolio_value", 0) or 0
        balance_usd = round(balance_cents / 100.0, 2)
        portfolio_usd = round(portfolio_cents / 100.0, 2)
        return {
            "balance_usd": balance_usd,
            "portfolio_value_usd": portfolio_usd,
            "total_usd": round(balance_usd + portfolio_usd, 2),
        }
    except Exception as exc:
        logger.warning("get_balance: API call failed — %s: %s", type(exc).__name__, exc)
        return None
