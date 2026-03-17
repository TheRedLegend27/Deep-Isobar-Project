"""Market microstructure scanner for Deep Isobar.

Computes order-book quality metrics from normalised
:class:`~deep_isobar.core.types.OrderBookSnapshot` objects.  This module
operates purely on already-fetched data — no network calls, no exchange
parsing, no signal generation.

Metric overview
---------------
All four public functions accept an ``OrderBookSnapshot`` whose ``best_bid``
and ``best_ask``, when present, are expected to be in **decimal probability
form** — i.e. floats in ``[0.0, 1.0]``.  (The :mod:`market_price_adapter`
handles conversion from cent-denominated exchanges like Kalshi before
snapshots reach this layer.)

+------------------------------+--------------------------------------------+
| Function                     | Direction                                  |
+==============================+============================================+
| ``compute_spread``           | lower = tighter / healthier                |
| ``compute_liquidity_score``  | higher = more liquid / healthier           |
| ``compute_staleness_seconds``| lower = fresher / healthier                |
| ``compute_microstructure_score`` | **higher = healthier market quality**  |
+------------------------------+--------------------------------------------+

``compute_microstructure_score`` is the composite output intended for
consumption by the alpha engine's ranking layer.  A score of 1.0 represents a
theoretically perfect market (zero spread, infinite liquidity, zero staleness);
0.0 represents the worst possible market on all three dimensions.

Example usage::

    from datetime import datetime, timezone
    from deep_isobar.core.types import OrderBookSnapshot
    from deep_isobar.market.microstructure_scanner import (
        compute_spread,
        compute_liquidity_score,
        compute_staleness_seconds,
        compute_microstructure_score,
    )

    snapshot = OrderBookSnapshot(
        timestamp_utc=datetime(2026, 3, 17, 12, 0, 0, tzinfo=timezone.utc),
        contract_id="CHI_HIGH_TEMP_F_GE_75_20260318",
        market_source="Kalshi",
        best_bid=0.47,
        best_ask=0.53,
        bid_size=150,
        ask_size=200,
        volume_24h=5000,
        open_interest=12000,
    )

    now = datetime(2026, 3, 17, 12, 5, 0, tzinfo=timezone.utc)
    spread   = compute_spread(snapshot)                      # 0.06
    liq      = compute_liquidity_score(snapshot)             # > 0
    staleness = compute_staleness_seconds(snapshot, now)     # 300.0
    score    = compute_microstructure_score(snapshot, now)   # ~0.85
"""

from __future__ import annotations

import logging
from datetime import datetime

from deep_isobar.core.types import OrderBookSnapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Composite-score tuning constants (module-level so tests can inspect them)
# ---------------------------------------------------------------------------

#: Spread at or above this value receives a spread_score of 0.0.
#: In decimal probability units [0, 1]; 0.50 = 50-cent spread.
MAX_SPREAD: float = 0.50

#: Staleness at or above this value receives a freshness_score of 0.0.
MAX_STALENESS_SECONDS: float = 3600.0

#: Liquidity score at or above this value receives a normalised score of 1.0.
#: Derived from: bid_size + ask_size + volume_24h * 0.01 + open_interest * 0.01
LIQUIDITY_CAP: float = 200.0


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _require_bid_ask(snapshot: OrderBookSnapshot) -> tuple[float, float]:
    """Return ``(best_bid, best_ask)`` or raise ``ValueError``.

    Args:
        snapshot: Order-book snapshot to inspect.

    Returns:
        ``(best_bid, best_ask)`` as floats.

    Raises:
        ValueError: If either price is ``None``, outside ``[0.0, 1.0]``, or
            ``best_bid > best_ask``.
    """
    bid = snapshot.best_bid
    ask = snapshot.best_ask

    if bid is None:
        raise ValueError(
            f"best_bid is None for contract={snapshot.contract_id!r}; "
            "cannot compute spread"
        )
    if ask is None:
        raise ValueError(
            f"best_ask is None for contract={snapshot.contract_id!r}; "
            "cannot compute spread"
        )

    if not (0.0 <= bid <= 1.0):
        raise ValueError(
            f"best_bid={bid!r} is outside [0.0, 1.0] "
            f"(contract={snapshot.contract_id!r}); "
            "ensure prices are in decimal probability form"
        )
    if not (0.0 <= ask <= 1.0):
        raise ValueError(
            f"best_ask={ask!r} is outside [0.0, 1.0] "
            f"(contract={snapshot.contract_id!r}); "
            "ensure prices are in decimal probability form"
        )
    if bid > ask:
        raise ValueError(
            f"best_bid={bid!r} > best_ask={ask!r} "
            f"(contract={snapshot.contract_id!r}); "
            "crossed order book is invalid"
        )

    return bid, ask


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def compute_spread(snapshot: OrderBookSnapshot) -> float:
    """Compute the bid-ask spread from a normalised order-book snapshot.

    Spread is defined as ``best_ask - best_bid`` in decimal probability units.
    A tighter (smaller) spread indicates a healthier market.

    Args:
        snapshot: Order-book snapshot with prices in ``[0.0, 1.0]``.  Both
            ``best_bid`` and ``best_ask`` must be present.

    Returns:
        Spread as a non-negative float in ``[0.0, 1.0]``.

    Raises:
        ValueError: If ``best_bid`` or ``best_ask`` is ``None``.
        ValueError: If either price is outside ``[0.0, 1.0]``.
        ValueError: If ``best_bid > best_ask`` (crossed order book).

    Example::

        spread = compute_spread(snapshot)   # → 0.06 for bid=0.47 ask=0.53
    """
    bid, ask = _require_bid_ask(snapshot)
    spread = ask - bid
    logger.debug(
        "compute_spread: contract=%s bid=%.4f ask=%.4f spread=%.4f",
        snapshot.contract_id,
        bid,
        ask,
        spread,
    )
    return spread


def compute_liquidity_score(snapshot: OrderBookSnapshot) -> float:
    """Compute a liquidity quality score from available order-book fields.

    **Score direction:** higher = more liquid / healthier market.

    Formula (MVP)::

        score = (bid_size or 0)
              + (ask_size or 0)
              + (volume_24h  or 0) * 0.01
              + (open_interest or 0) * 0.01

    Rationale for weights:

    - ``bid_size`` and ``ask_size`` are in contract units; their contribution
      is counted at face value.
    - ``volume_24h`` and ``open_interest`` are typically much larger numbers
      (dollar or contract totals), so they are down-weighted by 0.01 to keep
      them on a comparable scale to size fields.

    If all four optional fields are absent or zero the score is ``0.0``.  A
    missing field is treated as contributing zero rather than raising an error.

    Args:
        snapshot: Order-book snapshot.  All liquidity fields are optional.

    Returns:
        Liquidity score as a non-negative float.  Higher is more liquid.

    Example::

        score = compute_liquidity_score(snapshot)  # → 83.5 for typical values
    """
    score = 0.0

    if snapshot.bid_size is not None:
        score += snapshot.bid_size
    if snapshot.ask_size is not None:
        score += snapshot.ask_size
    if snapshot.volume_24h is not None:
        score += snapshot.volume_24h * 0.01
    if snapshot.open_interest is not None:
        score += snapshot.open_interest * 0.01

    logger.debug(
        "compute_liquidity_score: contract=%s bid_size=%s ask_size=%s "
        "volume_24h=%s open_interest=%s score=%.4f",
        snapshot.contract_id,
        snapshot.bid_size,
        snapshot.ask_size,
        snapshot.volume_24h,
        snapshot.open_interest,
        score,
    )
    return score


def compute_staleness_seconds(
    snapshot: OrderBookSnapshot,
    now_utc: datetime,
) -> float:
    """Compute the age of an order-book snapshot in seconds.

    Staleness is defined as ``(now_utc - snapshot.timestamp_utc).total_seconds()``.
    A lower value means the snapshot is fresher.

    Timezone handling:

    - Both ``now_utc`` and ``snapshot.timestamp_utc`` must have the same
      timezone awareness (both aware or both naive).  Mixing aware and naive
      datetimes raises ``ValueError``.

    Args:
        snapshot: Order-book snapshot containing ``timestamp_utc``.
        now_utc: The reference time to measure staleness against.  Must be
            timezone-consistent with ``snapshot.timestamp_utc``.

    Returns:
        Staleness in seconds as a non-negative float.

    Raises:
        ValueError: If ``now_utc`` is earlier than ``snapshot.timestamp_utc``
            (negative staleness is physically impossible).
        ValueError: If ``now_utc`` and ``snapshot.timestamp_utc`` have
            mismatched timezone awareness.

    Example::

        staleness = compute_staleness_seconds(snapshot, now_utc)  # → 300.0
    """
    snap_ts = snapshot.timestamp_utc

    # Guard against naive/aware mixing
    snap_aware = snap_ts.tzinfo is not None
    now_aware = now_utc.tzinfo is not None
    if snap_aware != now_aware:
        raise ValueError(
            "snapshot.timestamp_utc and now_utc must both be timezone-aware "
            "or both be timezone-naive; got "
            f"snapshot.tzinfo={snap_ts.tzinfo!r}, now_utc.tzinfo={now_utc.tzinfo!r}"
        )

    delta = (now_utc - snap_ts).total_seconds()
    if delta < 0:
        raise ValueError(
            f"now_utc ({now_utc.isoformat()}) is earlier than "
            f"snapshot.timestamp_utc ({snap_ts.isoformat()}); "
            "staleness cannot be negative"
        )

    logger.debug(
        "compute_staleness_seconds: contract=%s timestamp=%s now=%s staleness=%.1fs",
        snapshot.contract_id,
        snap_ts.isoformat(),
        now_utc.isoformat(),
        delta,
    )
    return delta


def compute_microstructure_score(
    snapshot: OrderBookSnapshot,
    now_utc: datetime,
) -> float:
    """Compute a composite market microstructure quality score.

    **Score direction:** higher = healthier market quality.  The alpha engine
    can use a low score to filter out or down-rank contracts that are difficult
    to trade (wide spreads, low liquidity, stale prices).

    The score is the unweighted mean of three normalised sub-scores, each in
    ``[0.0, 1.0]``::

        spread_score    = max(0, 1 - spread / MAX_SPREAD)
        freshness_score = max(0, 1 - staleness_seconds / MAX_STALENESS_SECONDS)
        liquidity_norm  = min(liquidity_score / LIQUIDITY_CAP, 1.0)

        microstructure_score = (spread_score + freshness_score + liquidity_norm) / 3

    Module-level constants ``MAX_SPREAD``, ``MAX_STALENESS_SECONDS``, and
    ``LIQUIDITY_CAP`` define the "worst acceptable" thresholds beyond which a
    sub-score is clamped to 0.0.

    If ``best_bid`` or ``best_ask`` is missing the spread sub-score is 0.0
    (worst case) rather than raising.  The function is intentionally lenient
    on missing data so a single missing field does not invalidate the whole
    score; callers can inspect ``compute_spread`` directly if they need strict
    validation.

    Args:
        snapshot: Order-book snapshot.  ``best_bid``/``best_ask`` should be in
            ``[0.0, 1.0]`` (decimal); see :func:`compute_spread` for details.
        now_utc: Current UTC time, timezone-consistent with
            ``snapshot.timestamp_utc``.

    Returns:
        Composite microstructure score in ``[0.0, 1.0]``.  Higher is healthier.

    Raises:
        ValueError: If ``now_utc`` is earlier than ``snapshot.timestamp_utc``.
        ValueError: If both bid and ask are present but invalid (out of range
            or crossed); see :func:`compute_spread`.
        ValueError: If ``now_utc`` and ``snapshot.timestamp_utc`` have
            mismatched timezone awareness.

    Example::

        score = compute_microstructure_score(snapshot, now_utc)  # → 0.88
    """
    # Spread sub-score: missing bid/ask → worst case (0.0).
    # Invalid or crossed prices (bid > ask, out of range) still raise so the
    # caller is not silently misled about a malformed snapshot.
    if snapshot.best_bid is None or snapshot.best_ask is None:
        spread_score = 0.0
        logger.debug(
            "compute_microstructure_score: contract=%s bid or ask missing — "
            "using spread_score=0.0",
            snapshot.contract_id,
        )
    else:
        spread = compute_spread(snapshot)  # raises on invalid/crossed
        spread_score = max(0.0, 1.0 - spread / MAX_SPREAD)

    # Liquidity sub-score (always succeeds; missing fields → 0 contribution)
    liq = compute_liquidity_score(snapshot)
    liquidity_norm = min(liq / LIQUIDITY_CAP, 1.0)

    # Freshness sub-score (may raise on bad timestamps)
    staleness = compute_staleness_seconds(snapshot, now_utc)
    freshness_score = max(0.0, 1.0 - staleness / MAX_STALENESS_SECONDS)

    composite = (spread_score + freshness_score + liquidity_norm) / 3.0

    logger.info(
        "compute_microstructure_score: contract=%s "
        "spread_score=%.4f freshness_score=%.4f liquidity_norm=%.4f composite=%.4f",
        snapshot.contract_id,
        spread_score,
        freshness_score,
        liquidity_norm,
        composite,
    )
    return composite
