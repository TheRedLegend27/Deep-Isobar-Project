"""Market price adapter for Deep Isobar.

Converts raw order-book snapshots into normalized market-implied probabilities.
This is a pure computation layer with no network calls — it expects an already-
fetched :class:`~deep_isobar.core.types.OrderBookSnapshot`.

Canonical interface (from INTERFACES.md)::

    validate_orderbook(snapshot) -> None
    compute_mid_price(snapshot)  -> float
    compute_market_probability(snapshot) -> float

Market probabilities are expressed as floats in [0.0, 1.0], consistent with
all other probability inputs in the system.

On Kalshi, contract prices are quoted in cents (0–100) and represent the
implied probability directly.  This adapter normalises the mid-price to a
[0, 1] probability by dividing by 100 when both bid and ask are detected to
be in cent-denominated form (i.e. > 1.0).  If bid and ask are already in
decimal form ([0, 1]), they are used as-is.  The adapter infers which regime
applies automatically.
"""

from __future__ import annotations

import logging

from deep_isobar.core.types import OrderBookSnapshot

logger = logging.getLogger(__name__)


def validate_orderbook(snapshot: OrderBookSnapshot) -> None:
    """Validate an order-book snapshot before probability extraction.

    Checks:
    - ``best_bid`` and ``best_ask``, if present, are in [0.0, 100.0]
    - ``best_bid <= best_ask`` when both are present

    Args:
        snapshot: The order-book snapshot to validate.

    Raises:
        ValueError: If bid or ask is out of range, or bid exceeds ask.

    Example::

        validate_orderbook(snapshot)   # raises if snapshot is malformed
    """
    bid = snapshot.best_bid
    ask = snapshot.best_ask

    if bid is not None and not (0.0 <= bid <= 100.0):
        raise ValueError(
            f"best_bid must be in [0, 100], got {bid} "
            f"(contract={snapshot.contract_id})"
        )
    if ask is not None and not (0.0 <= ask <= 100.0):
        raise ValueError(
            f"best_ask must be in [0, 100], got {ask} "
            f"(contract={snapshot.contract_id})"
        )
    if bid is not None and ask is not None and bid > ask:
        raise ValueError(
            f"best_bid ({bid}) must not exceed best_ask ({ask}) "
            f"(contract={snapshot.contract_id})"
        )

    logger.debug(
        "validate_orderbook: contract=%s bid=%s ask=%s OK",
        snapshot.contract_id,
        bid,
        ask,
    )


def compute_mid_price(snapshot: OrderBookSnapshot) -> float:
    """Compute the mid-price from a validated order-book snapshot.

    Uses the arithmetic midpoint ``(bid + ask) / 2`` when both sides are
    present.  Falls back to whichever side is available, or to
    ``last_trade_price`` if neither bid nor ask is set.

    Args:
        snapshot: A validated :class:`~deep_isobar.core.types.OrderBookSnapshot`.

    Returns:
        Mid-price in the same units as bid/ask (cents or decimal).

    Raises:
        ValueError: If no usable price is available.

    Example::

        mid = compute_mid_price(snapshot)   # → 55.0  (cents) or 0.55 (decimal)
    """
    bid = snapshot.best_bid
    ask = snapshot.best_ask

    if bid is not None and ask is not None:
        mid = (bid + ask) / 2.0
    elif bid is not None:
        mid = bid
    elif ask is not None:
        mid = ask
    elif snapshot.last_trade_price is not None:
        mid = snapshot.last_trade_price
    else:
        raise ValueError(
            f"No usable price in snapshot for contract={snapshot.contract_id}; "
            "best_bid, best_ask, and last_trade_price are all None"
        )

    logger.debug(
        "compute_mid_price: contract=%s bid=%s ask=%s mid=%.4f",
        snapshot.contract_id,
        bid,
        ask,
        mid,
    )
    return mid


def compute_market_probability(snapshot: OrderBookSnapshot) -> float:
    """Derive a market-implied probability from an order-book snapshot.

    Validates the snapshot, computes the mid-price, then normalises to
    [0.0, 1.0]:

    - If mid-price > 1.0 (cent-denominated, e.g. Kalshi): divide by 100.
    - If mid-price ≤ 1.0 (decimal form): use as-is.

    The result is clamped to [0.0, 1.0] to guard against bad tick data.

    Args:
        snapshot: A :class:`~deep_isobar.core.types.OrderBookSnapshot` with at
            least one of ``best_bid``, ``best_ask``, or ``last_trade_price``.

    Returns:
        Market-implied probability as a float in [0.0, 1.0].

    Raises:
        ValueError: If the snapshot is malformed or contains no usable price.

    Example::

        prob = compute_market_probability(snapshot)  # → 0.55
    """
    validate_orderbook(snapshot)
    mid = compute_mid_price(snapshot)

    # Normalise: Kalshi prices are 0–100 cents; decimal prices are 0.0–1.0
    if mid > 1.0:
        probability = mid / 100.0
    else:
        probability = mid

    probability = max(0.0, min(1.0, probability))

    logger.debug(
        "compute_market_probability: contract=%s mid=%.4f probability=%.4f",
        snapshot.contract_id,
        mid,
        probability,
    )
    return probability
