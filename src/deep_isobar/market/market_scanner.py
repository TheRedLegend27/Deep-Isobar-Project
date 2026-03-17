"""Market scanner for Deep Isobar.

Evaluates individual prediction-market contracts for alpha (mispricing)
and ranks the resulting opportunities.

This module is the orchestration layer that wires together:
- ``market_price_adapter`` — extracts market-implied probabilities
- ``alpha_engine``          — computes alpha and classifies the signal

It does **not** contain risk logic, order submission, or scheduling.

Canonical interface (from INTERFACES.md)::

    evaluate_contract_opportunity(
        contract, probability_surface, orderbook,
        signal_threshold, timestamp_utc
    ) -> TradeSignal

    rank_trade_signals(signals) -> list[TradeSignal]

Pipeline for a single contract::

    probability_surface[threshold_f]  →  model_probability
    orderbook (bid/ask)               →  market_probability   (via adapter)
    (model_prob, market_prob)         →  alpha + side         (via alpha_engine)
    all of the above                  →  TradeSignal
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from deep_isobar.core.types import MarketContract, OrderBookSnapshot, TradeSignal
from deep_isobar.market.market_price_adapter import compute_market_probability
from deep_isobar.trading.alpha_engine import build_trade_signal

logger = logging.getLogger(__name__)


def evaluate_contract_opportunity(
    contract: MarketContract,
    probability_surface: dict[int, float],
    orderbook: OrderBookSnapshot,
    signal_threshold: float,
    timestamp_utc: datetime | None = None,
) -> TradeSignal:
    """Evaluate a single contract for a trading opportunity.

    Looks up the model probability for ``contract.threshold_f`` in the
    probability surface, derives the market probability from the order book,
    then delegates to :func:`~deep_isobar.trading.alpha_engine.build_trade_signal`
    to compute alpha and produce a classified :class:`~deep_isobar.core.types.TradeSignal`.

    The ``confidence_score`` on the returned signal is set to the absolute
    value of alpha for the MVP (a simple proxy for conviction).

    Args:
        contract: The market contract to evaluate.
        probability_surface: Mapping of ``threshold_f`` → model probability.
            Must contain an entry for ``contract.threshold_f``.
        orderbook: Current order-book snapshot for this contract.
        signal_threshold: Minimum |alpha| required to trigger BUY or SELL.
        timestamp_utc: Evaluation timestamp.  Defaults to
            ``datetime.now(timezone.utc)`` if not supplied.

    Returns:
        A :class:`~deep_isobar.core.types.TradeSignal` with alpha, side, and
        all contract metadata populated.

    Raises:
        KeyError: If ``contract.threshold_f`` is not present in
            ``probability_surface``.
        ValueError: If the order book contains no usable price, probabilities
            are out of range, or ``signal_threshold`` is not positive.

    Example::

        signal = evaluate_contract_opportunity(
            contract=contract,
            probability_surface={80: 0.62, 85: 0.31},
            orderbook=orderbook,
            signal_threshold=0.10,
        )
        # signal.signal_side → "BUY" / "SELL" / "HOLD"
    """
    if timestamp_utc is None:
        timestamp_utc = datetime.now(timezone.utc)

    threshold = contract.threshold_f
    if threshold not in probability_surface:
        raise KeyError(
            f"threshold_f={threshold} not found in probability_surface for "
            f"contract={contract.contract_id}. "
            f"Available thresholds: {sorted(probability_surface)}"
        )

    model_probability = probability_surface[threshold]
    market_probability = compute_market_probability(orderbook)

    logger.debug(
        "evaluate_contract_opportunity: contract=%s threshold=%d "
        "model=%.4f market=%.4f",
        contract.contract_id,
        threshold,
        model_probability,
        market_probability,
    )

    # MVP confidence score: |alpha| as a proxy for conviction strength
    alpha = model_probability - market_probability
    confidence_score = abs(alpha)

    return build_trade_signal(
        timestamp_utc=timestamp_utc,
        contract=contract,
        model_probability=model_probability,
        market_probability=market_probability,
        signal_threshold=signal_threshold,
        confidence_score=confidence_score,
    )


def rank_trade_signals(signals: list[TradeSignal]) -> list[TradeSignal]:
    """Rank trade signals by descending absolute alpha.

    Higher ``absolute_alpha`` means stronger conviction that the market is
    mispriced, so those opportunities appear first.  Signals with equal
    absolute alpha are returned in stable order (their original relative
    ordering is preserved).

    Args:
        signals: List of :class:`~deep_isobar.core.types.TradeSignal` objects
            to rank.  May be empty.

    Returns:
        A new list sorted by ``absolute_alpha`` descending.  The input list
        is not mutated.

    Example::

        ranked = rank_trade_signals(signals)
        best = ranked[0]   # highest |alpha| opportunity
    """
    ranked = sorted(signals, key=lambda s: s.absolute_alpha, reverse=True)
    logger.debug(
        "rank_trade_signals: %d signals ranked, top absolute_alpha=%.4f",
        len(ranked),
        ranked[0].absolute_alpha if ranked else 0.0,
    )
    return ranked
