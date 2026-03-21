"""Market scanner for Deep Isobar.

Evaluates individual prediction-market contracts for alpha (mispricing)
and ranks the resulting opportunities.

This module is the orchestration layer that wires together:
- ``market_price_adapter``    — extracts market-implied probabilities
- ``alpha_engine``            — computes alpha, classifies the signal, and
                                derives a composite rank_score (including tail,
                                shift, lag, and microstructure bonuses)

It accepts pre-computed enhancement signals from the caller (scheduler or
research layer) and forwards them into the alpha engine.  It does **not**
itself call the enhancement modules — those are called upstream and their
boolean flags / scores passed in as optional keyword arguments.

It does **not** contain risk logic, order submission, or scheduling.

Canonical interface (from INTERFACES.md)::

    evaluate_contract_opportunity(
        contract, probability_surface, orderbook,
        signal_threshold, timestamp_utc,
        *, forecast_shift_flag, stale_market_flag,
        microstructure_score, tail_opportunity_flag, tail_multiplier
    ) -> TradeSignal

    rank_trade_signals(signals) -> list[TradeSignal]

Pipeline for a single contract::

    probability_surface[threshold_f]  →  model_probability
    orderbook (bid/ask)               →  market_probability   (via adapter)
    (model_prob, market_prob)         →  alpha + side         (via alpha_engine)
    alpha + enhancement flags/scores  →  rank_score           (via alpha_engine)
    all of the above                  →  TradeSignal

Enhancement inputs forwarded to alpha_engine::

    forecast_shift_flag    — from forecast_shift.compute_forecast_shift
    stale_market_flag      — from market_lag_detection.detect_market_lag
    microstructure_score   — from microstructure_scanner.compute_microstructure_score
    tail_opportunity_flag  — from distribution_tail_alpha.is_tail_threshold
    tail_multiplier        — from CityProfile.tail_multiplier

Ranking key (descending priority)::

    1. rank_score     — composite score from alpha_engine (preferred)
    2. absolute_alpha — raw mispricing magnitude (fallback when rank_score absent)
    3. contract_id    — alphabetical, for deterministic tie-breaking
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
    *,
    forecast_shift_flag: bool = False,
    stale_market_flag: bool = False,
    microstructure_score: float | None = None,
    tail_opportunity_flag: bool = False,
    tail_multiplier: float = 1.0,
) -> TradeSignal:
    """Evaluate a single contract for a trading opportunity.

    Looks up the model probability for ``contract.threshold_f`` in the
    probability surface, derives the market probability from the order book,
    then delegates to :func:`~deep_isobar.trading.alpha_engine.build_trade_signal`
    to compute alpha and a composite rank_score that incorporates all
    supplied enhancement signals.

    The ``confidence_score`` on the returned signal is ``|alpha|``.

    Enhancement flags and scores are optional.  When omitted the signal is
    evaluated with raw alpha only (MVP baseline behaviour).  Callers that
    run the enhancement modules upstream should pass their outputs here:

    - ``forecast_shift_flag`` from
      :func:`~deep_isobar.models.forecast_shift.compute_forecast_shift`
    - ``stale_market_flag`` from
      :func:`~deep_isobar.market.market_lag_detection.detect_market_lag`
    - ``microstructure_score`` from
      :func:`~deep_isobar.market.microstructure_scanner.compute_microstructure_score`
    - ``tail_opportunity_flag`` from
      :func:`~deep_isobar.trading.distribution_tail_alpha.is_tail_threshold`
    - ``tail_multiplier`` from
      :attr:`~deep_isobar.core.types.CityProfile.tail_multiplier`

    Args:
        contract: The market contract to evaluate.
        probability_surface: Mapping of ``threshold_f`` → model probability.
            Must contain an entry for ``contract.threshold_f``.
        orderbook: Current order-book snapshot for this contract.
        signal_threshold: Minimum |alpha| required to trigger BUY or SELL.
        timestamp_utc: Evaluation timestamp.  Defaults to
            ``datetime.now(timezone.utc)`` if not supplied.
        forecast_shift_flag: ``True`` when a significant forecast shift was
            detected for this contract.  Adds a shift bonus to rank_score.
        stale_market_flag: ``True`` when the market price has not reacted to
            the latest forecast shift.  Adds a lag bonus to rank_score.
        microstructure_score: Composite market-quality score in [0, 1] from
            the microstructure scanner.  Contributes an additive term to
            rank_score.  Must be >= 0 when provided.
        tail_opportunity_flag: ``True`` when the contract threshold lies in
            the tail of the forecast distribution.  Activates the tail
            multiplier in rank_score computation.
        tail_multiplier: City-specific multiplier applied to ``|alpha|`` when
            ``tail_opportunity_flag`` is ``True``.  Must be >= 0.
            Default ``1.0`` leaves the base score unchanged.

    Returns:
        A :class:`~deep_isobar.core.types.TradeSignal` with alpha, side,
        all contract metadata, and a composite rank_score populated.

    Raises:
        KeyError: If ``contract.threshold_f`` is not present in
            ``probability_surface``.
        ValueError: If the order book contains no usable price, probabilities
            are out of range, ``signal_threshold`` is not positive, or
            ``tail_multiplier`` is negative.

    Example::

        signal = evaluate_contract_opportunity(
            contract=contract,
            probability_surface={80: 0.62, 85: 0.31},
            orderbook=orderbook,
            signal_threshold=0.10,
            tail_opportunity_flag=True,
            tail_multiplier=1.5,
            forecast_shift_flag=True,
        )
        # signal.signal_side        → "BUY" / "SELL" / "HOLD"
        # signal.tail_opportunity_flag → True
        # signal.rank_score         → boosted by tail + shift
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
        "model=%.4f market=%.4f shift=%s lag=%s micro=%s tail=%s",
        contract.contract_id,
        threshold,
        model_probability,
        market_probability,
        forecast_shift_flag,
        stale_market_flag,
        microstructure_score,
        tail_opportunity_flag,
    )

    alpha = model_probability - market_probability
    confidence_score = abs(alpha)

    # Pass raw spread so alpha_engine can apply the wide-spread hard filter.
    raw_spread: float | None = None
    if orderbook.best_bid is not None and orderbook.best_ask is not None:
        raw_spread = orderbook.best_ask - orderbook.best_bid

    return build_trade_signal(
        timestamp_utc=timestamp_utc,
        contract=contract,
        model_probability=model_probability,
        market_probability=market_probability,
        signal_threshold=signal_threshold,
        confidence_score=confidence_score,
        forecast_shift_flag=forecast_shift_flag,
        stale_market_flag=stale_market_flag,
        microstructure_score=microstructure_score,
        spread=raw_spread,
        tail_opportunity_flag=tail_opportunity_flag,
        tail_multiplier=tail_multiplier,
    )


def rank_trade_signals(signals: list[TradeSignal]) -> list[TradeSignal]:
    """Rank trade signals by composite score, with deterministic tie-breaking.

    Ranking priority (descending):

    1. **rank_score** — composite prioritisation score from
       :func:`~deep_isobar.trading.alpha_engine.compute_rank_score`, which
       incorporates tail position, forecast shifts, market lag, and
       microstructure quality on top of |alpha|.
    2. **absolute_alpha** — raw mispricing magnitude, used when ``rank_score``
       is ``None`` *or* as a tie-breaker between equal ``rank_score`` values.
    3. **contract_id** — alphabetical ascending, for fully deterministic
       output regardless of input ordering.

    The input list is not mutated.

    Args:
        signals: List of :class:`~deep_isobar.core.types.TradeSignal` objects
            to rank.  May be empty.

    Returns:
        A new list sorted by the ranking key described above.

    Example::

        ranked = rank_trade_signals(signals)
        best = ranked[0]   # highest composite score
    """
    def _key(s: TradeSignal) -> tuple:
        primary = s.rank_score if s.rank_score is not None else s.absolute_alpha
        return (-primary, -s.absolute_alpha, s.contract_id)

    ranked = sorted(signals, key=_key)
    logger.debug(
        "rank_trade_signals: %d signals ranked, top rank_score=%s absolute_alpha=%s",
        len(ranked),
        ranked[0].rank_score if ranked else None,
        ranked[0].absolute_alpha if ranked else None,
    )
    return ranked
