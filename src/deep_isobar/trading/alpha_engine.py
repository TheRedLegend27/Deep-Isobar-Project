"""Alpha engine for Deep Isobar.

Computes the edge (alpha) between the model's estimated probability and the
market-implied probability, classifies it into a trading signal, and packages
the result as a :class:`~deep_isobar.core.types.TradeSignal`.

Canonical interface (from INTERFACES.md)::

    compute_alpha(model_probability, market_probability) -> float
    classify_signal(alpha, threshold) -> str          # "BUY" | "SELL" | "HOLD"
    build_trade_signal(timestamp_utc, contract, ...) -> TradeSignal

Signal rules:
    alpha > threshold  → BUY
    alpha < -threshold → SELL
    otherwise          → HOLD

All probability arguments must be in [0.0, 1.0].
"""

from __future__ import annotations

import logging
from datetime import datetime

from deep_isobar.core.types import MarketContract, TradeSignal

logger = logging.getLogger(__name__)

_SIGNAL_BUY = "BUY"
_SIGNAL_SELL = "SELL"
_SIGNAL_HOLD = "HOLD"


def compute_alpha(model_probability: float, market_probability: float) -> float:
    """Compute the signed edge between model and market probabilities.

    Alpha is the raw mispricing signal:

    .. code-block:: text

        alpha = model_probability − market_probability

    A positive alpha means the model believes the event is more likely than
    the market does (potential BUY edge).  A negative alpha means the model
    believes it is less likely (potential SELL edge).

    Args:
        model_probability: Probability estimated by the forecasting model.
            Must be in [0.0, 1.0].
        market_probability: Probability implied by the current market price.
            Must be in [0.0, 1.0].

    Returns:
        Signed float in the range [-1.0, 1.0].

    Raises:
        ValueError: If either probability is outside [0.0, 1.0].

    Example::

        alpha = compute_alpha(0.70, 0.55)  # → 0.15
    """
    _validate_probability(model_probability, "model_probability")
    _validate_probability(market_probability, "market_probability")

    alpha = model_probability - market_probability
    logger.debug(
        "compute_alpha: model=%.4f market=%.4f alpha=%.4f",
        model_probability,
        market_probability,
        alpha,
    )
    return alpha


def classify_signal(alpha: float, threshold: float) -> str:
    """Classify a trading signal from an alpha value and a threshold.

    Rules:
        - alpha > threshold  → ``"BUY"``
        - alpha < -threshold → ``"SELL"``
        - otherwise          → ``"HOLD"``

    Args:
        alpha: Signed edge value, typically from :func:`compute_alpha`.
        threshold: Minimum absolute alpha required to trigger BUY or SELL.
            Must be positive.

    Returns:
        One of ``"BUY"``, ``"SELL"``, or ``"HOLD"``.

    Raises:
        ValueError: If *threshold* is not positive.

    Example::

        classify_signal(alpha=0.15, threshold=0.10)  # → "BUY"
        classify_signal(alpha=-0.18, threshold=0.10) # → "SELL"
        classify_signal(alpha=0.03, threshold=0.10)  # → "HOLD"
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")

    if alpha > threshold:
        signal = _SIGNAL_BUY
    elif alpha < -threshold:
        signal = _SIGNAL_SELL
    else:
        signal = _SIGNAL_HOLD

    logger.debug(
        "classify_signal: alpha=%.4f threshold=%.4f → %s",
        alpha,
        threshold,
        signal,
    )
    return signal


def build_trade_signal(
    timestamp_utc: datetime,
    contract: MarketContract,
    model_probability: float,
    market_probability: float,
    signal_threshold: float,
    confidence_score: float,
    tail_opportunity_flag: bool = False,
    forecast_shift_flag: bool = False,
    stale_market_flag: bool = False,
    microstructure_score: float | None = None,
    rank_score: float | None = None,
    model_version: str = "v1",
) -> TradeSignal:
    """Assemble a fully-populated :class:`~deep_isobar.core.types.TradeSignal`.

    Calls :func:`compute_alpha` and :func:`classify_signal` internally and
    attaches all metadata from the contract and optional feature flags.

    Args:
        timestamp_utc: When this signal was generated.
        contract: The :class:`~deep_isobar.core.types.MarketContract` being
            evaluated.
        model_probability: Model's estimated probability.  Must be in [0, 1].
        market_probability: Market-implied probability.  Must be in [0, 1].
        signal_threshold: Minimum |alpha| to trigger BUY or SELL.  Must be
            positive.
        confidence_score: Caller-supplied confidence score for this signal.
        tail_opportunity_flag: True when the threshold lies in a distribution
            tail.
        forecast_shift_flag: True when a recent model forecast shift was
            detected.
        stale_market_flag: True when the market price appears lagged.
        microstructure_score: Optional market-microstructure quality score.
        rank_score: Optional composite rank score for signal ordering.
        model_version: Identifier for the model version that produced the
            probability estimate.

    Returns:
        A :class:`~deep_isobar.core.types.TradeSignal` with all fields
        populated.

    Raises:
        ValueError: If probabilities are out of [0, 1] or threshold is not
            positive.

    Example::

        signal = build_trade_signal(
            timestamp_utc=datetime.now(timezone.utc),
            contract=contract,
            model_probability=0.72,
            market_probability=0.55,
            signal_threshold=0.10,
            confidence_score=0.85,
        )
        # signal.signal_side → "BUY"
        # signal.alpha       → 0.17
    """
    alpha = compute_alpha(model_probability, market_probability)
    signal_side = classify_signal(alpha, signal_threshold)

    logger.info(
        "build_trade_signal: contract=%s side=%s alpha=%.4f model=%.4f market=%.4f",
        contract.contract_id,
        signal_side,
        alpha,
        model_probability,
        market_probability,
    )

    return TradeSignal(
        timestamp_utc=timestamp_utc,
        contract_id=contract.contract_id,
        city=contract.city,
        target_date=contract.target_date,
        metric=contract.metric,
        threshold_f=contract.threshold_f,
        comparison_operator=contract.comparison_operator,
        market_probability=market_probability,
        model_probability=model_probability,
        alpha=alpha,
        absolute_alpha=abs(alpha),
        signal_side=signal_side,
        confidence_score=confidence_score,
        tail_opportunity_flag=tail_opportunity_flag,
        forecast_shift_flag=forecast_shift_flag,
        stale_market_flag=stale_market_flag,
        microstructure_score=microstructure_score,
        rank_score=rank_score,
        model_version=model_version,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_probability(value: float, name: str) -> None:
    """Raise ValueError if *value* is not in [0.0, 1.0]."""
    if not (0.0 <= value <= 1.0):
        raise ValueError(
            f"{name} must be in [0.0, 1.0], got {value}"
        )
