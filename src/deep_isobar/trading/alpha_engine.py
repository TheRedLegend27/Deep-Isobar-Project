"""Alpha engine for Deep Isobar.

Computes the edge (alpha) between the model's estimated probability and the
market-implied probability, classifies it into a trading signal, and packages
the result as a :class:`~deep_isobar.core.types.TradeSignal`.

Canonical interface (from INTERFACES.md)::

    compute_alpha(model_probability, market_probability) -> float
    classify_signal(alpha, threshold) -> str          # "BUY" | "SELL" | "HOLD"
    compute_rank_score(alpha, ...) -> float
    build_trade_signal(timestamp_utc, contract, ...) -> TradeSignal

Signal rules:
    alpha > threshold  → BUY
    alpha < -threshold → SELL
    otherwise          → HOLD

Alpha vs rank_score
-------------------
``alpha`` is the raw mispricing signal: ``model_probability − market_probability``.
It is never modified by any ranking logic.

``rank_score`` is a composite prioritisation score used only to order
opportunities.  It starts from ``abs(alpha)`` and can be boosted by tail
position, forecast shifts, market lag, and microstructure quality.  Raw alpha
and signal-side classification are always preserved unchanged.

Enhancement integration
-----------------------
``build_trade_signal`` integrates four optional enhancement signals produced by
the upstream modules:

- **market_lag_detection** → ``stale_market_flag``
- **microstructure_scanner** → ``microstructure_score``
- **distribution_tail_alpha** → ``tail_opportunity_flag`` + ``tail_multiplier``
- **forecast_shift** → ``forecast_shift_flag``

Hard filters (forced HOLD):
    1. ``stale_market_flag=True`` — market price has not reacted to a
       significant forecast shift; observed price may not be executable.
    2. ``microstructure_score < _MIN_LIQUIDITY_THRESHOLD`` — market is too
       thin or stale to trade reliably.

Confidence adjustments (applied before the hard filter):
    - ``forecast_shift_flag=True``  → +``_SHIFT_CONFIDENCE_BOOST``
    - ``tail_opportunity_flag=True`` → +``_TAIL_CONFIDENCE_BOOST``
    Hard filter triggers → confidence multiplied by the corresponding
    penalty factor so the signal carries a truthful score even as HOLD.

All probability arguments must be in [0.0, 1.0].
"""

from __future__ import annotations

import logging
from datetime import datetime

from deep_isobar.core.types import MarketContract, TradeSignal
from deep_isobar.trading.distribution_tail_alpha import compute_tail_alpha_boost

logger = logging.getLogger(__name__)

_SIGNAL_BUY = "BUY"
_SIGNAL_SELL = "SELL"
_SIGNAL_HOLD = "HOLD"

# ---------------------------------------------------------------------------
# Enhancement integration constants
# ---------------------------------------------------------------------------

#: Microstructure score below this threshold forces the signal to HOLD.
#: Below 0.30 the spread, freshness, and liquidity combination is too poor
#: to rely on the observed price as executable.
_MIN_LIQUIDITY_THRESHOLD: float = 0.30

#: Confidence multiplier when stale_market_flag is True.  Heavy penalty
#: because the market price has not absorbed the latest forecast shift, so
#: we cannot trust it as a reliable fill price.
_STALE_CONFIDENCE_FACTOR: float = 0.15

#: Confidence multiplier when microstructure_score < _MIN_LIQUIDITY_THRESHOLD.
_LOW_LIQUIDITY_CONFIDENCE_FACTOR: float = 0.20

#: Bid-ask spread above this threshold forces HOLD regardless of composite score.
#: A 40-cent spread means the fill cost alone would consume most of any edge.
_MAX_SPREAD_THRESHOLD: float = 0.40

#: Additive confidence boost when a significant forecast shift is confirmed.
#: A shift in the same direction as our alpha strengthens conviction.
_SHIFT_CONFIDENCE_BOOST: float = 0.05

#: Additive confidence boost for tail-threshold contracts.
#: Tail opportunities are rarer but when they align with alpha they tend to
#: be more decisively mispriced.
_TAIL_CONFIDENCE_BOOST: float = 0.03


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


def compute_rank_score(
    alpha: float,
    tail_rank_score: float | None = None,
    microstructure_score: float | None = None,
    forecast_shift_flag: bool = False,
    stale_market_flag: bool = False,
    shift_bonus: float = 0.05,
    lag_bonus: float = 0.05,
    microstructure_weight: float = 0.10,
) -> float:
    """Compute a composite ranking score for opportunity prioritisation.

    The rank score is used **only** to order signals — it does not alter raw
    alpha or the BUY / SELL / HOLD classification.

    Computation::

        base  = max(abs(alpha), tail_rank_score)   # if tail_rank_score provided
              = abs(alpha)                          # otherwise
        score = base
              + shift_bonus   (if forecast_shift_flag)
              + lag_bonus     (if stale_market_flag)
              + microstructure_score * microstructure_weight  (if provided)

    The final score is clamped to be non-negative.

    Args:
        alpha: Raw signed alpha from :func:`compute_alpha`.  Used as the
            baseline via ``abs(alpha)``.
        tail_rank_score: Optional boosted score from
            ``distribution_tail_alpha.compute_tail_alpha_boost``.  When
            provided, ``max(abs(alpha), tail_rank_score)`` is used as the base
            so a tail boost never reduces the score below ``abs(alpha)``.
        microstructure_score: Optional market-quality score.  Added as
            ``microstructure_score * microstructure_weight``.  Must be >= 0.
        forecast_shift_flag: Add *shift_bonus* when ``True``.
        stale_market_flag: Add *lag_bonus* when ``True`` (market price lags
            the latest forecast shift).
        shift_bonus: Fixed additive bonus for detected forecast shifts.
            Must be >= 0.  Default 0.05.
        lag_bonus: Fixed additive bonus for stale market prices.
            Must be >= 0.  Default 0.05.
        microstructure_weight: Weight applied to *microstructure_score*.
            Must be >= 0.  Default 0.10.

    Returns:
        Non-negative composite ranking score.

    Raises:
        ValueError: If *shift_bonus*, *lag_bonus*, or *microstructure_weight*
            is negative.
        ValueError: If *microstructure_score* is provided and is negative.

    Example::

        score = compute_rank_score(
            alpha=0.25,
            tail_rank_score=0.375,   # from compute_tail_alpha_boost
            forecast_shift_flag=True,
            microstructure_score=0.8,
        )
        # base  = max(0.25, 0.375) = 0.375
        # +0.05 (shift bonus)
        # +0.8 × 0.10 = 0.08
        # score = 0.505
    """
    if shift_bonus < 0:
        raise ValueError(f"shift_bonus must be >= 0, got {shift_bonus}")
    if lag_bonus < 0:
        raise ValueError(f"lag_bonus must be >= 0, got {lag_bonus}")
    if microstructure_weight < 0:
        raise ValueError(f"microstructure_weight must be >= 0, got {microstructure_weight}")
    if microstructure_score is not None and microstructure_score < 0:
        raise ValueError(f"microstructure_score must be >= 0, got {microstructure_score}")

    base = abs(alpha)
    if tail_rank_score is not None:
        base = max(base, tail_rank_score)

    score = base
    if forecast_shift_flag:
        score += shift_bonus
    if stale_market_flag:
        score += lag_bonus
    if microstructure_score is not None:
        score += microstructure_score * microstructure_weight

    score = max(0.0, score)

    logger.debug(
        "compute_rank_score: alpha=%.4f base=%.4f shift=%s lag=%s micro=%s → %.4f",
        alpha,
        base,
        forecast_shift_flag,
        stale_market_flag,
        microstructure_score,
        score,
    )
    return score


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
    spread: float | None = None,
    rank_score: float | None = None,
    tail_multiplier: float = 1.0,
    model_version: str = "v1",
) -> TradeSignal:
    """Assemble a fully-populated :class:`~deep_isobar.core.types.TradeSignal`.

    Calls :func:`compute_alpha` and :func:`classify_signal` internally and
    attaches all metadata from the contract and optional feature flags.

    When ``rank_score`` is not supplied the score is auto-computed by wiring
    all available enhancement signals into :func:`compute_rank_score`:

    - ``tail_opportunity_flag`` + ``tail_multiplier`` →
      :func:`~deep_isobar.trading.distribution_tail_alpha.compute_tail_alpha_boost`
      → ``tail_rank_score`` base
    - ``forecast_shift_flag`` → shift bonus
    - ``stale_market_flag``   → lag bonus
    - ``microstructure_score`` → quality additive

    Raw alpha is never modified.  Signal side may be overridden to ``"HOLD"``
    by the hard filter logic in :func:`_apply_enhancement_adjustments` when
    ``stale_market_flag`` is ``True`` or ``microstructure_score`` is below
    :data:`_MIN_LIQUIDITY_THRESHOLD`.

    Args:
        timestamp_utc: When this signal was generated.
        contract: The :class:`~deep_isobar.core.types.MarketContract` being
            evaluated.
        model_probability: Model's estimated probability.  Must be in [0, 1].
        market_probability: Market-implied probability.  Must be in [0, 1].
        signal_threshold: Minimum |alpha| to trigger BUY or SELL.  Must be
            positive.
        confidence_score: Caller-supplied confidence score for this signal.
        tail_opportunity_flag: ``True`` when the threshold lies in the tail
            of the forecast distribution (as determined by
            :func:`~deep_isobar.trading.distribution_tail_alpha.is_tail_threshold`).
            Activates the tail-boost path in rank_score computation.
        forecast_shift_flag: ``True`` when a recent model forecast shift was
            detected (as determined by
            :func:`~deep_isobar.models.forecast_shift.compute_forecast_shift`).
        stale_market_flag: ``True`` when the market price appears lagged
            behind the latest forecast (as determined by
            :func:`~deep_isobar.market.market_lag_detection.detect_market_lag`).
        microstructure_score: Composite market-quality score from
            :func:`~deep_isobar.market.microstructure_scanner.compute_microstructure_score`.
            Must be >= 0 when provided.
        rank_score: Override for the composite rank score.  When ``None``
            (the default) the score is auto-computed from all available
            enhancement signals.  Pass an explicit value to bypass
            auto-computation entirely.
        tail_multiplier: Multiplier applied to ``|alpha|`` when
            ``tail_opportunity_flag`` is ``True``.  Sourced from
            :attr:`~deep_isobar.core.types.CityProfile.tail_multiplier`.
            Must be >= 0.  Default ``1.0`` leaves the base unchanged.
        model_version: Identifier for the model version that produced the
            probability estimate.

    Returns:
        A :class:`~deep_isobar.core.types.TradeSignal` with all fields
        populated.

    Raises:
        ValueError: If probabilities are out of [0, 1], threshold is not
            positive, or tail_multiplier is negative.

    Example::

        signal = build_trade_signal(
            timestamp_utc=datetime.now(timezone.utc),
            contract=contract,
            model_probability=0.72,
            market_probability=0.55,
            signal_threshold=0.10,
            confidence_score=0.85,
            tail_opportunity_flag=True,
            tail_multiplier=1.5,
        )
        # signal.signal_side → "BUY"
        # signal.alpha       → 0.17   (unchanged)
        # signal.rank_score  → 0.255  (0.17 × 1.5)
    """
    alpha = compute_alpha(model_probability, market_probability)
    signal_side = classify_signal(alpha, signal_threshold)

    # ── Enhancement integration ────────────────────────────────────────────
    # 1. Adjust confidence_score based on enhancement signal quality.
    # 2. Apply hard filters: stale market or low liquidity → force HOLD.
    # Raw alpha and absolute_alpha are never modified by this step.
    signal_side, adjusted_confidence, filter_reason = _apply_enhancement_adjustments(
        signal_side=signal_side,
        confidence_score=confidence_score,
        stale_market_flag=stale_market_flag,
        forecast_shift_flag=forecast_shift_flag,
        tail_opportunity_flag=tail_opportunity_flag,
        microstructure_score=microstructure_score,
        spread=spread,
    )

    # ── Rank score ─────────────────────────────────────────────────────────
    # Hard-filtered signals (forced HOLD) get rank_score=0.0 so they never
    # compete with legitimate opportunities in downstream ranking.
    # For tradeable signals, auto-compute from all enhancement signals when
    # not supplied explicitly.
    if filter_reason is not None:
        effective_rank_score = 0.0
    elif rank_score is not None:
        effective_rank_score = rank_score
    else:
        tail_rank_score: float | None = None
        if tail_opportunity_flag:
            tail_rank_score = compute_tail_alpha_boost(
                alpha=alpha,
                tail_multiplier=tail_multiplier,
                tail_flag=True,
            )
        effective_rank_score = compute_rank_score(
            alpha=alpha,
            tail_rank_score=tail_rank_score,
            microstructure_score=microstructure_score,
            forecast_shift_flag=forecast_shift_flag,
            stale_market_flag=stale_market_flag,
        )

    logger.info(
        "build_trade_signal: contract=%s side=%s alpha=%.4f model=%.4f market=%.4f "
        "rank=%.4f confidence=%.4f filter=%s",
        contract.contract_id,
        signal_side,
        alpha,
        model_probability,
        market_probability,
        effective_rank_score,
        adjusted_confidence,
        filter_reason or "none",
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
        confidence_score=adjusted_confidence,
        tail_opportunity_flag=tail_opportunity_flag,
        forecast_shift_flag=forecast_shift_flag,
        stale_market_flag=stale_market_flag,
        microstructure_score=microstructure_score,
        rank_score=effective_rank_score,
        model_version=model_version,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _apply_enhancement_adjustments(
    signal_side: str,
    confidence_score: float,
    stale_market_flag: bool,
    forecast_shift_flag: bool,
    tail_opportunity_flag: bool,
    microstructure_score: float | None,
    spread: float | None = None,
) -> tuple[str, float, str | None]:
    """Apply hard filters and confidence adjustments from enhancement signals.

    Confidence adjustments are applied first (they are always valid).  Hard
    filters are checked afterwards and override the signal side to HOLD when
    triggered; the already-adjusted confidence is then penalised further so
    the returned confidence truthfully reflects the quality of the signal.

    Hard filter rules (checked in order; first match wins):
        0. ``spread > _MAX_SPREAD_THRESHOLD`` → HOLD (fill cost consumes edge)
        1. ``stale_market_flag=True`` → HOLD + ``_STALE_CONFIDENCE_FACTOR``
        2. ``microstructure_score < _MIN_LIQUIDITY_THRESHOLD`` → HOLD +
           ``_LOW_LIQUIDITY_CONFIDENCE_FACTOR``

    Confidence boosts (additive, clamped to 1.0):
        - ``forecast_shift_flag=True`` → +``_SHIFT_CONFIDENCE_BOOST``
        - ``tail_opportunity_flag=True`` → +``_TAIL_CONFIDENCE_BOOST``

    Args:
        signal_side: Initial classification from :func:`classify_signal`.
        confidence_score: Caller-supplied confidence, typically ``|alpha|``.
        stale_market_flag: From ``market_lag_detection.detect_market_lag``.
        forecast_shift_flag: From ``forecast_shift.compute_forecast_shift``.
        tail_opportunity_flag: From ``distribution_tail_alpha.is_tail_threshold``.
        microstructure_score: From ``microstructure_scanner.compute_microstructure_score``.
        spread: Raw bid-ask spread (``best_ask - best_bid``) in [0, 1].  When
            provided and above :data:`_MAX_SPREAD_THRESHOLD`, forces HOLD.

    Returns:
        ``(effective_signal_side, effective_confidence, filter_reason)``
        where *filter_reason* is ``None`` when no hard filter triggered, or a
        short description string when HOLD was forced.
    """
    effective = confidence_score

    # ── Positive confidence adjustments ───────────────────────────────────
    if forecast_shift_flag:
        effective = min(1.0, effective + _SHIFT_CONFIDENCE_BOOST)
    if tail_opportunity_flag:
        effective = min(1.0, effective + _TAIL_CONFIDENCE_BOOST)

    # ── Hard filter 0: wide spread ─────────────────────────────────────────
    if spread is not None and spread > _MAX_SPREAD_THRESHOLD:
        effective *= _LOW_LIQUIDITY_CONFIDENCE_FACTOR
        reason = f"wide_spread({spread:.3f}>{_MAX_SPREAD_THRESHOLD})"
        logger.warning(
            "_apply_enhancement_adjustments: spread=%.3f > %.3f — "
            "forcing HOLD, confidence %.4f → %.4f",
            spread, _MAX_SPREAD_THRESHOLD, confidence_score, effective,
        )
        return _SIGNAL_HOLD, round(effective, 6), reason

    # ── Hard filter 1: stale market ───────────────────────────────────────
    if stale_market_flag:
        effective *= _STALE_CONFIDENCE_FACTOR
        logger.warning(
            "_apply_enhancement_adjustments: stale_market_flag=True — "
            "forcing HOLD, confidence %.4f → %.4f",
            confidence_score, effective,
        )
        return _SIGNAL_HOLD, round(effective, 6), "stale_market"

    # ── Hard filter 2: insufficient liquidity ─────────────────────────────
    if microstructure_score is not None and microstructure_score < _MIN_LIQUIDITY_THRESHOLD:
        effective *= _LOW_LIQUIDITY_CONFIDENCE_FACTOR
        reason = f"low_liquidity(score={microstructure_score:.3f}<{_MIN_LIQUIDITY_THRESHOLD})"
        logger.warning(
            "_apply_enhancement_adjustments: microstructure_score=%.3f < %.3f — "
            "forcing HOLD, confidence %.4f → %.4f",
            microstructure_score, _MIN_LIQUIDITY_THRESHOLD,
            confidence_score, effective,
        )
        return _SIGNAL_HOLD, round(effective, 6), reason

    return signal_side, round(effective, 6), None


def _validate_probability(value: float, name: str) -> None:
    """Raise ValueError if *value* is not in [0.0, 1.0]."""
    if not (0.0 <= value <= 1.0):
        raise ValueError(
            f"{name} must be in [0.0, 1.0], got {value}"
        )
