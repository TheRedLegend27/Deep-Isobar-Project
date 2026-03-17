"""Risk manager for Deep Isobar.

Gate that approves or rejects a proposed trade before execution.  All
decisions are deterministic and based solely on the inputs supplied —
no external state is read from disk or the network.

Canonical interface (from INTERFACES.md)::

    approve_trade(
        signal, proposed_quantity, proposed_price,
        current_position, max_position_per_contract,
        daily_exposure_before, max_daily_exposure,
    ) -> RiskDecision

Rejection priority (first matching rule wins):
    1. signal_side == "HOLD"            → no edge, nothing to trade
    2. proposed_quantity <= 0           → invalid size
    3. proposed_price not in [0, 1]    → invalid price
    4. projected position exceeds limit → position gate
    5. daily exposure would exceed limit → exposure gate

Precondition (raises ValueError, not a rejection):
    - limit arguments (max_position_per_contract, daily_exposure_before,
      max_daily_exposure) must be non-negative — indicates a misconfigured caller

All other cases are approved.

Position arithmetic (signed):
    BUY  → projected_position = current_position + proposed_quantity
    SELL → projected_position = current_position − proposed_quantity

The exposure gate uses notional cost:
    daily_exposure_after = daily_exposure_before + proposed_quantity * proposed_price
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from deep_isobar.core.types import RiskDecision, TradeSignal

logger = logging.getLogger(__name__)

_HOLD = "HOLD"
_BUY = "BUY"
_SELL = "SELL"


def approve_trade(
    signal: TradeSignal,
    proposed_quantity: float,
    proposed_price: float,
    current_position: float,
    max_position_per_contract: float,
    daily_exposure_before: float,
    max_daily_exposure: float,
) -> RiskDecision:
    """Evaluate a proposed trade and return an approval or rejection decision.

    All gate checks are applied in order; the first failing check produces the
    rejection reason and no further checks are performed.

    Args:
        signal: The :class:`~deep_isobar.core.types.TradeSignal` to evaluate.
            ``signal.signal_side`` must be ``"BUY"`` or ``"SELL"`` to pass the
            first gate; ``"HOLD"`` is always rejected.
        proposed_quantity: Number of contracts (or units) to trade.  Must be
            strictly positive.
        proposed_price: Intended execution price per unit as a decimal
            probability (e.g. ``0.55`` for a 55-cent Kalshi contract).
            Must be in ``[0.0, 1.0]``.
        current_position: Signed current position in this contract before the
            trade.  Positive = long, negative = short.
        max_position_per_contract: Maximum allowed absolute position in any
            single contract.  Must be non-negative.
        daily_exposure_before: Notional exposure already consumed today across
            all contracts, in dollar terms.  Must be non-negative.
        max_daily_exposure: Maximum total daily notional exposure allowed.
            Must be non-negative.

    Returns:
        A :class:`~deep_isobar.core.types.RiskDecision` with ``approved_flag``
        set and, if rejected, a ``rejection_reason`` explaining why.

    Raises:
        ValueError: If limit arguments are negative (misconfigured caller).

    Example::

        decision = approve_trade(
            signal=signal,
            proposed_quantity=10.0,
            proposed_price=0.55,
            current_position=0.0,
            max_position_per_contract=50.0,
            daily_exposure_before=200.0,
            max_daily_exposure=1000.0,
        )
        # decision.approved_flag → True
    """
    # ── Validate caller-supplied limits ──────────────────────────────────────
    if max_position_per_contract < 0:
        raise ValueError(
            f"max_position_per_contract must be non-negative, got {max_position_per_contract}"
        )
    if daily_exposure_before < 0:
        raise ValueError(
            f"daily_exposure_before must be non-negative, got {daily_exposure_before}"
        )
    if max_daily_exposure < 0:
        raise ValueError(
            f"max_daily_exposure must be non-negative, got {max_daily_exposure}"
        )

    # ── Pre-compute derived values ────────────────────────────────────────────
    if signal.signal_side == _BUY:
        projected_position = current_position + proposed_quantity
    elif signal.signal_side == _SELL:
        projected_position = current_position - proposed_quantity
    else:
        # HOLD — set a sensible value even though we'll reject below
        projected_position = current_position

    daily_exposure_after = daily_exposure_before + proposed_quantity * proposed_price

    # ── Gate 1: HOLD signals are never executable ─────────────────────────────
    if signal.signal_side == _HOLD:
        return _reject(
            signal=signal,
            proposed_quantity=proposed_quantity,
            proposed_price=proposed_price,
            current_position=current_position,
            projected_position=projected_position,
            daily_exposure_before=daily_exposure_before,
            daily_exposure_after=daily_exposure_after,
            reason="signal_side is HOLD — no actionable edge",
        )

    # ── Gate 2: quantity must be positive ────────────────────────────────────
    if proposed_quantity <= 0:
        return _reject(
            signal=signal,
            proposed_quantity=proposed_quantity,
            proposed_price=proposed_price,
            current_position=current_position,
            projected_position=projected_position,
            daily_exposure_before=daily_exposure_before,
            daily_exposure_after=daily_exposure_after,
            reason=f"proposed_quantity must be > 0, got {proposed_quantity}",
        )

    # ── Gate 3: price must be a valid probability ─────────────────────────────
    if not (0.0 <= proposed_price <= 1.0):
        return _reject(
            signal=signal,
            proposed_quantity=proposed_quantity,
            proposed_price=proposed_price,
            current_position=current_position,
            projected_position=projected_position,
            daily_exposure_before=daily_exposure_before,
            daily_exposure_after=daily_exposure_after,
            reason=f"proposed_price must be in [0, 1], got {proposed_price}",
        )

    # ── Gate 4: position limit ────────────────────────────────────────────────
    if abs(projected_position) > max_position_per_contract:
        return _reject(
            signal=signal,
            proposed_quantity=proposed_quantity,
            proposed_price=proposed_price,
            current_position=current_position,
            projected_position=projected_position,
            daily_exposure_before=daily_exposure_before,
            daily_exposure_after=daily_exposure_after,
            reason=(
                f"projected_position {projected_position:.2f} exceeds "
                f"max_position_per_contract {max_position_per_contract:.2f}"
            ),
        )

    # ── Gate 5: daily exposure limit ─────────────────────────────────────────
    if daily_exposure_after > max_daily_exposure:
        return _reject(
            signal=signal,
            proposed_quantity=proposed_quantity,
            proposed_price=proposed_price,
            current_position=current_position,
            projected_position=projected_position,
            daily_exposure_before=daily_exposure_before,
            daily_exposure_after=daily_exposure_after,
            reason=(
                f"daily_exposure_after {daily_exposure_after:.2f} would exceed "
                f"max_daily_exposure {max_daily_exposure:.2f}"
            ),
        )

    # ── Approved ──────────────────────────────────────────────────────────────
    logger.info(
        "approve_trade: APPROVED contract=%s side=%s qty=%.2f price=%.4f "
        "projected_pos=%.2f daily_exp_after=%.2f",
        signal.contract_id,
        signal.signal_side,
        proposed_quantity,
        proposed_price,
        projected_position,
        daily_exposure_after,
    )

    return RiskDecision(
        timestamp_utc=datetime.now(timezone.utc),
        contract_id=signal.contract_id,
        proposed_side=signal.signal_side,
        proposed_quantity=proposed_quantity,
        proposed_price=proposed_price,
        approved_flag=True,
        rejection_reason=None,
        current_position=current_position,
        projected_position=projected_position,
        daily_exposure_before=daily_exposure_before,
        daily_exposure_after=daily_exposure_after,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _reject(
    signal: TradeSignal,
    proposed_quantity: float,
    proposed_price: float,
    current_position: float,
    projected_position: float,
    daily_exposure_before: float,
    daily_exposure_after: float,
    reason: str,
) -> RiskDecision:
    """Build a rejected :class:`~deep_isobar.core.types.RiskDecision`."""
    logger.info(
        "approve_trade: REJECTED contract=%s side=%s qty=%.2f — %s",
        signal.contract_id,
        signal.signal_side,
        proposed_quantity,
        reason,
    )
    return RiskDecision(
        timestamp_utc=datetime.now(timezone.utc),
        contract_id=signal.contract_id,
        proposed_side=signal.signal_side,
        proposed_quantity=proposed_quantity,
        proposed_price=proposed_price,
        approved_flag=False,
        rejection_reason=reason,
        current_position=current_position,
        projected_position=projected_position,
        daily_exposure_before=daily_exposure_before,
        daily_exposure_after=daily_exposure_after,
    )
