"""Trade execution for Deep Isobar.

MVP implementation supporting paper trading only.  Live execution raises
``RuntimeError`` until a real exchange client is wired in.

Canonical interface (from INTERFACES.md)::

    execute_paper_trade(signal, quantity, price) -> ExecutedTrade
    submit_live_trade(market_source, contract_id, side, quantity, price,
                      order_type) -> ExecutedTrade

Paper trade behaviour:
- Validates quantity > 0 and price in [0, 1]
- Generates a deterministic ``trade_id`` from the contract ID + timestamp
- Sets ``execution_status = "filled"`` and ``paper_trade_flag = True``
- Sets ``fill_quantity`` equal to the requested quantity (instant fill,
  no slippage modelled at MVP stage)
- Sets ``avg_fill_price`` equal to ``price`` (no market impact)

Live trade behaviour:
- Raises ``RuntimeError`` unconditionally — placeholder for Phase 7
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone

from deep_isobar.core.types import ExecutedTrade, TradeSignal
from deep_isobar.ops import kill_switch

logger = logging.getLogger(__name__)

_PAPER_STATUS = "filled"
_PAPER_ORDER_TYPE = "simulated"


def execute_paper_trade(
    signal: TradeSignal,
    quantity: float,
    price: float,
) -> ExecutedTrade:
    """Simulate a paper trade for a given signal.

    Creates an :class:`~deep_isobar.core.types.ExecutedTrade` record that
    represents an instant full fill at the requested price with no market
    impact.  No network calls are made.

    Args:
        signal: The :class:`~deep_isobar.core.types.TradeSignal` being executed.
            ``signal.signal_side`` is used as the trade side.
        quantity: Number of units to trade.  Must be strictly positive.
        price: Execution price per unit as a decimal probability in [0.0, 1.0]
            (e.g. ``0.55`` for a 55-cent Kalshi contract).

    Returns:
        A fully-populated :class:`~deep_isobar.core.types.ExecutedTrade` with
        ``paper_trade_flag=True`` and ``execution_status="filled"``.

    Raises:
        ValueError: If ``quantity <= 0`` or ``price`` is outside [0.0, 1.0].

    Example::

        trade = execute_paper_trade(signal=signal, quantity=10.0, price=0.55)
        # trade.execution_status → "filled"
        # trade.paper_trade_flag → True
        # trade.fill_quantity    → 10.0
    """
    if quantity <= 0:
        raise ValueError(f"quantity must be > 0, got {quantity}")
    if not (0.0 <= price <= 1.0):
        raise ValueError(f"price must be in [0.0, 1.0], got {price}")

    timestamp_utc = datetime.now(timezone.utc)
    trade_id = _make_trade_id(signal.contract_id, timestamp_utc)

    logger.info(
        "execute_paper_trade: trade_id=%s contract=%s side=%s qty=%.2f price=%.4f",
        trade_id,
        signal.contract_id,
        signal.signal_side,
        quantity,
        price,
    )

    return ExecutedTrade(
        timestamp_utc=timestamp_utc,
        trade_id=trade_id,
        contract_id=signal.contract_id,
        market_source="paper",
        side=signal.signal_side,
        quantity=quantity,
        price=price,
        order_type=_PAPER_ORDER_TYPE,
        execution_status=_PAPER_STATUS,
        fill_quantity=quantity,
        avg_fill_price=price,
        exchange_order_id=None,
        paper_trade_flag=True,
        notes=f"Paper trade — alpha={signal.alpha:.4f}",
    )


def submit_live_trade(
    market_source: str,
    contract_id: str,
    side: str,
    quantity: float,
    price: float,
    order_type: str = "limit",
) -> ExecutedTrade:
    """Submit a live order to an exchange.

    .. warning::

        **Not implemented.**  Beyond the kill-switch guard below, this
        function always raises ``RuntimeError``.  It is a placeholder for
        Phase 7 (live execution) and will be wired to a real exchange
        client (e.g. ``kalshi_client``) in a future build step.  When that
        happens, the kill-switch check below MUST remain the final gate
        immediately before the network call — do not let any new logic
        land between it and the order submission.

    Args:
        market_source: Exchange identifier, e.g. ``"Kalshi"``.
        contract_id: Exchange contract ID.
        side: ``"BUY"`` or ``"SELL"``.
        quantity: Number of units to trade.
        price: Limit price per unit.
        order_type: Order type, default ``"limit"``.

    Raises:
        KillSwitchEngagedError: If the kill switch is engaged — checked
            before anything else, including the "not implemented" guard.
        RuntimeError: Otherwise, always — live trading is not yet implemented.
    """
    # ── Kill switch — the final guard before the (future) network call ────
    if kill_switch.is_engaged():
        state = kill_switch.get_state()
        raise kill_switch.KillSwitchEngagedError(
            "submit_live_trade refused: KILL SWITCH ENGAGED — "
            f"reason={state.reason or state.detail or 'unknown'!r} "
            f"source={state.source or 'unknown'!r}. "
            f"Attempted: market_source={market_source!r} contract={contract_id!r} "
            f"side={side!r} qty={quantity} price={price}"
        )

    raise RuntimeError(
        "Live trading not implemented yet. "
        f"Attempted: market_source={market_source!r} contract={contract_id!r} "
        f"side={side!r} qty={quantity} price={price} order_type={order_type!r}"
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _make_trade_id(contract_id: str, timestamp_utc: datetime) -> str:
    """Generate a short deterministic trade ID from contract + timestamp.

    Format: ``PAPER-{8-char hex}`` derived from SHA-256 of the combined string.

    Args:
        contract_id: The contract being traded.
        timestamp_utc: Execution timestamp.

    Returns:
        A string trade ID, e.g. ``"PAPER-3f9a1b2c"``.
    """
    raw = f"{contract_id}:{timestamp_utc.isoformat()}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:8]
    return f"PAPER-{digest}"
