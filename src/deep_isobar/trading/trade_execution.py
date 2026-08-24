"""Trade execution for Deep Isobar.

Paper trading plus gated live execution via ``kalshi_client.create_order``.
Live orders are refused with :class:`LiveTradingDisabledError` until
``runtime.paper_trade`` is set to exactly ``False`` in settings.yaml.

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

Live trade behaviour (see ``submit_live_trade`` for the full gate order):
- Refused while ``runtime.paper_trade`` is not exactly ``False``
- Kill switch checked immediately before the network call, always last
- SELL signals are placed as long-NO orders at the complementary price
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone

from deep_isobar.config import get_setting
from deep_isobar.core.types import ExecutedTrade, TradeSignal
from deep_isobar.market import kalshi_client
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


class LiveTradingDisabledError(RuntimeError):
    """Raised when live order placement is attempted while the system is
    configured for paper trading (``runtime.paper_trade`` is not exactly
    ``False`` in config/settings.yaml)."""


# Kalshi order statuses → our execution_status vocabulary.
_KALSHI_STATUS_MAP = {
    "executed": "filled",
    "resting":  "open",
    "pending":  "pending",
    "canceled": "canceled",
}


def submit_live_trade(
    market_source: str,
    contract_id: str,
    side: str,
    quantity: float,
    price: float,
    order_type: str = "limit",
    client_order_id: str | None = None,
) -> ExecutedTrade:
    """Submit a live order to Kalshi.

    Signal-side semantics match the paper book: ``side="BUY"`` buys the
    YES contract at *price*; ``side="SELL"`` is expressed as buying the NO
    contract at ``1 − price`` (Kalshi has no naked shorting — betting
    against IS buying NO; same convention as ``trading/kelly.py``).
    ``price`` is always in YES-space, exactly as recorded in
    ``paper_trades.csv``.

    Three gates run before any network traffic, in order:

    1. **Input validation** — ``ValueError`` on bad side/quantity/price.
    2. **Live-trading flag** — ``runtime.paper_trade`` in settings.yaml
       must be **exactly** ``False`` (not merely falsy, and absent means
       paper).  Otherwise :class:`LiveTradingDisabledError`.  This is the
       single switch that keeps the whole system paper-only today.
    3. **Position limit** — contracts must not exceed
       ``risk.max_position_per_contract``.

    The kill switch is checked LAST, immediately before the order call —
    per the go-live safety spec, no logic may ever land between that check
    and the network submission.

    Args:
        market_source: Exchange identifier — only ``"Kalshi"`` is supported.
        contract_id: Kalshi market ticker, e.g. ``"KXHIGHLAX-26AUG18-B77.5"``.
        side: ``"BUY"`` or ``"SELL"`` (signal-side vocabulary).
        quantity: Number of contracts — must be a positive whole number.
        price: YES-space limit price as a decimal probability in (0, 1).
        order_type: ``"limit"`` (default) or ``"market"``.
        client_order_id: Idempotency key passed through to Kalshi.  Reuse
            the SAME id when retrying after an ambiguous network failure —
            Kalshi rejects duplicates instead of double-filling.  Auto-
            generated (UUID4) when omitted.

    Raises:
        ValueError: On invalid arguments.
        LiveTradingDisabledError: While ``runtime.paper_trade`` is not
            exactly ``False``.
        KillSwitchEngagedError: If the kill switch is engaged.
        RuntimeError: On any exchange/API failure (from ``create_order``).
    """
    # ── Gate 1: input validation — no side effects ────────────────────────
    if market_source != "Kalshi":
        raise ValueError(f"unsupported market_source {market_source!r} — only 'Kalshi'")
    if side not in ("BUY", "SELL"):
        raise ValueError(f"side must be 'BUY' or 'SELL', got {side!r}")
    if quantity <= 0 or quantity != int(quantity):
        raise ValueError(f"quantity must be a positive whole number of contracts, got {quantity}")
    if not (0.0 < price < 1.0):
        raise ValueError(f"price must be strictly inside (0.0, 1.0), got {price}")

    count = int(quantity)

    # ── Gate 2: the paper/live switch ─────────────────────────────────────
    if get_setting("runtime.paper_trade", True) is not False:
        raise LiveTradingDisabledError(
            "submit_live_trade refused: runtime.paper_trade is not False — "
            "the system is configured for paper trading. Flip it in "
            "config/settings.yaml only as a deliberate go-live decision. "
            f"Attempted: contract={contract_id!r} side={side!r} "
            f"qty={quantity} price={price}"
        )

    # ── Gate 3: hard per-contract position limit ──────────────────────────
    max_contracts = int(get_setting("risk.max_position_per_contract", 100))
    if count > max_contracts:
        raise ValueError(
            f"quantity {count} exceeds risk.max_position_per_contract={max_contracts}"
        )

    # SELL = long NO at the complementary price (see docstring).
    if side == "BUY":
        kalshi_side, price_cents = "yes", round(price * 100)
    else:
        kalshi_side, price_cents = "no", round((1.0 - price) * 100)
    price_cents = min(99, max(1, price_cents))

    idem_key = client_order_id or str(uuid.uuid4())
    timestamp_utc = datetime.now(timezone.utc)

    # ── Kill switch — the FINAL guard before the network call ─────────────
    if kill_switch.is_engaged():
        state = kill_switch.get_state()
        raise kill_switch.KillSwitchEngagedError(
            "submit_live_trade refused: KILL SWITCH ENGAGED — "
            f"reason={state.reason or state.detail or 'unknown'!r} "
            f"source={state.source or 'unknown'!r}. "
            f"Attempted: market_source={market_source!r} contract={contract_id!r} "
            f"side={side!r} qty={quantity} price={price}"
        )

    order = kalshi_client.create_order(
        ticker=contract_id,
        action="buy",
        side=kalshi_side,
        count=count,
        price_cents=price_cents,
        client_order_id=idem_key,
        order_type=order_type,
    )

    status = _KALSHI_STATUS_MAP.get(str(order.get("status", "")).lower(), "unknown")
    filled = status == "filled"

    return ExecutedTrade(
        timestamp_utc=timestamp_utc,
        trade_id=f"LIVE-{idem_key[:8]}",
        contract_id=contract_id,
        market_source=market_source,
        side=side,
        quantity=float(count),
        price=price,
        order_type=order_type,
        execution_status=status,
        fill_quantity=float(count) if filled else None,
        avg_fill_price=price if filled else None,
        exchange_order_id=str(order.get("order_id")),
        paper_trade_flag=False,
        notes=(
            f"Live Kalshi order — side={kalshi_side} price={price_cents}¢ "
            f"client_order_id={idem_key}"
        ),
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
