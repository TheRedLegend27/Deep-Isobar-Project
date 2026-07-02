"""Multi-bracket position spreader for Deep Isobar.

Distributes a session's daily exposure cap across the top-N qualifying alpha
signals — BUY and SELL alike — using one of three allocation methods:

- ``proportional`` — risk dollars proportional to |alpha|
- ``equal``        — cap split evenly across selected signals
- ``kelly``        — fee-adjusted fractional Kelly with a correlation
                     haircut (see :mod:`deep_isobar.trading.kelly`), then
                     scaled down if the sum exceeds the daily cap

``allocated_usd`` is always **risk dollars** (max loss).  When
*entry_prices* are provided, ``contracts`` is derived as
``allocated_usd / risk_per_contract`` so downstream P&L (which counts
contracts) matches the dollars actually at risk.

Interface::

    build_spread(signals, daily_exposure_cap_usd, max_contracts,
                 min_alpha, allocation_method, entry_prices=None,
                 kelly_cfg=None) -> list[BracketAllocation]
    log_spread_summary(allocations, logger) -> None
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from deep_isobar.core.types import TradeSignal
from deep_isobar.trading.kelly import (
    correlation_haircut,
    kelly_fraction,
    net_edge,
    risk_per_contract,
)

logger = logging.getLogger(__name__)


@dataclass
class BracketAllocation:
    """Allocation for a single bracket in a multi-contract spread."""

    signal: TradeSignal           # the original TradeSignal object
    allocated_usd: float          # dollars at risk on this bracket (max loss)
    allocation_fraction: float    # fraction of total session exposure (0.0–1.0)
    rank: int                     # 1 = highest alpha
    contracts: float | None = None  # allocated_usd / risk-per-contract; None
                                    # when no entry price was available


def build_spread(
    signals: list[TradeSignal],
    daily_exposure_cap_usd: float,
    max_contracts: int,
    min_alpha: float,
    allocation_method: str,
    entry_prices: dict[str, float] | None = None,
    kelly_cfg: dict | None = None,
) -> list[BracketAllocation]:
    """Build a multi-bracket spread from a list of trade signals.

    BUY and SELL signals are both eligible (SELL = short YES); they compete
    for the same cap ranked by |alpha|.

    Args:
        signals: All candidate TradeSignal objects for the session.
        daily_exposure_cap_usd: Total risk USD to allocate across brackets.
        max_contracts: Maximum number of brackets to enter.
        min_alpha: Minimum |alpha| required for a signal to qualify.
        allocation_method: ``"proportional"`` | ``"equal"`` | ``"kelly"``.
        entry_prices: ``{contract_id: fill price}`` (BUY at ask / SELL at
            bid).  Required for ``kelly``; for the other methods it enables
            the ``contracts`` field on the result.
        kelly_cfg: For ``kelly`` — keys ``bankroll_usd`` (default 1000),
            ``fraction`` (default 0.25), ``avg_pairwise_correlation``
            (default 0.6), ``n_correlated_bets`` (default 1; pass the number
            of active cities so same-airmass days aren't overbet).

    Returns:
        List of :class:`BracketAllocation` sorted by rank ascending
        (rank 1 = highest alpha).  Empty when nothing qualifies — for
        ``kelly`` that includes signals whose edge dies to the taker fee.

    Raises:
        ValueError: For an unrecognised ``allocation_method``, or ``kelly``
            without *entry_prices*.
    """
    qualifying = [
        s for s in signals
        if s.signal_side in ("BUY", "SELL") and abs(s.alpha) >= min_alpha
    ]

    if not qualifying:
        return []

    # Sort descending by alpha magnitude; take top max_contracts.
    qualifying.sort(key=lambda s: abs(s.alpha), reverse=True)
    selected = qualifying[:max_contracts]

    if allocation_method == "proportional":
        allocations = _proportional_allocation(selected, daily_exposure_cap_usd)
    elif allocation_method == "equal":
        allocations = _equal_allocation(selected, daily_exposure_cap_usd)
    elif allocation_method == "kelly":
        if entry_prices is None:
            raise ValueError("kelly allocation requires entry_prices")
        allocations = _kelly_allocation(
            selected, daily_exposure_cap_usd, entry_prices, kelly_cfg or {}
        )
    else:
        raise ValueError(f"Unknown allocation_method: {allocation_method!r}")

    _fill_contracts(allocations, entry_prices)
    return allocations


def log_spread_summary(
    allocations: list[BracketAllocation],
    logger: logging.Logger,
) -> None:
    """Log a formatted spread summary table at INFO level.

    Example output::

        Bracket spread — 3 positions, $50.00 total exposure
        #1 KXHIGHCHI-26APR14-T58  alpha=0.534  alloc=$22.45
        #2 KXHIGHCHI-26APR14-T57  alpha=0.481  alloc=$20.21
        #3 KXHIGHCHI-26APR14-T56  alpha=0.401  alloc=$7.34

    Args:
        allocations: List of :class:`BracketAllocation` from :func:`build_spread`.
        logger: Logger instance to write to.
    """
    if not allocations:
        logger.info("Bracket spread — no allocations (no qualifying signals)")
        return

    n = len(allocations)
    total_usd = sum(a.allocated_usd for a in allocations)
    logger.info(
        "Bracket spread — %d position%s, $%.2f total exposure",
        n,
        "s" if n != 1 else "",
        total_usd,
    )
    for alloc in sorted(allocations, key=lambda a: a.rank):
        logger.info(
            "#%d %-4s %-40s  alpha=%+.3f  risk=$%.2f%s",
            alloc.rank,
            alloc.signal.signal_side,
            alloc.signal.contract_id,
            alloc.signal.alpha,
            alloc.allocated_usd,
            f"  ({alloc.contracts:.1f} contracts)" if alloc.contracts else "",
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _proportional_allocation(
    selected: list[TradeSignal],
    daily_exposure_cap_usd: float,
) -> list[BracketAllocation]:
    """Allocate proportionally by alpha magnitude with rounding correction.

    Weight each position by its alpha relative to the sum of all selected
    alphas::

        allocated_usd_i = (|alpha_i| / sum_alphas) × daily_exposure_cap_usd

    All values are rounded to 2 decimal places.  The largest allocation is
    adjusted so the total exactly equals ``daily_exposure_cap_usd``.
    """
    sum_alphas = sum(abs(s.alpha) for s in selected)
    if sum_alphas == 0.0:
        return []

    allocations: list[BracketAllocation] = []
    for rank, signal in enumerate(selected, start=1):
        allocated = round((abs(signal.alpha) / sum_alphas) * daily_exposure_cap_usd, 2)
        allocations.append(BracketAllocation(
            signal=signal,
            allocated_usd=allocated,
            allocation_fraction=0.0,  # computed after rounding correction below
            rank=rank,
        ))

    # Rounding correction: adjust the largest allocation so the total
    # equals the cap exactly (avoids off-by-a-cent drift).
    total_allocated = sum(a.allocated_usd for a in allocations)
    diff = round(daily_exposure_cap_usd - total_allocated, 2)
    if diff != 0.0:
        largest_idx = max(range(len(allocations)), key=lambda i: allocations[i].allocated_usd)
        allocations[largest_idx].allocated_usd = round(
            allocations[largest_idx].allocated_usd + diff, 2
        )

    # Set allocation_fraction from final USD amounts so it matches allocated_usd
    # exactly, rather than from the pre-rounding alpha weights.
    for alloc in allocations:
        alloc.allocation_fraction = alloc.allocated_usd / daily_exposure_cap_usd

    return allocations


def _equal_allocation(
    selected: list[TradeSignal],
    daily_exposure_cap_usd: float,
) -> list[BracketAllocation]:
    """Split the cap evenly across the selected signals."""
    n = len(selected)
    each = round(daily_exposure_cap_usd / n, 2)
    allocations = [
        BracketAllocation(
            signal=signal,
            allocated_usd=each,
            allocation_fraction=each / daily_exposure_cap_usd,
            rank=rank,
        )
        for rank, signal in enumerate(selected, start=1)
    ]
    # Absorb rounding drift into rank 1 so the total equals the cap.
    diff = round(daily_exposure_cap_usd - each * n, 2)
    if diff:
        allocations[0].allocated_usd = round(allocations[0].allocated_usd + diff, 2)
        allocations[0].allocation_fraction = (
            allocations[0].allocated_usd / daily_exposure_cap_usd
        )
    return allocations


def _kelly_allocation(
    selected: list[TradeSignal],
    daily_exposure_cap_usd: float,
    entry_prices: dict[str, float],
    kelly_cfg: dict,
) -> list[BracketAllocation]:
    """Fee-adjusted fractional Kelly with a correlation haircut.

    Per signal: ``risk_usd = f* × fraction × haircut × bankroll``, where
    ``f*`` is the fee-adjusted Kelly fraction on the actual fill price.
    Signals whose edge is non-positive after the taker fee size to zero and
    are dropped (logged).  If the summed risk exceeds the daily cap, all
    allocations are scaled down proportionally — the cap is a hard limit,
    Kelly decides the shape.
    """
    bankroll = float(kelly_cfg.get("bankroll_usd", 1000.0))
    fraction = float(kelly_cfg.get("fraction", 0.25))
    rho = float(kelly_cfg.get("avg_pairwise_correlation", 0.6))
    n_bets = int(kelly_cfg.get("n_correlated_bets", 1))
    haircut = correlation_haircut(n_bets, rho)

    allocations: list[BracketAllocation] = []
    rank = 0
    for signal in selected:
        price = entry_prices.get(signal.contract_id)
        if price is None or not (0.0 < price < 1.0):
            logger.warning(
                "kelly: no usable entry price for %s (%r) — skipping",
                signal.contract_id, price,
            )
            continue
        f_star = kelly_fraction(signal.model_probability, price, signal.signal_side)
        if f_star <= 0.0:
            logger.info(
                "kelly: %s %s edge %.4f dies to taker fee (net %.4f) — dropped",
                signal.signal_side, signal.contract_id, signal.alpha,
                net_edge(signal.model_probability, price, signal.signal_side),
            )
            continue
        rank += 1
        risk_usd = f_star * fraction * haircut * bankroll
        allocations.append(BracketAllocation(
            signal=signal,
            allocated_usd=risk_usd,
            allocation_fraction=0.0,   # set after cap scaling
            rank=rank,
        ))

    if not allocations:
        return []

    total = sum(a.allocated_usd for a in allocations)
    if total > daily_exposure_cap_usd:
        scale = daily_exposure_cap_usd / total
        logger.info(
            "kelly: summed risk $%.2f exceeds cap $%.2f — scaling all "
            "allocations by %.3f", total, daily_exposure_cap_usd, scale,
        )
        for a in allocations:
            a.allocated_usd *= scale
        total = daily_exposure_cap_usd

    for a in allocations:
        a.allocated_usd = round(a.allocated_usd, 2)
        a.allocation_fraction = (
            a.allocated_usd / daily_exposure_cap_usd if daily_exposure_cap_usd else 0.0
        )
    return allocations


def _fill_contracts(
    allocations: list[BracketAllocation],
    entry_prices: dict[str, float] | None,
) -> None:
    """Convert risk dollars to a contract count where a fill price is known.

    Downstream P&L counts contracts (`settle_paper_trades`), so risk
    dollars must be divided by the per-contract max loss: the entry cost
    for BUY, ``1 − bid`` for SELL — fee included.
    """
    if not entry_prices:
        return
    for a in allocations:
        price = entry_prices.get(a.signal.contract_id)
        if price is None or not (0.0 < price < 1.0):
            continue
        rpc = risk_per_contract(price, a.signal.signal_side)
        if rpc > 0:
            a.contracts = round(a.allocated_usd / rpc, 2)
