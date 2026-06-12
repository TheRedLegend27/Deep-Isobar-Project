"""Multi-bracket position spreader for Deep Isobar.

Distributes a session's daily exposure cap across the top-N qualifying alpha
signals using a proportional (or future) allocation method.

Interface::

    build_spread(signals, daily_exposure_cap_usd, max_contracts,
                 min_alpha, allocation_method) -> list[BracketAllocation]
    log_spread_summary(allocations, logger) -> None
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from deep_isobar.core.types import TradeSignal

logger = logging.getLogger(__name__)


@dataclass
class BracketAllocation:
    """Allocation for a single bracket in a multi-contract spread."""

    signal: TradeSignal           # the original TradeSignal object
    allocated_usd: float          # dollar amount to risk on this bracket
    allocation_fraction: float    # fraction of total session exposure (0.0–1.0)
    rank: int                     # 1 = highest alpha


def build_spread(
    signals: list[TradeSignal],
    daily_exposure_cap_usd: float,
    max_contracts: int,
    min_alpha: float,
    allocation_method: str,
) -> list[BracketAllocation]:
    """Build a multi-bracket spread from a list of trade signals.

    Only BUY signals are spread.  SELL spreading is not implemented — SELL
    signals are silently skipped rather than raising, so a mixed session
    degrades gracefully to BUY-only spreading.

    Args:
        signals: All candidate TradeSignal objects for the session.
        daily_exposure_cap_usd: Total USD to allocate across all brackets.
        max_contracts: Maximum number of brackets to enter.
        min_alpha: Minimum |alpha| required for a signal to qualify.
        allocation_method: ``"proportional"`` | ``"equal"`` | ``"kelly"``.
            Only ``"proportional"`` is implemented; others raise
            ``NotImplementedError``.

    Returns:
        List of :class:`BracketAllocation` sorted by rank ascending
        (rank 1 = highest alpha).  Returns an empty list when no signals
        pass the filter.

    Raises:
        NotImplementedError: For ``"equal"`` or ``"kelly"`` methods.
        ValueError: For an unrecognised ``allocation_method``.
    """
    # Filter: BUY only (SELL spreading not implemented — future enhancement).
    sell_signals = [s for s in signals if s.signal_side == "SELL"]
    if sell_signals:
        logger.warning(
            "build_spread: %d SELL signal(s) skipped — SELL spreading not yet implemented "
            "(contracts: %s)",
            len(sell_signals),
            [s.contract_id for s in sell_signals],
        )

    qualifying = [
        s for s in signals
        if s.signal_side == "BUY" and abs(s.alpha) >= min_alpha
    ]

    if not qualifying:
        return []

    # Sort descending by alpha magnitude; take top max_contracts.
    qualifying.sort(key=lambda s: abs(s.alpha), reverse=True)
    selected = qualifying[:max_contracts]

    if allocation_method == "proportional":
        return _proportional_allocation(selected, daily_exposure_cap_usd)
    elif allocation_method == "equal":
        raise NotImplementedError("equal allocation not yet implemented")
    elif allocation_method == "kelly":
        raise NotImplementedError("kelly allocation not yet implemented")
    else:
        raise ValueError(f"Unknown allocation_method: {allocation_method!r}")


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
        logger.info("Bracket spread — no allocations (no qualifying BUY signals)")
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
            "#%d %-40s  alpha=%.3f  alloc=$%.2f",
            alloc.rank,
            alloc.signal.contract_id,
            alloc.signal.alpha,
            alloc.allocated_usd,
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
