"""Kelly sizing with fee-adjusted edges and a correlation haircut.

Implements the sizing framework from the 2026-06 research report (Q5/Q6):

- **Kelly fraction** for a binary contract, computed on the *fee-adjusted*
  entry price so a thin edge that dies to the taker fee sizes to zero.
- **Fractional Kelly** (default 0.25x) — raw Kelly overbets out-of-sample
  because the model probability is itself an estimate (Baker & McHale 2013).
- **Correlation haircut** — same-day bets across cities share one synoptic
  airmass.  Each bet's fraction is divided by ``1 + (N−1)·rho_bar``, i.e. the
  portfolio is treated as ``N_eff = N / (1 + (N−1)·rho_bar)`` independent
  bets rather than N.

Fee model — Kalshi charges **takers** ``0.07 · P · (1−P)`` per contract at
the execution price P (peaks at 1.75 cents at P=0.5); makers pay nothing.
Paper fills cross the spread (BUY at ask / SELL at bid), so every paper
trade is a taker trade.

Conventions: prices and probabilities in [0, 1]; one contract pays $1.
A SELL (short YES at bid ``b``) is treated as buying NO at ``1 − b``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

KALSHI_TAKER_FEE_COEFF: float = 0.07


def kalshi_taker_fee(price: float, coeff: float = KALSHI_TAKER_FEE_COEFF) -> float:
    """Taker fee per contract at execution price *price*."""
    return coeff * price * (1.0 - price)


def effective_entry(price: float, side: str, fee_coeff: float = KALSHI_TAKER_FEE_COEFF) -> float:
    """Fee-adjusted cost per contract of the position's *risk* side.

    BUY:  pay the ask plus the fee → effective long-YES cost.
    SELL: short YES at the bid ``b`` == long NO at ``1 − b``; the fee adds to
    that cost the same way.
    """
    fee = kalshi_taker_fee(price, fee_coeff)
    if side == "BUY":
        return price + fee
    if side == "SELL":
        return (1.0 - price) + fee
    raise ValueError(f"side must be BUY or SELL, got {side!r}")


def net_edge(model_prob: float, price: float, side: str,
             fee_coeff: float = KALSHI_TAKER_FEE_COEFF) -> float:
    """Expected value per contract after the taker fee.

    BUY:  p − ask − fee.   SELL: bid − p − fee.
    """
    fee = kalshi_taker_fee(price, fee_coeff)
    if side == "BUY":
        return model_prob - price - fee
    if side == "SELL":
        return price - model_prob - fee
    raise ValueError(f"side must be BUY or SELL, got {side!r}")


def kelly_fraction(model_prob: float, price: float, side: str,
                   fee_coeff: float = KALSHI_TAKER_FEE_COEFF) -> float:
    """Full-Kelly bankroll fraction for one contract, fee-adjusted.

    For a binary paying 1 bought at effective price ``q``::

        f* = (p − q) / (1 − q)

    where ``p`` is the win probability of the position (``model_prob`` for
    BUY, ``1 − model_prob`` for SELL) and ``q`` the fee-adjusted cost from
    :func:`effective_entry`.  Returns 0.0 when the edge is non-positive
    after fees (never a negative "anti-bet").
    """
    q = effective_entry(price, side, fee_coeff)
    p = model_prob if side == "BUY" else 1.0 - model_prob
    if q >= 1.0 or q <= 0.0:
        return 0.0
    f = (p - q) / (1.0 - q)
    return max(f, 0.0)


def correlation_haircut(n_bets: int, avg_pairwise_rho: float) -> float:
    """Per-bet scaling for *n_bets* simultaneous correlated bets.

    ``1 / (1 + (N−1)·rho_bar)`` — equals 1 for a single bet or rho=0, and
    1/N for perfectly correlated bets (the day is really one bet split N
    ways).  rho is clamped to [0, 1]: negative same-day weather correlation
    across these cities is not a regime we size up on.
    """
    if n_bets <= 1:
        return 1.0
    rho = min(max(avg_pairwise_rho, 0.0), 1.0)
    return 1.0 / (1.0 + (n_bets - 1) * rho)


def risk_per_contract(price: float, side: str,
                      fee_coeff: float = KALSHI_TAKER_FEE_COEFF) -> float:
    """Maximum loss per contract including the entry fee.

    BUY loses the ask (plus fee) when YES fails; SELL loses ``1 − bid``
    (plus fee) when YES hits.  This converts risk dollars to contracts.
    """
    return effective_entry(price, side, fee_coeff)
