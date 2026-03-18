"""MVP backtest engine for Deep Isobar.

Replays historical opportunities and simulates paper trades so the strategy
can be evaluated on realized outcomes without touching live execution systems.

This module is for **research only**.  It does not import or interact with
live market clients, the scheduler, or the execution layer.

Canonical interface (from INTERFACES.md)::

    simulate_backtest_trades(opportunities_df, quantity) -> pd.DataFrame
    summarize_backtest_results(trades_df) -> dict

Pipeline::

    opportunities_df  →  simulate_backtest_trades  →  trades_df
    trades_df         →  summarize_backtest_results →  metrics dict

Settlement logic (MVP, no transaction costs or slippage)::

    BUY:  pnl_per_unit = realized_outcome − simulated_price
    SELL: pnl_per_unit = simulated_price  − realized_outcome
    HOLD: pnl_per_unit = 0
    realized_pnl = pnl_per_unit × quantity

Ranking metrics returned by summarize_backtest_results::

    total_trades  — non-HOLD rows
    win_rate      — proportion of executed trades with realized_pnl > 0
    gross_pnl     — sum of realized_pnl across all rows
    max_drawdown  — maximum peak-to-trough decline in cumulative PnL
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_REQUIRED_INPUT_COLS: frozenset[str] = frozenset({
    "contract_id",
    "side",
    "model_probability",
    "market_probability",
    "simulated_price",
    "realized_outcome",
})

_VALID_SIDES: frozenset[str] = frozenset({"BUY", "SELL", "HOLD"})

# Guaranteed output columns from simulate_backtest_trades.
_OUTPUT_COLS: list[str] = [
    "contract_id",
    "side",
    "quantity",
    "simulated_price",
    "realized_outcome",
    "pnl_per_unit",
    "realized_pnl",
]

# Optional columns passed through from input when present.
_PASSTHROUGH_COLS: list[str] = [
    "model_probability",
    "market_probability",
    "alpha",
    "rank_score",
    "target_date",
]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def simulate_backtest_trades(
    opportunities_df: pd.DataFrame,
    quantity: float,
) -> pd.DataFrame:
    """Simulate paper trades from a historical opportunities DataFrame.

    Applies simple contract-outcome settlement logic row by row:

    - **BUY**: ``pnl_per_unit = realized_outcome - simulated_price``
    - **SELL**: ``pnl_per_unit = simulated_price - realized_outcome``
    - **HOLD**: ``pnl_per_unit = 0``
    - ``realized_pnl = pnl_per_unit * quantity``

    This is the MVP simulation model.  No transaction costs, slippage, or
    partial fills are applied.  All row-level validation is performed before
    any computation begins, so either all rows succeed or a ``ValueError``
    is raised for the entire batch.

    Args:
        opportunities_df: Historical opportunity rows.  Must contain at
            minimum: ``contract_id``, ``side``, ``model_probability``,
            ``market_probability``, ``simulated_price``,
            ``realized_outcome``.  Optional columns (``alpha``,
            ``rank_score``, ``target_date``) are preserved in the output
            when present.
        quantity: Number of contracts to simulate per row.  Must be > 0.

    Returns:
        DataFrame with one row per input opportunity, containing at minimum:
        ``contract_id``, ``side``, ``quantity``, ``simulated_price``,
        ``realized_outcome``, ``pnl_per_unit``, ``realized_pnl``.
        Returns an empty DataFrame with those columns when
        ``opportunities_df`` is empty.

    Raises:
        ValueError: If ``quantity`` is not positive, required columns are
            missing, any ``side`` value is not one of BUY/SELL/HOLD, any
            ``simulated_price`` is outside [0, 1], or any
            ``realized_outcome`` is not 0 or 1.

    Example::

        trades = simulate_backtest_trades(opportunities_df, quantity=10.0)
        # trades.columns includes:
        #   contract_id, side, quantity, simulated_price, realized_outcome,
        #   pnl_per_unit, realized_pnl
    """
    if quantity <= 0:
        raise ValueError(f"quantity must be > 0, got {quantity}")

    # ── Required column check ──────────────────────────────────────────────
    missing = _REQUIRED_INPUT_COLS - set(opportunities_df.columns)
    if missing:
        raise ValueError(
            f"opportunities_df is missing required columns: {sorted(missing)}"
        )

    # ── Empty-input fast path ──────────────────────────────────────────────
    if opportunities_df.empty:
        logger.debug("simulate_backtest_trades: empty input, returning empty result")
        return pd.DataFrame(columns=_OUTPUT_COLS)

    # ── Row-level validation (all rows validated before computation) ───────
    _validate_rows(opportunities_df)

    # ── Compute pnl_per_unit ───────────────────────────────────────────────
    df = opportunities_df.copy()

    pnl_per_unit = pd.Series(0.0, index=df.index)
    buy_mask = df["side"] == "BUY"
    sell_mask = df["side"] == "SELL"

    pnl_per_unit[buy_mask] = (
        df.loc[buy_mask, "realized_outcome"] - df.loc[buy_mask, "simulated_price"]
    )
    pnl_per_unit[sell_mask] = (
        df.loc[sell_mask, "simulated_price"] - df.loc[sell_mask, "realized_outcome"]
    )

    df["quantity"] = quantity
    df["pnl_per_unit"] = pnl_per_unit
    df["realized_pnl"] = pnl_per_unit * quantity

    # ── Assemble output columns ────────────────────────────────────────────
    keep_cols = list(_OUTPUT_COLS)
    for col in _PASSTHROUGH_COLS:
        if col not in keep_cols and col in df.columns:
            keep_cols.append(col)

    result = df[keep_cols].reset_index(drop=True)

    logger.info(
        "simulate_backtest_trades: %d rows simulated, quantity=%.2f, "
        "gross_realized_pnl=%.4f",
        len(result),
        quantity,
        result["realized_pnl"].sum(),
    )
    return result


def summarize_backtest_results(trades_df: pd.DataFrame) -> dict:
    """Summarize backtest performance metrics from a simulated trades DataFrame.

    Computes four aggregate metrics:

    - **total_trades**: number of non-HOLD rows (executed trades).  When a
      ``side`` column is absent every row is counted as executed.
    - **win_rate**: proportion of executed trades where
      ``realized_pnl > 0``.  Returns 0.0 when there are no executed trades.
    - **gross_pnl**: ``sum(realized_pnl)`` across all rows, including HOLD
      rows (which contribute 0).
    - **max_drawdown**: maximum peak-to-trough decline in cumulative PnL,
      computed in DataFrame row order.  Always >= 0; 0.0 means there was no
      drawdown period.

    Args:
        trades_df: Output from :func:`simulate_backtest_trades`, or any
            DataFrame with at least a ``realized_pnl`` column.  If a
            ``side`` column is present HOLD rows are excluded from
            ``total_trades`` and ``win_rate`` calculations.

    Returns:
        Dictionary with keys ``total_trades`` (int), ``win_rate`` (float),
        ``gross_pnl`` (float), ``max_drawdown`` (float).  All values are
        zero when ``trades_df`` is empty.

    Raises:
        ValueError: If ``realized_pnl`` column is missing.

    Example::

        summary = summarize_backtest_results(trades_df)
        # {"total_trades": 42, "win_rate": 0.619,
        #  "gross_pnl": 7.3,   "max_drawdown": 1.2}
    """
    if "realized_pnl" not in trades_df.columns:
        raise ValueError("trades_df must contain a 'realized_pnl' column")

    if trades_df.empty:
        logger.debug("summarize_backtest_results: empty input, returning zeros")
        return {"total_trades": 0, "win_rate": 0.0, "gross_pnl": 0.0, "max_drawdown": 0.0}

    # ── Gross PnL (all rows; HOLD rows contribute 0) ───────────────────────
    gross_pnl = float(trades_df["realized_pnl"].sum())

    # ── Max drawdown (computed over all rows in order) ─────────────────────
    max_drawdown = _compute_max_drawdown(trades_df["realized_pnl"])

    # ── Executed trades (non-HOLD subset) ─────────────────────────────────
    if "side" in trades_df.columns:
        executed = trades_df[trades_df["side"] != "HOLD"]
    else:
        executed = trades_df

    total_trades = len(executed)

    if total_trades == 0:
        logger.debug(
            "summarize_backtest_results: no executed trades (all HOLD or empty)"
        )
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "gross_pnl": gross_pnl,
            "max_drawdown": max_drawdown,
        }

    win_rate = float((executed["realized_pnl"] > 0).sum() / total_trades)

    result = {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "gross_pnl": gross_pnl,
        "max_drawdown": max_drawdown,
    }

    logger.info(
        "summarize_backtest_results: total_trades=%d win_rate=%.3f "
        "gross_pnl=%.4f max_drawdown=%.4f",
        total_trades,
        win_rate,
        gross_pnl,
        max_drawdown,
    )
    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_rows(df: pd.DataFrame) -> None:
    """Validate per-row constraints in opportunities_df.

    Checks are performed in this order:
    1. ``side`` must be in {BUY, SELL, HOLD}
    2. ``simulated_price`` must be in [0.0, 1.0]
    3. ``realized_outcome`` must be 0 or 1

    Raises:
        ValueError: On the first constraint that is violated, listing all
            offending values found.
    """
    # 1. side
    invalid_side_mask = ~df["side"].isin(_VALID_SIDES)
    if invalid_side_mask.any():
        bad = sorted(df.loc[invalid_side_mask, "side"].unique())
        raise ValueError(
            f"side must be one of {sorted(_VALID_SIDES)}, "
            f"found invalid values: {bad}"
        )

    # 2. simulated_price  (NaN is treated as out-of-range: NaN < 0 and NaN > 1
    #    both evaluate to False in pandas, so we must check isna() explicitly)
    price_mask = (
        df["simulated_price"].isna()
        | (df["simulated_price"] < 0.0)
        | (df["simulated_price"] > 1.0)
    )
    if price_mask.any():
        bad = df.loc[price_mask, "simulated_price"].unique().tolist()
        raise ValueError(
            f"simulated_price must be in [0, 1], "
            f"found out-of-range values: {bad}"
        )

    # 3. realized_outcome
    outcome_mask = ~df["realized_outcome"].isin([0, 1])
    if outcome_mask.any():
        bad = df.loc[outcome_mask, "realized_outcome"].unique().tolist()
        raise ValueError(
            f"realized_outcome must be 0 or 1, "
            f"found invalid values: {bad}"
        )


def _compute_max_drawdown(pnl_series: pd.Series) -> float:
    """Compute the maximum peak-to-trough drawdown of a PnL series.

    The portfolio is assumed to start at 0 before the first trade.  Drawdown
    at each step is the difference between the running cumulative peak
    (starting from 0) and the current cumulative PnL.  The series is
    processed in its existing row order.

    Args:
        pnl_series: Per-trade PnL values in chronological order.

    Returns:
        Maximum observed drawdown (>= 0).  Returns 0.0 for empty input.
    """
    if pnl_series.empty:
        return 0.0
    cumulative = pnl_series.cumsum()
    # Prepend 0 to represent portfolio value before any trade is made, so
    # that an opening loss registers as a drawdown from the origin.
    cumulative_with_origin = pd.concat(
        [pd.Series([0.0], dtype=float), cumulative.reset_index(drop=True)],
        ignore_index=True,
    )
    running_peak = cumulative_with_origin.cummax()
    drawdown = running_peak - cumulative_with_origin
    return float(drawdown.max())
