"""Tests for deep_isobar.research.backtest_engine.

Covers:
- simulate_backtest_trades: BUY / SELL / HOLD pnl logic
- simulate_backtest_trades: quantity validation
- simulate_backtest_trades: side / price / realized_outcome validation
- simulate_backtest_trades: missing required columns
- simulate_backtest_trades: empty input fast path
- simulate_backtest_trades: optional column pass-through
- simulate_backtest_trades: output column set and row count
- summarize_backtest_results: total_trades / win_rate / gross_pnl / max_drawdown
- summarize_backtest_results: empty input → zeros
- summarize_backtest_results: all-HOLD → total_trades=0
- summarize_backtest_results: missing realized_pnl → ValueError
- summarize_backtest_results: max_drawdown on loss streaks and recovering series
"""

import pytest
import pandas as pd

from deep_isobar.research.backtest_engine import (
    simulate_backtest_trades,
    summarize_backtest_results,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_QUANTITY = 10.0


def _row(**overrides) -> dict:
    """Build a minimal valid opportunity row."""
    defaults = dict(
        contract_id="CHI_HIGH_TEMP_F_GE_80_20260320",
        side="BUY",
        model_probability=0.70,
        market_probability=0.41,
        simulated_price=0.41,
        realized_outcome=1,
    )
    defaults.update(overrides)
    return defaults


def _df(*rows) -> pd.DataFrame:
    """Build a DataFrame from one or more row dicts."""
    return pd.DataFrame(list(rows))


# ---------------------------------------------------------------------------
# simulate_backtest_trades — pnl logic
# ---------------------------------------------------------------------------


def test_buy_win_pnl():
    """BUY + outcome=1: pnl_per_unit = 1 - simulated_price."""
    df = _df(_row(side="BUY", simulated_price=0.40, realized_outcome=1))
    result = simulate_backtest_trades(df, _QUANTITY)
    assert result.loc[0, "pnl_per_unit"] == pytest.approx(0.60)
    assert result.loc[0, "realized_pnl"] == pytest.approx(0.60 * _QUANTITY)


def test_buy_loss_pnl():
    """BUY + outcome=0: pnl_per_unit = 0 - simulated_price (negative)."""
    df = _df(_row(side="BUY", simulated_price=0.70, realized_outcome=0))
    result = simulate_backtest_trades(df, _QUANTITY)
    assert result.loc[0, "pnl_per_unit"] == pytest.approx(-0.70)
    assert result.loc[0, "realized_pnl"] == pytest.approx(-0.70 * _QUANTITY)


def test_sell_win_pnl():
    """SELL + outcome=0: pnl_per_unit = simulated_price - 0."""
    df = _df(_row(side="SELL", simulated_price=0.70, realized_outcome=0))
    result = simulate_backtest_trades(df, _QUANTITY)
    assert result.loc[0, "pnl_per_unit"] == pytest.approx(0.70)
    assert result.loc[0, "realized_pnl"] == pytest.approx(0.70 * _QUANTITY)


def test_sell_loss_pnl():
    """SELL + outcome=1: pnl_per_unit = simulated_price - 1 (negative)."""
    df = _df(_row(side="SELL", simulated_price=0.30, realized_outcome=1))
    result = simulate_backtest_trades(df, _QUANTITY)
    assert result.loc[0, "pnl_per_unit"] == pytest.approx(-0.70)
    assert result.loc[0, "realized_pnl"] == pytest.approx(-0.70 * _QUANTITY)


def test_hold_pnl_is_zero():
    """HOLD always produces pnl_per_unit = 0 regardless of outcome."""
    df = _df(_row(side="HOLD", simulated_price=0.50, realized_outcome=1))
    result = simulate_backtest_trades(df, _QUANTITY)
    assert result.loc[0, "pnl_per_unit"] == pytest.approx(0.0)
    assert result.loc[0, "realized_pnl"] == pytest.approx(0.0)


def test_quantity_is_multiplied():
    """realized_pnl = pnl_per_unit * quantity."""
    df = _df(_row(side="BUY", simulated_price=0.40, realized_outcome=1))
    result = simulate_backtest_trades(df, quantity=5.0)
    assert result.loc[0, "quantity"] == pytest.approx(5.0)
    assert result.loc[0, "realized_pnl"] == pytest.approx(0.60 * 5.0)


def test_mixed_sides_all_computed():
    """BUY, SELL, HOLD in one dataframe all compute correctly."""
    df = _df(
        _row(side="BUY", simulated_price=0.40, realized_outcome=1),   # +0.60
        _row(side="SELL", simulated_price=0.70, realized_outcome=0),  # +0.70
        _row(side="HOLD", simulated_price=0.55, realized_outcome=1),  # 0.00
    )
    result = simulate_backtest_trades(df, quantity=1.0)
    assert result.loc[0, "pnl_per_unit"] == pytest.approx(0.60)
    assert result.loc[1, "pnl_per_unit"] == pytest.approx(0.70)
    assert result.loc[2, "pnl_per_unit"] == pytest.approx(0.00)


# ---------------------------------------------------------------------------
# simulate_backtest_trades — price boundary values
# ---------------------------------------------------------------------------


def test_simulated_price_boundary_zero():
    """simulated_price=0.0 is valid (lower bound inclusive)."""
    df = _df(_row(side="BUY", simulated_price=0.0, realized_outcome=1))
    result = simulate_backtest_trades(df, _QUANTITY)
    assert result.loc[0, "pnl_per_unit"] == pytest.approx(1.0)


def test_simulated_price_boundary_one():
    """simulated_price=1.0 is valid (upper bound inclusive)."""
    df = _df(_row(side="BUY", simulated_price=1.0, realized_outcome=1))
    result = simulate_backtest_trades(df, _QUANTITY)
    assert result.loc[0, "pnl_per_unit"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# simulate_backtest_trades — output shape and column preservation
# ---------------------------------------------------------------------------


def test_output_contains_required_columns():
    """Output always contains the guaranteed column set."""
    required = {
        "contract_id", "side", "quantity", "simulated_price",
        "realized_outcome", "pnl_per_unit", "realized_pnl",
    }
    df = _df(_row())
    result = simulate_backtest_trades(df, _QUANTITY)
    assert required.issubset(set(result.columns))


def test_output_row_count_matches_input():
    """One output row per input row."""
    df = _df(_row(), _row(side="SELL", realized_outcome=0), _row(side="HOLD"))
    result = simulate_backtest_trades(df, _QUANTITY)
    assert len(result) == 3


def test_optional_columns_passed_through():
    """model_probability, market_probability, alpha, rank_score are preserved."""
    df = _df(_row(alpha=0.29, rank_score=0.31))
    result = simulate_backtest_trades(df, _QUANTITY)
    assert "alpha" in result.columns
    assert "rank_score" in result.columns
    assert result.loc[0, "alpha"] == pytest.approx(0.29)
    assert result.loc[0, "rank_score"] == pytest.approx(0.31)


def test_model_and_market_probability_preserved():
    """model_probability and market_probability are kept in output."""
    df = _df(_row(model_probability=0.70, market_probability=0.41))
    result = simulate_backtest_trades(df, _QUANTITY)
    assert "model_probability" in result.columns
    assert result.loc[0, "model_probability"] == pytest.approx(0.70)


def test_output_index_is_reset():
    """Output index starts from 0 regardless of input index."""
    df = _df(_row(), _row(side="SELL", realized_outcome=0))
    df.index = [10, 20]
    result = simulate_backtest_trades(df, _QUANTITY)
    assert list(result.index) == [0, 1]


# ---------------------------------------------------------------------------
# simulate_backtest_trades — empty input
# ---------------------------------------------------------------------------


def test_empty_df_returns_empty_with_required_columns():
    """Empty input returns empty DataFrame with the expected column set."""
    empty = pd.DataFrame(columns=list(
        {"contract_id", "side", "model_probability", "market_probability",
         "simulated_price", "realized_outcome"}
    ))
    result = simulate_backtest_trades(empty, _QUANTITY)
    assert result.empty
    required = {
        "contract_id", "side", "quantity", "simulated_price",
        "realized_outcome", "pnl_per_unit", "realized_pnl",
    }
    assert required.issubset(set(result.columns))


# ---------------------------------------------------------------------------
# simulate_backtest_trades — validation errors
# ---------------------------------------------------------------------------


def test_quantity_zero_raises():
    """quantity=0 raises ValueError."""
    with pytest.raises(ValueError, match="quantity"):
        simulate_backtest_trades(_df(_row()), quantity=0.0)


def test_quantity_negative_raises():
    """Negative quantity raises ValueError."""
    with pytest.raises(ValueError, match="quantity"):
        simulate_backtest_trades(_df(_row()), quantity=-5.0)


def test_missing_required_column_raises():
    """Missing required column raises ValueError listing the column."""
    df = _df(_row())
    df = df.drop(columns=["simulated_price"])
    with pytest.raises(ValueError, match="simulated_price"):
        simulate_backtest_trades(df, _QUANTITY)


def test_invalid_side_raises():
    """Unrecognised side value raises ValueError."""
    df = _df(_row(side="LONG"))
    with pytest.raises(ValueError, match="side"):
        simulate_backtest_trades(df, _QUANTITY)


def test_simulated_price_above_one_raises():
    """simulated_price > 1 raises ValueError."""
    df = _df(_row(simulated_price=1.01))
    with pytest.raises(ValueError, match="simulated_price"):
        simulate_backtest_trades(df, _QUANTITY)


def test_simulated_price_below_zero_raises():
    """simulated_price < 0 raises ValueError."""
    df = _df(_row(simulated_price=-0.01))
    with pytest.raises(ValueError, match="simulated_price"):
        simulate_backtest_trades(df, _QUANTITY)


def test_invalid_realized_outcome_raises():
    """realized_outcome not in {0, 1} raises ValueError."""
    df = _df(_row(realized_outcome=0.5))
    with pytest.raises(ValueError, match="realized_outcome"):
        simulate_backtest_trades(df, _QUANTITY)


def test_invalid_realized_outcome_two_raises():
    """realized_outcome=2 raises ValueError."""
    df = _df(_row(realized_outcome=2))
    with pytest.raises(ValueError, match="realized_outcome"):
        simulate_backtest_trades(df, _QUANTITY)


def test_nan_simulated_price_raises():
    """NaN simulated_price raises ValueError rather than propagating silently."""
    import math
    df = _df(_row(simulated_price=math.nan))
    with pytest.raises(ValueError, match="simulated_price"):
        simulate_backtest_trades(df, _QUANTITY)


# ---------------------------------------------------------------------------
# summarize_backtest_results — basic metrics
# ---------------------------------------------------------------------------

# Hand-crafted scenario:
#   row 0: BUY win  — simulated_price=0.40, outcome=1 → pnl_per_unit=+0.60 → realized_pnl=+6.0
#   row 1: BUY loss — simulated_price=0.70, outcome=0 → pnl_per_unit=−0.70 → realized_pnl=−7.0
#   row 2: SELL win — simulated_price=0.60, outcome=0 → pnl_per_unit=+0.60 → realized_pnl=+6.0
#   row 3: HOLD     — pnl_per_unit=0 → realized_pnl=0.0
#
# total_trades  = 3   (non-HOLD)
# win_rate      = 2/3 ≈ 0.6667
# gross_pnl     = 6 − 7 + 6 + 0 = 5.0
# cumulative    = [6.0, −1.0, 5.0, 5.0]
# running_peak  = [6.0, 6.0,  6.0, 6.0]
# drawdown      = [0.0, 7.0,  1.0, 1.0]
# max_drawdown  = 7.0


def _make_summary_trades() -> pd.DataFrame:
    df = _df(
        _row(side="BUY",  simulated_price=0.40, realized_outcome=1),
        _row(side="BUY",  simulated_price=0.70, realized_outcome=0),
        _row(side="SELL", simulated_price=0.60, realized_outcome=0),
        _row(side="HOLD", simulated_price=0.50, realized_outcome=1),
    )
    return simulate_backtest_trades(df, quantity=10.0)


def test_summary_total_trades():
    """total_trades counts only non-HOLD rows."""
    summary = summarize_backtest_results(_make_summary_trades())
    assert summary["total_trades"] == 3


def test_summary_win_rate():
    """win_rate = wins / total_trades."""
    summary = summarize_backtest_results(_make_summary_trades())
    assert summary["win_rate"] == pytest.approx(2 / 3)


def test_summary_gross_pnl():
    """gross_pnl = sum of all realized_pnl (including HOLD zeros)."""
    summary = summarize_backtest_results(_make_summary_trades())
    assert summary["gross_pnl"] == pytest.approx(5.0)


def test_summary_max_drawdown():
    """max_drawdown = max(running_peak − cumulative_pnl)."""
    summary = summarize_backtest_results(_make_summary_trades())
    # Cumulative pnl (quantity=10): [6.0, -1.0, 5.0, 5.0]
    # peak: [6.0, 6.0, 6.0, 6.0] → drawdown: [0, 7, 1, 1] → max = 7.0
    assert summary["max_drawdown"] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# summarize_backtest_results — edge cases
# ---------------------------------------------------------------------------


def test_summary_empty_df_returns_zeros():
    """Empty trades_df returns all-zero metrics without error."""
    empty = pd.DataFrame(columns=["side", "realized_pnl"])
    result = summarize_backtest_results(empty)
    assert result["total_trades"] == 0
    assert result["win_rate"] == pytest.approx(0.0)
    assert result["gross_pnl"] == pytest.approx(0.0)
    assert result["max_drawdown"] == pytest.approx(0.0)


def test_summary_all_hold_returns_zero_trades():
    """All-HOLD input: total_trades=0, win_rate=0.0; gross_pnl and drawdown still computed."""
    df = _df(
        _row(side="HOLD", simulated_price=0.50, realized_outcome=1),
        _row(side="HOLD", simulated_price=0.55, realized_outcome=0),
    )
    trades = simulate_backtest_trades(df, quantity=10.0)
    result = summarize_backtest_results(trades)
    assert result["total_trades"] == 0
    assert result["win_rate"] == pytest.approx(0.0)
    assert result["gross_pnl"] == pytest.approx(0.0)
    assert result["max_drawdown"] == pytest.approx(0.0)


def test_summary_missing_realized_pnl_raises():
    """ValueError when realized_pnl column is absent."""
    bad = pd.DataFrame({"side": ["BUY"], "contract_id": ["X"]})
    with pytest.raises(ValueError, match="realized_pnl"):
        summarize_backtest_results(bad)


def test_summary_all_winners_max_drawdown_is_zero():
    """Monotonically increasing cumulative PnL → max_drawdown = 0."""
    df = _df(
        _row(side="BUY", simulated_price=0.40, realized_outcome=1),
        _row(side="BUY", simulated_price=0.30, realized_outcome=1),
        _row(side="SELL", simulated_price=0.80, realized_outcome=0),
    )
    trades = simulate_backtest_trades(df, quantity=1.0)
    result = summarize_backtest_results(trades)
    assert result["max_drawdown"] == pytest.approx(0.0)


def test_summary_all_losers_win_rate_is_zero():
    """All losing trades → win_rate = 0.0."""
    df = _df(
        _row(side="BUY", simulated_price=0.80, realized_outcome=0),
        _row(side="BUY", simulated_price=0.90, realized_outcome=0),
    )
    trades = simulate_backtest_trades(df, quantity=1.0)
    result = summarize_backtest_results(trades)
    assert result["win_rate"] == pytest.approx(0.0)


def test_summary_all_winners_win_rate_is_one():
    """All winning trades → win_rate = 1.0."""
    df = _df(
        _row(side="BUY",  simulated_price=0.30, realized_outcome=1),
        _row(side="SELL", simulated_price=0.70, realized_outcome=0),
    )
    trades = simulate_backtest_trades(df, quantity=1.0)
    result = summarize_backtest_results(trades)
    assert result["win_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# summarize_backtest_results — max_drawdown detailed scenarios
# ---------------------------------------------------------------------------


def test_max_drawdown_recovering_series():
    """Series that falls then recovers past old peak produces correct drawdown.

    PnL sequence (quantity=1):
      trade 0:  +5.0  → cum=5.0,  peak=5.0, dd=0.0
      trade 1:  −3.0  → cum=2.0,  peak=5.0, dd=3.0
      trade 2:  −2.0  → cum=0.0,  peak=5.0, dd=5.0  ← maximum
      trade 3:  +7.0  → cum=7.0,  peak=7.0, dd=0.0
      trade 4:  −1.0  → cum=6.0,  peak=7.0, dd=1.0
    max_drawdown = 5.0
    """
    # Build trades_df directly (not via simulate) to control exact pnl values.
    trades_df = pd.DataFrame({
        "side": ["BUY", "BUY", "BUY", "BUY", "BUY"],
        "realized_pnl": [5.0, -3.0, -2.0, 7.0, -1.0],
    })
    result = summarize_backtest_results(trades_df)
    assert result["max_drawdown"] == pytest.approx(5.0)


def test_max_drawdown_single_losing_trade():
    """A single losing trade produces a drawdown equal to its absolute loss."""
    trades_df = pd.DataFrame({
        "side": ["BUY"],
        "realized_pnl": [-4.0],
    })
    result = summarize_backtest_results(trades_df)
    assert result["max_drawdown"] == pytest.approx(4.0)


def test_max_drawdown_flat_series_is_zero():
    """All-zero PnL → max_drawdown = 0."""
    trades_df = pd.DataFrame({
        "side": ["HOLD", "HOLD", "HOLD"],
        "realized_pnl": [0.0, 0.0, 0.0],
    })
    result = summarize_backtest_results(trades_df)
    assert result["max_drawdown"] == pytest.approx(0.0)
