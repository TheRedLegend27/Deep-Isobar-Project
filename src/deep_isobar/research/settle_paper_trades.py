"""Settle paper trades against NOAA actual daily high temperature.

Run after ~18:00 local time (6 PM CDT) once NOAA/ACIS has posted the
official daily high for each traded city::

    python -m deep_isobar.research.settle_paper_trades
    python -m deep_isobar.research.settle_paper_trades --date 2026-04-01

Pipeline
--------
1. Read ``data/paper_trades/paper_trades.csv``.
2. Find OPEN rows whose ``date`` matches the settle date (default: today).
3. Group OPEN rows by city; for each city fetch the actual high_temp_f from
   NOAA/ACIS using the city's ``acis_station_id`` (from ``config/cities.yaml``).
4. For each open trade determine the YES outcome by strike type
   (``less``: actual < cap; ``greater``: actual > floor; ``between``:
   floor <= actual <= cap, cap INCLUSIVE — Kalshi B82.5 "82-83" pays on
   82 AND 83), then WIN/LOSS by direction: BUY wins when YES
   settled 1, SELL (short YES) wins when YES settled 0.

5. Compute realized P&L, charging Kalshi's actual taker fee
   ``0.07 × entry × (1 − entry)`` per contract at entry — on every trade,
   win or lose (paper fills cross the spread, so they are taker fills):

   - BUY:  ``(outcome − entry) × qty − fee(entry) × qty``
   - SELL: ``(entry − outcome) × qty − fee(entry) × qty``

   (Until 2026-07-02 this was a flat 7% haircut on winning trades only —
   changed before any real trades settled, so the track record is
   consistent under the new model.)

6. Write ``status``, ``realized_pnl``, and ``settled_temp`` back to the CSV.
7. Print a running P&L summary to stdout.

ACIS availability note
----------------------
NOAA/ACIS typically posts the official daily high by 20:00–23:00 local
time, though Chicago (KMDW) is usually available by ~18:00 CDT.  If the
script reports missing data, wait 30–60 minutes and retry.

The settlement station used per city is ``acis_station_id`` from
``config/cities.yaml``.  For example, Dallas uses ``CLIDFW`` (the Kalshi
official settlement station) rather than the METAR station ``KDFW``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")  # load KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PATH, etc.

from deep_isobar.calibration import bias_loader
from deep_isobar.config import get_setting
from deep_isobar.data.city_universe import get_city_profile, load_city_profiles
from deep_isobar.data.historical_noaa_ingest import fetch_settlement_observations
from deep_isobar.notifications.discord_notifier import (
    COLOR_GREEN,
    COLOR_RED,
    COLOR_BLUE,
    post_embed,
)
from deep_isobar.ops import kill_switch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PAPER_TRADES_CSV = _PROJECT_ROOT / "data" / "paper_trades" / "paper_trades.csv"

# Kalshi taker-fee coefficient: fee = 0.07 × entry × (1 − entry) per
# contract, charged at entry on every taker fill (win or lose).  Peaks at
# 1.75 cents at entry=0.50.  Replaced the flat 7%-of-winnings model on
# 2026-07-02, before any real trades settled.
KALSHI_FEE_RATE = 0.07
_DEFAULT_POSITION_SIZE = 10.0
# Fallback city when the 'city' column is absent or empty (legacy rows written
# before the multi-city refactor — all of which are Chicago trades).
_LEGACY_CITY_DEFAULT = "Chicago"


# ---------------------------------------------------------------------------
# NOAA settlement fetch
# ---------------------------------------------------------------------------


def _fetch_noaa_actual(city_name: str, settle_date: date) -> float | None:
    """Return the official NOAA high_temp_f for *settle_date* from ACIS.

    Uses ``acis_station_id`` from the city config (falling back to
    ``station_id``) via
    :func:`~deep_isobar.data.historical_noaa_ingest.fetch_settlement_observations`.

    Args:
        city_name: Exact city name as in ``config/cities.yaml``.
        settle_date: The date to fetch the observed high for.

    Returns:
        High temperature in °F, or ``None`` if data is unavailable.
    """
    try:
        df = fetch_settlement_observations(
            city=city_name,
            start_date=settle_date,
            end_date=settle_date,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("[%s] ACIS fetch failed for %s: %s", city_name, settle_date, exc)
        return None

    if df.empty:
        logger.warning("[%s] ACIS returned no rows for %s", city_name, settle_date)
        return None

    df = df[df["quality_flag"] != "missing"]
    if df.empty:
        logger.warning("[%s] ACIS data for %s has quality_flag=missing", city_name, settle_date)
        return None

    return float(df.iloc[0]["high_temp_f"])


# ---------------------------------------------------------------------------
# Settlement rule
# ---------------------------------------------------------------------------


def realized_yes_outcome(
    strike_type: str,
    floor_strike: int | None,
    cap_strike: int | None,
    settled_temp: float,
) -> int:
    """Kalshi YES outcome (1/0) for a contract given the settled temperature.

    The single source of truth for the exchange's settlement conventions
    (verified against live API titles and official results 2026-07-05):

    - ``less``    : YES iff ``settled_temp <  cap_strike``
    - ``greater`` : YES iff ``settled_temp >  floor_strike``
    - ``between`` : YES iff ``floor_strike <= settled_temp <= cap_strike``
      (cap INCLUSIVE — B82.5 = "82-83°" pays on 82 AND 83)

    Mirrors :func:`~deep_isobar.models.probability_engine.probability_for_contract`
    exactly; the golden-fixture suite asserts the two never diverge.

    Raises:
        ValueError: On an unknown strike_type or a missing required strike —
            grading must never guess (T-tickers are "less" OR "greater").
    """
    st = (strike_type or "").lower().strip()
    if st == "less":
        if cap_strike is None:
            raise ValueError("less contract requires cap_strike")
        return 1 if settled_temp < cap_strike else 0
    if st == "greater":
        if floor_strike is None:
            raise ValueError("greater contract requires floor_strike")
        return 1 if settled_temp > floor_strike else 0
    if st == "between":
        if floor_strike is None or cap_strike is None:
            raise ValueError("between contract requires floor_strike and cap_strike")
        return 1 if floor_strike <= settled_temp <= cap_strike else 0
    raise ValueError(f"unknown strike_type {strike_type!r}")


# ---------------------------------------------------------------------------
# P&L calculation
# ---------------------------------------------------------------------------


def _compute_pnl(
    direction: str,
    entry_price: float,
    realized_outcome: int,
    position_size: float,
    fee_rate: float = KALSHI_FEE_RATE,
) -> float:
    """Compute net realized P&L for one paper trade after the Kalshi taker fee.

    Settlement convention (Kalshi YES binary):
    - ``realized_outcome = 1``  → YES settled (high >= threshold)
    - ``realized_outcome = 0``  → NO settled  (high <  threshold)

    The taker fee ``fee_rate × entry × (1 − entry)`` per contract is
    charged on every trade (paper fills cross the spread), matching the
    published Kalshi schedule — not just on winners.

    Args:
        direction: ``"BUY"`` (long YES) or ``"SELL"`` (short YES).
        entry_price: Price paid / received per contract, in [0, 1].
        realized_outcome: 1 if YES settled, 0 if NO settled.
        position_size: Number of contracts.
        fee_rate: Kalshi taker-fee coefficient (default 0.07).

    Returns:
        Net realized P&L in probability-point × contracts units.
    """
    if direction == "BUY":
        pnl_per_unit = realized_outcome - entry_price
    elif direction == "SELL":
        pnl_per_unit = entry_price - realized_outcome
    else:
        return 0.0

    fee_per_contract = fee_rate * entry_price * (1.0 - entry_price)
    return (pnl_per_unit - fee_per_contract) * position_size


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------


def _settle_open_trades(
    df: pd.DataFrame,
    settle_date: date,
    settled_temp: float,
    city_name: str,
) -> tuple[pd.DataFrame, int]:
    """Mark OPEN trades for *settle_date* and *city_name* as WIN/LOSS.

    Args:
        df: Full paper_trades DataFrame.
        settle_date: Date being settled.
        settled_temp: Observed high temperature in °F.
        city_name: City whose OPEN rows should be settled.

    Returns:
        ``(updated_df, n_settled)`` where *n_settled* is the count of rows
        that were updated.
    """
    df = df.copy()

    # Determine which rows belong to this city.  The 'city' column was added
    # in the multi-city refactor.  Legacy rows (empty city) default to Chicago.
    if "city" in df.columns:
        city_col = df["city"].fillna("").str.strip()
        city_mask = city_col.apply(
            lambda v: (v == city_name) or (v == "" and city_name == _LEGACY_CITY_DEFAULT)
        )
    else:
        # Column absent entirely — treat all rows as Chicago.
        city_mask = pd.Series(
            [city_name == _LEGACY_CITY_DEFAULT] * len(df), index=df.index
        )

    open_mask = (
        (df["date"].astype(str) == str(settle_date))
        & (df["status"] == "OPEN")
        & city_mask
    )

    def _parse_optional_int(val) -> int | None:
        try:
            s = str(val).strip()
            if s in ("", "nan", "None"):
                return None
            return int(float(s))
        except (ValueError, TypeError):
            return None

    n_settled = 0
    for idx in df[open_mask].index:
        row = df.loc[idx]

        entry_price = float(row["entry_price"])
        direction = str(row["direction"])
        position_size = (
            float(row["position_size"])
            if pd.notna(row.get("position_size"))
            else _DEFAULT_POSITION_SIZE
        )

        strike_type = str(row.get("strike_type", "") or "").lower().strip()
        if not strike_type:
            # NEVER guess from the ticker: T-tickers are "less" OR "greater"
            # (live API 2026-07-05: KXHIGHCHI T76 = less, T83 = greater).
            # The old fallback inferred every T as "less", which would grade
            # a greater-type contract exactly backwards.  Leave the row OPEN
            # for manual review instead.
            logger.error(
                "settle: missing strike_type for %r — left OPEN for manual "
                "review (ticker letters do NOT determine direction).",
                row.get("contract_ticker"),
            )
            continue

        floor_strike = _parse_optional_int(row.get("floor_strike"))
        cap_strike   = _parse_optional_int(row.get("cap_strike"))

        # Legacy rows may lack tail strikes — threshold_f carried the value.
        if strike_type == "less" and cap_strike is None:
            cap_strike = int(float(row["threshold_f"]))
            logger.warning(
                "settle: missing cap_strike for %r — using threshold_f=%s",
                row.get("contract_ticker"), cap_strike,
            )
        elif strike_type == "greater" and floor_strike is None:
            floor_strike = int(float(row["threshold_f"]))
            logger.warning(
                "settle: missing floor_strike for %r — using threshold_f=%s",
                row.get("contract_ticker"), floor_strike,
            )

        try:
            realized_outcome = realized_yes_outcome(
                strike_type, floor_strike, cap_strike, settled_temp
            )
        except ValueError as exc:
            logger.warning(
                "settle: cannot grade %r (%s) — skipping.",
                row.get("contract_ticker"), exc,
            )
            continue

        pnl = _compute_pnl(direction, entry_price, realized_outcome, position_size)

        if direction == "BUY":
            status = "WIN" if realized_outcome == 1 else "LOSS"
        else:
            status = "WIN" if realized_outcome == 0 else "LOSS"

        df.loc[idx, "status"] = status
        df.loc[idx, "realized_pnl"] = round(pnl, 6)
        df.loc[idx, "settled_temp"] = settled_temp
        n_settled += 1

    return df, n_settled


# ---------------------------------------------------------------------------
# P&L summary
# ---------------------------------------------------------------------------


def _print_pnl_summary(df: pd.DataFrame) -> None:
    """Print a running P&L summary over all settled trades to stdout."""
    settled = df[df["status"].isin(["WIN", "LOSS"])].copy()

    if settled.empty:
        print("\n  No settled trades in paper_trades.csv.")
        return

    settled["realized_pnl"] = pd.to_numeric(
        settled["realized_pnl"], errors="coerce"
    ).fillna(0.0)

    total = len(settled)
    wins = int((settled["status"] == "WIN").sum())
    losses = int((settled["status"] == "LOSS").sum())
    win_rate = wins / total if total > 0 else 0.0
    running_pnl = settled["realized_pnl"].sum()

    print()
    print("=" * 58)
    print("  PAPER TRADE P&L SUMMARY  (all settled trades)")
    print("=" * 58)
    print(f"  Total trades   : {total}")
    print(f"  Wins / Losses  : {wins} / {losses}")
    print(f"  Win rate       : {win_rate * 100:.1f}%")
    print(
        f"  Running P&L    : {running_pnl:+.4f}"
        f"  (prob-pts × contracts, after {KALSHI_FEE_RATE:.2f}·P·(1−P) taker fee)"
    )

    if "date" in settled.columns:
        print("\n  By date:")
        by_date = (
            settled.groupby("date")["realized_pnl"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "pnl", "count": "trades"})
            .sort_index()
        )
        for dt, agg_row in by_date.iterrows():
            print(
                f"    {dt}  trades={int(agg_row['trades'])}  "
                f"pnl={agg_row['pnl']:+.4f}"
            )

    print("=" * 58)


def _evaluate_kill_switch_triggers(df: pd.DataFrame, session_pnl: float) -> None:
    """Evaluate the daily-loss / drawdown / consecutive-loss kill-switch
    triggers (go-live safety build, Part 1) after a settlement pass.

    Uses today's realized P&L directly for the daily-loss trigger, and
    derives the consecutive-loss streak and drawdown-from-peak from the
    full settled history in *df* (all cities pooled, chronological by
    target date). Never raises — an exception here must not take down
    settlement itself; kill_switch.engage() failures are already
    best-effort internally, but the derivation here is wrapped defensively
    too since it runs unattended at 18:00.
    """
    try:
        settled = df[df["status"].isin(["WIN", "LOSS"])].copy()
        if settled.empty:
            kill_switch.evaluate_trading_triggers(daily_realized_pnl_usd=session_pnl)
            return

        settled["realized_pnl"] = pd.to_numeric(
            settled["realized_pnl"], errors="coerce"
        ).fillna(0.0)
        settled = settled.sort_values("date", kind="stable")

        # Consecutive losses: trailing run of LOSS rows in settlement order.
        consecutive_losses = 0
        for status in reversed(settled["status"].tolist()):
            if status == "LOSS":
                consecutive_losses += 1
            else:
                break

        # Equity curve for drawdown: bankroll + cumulative realized P&L by
        # date, using the single authoritative bankroll figure the smart
        # sizer also reads (risk.position_sizing.bankroll_usd).
        bankroll_usd = float(get_setting("risk.position_sizing.bankroll_usd", 500.0))
        daily_pnl = settled.groupby("date")["realized_pnl"].sum().sort_index()
        equity_curve = bankroll_usd + daily_pnl.cumsum()
        peak_equity_usd = float(equity_curve.cummax().iloc[-1])
        current_equity_usd = float(equity_curve.iloc[-1])

        kill_switch.evaluate_trading_triggers(
            daily_realized_pnl_usd=session_pnl,
            current_equity_usd=current_equity_usd,
            peak_equity_usd=peak_equity_usd,
            consecutive_losses=consecutive_losses,
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "kill-switch trigger evaluation failed — settlement continues, "
            "but automatic triggers did not run this pass"
        )


# ---------------------------------------------------------------------------
# Bias-profile update
# ---------------------------------------------------------------------------


def _update_bias_profile(
    city_name: str,
    settle_date: date,
    settled_rows: pd.DataFrame,
    settled_temp: float,
) -> None:
    """Push one calibration point into the bias_loader parquet for this date.

    Uses ``station_id`` (METAR station) for the parquet file key — this is
    intentional; the bias profile tracks model vs. METAR-collocated errors,
    independent of the ACIS settlement station.

    Never raises — a failure here must not abort settlement.
    """
    try:
        if "ensemble_mean_f" not in settled_rows.columns:
            logger.debug(
                "[%s] Skipping bias update for %s — ensemble_mean_f column absent (legacy row)",
                city_name, settle_date,
            )
            return

        raw_mean_series = pd.to_numeric(
            settled_rows["ensemble_mean_f"], errors="coerce"
        ).dropna()

        if raw_mean_series.empty:
            logger.debug(
                "[%s] Skipping bias update for %s — ensemble_mean_f is NaN (legacy row)",
                city_name, settle_date,
            )
            return

        raw_mean_f = float(raw_mean_series.iloc[0])
        # Use station_id (METAR), not acis_station_id, for bias profile key.
        profile = get_city_profile(city_name)
        station_id = profile.station_id

        bias_loader.update_profile_after_settlement(
            station_id=station_id,
            settlement_date=settle_date,
            raw_mean_f=raw_mean_f,
            actual_f=settled_temp,
        )
        logger.info(
            "[%s] Bias profile updated — %s  raw_mean=%.2f°F  actual=%.2f°F  error=%.2f°F",
            city_name, station_id, raw_mean_f, settled_temp, raw_mean_f - settled_temp,
        )
    except Exception:
        logger.exception(
            "[%s] Bias profile update failed for %s — settlement is unaffected",
            city_name, settle_date,
        )


def _refresh_emos_calibration() -> None:
    """Nightly EMOS refresh: extend training parquets and refit all stations.

    Re-fetches the previous-runs forecast window (so the training history
    keeps growing past the API's ~92-day limit) and refits the rolling-window
    EMOS parameters used by tomorrow's session.  Never raises — a refresh
    failure must not abort settlement; the session simply keeps using the
    last good parameters.
    """
    try:
        from deep_isobar.calibration.emos_training import (
            build_training_data,
            fit_station,
        )
        from deep_isobar.data.city_universe import get_city_universe

        for city in get_city_universe():
            if not city.active:
                continue
            try:
                build_training_data(city)
                params = fit_station(city)
                logger.info(
                    "[%s] EMOS refit: n=%d  train CRPS=%.3f°F",
                    city.city, params.n_train, params.train_crps,
                )
            except Exception:
                logger.exception(
                    "[%s] EMOS refresh failed — keeping previous params", city.city
                )
    except Exception:
        logger.exception("EMOS calibration refresh failed entirely")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def settle(settle_date: date) -> None:
    """Settle all OPEN paper trades for *settle_date*.

    Groups OPEN rows by city, fetches the NOAA actual for each city using
    the city's ``acis_station_id``, and settles each group independently.

    Args:
        settle_date: The date to settle (typically today, after ~18:00 local).

    Exits with code 1 if ``paper_trades.csv`` is missing.
    """
    if not _PAPER_TRADES_CSV.exists():
        logger.error(
            "paper_trades.csv not found at %s\n"
            "Run paper_trade_session.py first.",
            _PAPER_TRADES_CSV,
        )
        sys.exit(1)

    df = pd.read_csv(
        _PAPER_TRADES_CSV,
        dtype={"threshold_f": float, "position_size": float},
    )

    # Determine which cities have OPEN trades on this date.  Bracket
    # (between) contracts settle like any other row — the old "B-type not
    # implemented" skip left the entire bracket book permanently OPEN.
    open_today_mask = (
        (df["date"].astype(str) == str(settle_date))
        & (df["status"] == "OPEN")
    )
    open_today = df[open_today_mask]

    if open_today.empty:
        logger.info("No OPEN trades for %s — nothing to settle.", settle_date)
        _print_pnl_summary(df)
        _refresh_emos_calibration()
        return

    # Resolve the city for each open row.  The 'city' column was added in the
    # multi-city refactor; legacy rows (empty or absent) default to Chicago.
    if "city" in open_today.columns:
        cities_with_open = (
            open_today["city"].fillna("").str.strip()
            .apply(lambda v: v if v else _LEGACY_CITY_DEFAULT)
            .unique()
            .tolist()
        )
    else:
        cities_with_open = [_LEGACY_CITY_DEFAULT]

    # Settle each city independently.
    n_total_settled = 0
    all_newly_settled_rows: list[pd.DataFrame] = []

    for city_name in cities_with_open:
        logger.info(
            "[%s] Fetching NOAA actual high_temp_f for %s…",
            city_name, settle_date,
        )
        settled_temp = _fetch_noaa_actual(city_name, settle_date)

        if settled_temp is None:
            logger.error(
                "[%s] NOAA actual high_temp_f not yet available for %s.\n"
                "ACIS typically posts by 18:00–23:00 CDT. Try again later.",
                city_name, settle_date,
            )
            # Continue with other cities rather than exiting.
            continue

        logger.info(
            "[%s] NOAA actual high_temp_f for %s: %.1f°F",
            city_name, settle_date, settled_temp,
        )

        df, n_settled = _settle_open_trades(df, settle_date, settled_temp, city_name)
        n_total_settled += n_settled

        newly_settled = df[
            (df["date"].astype(str) == str(settle_date))
            & df["status"].isin(["WIN", "LOSS"])
            & (
                df["city"].fillna("").str.strip().apply(
                    lambda v: (v == city_name) or (v == "" and city_name == _LEGACY_CITY_DEFAULT)
                )
                if "city" in df.columns
                else pd.Series([city_name == _LEGACY_CITY_DEFAULT] * len(df), index=df.index)
            )
        ]

        if not newly_settled.empty:
            all_newly_settled_rows.append(newly_settled)

            # Push calibration point into bias_loader.
            _update_bias_profile(city_name, settle_date, newly_settled, settled_temp)

            # Print newly settled trades for this city.
            print(f"\n  [{city_name}] Settled {len(newly_settled)} trade(s) for {settle_date}:")
            for _, trade_row in newly_settled.iterrows():
                print(
                    f"    {trade_row['direction']:<4}  "
                    f"{trade_row['contract_ticker']:<40}  "
                    f"thr={trade_row['threshold_f']:.0f}°F  "
                    f"actual={settled_temp:.1f}°F  "
                    f"{trade_row['status']:<4}  "
                    f"pnl={float(trade_row['realized_pnl']):+.4f}"
                )

            # ── Per-trade Discord embeds ───────────────────────────────────
            for _, trade_row in newly_settled.iterrows():
                is_win = trade_row["status"] == "WIN"
                pnl = float(trade_row["realized_pnl"])
                post_embed(
                    title=f"{'WIN' if is_win else 'LOSS'} \u2014 [{city_name}] {trade_row['contract_ticker']}",
                    color=COLOR_GREEN if is_win else COLOR_RED,
                    fields=[
                        {"name": "Settled temp",  "value": f"{settled_temp:.1f}\u00b0F"},
                        {"name": "Threshold",     "value": f"{trade_row['threshold_f']:.0f}\u00b0F"},
                        {"name": "Realized P&L",  "value": f"{pnl:+.4f}"},
                        {"name": "Entry price",   "value": f"{float(trade_row['entry_price']):.4f}"},
                    ],
                )

    # Write the fully updated CSV once (after all cities settled).
    df.to_csv(_PAPER_TRADES_CSV, index=False)
    logger.info("Wrote %d settled row(s) to %s", n_total_settled, _PAPER_TRADES_CSV)

    # ── Summary Discord embed ─────────────────────────────────────────────
    if n_total_settled > 0:
        all_settled = df[df["status"].isin(["WIN", "LOSS"])].copy()
        all_settled["realized_pnl"] = pd.to_numeric(
            all_settled["realized_pnl"], errors="coerce"
        ).fillna(0.0)
        n_wins = int((all_settled["status"] == "WIN").sum())
        session_pnl = sum(
            float(r["realized_pnl"].astype(float).sum())
            for r in all_newly_settled_rows
        ) if all_newly_settled_rows else 0.0
        cumulative_pnl = float(all_settled["realized_pnl"].sum())
        post_embed(
            title=f"Settlement complete \u2014 {settle_date}",
            color=COLOR_BLUE,
            fields=[
                {"name": "Trades settled",  "value": str(n_total_settled)},
                {"name": "Wins",            "value": str(n_wins)},
                {"name": "Losses",          "value": str(len(all_settled) - n_wins)},
                {"name": "Session P&L",     "value": f"{session_pnl:+.4f}"},
                {"name": "Cumulative P&L",  "value": f"{cumulative_pnl:+.4f}"},
            ],
        )

        # Kill-switch automatic triggers (go-live safety build, Part 1):
        # daily loss, drawdown-from-peak, consecutive losses.
        _evaluate_kill_switch_triggers(df, session_pnl)

    _print_pnl_summary(df)

    # ── Nightly EMOS calibration refresh ──────────────────────────────────
    _refresh_emos_calibration()


def settle_all_pending(up_to: date | None = None) -> None:
    """Settle every date that still has OPEN rows, oldest first.

    ACIS publishes a day's official high the NEXT morning, so a row dated D
    is normally settleable from the morning of D+1 — the old single-date
    logic asked for *today's* high at 18:00 (never available) and dropped
    prior dates out of scope forever.  Dates whose observation is still
    missing are left OPEN and retried on the next run.
    """
    if not _PAPER_TRADES_CSV.exists():
        logger.error("paper_trades.csv not found at %s", _PAPER_TRADES_CSV)
        sys.exit(1)

    up_to = up_to or date.today()
    df = pd.read_csv(_PAPER_TRADES_CSV)
    pending = sorted(
        {
            d for d in pd.to_datetime(df.loc[df["status"] == "OPEN", "date"]).dt.date
            if d <= up_to
        }
    )
    if not pending:
        logger.info("No OPEN rows dated on/before %s — nothing to settle.", up_to)
        _refresh_emos_calibration()
        return

    logger.info("Pending settlement dates: %s", [str(d) for d in pending])
    for d in pending:
        settle(d)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Settle paper trades against NOAA actual daily high temperature.\n\n"
            "Run after ~18:00 local time when ACIS has posted the official\n"
            "daily highs.  Handles all active cities automatically using\n"
            "each city's acis_station_id from config/cities.yaml."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--date",
        default=str(date.today()),
        metavar="YYYY-MM-DD",
        help="Date to settle (default: today).",
    )
    parser.add_argument(
        "--pending",
        action="store_true",
        default=True,
        help=(
            "Settle every date that still has OPEN rows (default).  ACIS "
            "posts a day's official high the NEXT morning, so the 18:00 run "
            "mainly clears yesterday; unsettleable dates are retried on the "
            "next run automatically."
        ),
    )
    parser.add_argument(
        "--only-date",
        action="store_true",
        help="Settle strictly the --date given (legacy single-date behaviour).",
    )
    args = parser.parse_args()

    try:
        settle_date = date.fromisoformat(args.date)
    except ValueError:
        print(
            f"ERROR: invalid date {args.date!r}. Use YYYY-MM-DD format.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.only_date:
        settle(settle_date)
    else:
        settle_all_pending(up_to=settle_date)
