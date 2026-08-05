"""Smart position sizer for Deep Isobar — go-live safety build, Part 2.

Replaces the previous sizer, which only scaled a static $50 base down via
anomaly and spread multipliers, clamped to ``[min_exposure, base]``. Across
203 settled paper trades (2026-07-04 .. 2026-07-31; the historical record
grows daily so exact counts drift, see
:mod:`deep_isobar.research.validate_position_sizer` for a live rerun), that
sizer barely moved: ``sizing_final_usd`` was mean $48.57, min $40, max $50 —
effectively a constant regardless of edge, entry price, or station quality.

What the paper record shows (driving the design below)
--------------------------------------------------------
- Profit is dangerously concentrated: one trade
  (``KXHIGHTDAL-26JUL07-B100.5``, $0.05 entry) returned +$252.09 — roughly
  half of all realized profit.
- The cheap tail is where that trade lives *and* where most of the losses
  live: entries <= $0.10 excluding that one outlier lost money at a ~35%
  win rate. Same segment, opposite outcomes — that is a sizing problem
  (how much to risk at extreme prices), not a signal problem (whether to
  trade at all).
- Realized max drawdown was $154.47 at the old flat ~$50 stakes — on a
  $500 bankroll that is over 30%.
- The old exposure math did not fit a $500 bankroll: ``base_exposure_usd:
  50.0`` per city x 5 trading cities is ~$250/day at risk, half the
  bankroll, every day.

Design
------
Two functions do the work, both pure and deterministic (same inputs, same
output — every multiplier here is logged into the reasoning string that
:func:`adjust_allocations` returns, so a stake can always be reconstructed
by hand from the CSV):

:func:`compute_city_daily_cap`
    A bankroll-relative ceiling for one city's total risk for the day.
    Feeds into :func:`deep_isobar.trading.bracket_spreader.build_spread`
    exactly as the old sizer's output did, but now derived from one
    authoritative bankroll figure (``risk.position_sizing.bankroll_usd``)
    instead of the old, independently-configured
    ``dynamic_sizing.base_exposure_usd`` (which could silently disagree
    with ``kelly.bankroll_usd`` — the two numbers had no relationship to
    each other). Clamps two hard caps at once — this city's own daily cap,
    and an equal split of the total daily cap across every active trading
    city, computed once by the caller before cities run concurrently in
    ``paper_trade_session.main``'s ``ThreadPoolExecutor`` — so the sum
    across cities can never exceed the total cap without needing any
    shared mutable state between threads.

:func:`adjust_allocations`
    Takes the Kelly-derived :class:`~deep_isobar.trading.bracket_spreader.BracketAllocation`
    list that ``build_spread(..., allocation_method="kelly", ...)`` already
    produces — fee-adjusted edge and the correlation haircut are unchanged,
    reused via :mod:`deep_isobar.trading.kelly` exactly as before, not
    reimplemented — and layers shrink-only evidence multipliers onto each
    trade individually:

    - anomaly confidence / per-flag penalties / ensemble spread (unchanged
      logic from the old sizer, just applied per-trade instead of once
      per city)
    - entry-price tail haircut (new — :func:`_tail_multiplier`)
    - per-station calibration quality (new — :func:`_calibration_multiplier`)
    - per-station realized track record, risk-off only (new —
      :func:`_track_record_multiplier`)

    then clamps the result to a hard per-trade cap
    (``risk.position_sizing.max_risk_per_trade_pct`` of bankroll). This is
    the fix for the concentration problem above: under the old design a
    single huge Kelly edge could claim the *entire* city cap (Kelly's own
    proportional-scale-down only fires when the summed request exceeds the
    cap — with one signal, there is nothing to scale against). The
    per-trade cap bounds any single trade's stake independent of how large
    Kelly's edge estimate is.

    Fee-adjusted net edge itself is **not** a separate multiplier here —
    that scaling already comes from ``kelly_fraction()`` inside
    ``build_spread``'s ``kelly`` method (a bigger edge already produces a
    bigger Kelly stake). Re-scaling by edge again here would double-count
    it.

Every multiplier below only ever shrinks a stake (max 1.0); nothing here
ever sizes a trade up beyond what Kelly + the hard caps already allow. All
caps are ceilings, never floors — a signal that would size to zero for any
reason (dead edge, disqualified) simply gets zero, not a floor.

See ``research/validate_position_sizer.py`` for the required historical
replay: total P&L under this sizing vs. actual (with and without the
outlier), max drawdown, cheap-tail behaviour, and worst-case daily
exposure against the $500 bankroll.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any

from deep_isobar.trading.kelly import risk_per_contract

if TYPE_CHECKING:
    import pandas as pd

    from deep_isobar.trading.bracket_spreader import BracketAllocation

logger = logging.getLogger(__name__)

_DEFAULT_MAX_RISK_PER_TRADE_PCT = 0.025
_DEFAULT_MAX_CITY_DAILY_PCT = 0.03
_DEFAULT_MAX_TOTAL_DAILY_PCT = 0.10

_DEFAULT_CHEAP_TAIL_PRICE = 0.10
_DEFAULT_TAIL_MULTIPLIER = 0.35

_DEFAULT_CALIBRATION_FLOOR = 0.50

_DEFAULT_TRACK_RECORD_MIN_TRADES = 20
_DEFAULT_TRACK_RECORD_FLOOR = 0.50
_DEFAULT_TRACK_RECORD_WORST_EDGE_USD = -0.10


# ---------------------------------------------------------------------------
# Evidence inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StationCalibration:
    """Per-station calibration summary — see
    :func:`deep_isobar.research.daily_scorecard.station_calibration`.

    Deliberately narrow: only the two fields the multiplier needs, so this
    module never has to import pandas/parquet-reading code to be testable.
    Either half may be ``None`` (no NBM benchmark for this station, or too
    few observations for a PIT histogram) — a missing half drops out of the
    quality average rather than penalizing the station for a data gap.
    """

    mae: float
    nbm_mae: float | None = None
    pit_hist: tuple[float, ...] | None = None


@dataclass(frozen=True)
class StationTrackRecord:
    """Per-station realized paper-trading record, built by
    :func:`build_station_track_record` from settled rows strictly *before*
    the date being sized — never a lookahead into the future.
    """

    n_trades: int
    realized_edge_per_contract: float


def build_station_track_record(
    settled_df: "pd.DataFrame",
    city: str,
    asof: date,
) -> StationTrackRecord | None:
    """Build a station's track record from trades settled strictly before *asof*.

    Safe to call both live (today's session, using the trades CSV as it
    stands this morning) and inside a historical replay (using only the
    portion of the CSV that would have existed on that day) — the ``date <
    asof`` filter is what prevents lookahead bias in either case.

    Args:
        settled_df: Rows with ``status in {WIN, LOSS}``, and columns
            ``city``, ``date`` (comparable to *asof*), ``position_size``
            (contracts), ``realized_pnl``.
        city: City name to filter to.
        asof: Only trades with ``date < asof`` are included.

    Returns:
        ``None`` when there is nothing to build a record from (empty input,
        no matching rows, or zero total contracts).
    """
    if settled_df.empty:
        return None
    rows = settled_df[(settled_df["city"] == city) & (settled_df["date"] < asof)]
    if rows.empty:
        return None
    total_contracts = float(rows["position_size"].astype(float).sum())
    if total_contracts <= 0:
        return None
    edge_per_contract = float(rows["realized_pnl"].astype(float).sum()) / total_contracts
    return StationTrackRecord(n_trades=len(rows), realized_edge_per_contract=edge_per_contract)


# ---------------------------------------------------------------------------
# Sizing decision
# ---------------------------------------------------------------------------


@dataclass
class SizingDecision:
    """Final per-trade sizing decision, with a full human-readable derivation."""

    contract_id: str
    kelly_usd: float             # Kelly-derived stake before evidence adjustments
    evidence_multiplier: float   # product of every multiplier below (<= 1.0)
    final_stake_usd: float       # kelly_usd * evidence_multiplier, capped per-trade
    capped: bool                 # True when the per-trade cap actually bound
    reasoning: str


def log_sizing_decisions(decisions: list[SizingDecision], logger: logging.Logger) -> None:
    """Log each sizing decision at INFO (or DEBUG when nothing was adjusted)."""
    for d in decisions:
        if d.evidence_multiplier == 1.0 and not d.capped:
            logger.debug("Position sizing [%s]: %s", d.contract_id, d.reasoning)
        else:
            logger.info("Position sizing [%s]: %s", d.contract_id, d.reasoning)


# ---------------------------------------------------------------------------
# City daily cap
# ---------------------------------------------------------------------------


def compute_city_daily_cap(
    bankroll_usd: float,
    n_active_trading_cities: int,
    cfg: dict,
) -> float:
    """Bankroll-relative daily risk ceiling for one city.

    Clamps two hard caps simultaneously, both expressed in *cfg* as
    fractions of bankroll:

    - ``max_city_daily_pct`` — this city alone (default 3%).
    - ``max_total_daily_pct`` — split evenly across every active trading
      city (default 10% total). Every city computes this the exact same
      way from the same *n_active_trading_cities*, so the sum across
      cities is bounded by the total cap by construction — no shared
      mutable state is needed across the session's concurrent city
      threads.

    Args:
        bankroll_usd: The single authoritative bankroll figure
            (``risk.position_sizing.bankroll_usd``).
        n_active_trading_cities: Count of cities that will actually run a
            session today (``active and trade`` in ``cities.yaml``) — must
            match what the caller passes to every city, or the total-cap
            guarantee breaks.
        cfg: The ``risk.position_sizing`` config dict.

    Returns:
        The city's daily risk cap in dollars, rounded to cents. ``0.0``
        when there are no active trading cities.
    """
    if n_active_trading_cities <= 0:
        return 0.0
    max_city_pct = float(cfg.get("max_city_daily_pct", _DEFAULT_MAX_CITY_DAILY_PCT))
    max_total_pct = float(cfg.get("max_total_daily_pct", _DEFAULT_MAX_TOTAL_DAILY_PCT))
    per_city_share_of_total = max_total_pct / n_active_trading_cities
    city_pct = min(max_city_pct, per_city_share_of_total)
    return round(bankroll_usd * city_pct, 2)


# ---------------------------------------------------------------------------
# Evidence multipliers — each shrinks only (max 1.0), never boosts
# ---------------------------------------------------------------------------


def _anomaly_spread_multiplier(
    anomaly_report: Any | None,
    ensemble_std_f: float | None,
    cfg: dict,
) -> tuple[float, str | None]:
    """Confidence x per-flag x ensemble-spread multiplier (ported unchanged
    from the old sizer's logic — this part of the design was already
    correct, just applied per-trade here instead of once per city).
    """
    anomaly_multipliers: dict = cfg.get("anomaly_multipliers", {})
    flag_penalties: dict = cfg.get("flag_penalties", {})
    thresholds: dict = cfg.get("spread_thresholds", {})

    confidence_key = "NONE" if anomaly_report is None else (anomaly_report.confidence or "NONE").upper()
    confidence_mult = float(anomaly_multipliers.get(confidence_key, 1.0))

    flag_mult = 1.0
    applied_flags: list[str] = []
    if anomaly_report is not None and anomaly_report.flags:
        for flag in anomaly_report.flags:
            code = flag.code if hasattr(flag, "code") else str(flag)
            if code in flag_penalties:
                flag_mult *= float(flag_penalties[code])
                applied_flags.append(code)

    clean_thresh = float(thresholds.get("clean", 3.0))
    moderate_thresh = float(thresholds.get("moderate", 5.0))
    if ensemble_std_f is None or ensemble_std_f < clean_thresh:
        spread_mult, spread_label = 1.0, None
    elif ensemble_std_f < moderate_thresh:
        spread_mult, spread_label = 0.80, "moderate spread"
    else:
        spread_mult, spread_label = 0.60, "wide spread"

    mult = confidence_mult * flag_mult * spread_mult
    parts: list[str] = []
    if confidence_mult != 1.0:
        parts.append(f"×{confidence_mult:.2f}({confidence_key.lower()} confidence)")
    for code in applied_flags:
        parts.append(f"×{float(flag_penalties[code]):.2f}({code})")
    if spread_label:
        parts.append(f"×{spread_mult:.2f}({spread_label})")
    return mult, " ".join(parts) if parts else None


def _tail_multiplier(price: float | None, cfg: dict) -> tuple[float, str | None]:
    """Haircut (not exclusion) for entries in the poorly-calibrated tail.

    Chosen over full exclusion or a price floor: across the settled paper
    record, entries <= $0.10 lost money at roughly a one-in-three win rate
    *excluding* the one outlier that is simultaneously the single biggest
    win — the tail is both the worst-calibrated segment and the source of
    the best trade. Excluding it entirely would forfeit that upside along
    with the leak; Kelly's own fraction formula already demands an
    outsized probability gap before sizing anything this far from 0.50, so
    a flat haircut targets the calibration-error component specifically
    (KDE/EMOS tail probabilities are the least trustworthy part of the
    distribution) without zeroing out a genuine large mispricing.
    """
    if price is None:
        return 1.0, None
    threshold = float(cfg.get("cheap_tail_price_threshold", _DEFAULT_CHEAP_TAIL_PRICE))
    mult = float(cfg.get("cheap_tail_multiplier", _DEFAULT_TAIL_MULTIPLIER))
    if price <= threshold or price >= 1.0 - threshold:
        return mult, f"×{mult:.2f}(tail entry {price:.2f})"
    return 1.0, None


def _calibration_multiplier(
    calib: StationCalibration | None, cfg: dict,
) -> tuple[float, str | None]:
    """Scale down (never up) for a station whose calibration is currently poor.

    Two components, averaged when both are available:

    - Accuracy vs. the NBM benchmark: ``nbm_mae / mae``, capped at 1.0 so
      beating the benchmark never pushes the multiplier *above* 1.0 — it
      only stops it being penalized.
    - PIT flatness: one minus the total-variation distance from a uniform
      histogram, normalized to [0, 1]. A well-calibrated station's PIT
      values land roughly evenly across bins; a station that is
      over/under-confident concentrates in the tail bins and scores lower.

    A missing half drops out of the average rather than penalizing a
    station for a data gap; a station with no calibration data at all
    (``calib=None``) gets a neutral 1.0 — evidence you don't have isn't
    evidence of a problem.
    """
    if calib is None:
        return 1.0, None
    floor = float(cfg.get("calibration_multiplier_floor", _DEFAULT_CALIBRATION_FLOOR))
    scores: list[float] = []
    if calib.nbm_mae is not None and calib.nbm_mae > 0:
        accuracy = calib.nbm_mae / calib.mae if calib.mae > 0 else 1.0
        scores.append(min(1.0, accuracy))
    if calib.pit_hist:
        n = len(calib.pit_hist)
        uniform = 1.0 / n
        tvd = sum(abs(f - uniform) for f in calib.pit_hist) / 2.0
        max_tvd = 1.0 - uniform
        flatness = 1.0 - (tvd / max_tvd if max_tvd > 0 else 0.0)
        scores.append(max(0.0, flatness))
    if not scores:
        return 1.0, None
    quality = sum(scores) / len(scores)
    mult = floor + (1.0 - floor) * quality
    return mult, f"×{mult:.2f}(station calib quality {quality:.2f})"


def _track_record_multiplier(
    record: StationTrackRecord | None, cfg: dict,
) -> tuple[float, str | None]:
    """Risk-off only: shrink size for a station with a large-enough sample
    of negative realized edge. Never boosts size for a hot streak — with
    paper-trading samples this small, a good run is noise a sizer should
    not compound on, but a large enough bad run is a signal worth
    respecting.
    """
    min_n = int(cfg.get("track_record_min_trades", _DEFAULT_TRACK_RECORD_MIN_TRADES))
    if record is None or record.n_trades < min_n or record.realized_edge_per_contract >= 0:
        return 1.0, None
    floor = float(cfg.get("track_record_floor", _DEFAULT_TRACK_RECORD_FLOOR))
    worst_edge = float(cfg.get("track_record_worst_edge_usd", _DEFAULT_TRACK_RECORD_WORST_EDGE_USD))
    severity = min(1.0, record.realized_edge_per_contract / worst_edge) if worst_edge != 0 else 0.0
    mult = 1.0 - severity * (1.0 - floor)
    return mult, (
        f"×{mult:.2f}(station track record {record.n_trades}t "
        f"edge {record.realized_edge_per_contract:+.3f}/contract)"
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def adjust_allocations(
    allocations: "list[BracketAllocation]",
    *,
    bankroll_usd: float,
    entry_prices: dict[str, float],
    anomaly_report: Any | None,
    ensemble_std_f: float | None,
    station_calibration: StationCalibration | None,
    station_track_record: StationTrackRecord | None,
    cfg: dict,
) -> list[SizingDecision]:
    """Apply evidence multipliers and the hard per-trade cap to each allocation.

    Args:
        allocations: Kelly-derived allocations from
            ``build_spread(..., allocation_method="kelly", ...)`` — one
            :class:`SizingDecision` is returned per allocation, same order.
        bankroll_usd: The single authoritative bankroll figure.
        entry_prices: ``{contract_id: fill price}``, same mapping passed to
            ``build_spread``.
        anomaly_report: This city's anomaly report for the session (or
            ``None``) — applied identically to every trade in the city, as
            before.
        ensemble_std_f: This city's ensemble spread for the session.
        station_calibration: This city's calibration summary (or ``None``).
        station_track_record: This city's realized track record (or
            ``None``), from :func:`build_station_track_record`.
        cfg: The ``risk.position_sizing`` config dict.

    Returns:
        One :class:`SizingDecision` per input allocation, in the same order.
    """
    anomaly_mult, anomaly_note = _anomaly_spread_multiplier(anomaly_report, ensemble_std_f, cfg)
    calib_mult, calib_note = _calibration_multiplier(station_calibration, cfg)
    track_mult, track_note = _track_record_multiplier(station_track_record, cfg)
    max_trade_usd = round(
        bankroll_usd * float(cfg.get("max_risk_per_trade_pct", _DEFAULT_MAX_RISK_PER_TRADE_PCT)), 2
    )

    decisions: list[SizingDecision] = []
    for alloc in allocations:
        price = entry_prices.get(alloc.signal.contract_id)
        tail_mult, tail_note = _tail_multiplier(price, cfg)

        evidence_mult = anomaly_mult * tail_mult * calib_mult * track_mult
        adjusted = round(alloc.allocated_usd * evidence_mult, 2)
        final = min(adjusted, max_trade_usd)
        capped = final < adjusted - 1e-9

        parts = [f"kelly=${alloc.allocated_usd:.2f}"]
        parts.extend(n for n in (anomaly_note, tail_note, calib_note, track_note) if n)
        parts.append(f"={adjusted:.2f}")
        if capped:
            parts.append(f"→capped to ${final:.2f}(max/trade ${max_trade_usd:.2f})")
        reasoning = " ".join(parts)

        decisions.append(SizingDecision(
            contract_id=alloc.signal.contract_id,
            kelly_usd=alloc.allocated_usd,
            evidence_multiplier=round(evidence_mult, 6),
            final_stake_usd=final,
            capped=capped,
            reasoning=reasoning,
        ))

    return decisions


def apply_sizing_decisions(
    allocations: "list[BracketAllocation]",
    decisions: list[SizingDecision],
    entry_prices: dict[str, float],
) -> None:
    """Write each decision's final stake back onto its allocation, in place.

    Recomputes ``contracts`` from the adjusted ``allocated_usd`` — the
    value ``build_spread`` filled in is stale once the stake changes here.

    Args:
        allocations: Same list (and order) passed to
            :func:`adjust_allocations`.
        decisions: Its return value.
        entry_prices: Same mapping used to build *decisions*.
    """
    for alloc, decision in zip(allocations, decisions):
        alloc.allocated_usd = decision.final_stake_usd
        price = entry_prices.get(alloc.signal.contract_id)
        if price is None or not (0.0 < price < 1.0):
            alloc.contracts = None
            continue
        rpc = risk_per_contract(price, alloc.signal.signal_side)
        alloc.contracts = round(alloc.allocated_usd / rpc, 2) if rpc > 0 else None
