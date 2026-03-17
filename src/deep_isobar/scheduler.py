"""Deep Isobar MVP orchestration scheduler.

Wires together the full pipeline for one city / one metric / paper trading:

    forecast generation
    → temperature ensemble
    → KDE probability surface
    → Kalshi contract fetch
    → market scanner (alpha evaluation)
    → ranked TradeSignal list

Canonical interface (from INTERFACES.md)::

    run_once(now_utc=None) -> list[TradeSignal]
    run_loop(interval_seconds=300) -> None

MVP scope:
    - City:   Chicago
    - Metric: high_temp_f
    - Source: Kalshi (stub)
    - Mode:   paper trading only

run_once orchestration steps:
    1.  Load the Chicago city profile.
    2.  Fetch stub forecasts for tomorrow (GFS, ECMWF, NAM).
    3.  Build a weighted temperature ensemble.
    4.  Fit a KDE temperature distribution.
    5.  Generate a probability surface over the configured threshold range.
    6.  Fetch live Kalshi contracts (stub).
    7.  For each contract whose threshold is in the surface, fetch its
        order book and evaluate for alpha.
    8.  Return the ranked TradeSignal list.

run_loop calls run_once repeatedly, sleeping between iterations.  Non-fatal
exceptions are logged and the loop continues.  A KeyError from a missing
city configuration is re-raised immediately (fatal configuration error).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

from deep_isobar.config import get_setting
from deep_isobar.core.types import TradeSignal
from deep_isobar.data.city_universe import get_city_profile
from deep_isobar.market.kalshi_client import (
    fetch_live_contracts,
    fetch_orderbook_for_contract,
)
from deep_isobar.market.market_scanner import (
    evaluate_contract_opportunity,
    rank_trade_signals,
)
from deep_isobar.models.forecast_generation import fetch_forecasts_for_city
from deep_isobar.models.kde_temperature_distribution import build_kde_distribution
from deep_isobar.models.probability_surface import generate_probability_surface_kde
from deep_isobar.models.temperature_ensemble import build_temperature_ensemble

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MVP constants
# ---------------------------------------------------------------------------

_CITY = "Chicago"
_METRIC = "high_temp_f"
_OPERATOR = "ge"
_MODELS = ["GFS", "ECMWF", "NAM"]
_MARKET_SOURCE = "Kalshi"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_once(now_utc: datetime | None = None) -> list[TradeSignal]:
    """Execute one full scan cycle and return ranked trade opportunities.

    This is the core orchestration function.  It does not perform any
    trade execution — that responsibility belongs to the caller (or a
    future integrated loop).

    Args:
        now_utc: Reference timestamp for this run.  Defaults to
            ``datetime.now(timezone.utc)`` when not supplied.

    Returns:
        List of :class:`~deep_isobar.core.types.TradeSignal` objects
        sorted by descending ``absolute_alpha``.  Returns an empty list
        if no Kalshi contracts are available or no contracts have a
        threshold represented in the probability surface.

    Raises:
        KeyError: If ``"Chicago"`` is not found in the city profile
            configuration (fatal — misconfigured environment).

    Example::

        signals = run_once()
        for s in signals:
            print(s.contract_id, s.signal_side, f"alpha={s.alpha:.3f}")
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    target_date = (now_utc + timedelta(days=1)).date()

    logger.info(
        "run_once: city=%s metric=%s target_date=%s",
        _CITY,
        _METRIC,
        target_date,
    )

    # ── Step 1: City profile ──────────────────────────────────────────────
    # KeyError here is fatal — re-raised immediately.
    profile = get_city_profile(_CITY)

    # ── Step 2: Forecasts ─────────────────────────────────────────────────
    forecasts = fetch_forecasts_for_city(
        city=_CITY,
        target_date=target_date,
        metric=_METRIC,
        model_names=_MODELS,
    )
    logger.debug("run_once: fetched %d forecast points", len(forecasts))

    # ── Step 3: Temperature ensemble ──────────────────────────────────────
    ensemble = build_temperature_ensemble(
        city_profile=profile,
        forecasts=forecasts,
        target_date=target_date,
        metric=_METRIC,
        ensemble_run_time_utc=now_utc,
    )
    logger.info(
        "run_once: ensemble mean=%.1f°F std=%.2f°F methodology=%s",
        ensemble.bias_corrected_mean_f,
        ensemble.adjusted_std_f,
        ensemble.methodology,
    )

    # ── Step 4: KDE distribution ──────────────────────────────────────────
    forecast_values = [fp.forecast_value_f for fp in forecasts]
    kde = build_kde_distribution(
        forecast_values_f=forecast_values,
        bandwidth=profile.kde_bandwidth,
    )

    # ── Step 5: Live contracts (fetched before surface so we can derive range) ──
    contracts = fetch_live_contracts(_MARKET_SOURCE)
    if not contracts:
        logger.warning("run_once: no contracts returned by %s — returning empty", _MARKET_SOURCE)
        return []

    logger.info("run_once: received %d contracts from %s", len(contracts), _MARKET_SOURCE)

    # ── Step 6: Probability surface (range auto-derived from contract thresholds) ─
    cfg_min: int = get_setting("models.probability_surface.min_temp_f", 10)
    cfg_max: int = get_setting("models.probability_surface.max_temp_f", 120)
    contract_thresholds = [c.threshold_f for c in contracts]
    min_temp_f = min(cfg_min, min(contract_thresholds))
    max_temp_f = max(cfg_max, max(contract_thresholds))
    surface = generate_probability_surface_kde(
        kde_distribution=kde,
        min_temp_f=min_temp_f,
        max_temp_f=max_temp_f,
        comparison_operator=_OPERATOR,
    )
    logger.debug(
        "run_once: probability surface built over %d–%d°F (%d thresholds, contracts min/max=%d/%d)",
        min_temp_f,
        max_temp_f,
        len(surface),
        min(contract_thresholds),
        max(contract_thresholds),
    )

    # ── Steps 7–8: Evaluate each contract ────────────────────────────────
    signal_threshold: float = get_setting("risk.alpha_threshold", 0.10)
    signals: list[TradeSignal] = []

    for contract in contracts:
        try:
            orderbook = fetch_orderbook_for_contract(_MARKET_SOURCE, contract.contract_id)
            signal = evaluate_contract_opportunity(
                contract=contract,
                probability_surface=surface,
                orderbook=orderbook,
                signal_threshold=signal_threshold,
                timestamp_utc=now_utc,
            )
            signals.append(signal)
            logger.debug(
                "run_once: %s α=%.4f side=%s",
                contract.contract_id,
                signal.alpha,
                signal.signal_side,
            )
        except Exception as exc:
            logger.warning(
                "run_once: skipping contract %s — %s: %s",
                contract.contract_id,
                type(exc).__name__,
                exc,
            )

    # ── Step 9: Rank ──────────────────────────────────────────────────────
    ranked = rank_trade_signals(signals)
    logger.info(
        "run_once: evaluated %d contracts → %d signals (top |α|=%.4f)",
        len(signals),
        len(ranked),
        ranked[0].absolute_alpha if ranked else 0.0,
    )
    return ranked


def run_loop(interval_seconds: int = 300) -> None:
    """Run the scan loop indefinitely, sleeping between iterations.

    Calls :func:`run_once` on each iteration.  Non-fatal exceptions are
    logged and the loop continues.  A :exc:`KeyError` from a missing city
    profile is re-raised immediately since it indicates a misconfigured
    environment that will never recover without human intervention.

    Args:
        interval_seconds: Seconds to sleep between scan iterations.
            Defaults to 300 (5 minutes), matching the system design spec.

    Example::

        # Start the paper-trading loop:
        run_loop(interval_seconds=300)
    """
    logger.info("run_loop: starting — interval=%ds city=%s metric=%s", interval_seconds, _CITY, _METRIC)

    while True:
        try:
            signals = run_once()
            logger.info(
                "run_loop: cycle complete — %d signals returned",
                len(signals),
            )
        except KeyError as exc:
            # Fatal: city not configured — nothing will fix this at runtime.
            logger.critical("run_loop: fatal configuration error — %s", exc)
            raise
        except Exception as exc:
            logger.error(
                "run_loop: unhandled exception in run_once — %s: %s — continuing",
                type(exc).__name__,
                exc,
            )

        logger.debug("run_loop: sleeping %ds", interval_seconds)
        time.sleep(interval_seconds)
