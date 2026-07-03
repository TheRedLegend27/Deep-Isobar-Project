"""Pre-trade circuit breakers — the "is this sane?" gate before any signal.

Motivated by the 2026-07-02 incident: a disintegrating runtime produced a
103.8°F Philadelphia forecast and traded 16 contracts against deterministic
stub markets, because nothing between the data layer and the order log
sanity-checks the day.  Each check here is a refusal, never a trade: a
failed preflight means the city logs nothing and alerts.

Checks (config under ``risk.preflight``):

1. **live_market**   — the Kalshi client must be in live mode.  Stub books
   price everything at 0.50 and any "edge" against them is fiction.
2. **forecast_bounds** — the calibrated mean must sit within the observed
   historical range of the station's training data ± ``bounds_margin_f``.
3. **nbm_divergence** — |mu − NBM MaxT| beyond ``max_nbm_divergence_f`` is
   a hard stop (warn-level divergence is handled in the session already).
   NBM is NOAA's calibrated blend; a huge gap means our inputs are broken
   far more often than it means we found a 6°F edge nobody else sees.
4. **params_freshness** — EMOS params older than ``max_params_age_days``
   mean the 06:15 refit has been failing; stale coefficients drift.
5. **sigma_sanity** — predictive sigma outside [``min_sigma_f``,
   ``max_sigma_f``] indicates a broken variance input.
6. **contract_count** — fewer than ``min_contracts`` live contracts means
   the market isn't really open for that day.

Usage::

    result = run_preflight(...)
    if not result.ok:
        # refuse to trade this city today
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from deep_isobar.calibration.emos import EMOSParams
from deep_isobar.config import get_setting

logger = logging.getLogger(__name__)

# Defaults — all overridable via risk.preflight in settings.yaml.
_DEFAULTS = {
    "enabled": True,
    "allow_stub_market": False,   # True only for offline development
    "bounds_margin_f": 15.0,
    "max_nbm_divergence_f": 6.0,
    "max_params_age_days": 3,
    "min_sigma_f": 0.5,
    "max_sigma_f": 15.0,
    "min_contracts": 3,
}


@dataclass
class PreflightResult:
    """Outcome of the pre-trade gate for one city."""

    ok: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = []
        if self.failures:
            parts.append("BLOCKED: " + "; ".join(self.failures))
        if self.warnings:
            parts.append("warnings: " + "; ".join(self.warnings))
        return " | ".join(parts) if parts else "all checks passed"


def _cfg(key: str) -> object:
    cfg = get_setting("risk.preflight", default={}) or {}
    return cfg.get(key, _DEFAULTS[key])


def run_preflight(
    *,
    city_name: str,
    effective_mean: float,
    effective_std: float,
    dist_source: str,
    nbm_max_f: float | None,
    emos_params: EMOSParams | None,
    n_contracts: int,
    market_is_live: bool,
    hist_min_f: float | None = None,
    hist_max_f: float | None = None,
) -> PreflightResult:
    """Run every circuit breaker for one city; any failure blocks trading.

    Args:
        city_name: For log messages.
        effective_mean / effective_std: The distribution about to be traded.
        dist_source: ``"EMOS"`` or ``"legacy"``.
        nbm_max_f: NBM daily-max forecast, when available.
        emos_params: Fitted params (freshness check); None on legacy path.
        n_contracts: Live contracts found for the target date.
        market_is_live: From :func:`kalshi_client.is_live_mode`.
        hist_min_f / hist_max_f: Observed min/max settlement highs from the
            station's training history (bounds check); None skips with a
            warning.
    """
    if not bool(_cfg("enabled")):
        return PreflightResult(ok=True, warnings=["preflight disabled via config"])

    failures: list[str] = []
    warnings: list[str] = []

    # 1. Live market
    if not market_is_live and not bool(_cfg("allow_stub_market")):
        failures.append("Kalshi client is in STUB mode — refusing to trade fake markets")

    # 2. Forecast bounds
    margin = float(_cfg("bounds_margin_f"))
    if hist_min_f is not None and hist_max_f is not None:
        lo, hi = hist_min_f - margin, hist_max_f + margin
        if not (lo <= effective_mean <= hi):
            failures.append(
                f"calibrated mean {effective_mean:.1f}°F outside sane bounds "
                f"[{lo:.0f}, {hi:.0f}] (history {hist_min_f:.0f}–{hist_max_f:.0f} ± {margin:.0f})"
            )
    else:
        warnings.append("no training history for bounds check")

    # 3. NBM divergence hard stop
    max_div = float(_cfg("max_nbm_divergence_f"))
    if nbm_max_f is not None:
        div = abs(effective_mean - nbm_max_f)
        if div > max_div:
            failures.append(
                f"|mu − NBM| = {div:.1f}°F exceeds hard stop {max_div:.1f}°F "
                f"(mu {effective_mean:.1f} vs NBM {nbm_max_f:.1f})"
            )
    elif dist_source == "EMOS":
        warnings.append("NBM benchmark unavailable")

    # 4. Params freshness
    max_age = int(_cfg("max_params_age_days"))
    if dist_source == "EMOS" and emos_params is not None:
        try:
            fitted = datetime.fromisoformat(emos_params.fitted_at_utc)
            if fitted.tzinfo is None:
                fitted = fitted.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - fitted).total_seconds() / 86400.0
            if age_days > max_age:
                failures.append(
                    f"EMOS params are {age_days:.1f} days old (max {max_age}) — "
                    "is the 06:15 refit failing?"
                )
        except ValueError:
            warnings.append("could not parse params fitted_at_utc")

    # 5. Sigma sanity
    min_sig, max_sig = float(_cfg("min_sigma_f")), float(_cfg("max_sigma_f"))
    if not (min_sig <= effective_std <= max_sig):
        failures.append(
            f"sigma {effective_std:.2f}°F outside sane range [{min_sig}, {max_sig}]"
        )

    # 6. Contract count
    min_contracts = int(_cfg("min_contracts"))
    if n_contracts < min_contracts:
        failures.append(f"only {n_contracts} live contracts (min {min_contracts})")

    result = PreflightResult(ok=not failures, failures=failures, warnings=warnings)
    if not result.ok:
        logger.critical("[%s] PREFLIGHT BLOCKED — %s", city_name, result.summary())
    elif warnings:
        logger.warning("[%s] preflight passed with %s", city_name, result.summary())
    else:
        logger.info("[%s] preflight passed", city_name)
    return result


def training_history_bounds(station_id: str) -> tuple[float | None, float | None]:
    """Observed (min, max) settlement highs from the station's training parquet.

    Best-effort — (None, None) when the parquet is missing or unreadable,
    which downgrades the bounds check to a warning.
    """
    try:
        import pandas as pd
        from deep_isobar.calibration.emos_training import _training_path

        path = _training_path(station_id)
        if not path.exists():
            return None, None
        actual = pd.read_parquet(path)["actual_f"].dropna()
        if actual.empty:
            return None, None
        return float(actual.min()), float(actual.max())
    except Exception:  # noqa: BLE001
        logger.exception("could not load training bounds for %s", station_id)
        return None, None
