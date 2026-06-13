"""EMOS / Non-homogeneous Gaussian Regression calibration for Deep Isobar.

Implements Ensemble Model Output Statistics (Gneiting et al. 2005) fit by
minimum-CRPS estimation, the field-standard postprocessor for turning a
small multi-model ensemble into a calibrated Gaussian predictive
distribution for station daily-max temperature.

Model::

    mu      = a0 + a1*m1 + ... + aK*mK          (member forecasts m1..mK)
    sigma^2 = c + d * S^2                       (S^2 = ensemble variance)

Parameters ``(a0, a, c, d)`` are fit by minimising the mean closed-form
Gaussian CRPS over a rolling training window.  ``c`` and ``d`` are kept
non-negative by optimising their square roots.

The predictive sigma is floored at ``SIGMA_FLOOR_F`` (default 1.3 deg F),
the irreducible observation + integer-settlement error — NOT the legacy
5.5 deg F floor, which forced chronic over-dispersion (see research report
2026-06-11, Q1).

Public interface::

    crps_gaussian(mu, sigma, y)                  -> float | ndarray
    fit_emos(members, actuals, model_names)      -> EMOSParams
    emos_predict(params, member_values)          -> (mu_f, sigma_f)
    save_params(params, station_id)              -> Path
    load_params(station_id)                      -> EMOSParams | None
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PARAMS_DIR = _PROJECT_ROOT / "data" / "emos_params"

# Irreducible error floor: ASOS sensor (~0.5 F) + integer-settlement
# discretization (1/sqrt(12) ~ 0.29 F) + residual representation error.
SIGMA_FLOOR_F: float = 1.3

# Minimum paired (forecast, actual) days required to fit.
MIN_TRAIN_DAYS: int = 25


@dataclass
class EMOSParams:
    """Fitted EMOS parameters for one station / metric."""

    station_id: str
    model_names: list[str]          # member order for the `a` coefficients
    a0: float                       # intercept
    a: list[float]                  # per-member mean coefficients
    c: float                        # variance intercept (deg F^2)
    d: float                        # variance spread coefficient
    sigma_floor_f: float = SIGMA_FLOOR_F
    n_train: int = 0
    train_start: str = ""           # ISO dates of the training window
    train_end: str = ""
    train_crps: float = float("nan")
    fitted_at_utc: str = ""
    metric: str = "high_temp_f"


def crps_gaussian(
    mu: float | np.ndarray,
    sigma: float | np.ndarray,
    y: float | np.ndarray,
) -> float | np.ndarray:
    """Closed-form CRPS of a Gaussian N(mu, sigma^2) against observation *y*.

    CRPS = sigma * ( z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ),  z = (y-mu)/sigma

    Lower is better; units are the same as *y* (deg F).
    """
    sigma = np.asarray(sigma, dtype=float)
    if np.any(sigma <= 0):
        raise ValueError("sigma must be positive")
    z = (np.asarray(y, dtype=float) - np.asarray(mu, dtype=float)) / sigma
    out = sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / math.sqrt(math.pi))
    return float(out) if out.ndim == 0 else out


def _emos_mu_sigma(
    theta: np.ndarray,
    members: np.ndarray,
    spread_var: np.ndarray,
    sigma_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (mu, sigma) vectors from packed parameters.

    ``theta`` = [a0, a1..aK, sqrt_c, sqrt_d]; c and d enter squared so the
    optimiser is unconstrained while sigma^2 stays non-negative.
    """
    k = members.shape[1]
    a0 = theta[0]
    a = theta[1 : 1 + k]
    c = theta[1 + k] ** 2
    d = theta[2 + k] ** 2
    mu = a0 + members @ a
    sigma = np.sqrt(np.maximum(c + d * spread_var, sigma_floor**2))
    return mu, sigma


def fit_emos(
    members: np.ndarray,
    actuals: np.ndarray,
    model_names: list[str],
    station_id: str = "",
    sigma_floor_f: float = SIGMA_FLOOR_F,
    train_dates: tuple[date, date] | None = None,
) -> EMOSParams:
    """Fit EMOS parameters by minimum-CRPS over a training window.

    Args:
        members: ``(n_days, n_models)`` member daily-max forecasts (deg F).
            Rows with any NaN are dropped.
        actuals: ``(n_days,)`` observed settlement daily highs (deg F).
        model_names: Names matching the member column order.
        station_id: Station the fit belongs to (metadata only).
        sigma_floor_f: Hard floor for predictive sigma.
        train_dates: Optional ``(start, end)`` of the window for metadata.

    Returns:
        Fitted :class:`EMOSParams`.

    Raises:
        ValueError: On shape mismatch or fewer than ``MIN_TRAIN_DAYS``
            complete rows.
    """
    members = np.asarray(members, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    if members.ndim != 2 or members.shape[1] != len(model_names):
        raise ValueError(
            f"members must be (n_days, {len(model_names)}), got {members.shape}"
        )
    if members.shape[0] != actuals.shape[0]:
        raise ValueError("members and actuals must have the same number of rows")

    ok = ~(np.isnan(members).any(axis=1) | np.isnan(actuals))
    members, actuals = members[ok], actuals[ok]
    n = members.shape[0]
    if n < MIN_TRAIN_DAYS:
        raise ValueError(
            f"Need at least {MIN_TRAIN_DAYS} complete training days, got {n}"
        )

    k = members.shape[1]
    spread_var = members.var(axis=1, ddof=1) if k > 1 else np.zeros(n)

    # ── Initial guess: OLS for the mean, residual variance for c ─────────
    X = np.column_stack([np.ones(n), members])
    coef, *_ = np.linalg.lstsq(X, actuals, rcond=None)
    resid = actuals - X @ coef
    resid_var = float(resid.var(ddof=1))
    theta0 = np.concatenate([
        coef,
        [math.sqrt(max(resid_var * 0.8, sigma_floor_f**2)), math.sqrt(0.5)],
    ])

    def objective(theta: np.ndarray) -> float:
        mu, sigma = _emos_mu_sigma(theta, members, spread_var, sigma_floor_f)
        return float(np.mean(crps_gaussian(mu, sigma, actuals)))

    result = minimize(objective, theta0, method="Nelder-Mead",
                      options={"maxiter": 20000, "xatol": 1e-5, "fatol": 1e-7})
    if not result.success:
        logger.warning(
            "EMOS fit for %s did not fully converge (%s) — using best iterate",
            station_id, result.message,
        )

    theta = result.x
    train_crps = objective(theta)
    a0 = float(theta[0])
    a = [float(v) for v in theta[1 : 1 + k]]
    c = float(theta[1 + k] ** 2)
    d = float(theta[2 + k] ** 2)

    start, end = ("", "")
    if train_dates is not None:
        start, end = train_dates[0].isoformat(), train_dates[1].isoformat()

    params = EMOSParams(
        station_id=station_id,
        model_names=list(model_names),
        a0=a0, a=a, c=c, d=d,
        sigma_floor_f=sigma_floor_f,
        n_train=n,
        train_start=start,
        train_end=end,
        train_crps=round(train_crps, 4),
        fitted_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    logger.info(
        "EMOS fit %s: n=%d  a0=%.2f  a=%s  c=%.3f  d=%.3f  train CRPS=%.3f degF",
        station_id, n, a0, [round(v, 3) for v in a], c, d, train_crps,
    )
    return params


def emos_predict(
    params: EMOSParams,
    member_values: dict[str, float],
) -> tuple[float, float]:
    """Compute the calibrated ``(mu, sigma)`` for one forecast day.

    Missing members are substituted with the mean of the available members
    (logged at WARNING) so a single model outage does not kill the session.
    At least half the members must be present.

    Args:
        params: Fitted EMOS parameters.
        member_values: ``{model_name: daily_max_forecast_f}``; keys matching
            ``params.model_names``.

    Returns:
        ``(mu_f, sigma_f)`` of the calibrated Gaussian predictive
        distribution, with sigma floored at ``params.sigma_floor_f``.

    Raises:
        ValueError: If fewer than half the members are available.
    """
    available = [
        member_values[m] for m in params.model_names
        if m in member_values and member_values[m] is not None
    ]
    if len(available) < max(2, len(params.model_names) // 2):
        raise ValueError(
            f"EMOS predict for {params.station_id}: only {len(available)}/"
            f"{len(params.model_names)} members available"
        )
    fill = float(np.mean(available))

    values = []
    for m in params.model_names:
        v = member_values.get(m)
        if v is None:
            logger.warning(
                "EMOS predict %s: member %s missing — substituting ensemble mean %.1f",
                params.station_id, m, fill,
            )
            v = fill
        values.append(float(v))

    members = np.asarray(values)
    mu = params.a0 + float(np.dot(members, params.a))
    spread_var = float(members.var(ddof=1)) if len(members) > 1 else 0.0
    sigma = math.sqrt(max(params.c + params.d * spread_var, params.sigma_floor_f**2))
    return mu, sigma


# ── Persistence ──────────────────────────────────────────────────────────────


def _params_path(station_id: str, params_dir: Path | None = None) -> Path:
    d = params_dir or _DEFAULT_PARAMS_DIR
    return d / f"{station_id}_emos.json"


def save_params(params: EMOSParams, params_dir: Path | None = None) -> Path:
    """Write *params* to ``data/emos_params/{station_id}_emos.json``."""
    path = _params_path(params.station_id, params_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(params), indent=2), encoding="utf-8")
    tmp.replace(path)
    logger.info("Saved EMOS params for %s to %s", params.station_id, path)
    return path


def load_params(
    station_id: str,
    params_dir: Path | None = None,
) -> EMOSParams | None:
    """Load fitted params for *station_id*, or ``None`` if absent/corrupt.

    Never raises — a missing or unreadable file means the caller should
    fall back to the legacy bias-profile path.
    """
    path = _params_path(station_id, params_dir)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return EMOSParams(**raw)
    except Exception:
        logger.exception("Failed to load EMOS params from %s", path)
        return None
