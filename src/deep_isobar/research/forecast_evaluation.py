"""Probabilistic forecast evaluation for Deep Isobar.

Implements the proper-scoring trio from the 2026-06-11 research report (Q7):

- **CRPS** (closed-form Gaussian) — primary metric.
- **Brier score with Murphy decomposition** — reliability / resolution /
  uncertainty, computed on bracket-event probabilities.
- **PIT histogram** — the fastest read on dispersion errors:
  ∪-shape = under-dispersed, ∩-shape = over-dispersed, slope = bias.

Also ships a holdout backtest that compares the EMOS distribution against
the legacy pipeline proxy (weighted member mean + monthly bias profile +
variance multiplier + 5.5 deg F floor) on the same training parquet::

    python -m deep_isobar.research.forecast_evaluation
    python -m deep_isobar.research.forecast_evaluation --holdout-days 21
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from deep_isobar.calibration import bias_loader
from deep_isobar.calibration.emos import crps_gaussian, fit_emos
from deep_isobar.calibration.emos_training import EMOS_MODELS, _training_path
from deep_isobar.core.types import CityProfile
from deep_isobar.data.city_universe import get_city_universe

logger = logging.getLogger(__name__)

# Legacy live-session floor, reproduced here only for the comparison baseline.
_LEGACY_STD_FLOOR_F = 5.5


# ── Scoring primitives ───────────────────────────────────────────────────────


def pit_values(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Probability integral transform: F(y) under N(mu, sigma^2).

    A calibrated forecast yields PIT values ~ Uniform(0, 1).
    """
    return norm.cdf(np.asarray(y), loc=np.asarray(mu), scale=np.asarray(sigma))


def pit_histogram(pit: np.ndarray, bins: int = 10) -> np.ndarray:
    """Relative frequencies of PIT values in *bins* equal bins (sums to 1)."""
    counts, _ = np.histogram(pit, bins=bins, range=(0.0, 1.0))
    return counts / max(len(pit), 1)


def brier_decomposition(
    probs: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Murphy decomposition of the Brier score.

    BS = reliability − resolution + uncertainty.  *probs* are forecast
    probabilities of binary *outcomes* (0/1).  Returns all four numbers.
    """
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    n = len(probs)
    if n == 0:
        return {k: float("nan") for k in ("brier", "reliability", "resolution", "uncertainty")}

    base_rate = outcomes.mean()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)

    reliability = 0.0
    resolution = 0.0
    for b in range(n_bins):
        mask = idx == b
        nk = int(mask.sum())
        if nk == 0:
            continue
        pk = probs[mask].mean()
        ok = outcomes[mask].mean()
        reliability += nk * (pk - ok) ** 2
        resolution += nk * (ok - base_rate) ** 2
    reliability /= n
    resolution /= n
    uncertainty = base_rate * (1.0 - base_rate)

    return {
        "brier": float(np.mean((probs - outcomes) ** 2)),
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
    }


def modal_bucket_prob(mu: float, sigma: float) -> float:
    """P(daily high lands in the 2 deg F bracket centred on the mode).

    The report's headline symptom: this was ~0.14 with the 5.5 floor while
    the market priced ~0.45.  Brackets are [k, k+2) on integer settles, so
    use the bracket whose floor is round(mu) - 1 with the ±0.5 correction.
    """
    lo = round(mu) - 1
    return float(norm.cdf(lo + 1.5, mu, sigma) - norm.cdf(lo - 0.5, mu, sigma))


# ── Holdout comparison ───────────────────────────────────────────────────────


@dataclass
class HoldoutResult:
    station_id: str
    n_train: int
    n_test: int
    emos_crps: float
    legacy_crps: float
    emos_mae: float
    legacy_mae: float
    emos_modal_prob: float
    legacy_modal_prob: float
    emos_pit: np.ndarray
    legacy_pit: np.ndarray


def _legacy_distribution(
    city: CityProfile,
    members: np.ndarray,
    dates: list[date],
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the legacy live-session distribution from member forecasts.

    Weighted member mean (cities.yaml static weights) + monthly bias-profile
    correction, std = member spread x variance_multiplier floored at 5.5 F.
    This is a proxy — the production legacy path fed 18z snapshots rather
    than daily maxes, which made its raw inputs *colder*; scoring it on
    max-of-trace inputs flatters the legacy baseline if anything.
    """
    static = {
        "GFS": city.model_weight_gfs,
        "ECMWF": city.model_weight_ecmwf,
        "ICON": city.model_weight_icon,
        "GEM": city.model_weight_gem,
    }
    w = np.array([static.get(m) or 1.0 for m in EMOS_MODELS], dtype=float)
    w = w / w.sum()

    mean = members @ w
    spread = members.std(axis=1, ddof=1)

    mu = np.empty(len(dates))
    sigma = np.empty(len(dates))
    for i, d in enumerate(dates):
        bias_f, vm = bias_loader.get_current_bias(city.station_id, d.month)
        mu[i] = mean[i] + bias_f
        sigma[i] = max(spread[i] * vm, _LEGACY_STD_FLOOR_F)
    return mu, sigma


def evaluate_station_holdout(
    city: CityProfile,
    holdout_days: int = 15,
    training_dir: Path | None = None,
) -> HoldoutResult:
    """Time-split holdout: fit EMOS on all but the last *holdout_days* rows,
    score EMOS vs the legacy proxy on the held-out tail.
    """
    path = _training_path(city.station_id, training_dir)
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").dropna().reset_index(drop=True)

    model_names = list(EMOS_MODELS.keys())
    train, test = df.iloc[:-holdout_days], df.iloc[-holdout_days:]

    params = fit_emos(
        train[model_names].to_numpy(float),
        train["actual_f"].to_numpy(float),
        model_names,
        station_id=city.station_id,
        train_dates=(train["date"].min(), train["date"].max()),
    )

    m_test = test[model_names].to_numpy(float)
    y = test["actual_f"].to_numpy(float)
    test_dates = list(test["date"])

    a = np.asarray(params.a)
    emos_mu = params.a0 + m_test @ a
    spread_var = m_test.var(axis=1, ddof=1)
    emos_sigma = np.sqrt(np.maximum(params.c + params.d * spread_var, params.sigma_floor_f**2))

    leg_mu, leg_sigma = _legacy_distribution(city, m_test, test_dates)

    return HoldoutResult(
        station_id=city.station_id,
        n_train=len(train),
        n_test=len(test),
        emos_crps=float(np.mean(crps_gaussian(emos_mu, emos_sigma, y))),
        legacy_crps=float(np.mean(crps_gaussian(leg_mu, leg_sigma, y))),
        emos_mae=float(np.mean(np.abs(emos_mu - y))),
        legacy_mae=float(np.mean(np.abs(leg_mu - y))),
        emos_modal_prob=float(np.mean([modal_bucket_prob(m, s) for m, s in zip(emos_mu, emos_sigma)])),
        legacy_modal_prob=float(np.mean([modal_bucket_prob(m, s) for m, s in zip(leg_mu, leg_sigma)])),
        emos_pit=pit_values(emos_mu, emos_sigma, y),
        legacy_pit=pit_values(leg_mu, leg_sigma, y),
    )


def _format_pit(pit: np.ndarray, bins: int = 5) -> str:
    freq = pit_histogram(pit, bins=bins)
    return "[" + " ".join(f"{f:.2f}" for f in freq) + "]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EMOS vs legacy holdout evaluation")
    parser.add_argument("--holdout-days", type=int, default=15)
    parser.add_argument("--city", help="Single city name (default: all active)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)

    cities = [c for c in get_city_universe() if c.active]
    if args.city:
        cities = [c for c in cities if c.city.lower() == args.city.lower()]
        if not cities:
            print(f"City {args.city!r} not found", file=sys.stderr)
            return 2

    print(f"\nHoldout: last {args.holdout_days} days   (PIT bins should be flat ~0.20 each)\n")
    header = (
        f"{'station':>8} | {'CRPS emos':>9} {'legacy':>7} | "
        f"{'MAE emos':>8} {'legacy':>7} | {'modalP emos':>11} {'legacy':>7}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for city in cities:
        try:
            r = evaluate_station_holdout(city, holdout_days=args.holdout_days)
        except Exception as exc:  # noqa: BLE001
            print(f"{city.station_id:>8} | FAILED — {exc}")
            continue
        rows.append(r)
        print(
            f"{r.station_id:>8} | {r.emos_crps:9.3f} {r.legacy_crps:7.3f} | "
            f"{r.emos_mae:8.3f} {r.legacy_mae:7.3f} | "
            f"{r.emos_modal_prob:11.3f} {r.legacy_modal_prob:7.3f}"
        )

    print("\nPIT histograms (5 bins):")
    for r in rows:
        print(f"{r.station_id:>8}  emos {_format_pit(r.emos_pit)}   legacy {_format_pit(r.legacy_pit)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
