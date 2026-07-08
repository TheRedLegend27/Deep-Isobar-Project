"""Holdout race for EMOS fit variants — nothing ships unless it wins here.

Variants under test (2026-07-08):

- ``base``      — current production fit (unconstrained weights, no trend)
- ``nonneg``    — member weights constrained >= 0 (Gneiting's original EMOS;
                  kills the nonsense negative NBM weights on short windows)
- ``ridge``     — weights shrunk toward equal (collinearity control)
- ``trend``     — sigma^2 += e * (run-to-run forecast change)^2 — widens the
                  distribution on regime-change days (the 2026-07-05 bust)
- ``all``       — nonneg + ridge + trend

Protocol: last ``--holdout`` days held out, fit on the rest, score holdout
CRPS with the exact predict-time math.  Report per station and mean.

CLI::

    python -m deep_isobar.research.validate_emos_variants --cities Chicago "New York" Dallas
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from deep_isobar.calibration.emos import crps_gaussian, fit_emos
from deep_isobar.calibration.emos_training import EMOS_MODELS, _training_path
from deep_isobar.data.city_universe import get_city_universe

logger = logging.getLogger(__name__)

VARIANTS: dict[str, dict] = {
    "base":   {},
    "nonneg": {"nonneg_weights": True},
    "ridge":  {"ridge_lambda": 0.1},
    "trend":  {"use_trend": True},
    "all":    {"nonneg_weights": True, "ridge_lambda": 0.1, "use_trend": True},
}


def _holdout_crps(df: pd.DataFrame, holdout: int, variant: dict) -> float | None:
    model_names = [m for m in EMOS_MODELS if m in df.columns]
    df = df.dropna(subset=model_names + ["actual_f"]).sort_values("date")
    if len(df) < holdout + 30:
        return None
    train, test = df.iloc[:-holdout], df.iloc[-holdout:]

    use_trend = variant.get("use_trend", False) and "trend_f" in df.columns
    params = fit_emos(
        train[model_names].to_numpy(float),
        train["actual_f"].to_numpy(float),
        model_names,
        station_id="VAL",
        trend=train["trend_f"].to_numpy(float) if use_trend else None,
        nonneg_weights=variant.get("nonneg_weights", False),
        ridge_lambda=variant.get("ridge_lambda", 0.0),
    )

    m = test[model_names].to_numpy(float)
    y = test["actual_f"].to_numpy(float)
    mu = params.a0 + m @ np.asarray(params.a)
    var = params.c + params.d * m.var(axis=1, ddof=1)
    if use_trend and params.e_trend > 0:
        t = np.nan_to_num(test["trend_f"].to_numpy(float), nan=0.0)
        var = var + params.e_trend * t**2
    sigma = np.sqrt(np.maximum(var, params.sigma_floor_f**2))
    return float(np.mean(crps_gaussian(mu, sigma, y)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities", nargs="*", default=None)
    parser.add_argument("--holdout", type=int, default=15)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)

    cities = [c for c in get_city_universe() if c.active]
    if args.cities:
        wanted = {w.lower() for w in args.cities}
        cities = [c for c in cities if c.city.lower() in wanted]

    results: dict[str, dict[str, float]] = {}
    for city in cities:
        path = _training_path(city.station_id)
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        row = {}
        for name, opts in VARIANTS.items():
            try:
                crps = _holdout_crps(df.copy(), args.holdout, opts)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s/%s] failed: %s", city.city, name, exc)
                crps = None
            if crps is not None:
                row[name] = crps
        if row:
            results[city.city] = row

    names = list(VARIANTS)
    print(f"\nHoldout CRPS (°F, last {args.holdout} days) — lower is better\n")
    print(f"{'station':<14}" + "".join(f"{n:>9}" for n in names))
    for city, row in results.items():
        base = row.get("base")
        cells = []
        for n in names:
            v = row.get(n)
            mark = "*" if v is not None and base is not None and v == min(
                x for x in row.values() if x is not None
            ) else " "
            cells.append(f"{v:>8.3f}{mark}" if v is not None else f"{'—':>9}")
        print(f"{city:<14}" + "".join(cells))
    means = {n: np.mean([r[n] for r in results.values() if n in r]) for n in names}
    print(f"{'MEAN':<14}" + "".join(f"{means[n]:>8.3f} " for n in names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
