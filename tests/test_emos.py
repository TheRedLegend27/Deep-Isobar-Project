"""Tests for deep_isobar.calibration.emos."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from deep_isobar.calibration.emos import (
    MIN_TRAIN_DAYS,
    SIGMA_FLOOR_F,
    EMOSParams,
    crps_gaussian,
    emos_predict,
    fit_emos,
    load_params,
    save_params,
)

MODELS = ["GFS", "ECMWF", "ICON", "GEM"]


# ── crps_gaussian ────────────────────────────────────────────────────────────


def test_crps_matches_numerical_integration():
    """Closed form must agree with brute-force integral of (F(x)-1{x>=y})^2."""
    mu, sigma, y = 71.3, 2.1, 74.0
    xs = np.linspace(mu - 12 * sigma, mu + 12 * sigma, 200_001)
    F = norm.cdf(xs, mu, sigma)
    H = (xs >= y).astype(float)
    numerical = np.trapezoid((F - H) ** 2, xs)
    assert crps_gaussian(mu, sigma, y) == pytest.approx(numerical, abs=1e-4)


def test_crps_perfect_forecast_is_small():
    # CRPS at y == mu equals sigma * (2/sqrt(2pi) - 1/sqrt(pi)) ≈ 0.2337 * sigma
    assert crps_gaussian(70.0, 1.0, 70.0) == pytest.approx(
        2 / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi), abs=1e-9
    )


def test_crps_penalises_distance_and_rewards_sharpness():
    assert crps_gaussian(70, 2, 75) > crps_gaussian(70, 2, 71)
    # When the forecast is right, sharper is better.
    assert crps_gaussian(70, 1, 70) < crps_gaussian(70, 4, 70)


def test_crps_vectorised():
    out = crps_gaussian(np.array([70.0, 71.0]), np.array([2.0, 2.0]), np.array([70.0, 75.0]))
    assert out.shape == (2,)
    assert out[1] > out[0]


def test_crps_rejects_nonpositive_sigma():
    with pytest.raises(ValueError):
        crps_gaussian(70.0, 0.0, 70.0)


# ── fit_emos ─────────────────────────────────────────────────────────────────


def _synthetic_data(n=120, bias=4.0, noise=1.8, seed=7):
    """Members = truth + bias + member noise; actual = truth + obs noise."""
    rng = np.random.default_rng(seed)
    truth = 70 + 15 * np.sin(np.linspace(0, 6, n)) + rng.normal(0, 3, n)
    members = truth[:, None] + bias + rng.normal(0, noise, (n, 4))
    actuals = truth + rng.normal(0, 0.5, n)
    return members, actuals


def test_fit_recovers_bias_and_sharp_sigma():
    members, actuals = _synthetic_data()
    params = fit_emos(members, actuals, MODELS, station_id="TEST")

    # The fitted mean must remove the +4F bias: predict on a fresh day.
    mu, sigma = emos_predict(
        params, {m: 80.0 for m in MODELS}  # all members say 80 → truth ~76
    )
    assert mu == pytest.approx(76.0, abs=1.0)
    # Calibrated sigma should be sharp (~noise level), nowhere near 5.5.
    assert SIGMA_FLOOR_F <= sigma < 3.0


def test_fit_drops_nan_rows():
    members, actuals = _synthetic_data(n=60)
    members[5, 2] = np.nan
    actuals[10] = np.nan
    params = fit_emos(members, actuals, MODELS, station_id="TEST")
    assert params.n_train == 58


def test_fit_requires_min_days():
    members, actuals = _synthetic_data(n=MIN_TRAIN_DAYS - 1)
    with pytest.raises(ValueError, match="training days"):
        fit_emos(members, actuals, MODELS)


def test_fit_shape_validation():
    members, actuals = _synthetic_data(n=40)
    with pytest.raises(ValueError):
        fit_emos(members[:, :3], actuals, MODELS)
    with pytest.raises(ValueError):
        fit_emos(members, actuals[:-1], MODELS)


def test_fit_beats_naive_ensemble_crps():
    """EMOS train CRPS must beat the biased raw ensemble with a 5.5F floor."""
    members, actuals = _synthetic_data()
    params = fit_emos(members, actuals, MODELS, station_id="TEST")
    raw_mu = members.mean(axis=1)
    naive = float(np.mean(crps_gaussian(raw_mu, 5.5, actuals)))
    assert params.train_crps < naive


# ── emos_predict ─────────────────────────────────────────────────────────────


def _fixed_params() -> EMOSParams:
    return EMOSParams(
        station_id="TEST", model_names=MODELS,
        a0=-2.0, a=[0.25, 0.25, 0.25, 0.25], c=1.0, d=0.5,
    )


def test_predict_known_values():
    p = _fixed_params()
    vals = {"GFS": 80.0, "ECMWF": 82.0, "ICON": 78.0, "GEM": 80.0}
    mu, sigma = emos_predict(p, vals)
    assert mu == pytest.approx(-2.0 + 0.25 * (80 + 82 + 78 + 80))
    spread_var = np.var([80.0, 82.0, 78.0, 80.0], ddof=1)
    assert sigma == pytest.approx(math.sqrt(1.0 + 0.5 * spread_var))


def test_predict_applies_sigma_floor():
    p = _fixed_params()
    p.c, p.d = 0.0, 0.0
    mu, sigma = emos_predict(p, {m: 75.0 for m in MODELS})
    assert sigma == pytest.approx(SIGMA_FLOOR_F)


def test_predict_substitutes_missing_member():
    p = _fixed_params()
    mu, sigma = emos_predict(p, {"GFS": 80.0, "ECMWF": 80.0, "ICON": 80.0})
    assert mu == pytest.approx(-2.0 + 80.0)


def test_predict_rejects_too_few_members():
    p = _fixed_params()
    with pytest.raises(ValueError, match="members available"):
        emos_predict(p, {"GFS": 80.0})


# ── persistence ──────────────────────────────────────────────────────────────


def test_save_load_roundtrip(tmp_path):
    members, actuals = _synthetic_data(n=40)
    params = fit_emos(members, actuals, MODELS, station_id="KTST")
    save_params(params, params_dir=tmp_path)
    loaded = load_params("KTST", params_dir=tmp_path)
    assert loaded == params


def test_load_missing_returns_none(tmp_path):
    assert load_params("KNOPE", params_dir=tmp_path) is None


def test_load_corrupt_returns_none(tmp_path):
    (tmp_path / "KBAD_emos.json").write_text("{not json", encoding="utf-8")
    assert load_params("KBAD", params_dir=tmp_path) is None
