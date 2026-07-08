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


# ── ensemble spread source ───────────────────────────────────────────────────


def test_fit_with_ensemble_spread_marks_source_and_climatology():
    members, actuals = _synthetic_data()
    ens_var = np.full(len(actuals), 4.0)
    params = fit_emos(
        members, actuals, MODELS, station_id="TEST", ensemble_spread_var=ens_var
    )
    assert params.spread_source == "ensemble"
    assert params.climatological_spread_var == pytest.approx(4.0, abs=1e-3)


def test_fit_without_ensemble_spread_is_members_source():
    members, actuals = _synthetic_data()
    params = fit_emos(members, actuals, MODELS, station_id="TEST")
    assert params.spread_source == "members"
    member_var = members.var(axis=1, ddof=1).mean()
    assert params.climatological_spread_var == pytest.approx(member_var, abs=1e-2)


def test_fit_ensemble_spread_nan_falls_back_to_member_variance():
    members, actuals = _synthetic_data(n=60)
    ens_var = np.full(len(actuals), 4.0)
    ens_var[::2] = np.nan  # half the days missing — must not raise
    params = fit_emos(
        members, actuals, MODELS, station_id="TEST", ensemble_spread_var=ens_var
    )
    assert params.spread_source == "ensemble"


def test_fit_ensemble_spread_shape_validation():
    members, actuals = _synthetic_data(n=40)
    with pytest.raises(ValueError, match="ensemble_spread_var"):
        fit_emos(members, actuals, MODELS, ensemble_spread_var=np.zeros(39))


def _ensemble_params() -> EMOSParams:
    return EMOSParams(
        station_id="TEST", model_names=MODELS,
        a0=-2.0, a=[0.25, 0.25, 0.25, 0.25], c=1.0, d=0.5,
        spread_source="ensemble", climatological_spread_var=3.0,
    )


def test_predict_uses_live_ensemble_spread():
    p = _ensemble_params()
    mu, sigma = emos_predict(p, {m: 80.0 for m in MODELS}, ensemble_spread_var=6.0)
    assert sigma == pytest.approx(math.sqrt(1.0 + 0.5 * 6.0))


def test_predict_falls_back_to_climatological_spread():
    p = _ensemble_params()
    # No live spread → climatological 3.0, NOT the (zero) member variance.
    mu, sigma = emos_predict(p, {m: 80.0 for m in MODELS})
    assert sigma == pytest.approx(math.sqrt(1.0 + 0.5 * 3.0))


def test_predict_members_source_ignores_ensemble_spread():
    p = _fixed_params()  # spread_source defaults to "members"
    vals = {"GFS": 80.0, "ECMWF": 82.0, "ICON": 78.0, "GEM": 80.0}
    mu, sigma = emos_predict(p, vals, ensemble_spread_var=100.0)
    spread_var = np.var([80.0, 82.0, 78.0, 80.0], ddof=1)
    assert sigma == pytest.approx(math.sqrt(1.0 + 0.5 * spread_var))


def test_params_backward_compat_pre_ensemble_json(tmp_path):
    """Params files written before the 2026-07 upgrade lack the new fields."""
    import json as _json

    old = {
        "station_id": "KOLD", "model_names": MODELS,
        "a0": -1.0, "a": [0.25, 0.25, 0.25, 0.25], "c": 1.0, "d": 0.5,
        "sigma_floor_f": 1.3, "n_train": 60, "train_start": "", "train_end": "",
        "train_crps": 1.1, "fitted_at_utc": "", "metric": "high_temp_f",
    }
    (tmp_path / "KOLD_emos.json").write_text(_json.dumps(old), encoding="utf-8")
    loaded = load_params("KOLD", params_dir=tmp_path)
    assert loaded is not None
    assert loaded.spread_source == "members"
    vals = {"GFS": 72.0, "ECMWF": 78.0, "ICON": 72.0, "GEM": 78.0}  # var = 12
    mu, sigma = emos_predict(loaded, vals)
    assert sigma == pytest.approx(math.sqrt(1.0 + 0.5 * 12.0))


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


def test_nonneg_weights_are_nonnegative():
    """Validated 2026-07-08 (holdout CRPS 1.021->1.004): constrained fits
    must never emit the nonsense negative member weights collinearity
    produced on short windows."""
    members, actuals = _synthetic_data()
    params = fit_emos(members, actuals, MODELS, station_id="TEST",
                      nonneg_weights=True)
    assert all(w >= 0 for w in params.a)
    mu, _ = emos_predict(params, {m: 80.0 for m in MODELS})
    assert mu == pytest.approx(76.0, abs=1.5)
