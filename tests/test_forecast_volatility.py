import pytest
import numpy as np
from deep_isobar.models.forecast_volatility import (
    compute_forecast_std,
    compute_forecast_variance,
    compute_forecast_volatility_score,
)

def test_compute_forecast_std_basic():
    values = [70.0, 72.0, 71.0, 69.0, 73.0]
    expected = float(np.std(values, ddof=1))
    result = compute_forecast_std(values)
    assert np.isclose(result, expected)

def test_compute_forecast_std_single_value():
    assert compute_forecast_std([75.0]) == 0.0

def test_compute_forecast_std_empty():
    with pytest.raises(ValueError, match="cannot be empty"):
        compute_forecast_std([])

def test_compute_forecast_variance_basic():
    values = [70.0, 72.0, 71.0, 69.0, 73.0]
    expected = float(np.var(values, ddof=1))
    result = compute_forecast_variance(values)
    assert np.isclose(result, expected)

def test_compute_forecast_variance_single_value():
    assert compute_forecast_variance([75.0]) == 0.0

def test_compute_forecast_variance_empty():
    with pytest.raises(ValueError, match="cannot be empty"):
        compute_forecast_variance([])

def test_compute_forecast_volatility_score_basic():
    values = [70.0, 72.0, 71.0, 69.0, 73.0]
    # Score should match std
    expected = compute_forecast_std(values)
    result = compute_forecast_volatility_score(values)
    assert np.isclose(result, expected)

def test_compute_forecast_volatility_score_empty():
    with pytest.raises(ValueError, match="cannot be empty"):
        compute_forecast_volatility_score([])
