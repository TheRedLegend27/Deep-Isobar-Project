import numpy as np


def compute_forecast_std(forecast_values_f: list[float]) -> float:
    if not forecast_values_f:
        raise ValueError("forecast_values_f cannot be empty")
    return float(np.std(forecast_values_f))


def compute_forecast_variance(forecast_values_f: list[float]) -> float:
    if not forecast_values_f:
        raise ValueError("forecast_values_f cannot be empty")
    return float(np.var(forecast_values_f))


def compute_forecast_volatility_score(forecast_values_f: list[float]) -> float:
    return compute_forecast_std(forecast_values_f)