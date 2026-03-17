import logging
import numpy as np

logger = logging.getLogger(__name__)

def compute_forecast_std(forecast_values_f: list[float]) -> float:
    """
    Computes the sample standard deviation of a list of forecast temperatures.

    Args:
        forecast_values_f (list[float]): A list of forecast temperatures in Fahrenheit.

    Returns:
        float: The sample standard deviation of the forecast values.

    Raises:
        ValueError: If the input list is empty.
    """
    if not forecast_values_f:
        logger.error("Empty list provided to compute_forecast_std")
        raise ValueError("forecast_values_f cannot be empty")
        
    if len(forecast_values_f) == 1:
        logger.debug("Only one forecast value provided; standard deviation is 0.0")
        return 0.0
        
    std_val = float(np.std(forecast_values_f, ddof=1))
    logger.debug(f"Computed forecast standard deviation: {std_val:.4f}")
    return std_val

def compute_forecast_variance(forecast_values_f: list[float]) -> float:
    """
    Computes the sample variance of a list of forecast temperatures.

    Args:
        forecast_values_f (list[float]): A list of forecast temperatures in Fahrenheit.

    Returns:
        float: The sample variance of the forecast values.

    Raises:
        ValueError: If the input list is empty.
    """
    if not forecast_values_f:
        logger.error("Empty list provided to compute_forecast_variance")
        raise ValueError("forecast_values_f cannot be empty")
        
    if len(forecast_values_f) == 1:
        logger.debug("Only one forecast value provided; variance is 0.0")
        return 0.0
        
    var_val = float(np.var(forecast_values_f, ddof=1))
    logger.debug(f"Computed forecast variance: {var_val:.4f}")
    return var_val

def compute_forecast_volatility_score(forecast_values_f: list[float]) -> float:
    """
    Computes a volatility score for the forecast values.
    For the MVP, this is equivalent to the standard deviation.

    Args:
        forecast_values_f (list[float]): A list of forecast temperatures in Fahrenheit.

    Returns:
        float: The computed volatility score.

    Raises:
        ValueError: If the input list is empty.
    """
    if not forecast_values_f:
        logger.error("Empty list provided to compute_forecast_volatility_score")
        raise ValueError("forecast_values_f cannot be empty")
        
    score = compute_forecast_std(forecast_values_f)
    logger.debug(f"Computed forecast volatility score: {score:.4f}")
    return score