import numpy as np


def compute_ensemble(forecasts):

    mean = np.mean(forecasts)
    std = np.std(forecasts)

    return {
        "mean": mean,
        "std": std,
        "count": len(forecasts)
    }