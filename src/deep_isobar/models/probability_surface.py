import numpy as np

from deep_isobar.models.probability_engine import probability_ge_normal, probability_le_normal


def generate_probability_surface_normal(
    mean_f: float,
    std_f: float,
    min_temp_f: int,
    max_temp_f: int,
    comparison_operator: str = "ge",
) -> dict[int, float]:
    surface: dict[int, float] = {}
    for threshold in range(min_temp_f, max_temp_f + 1):
        if comparison_operator == "ge":
            surface[threshold] = probability_ge_normal(mean_f, std_f, threshold)
        elif comparison_operator == "le":
            surface[threshold] = probability_le_normal(mean_f, std_f, threshold)
        else:
            raise ValueError("comparison_operator must be 'ge' or 'le'")
    return surface


def generate_probability_surface_kde(
    kde_distribution,
    min_temp_f: int,
    max_temp_f: int,
    comparison_operator: str = "ge",
) -> dict[int, float]:
    xs = np.linspace(min_temp_f - 20, max_temp_f + 20, 2000)
    pdf = kde_distribution(xs)

    surface: dict[int, float] = {}
    for threshold in range(min_temp_f, max_temp_f + 1):
        if comparison_operator == "ge":
            mask = xs >= threshold
        elif comparison_operator == "le":
            mask = xs <= threshold
        else:
            raise ValueError("comparison_operator must be 'ge' or 'le'")

        prob = np.trapz(pdf[mask], xs[mask]) if mask.any() else 0.0
        surface[threshold] = float(max(0.0, min(1.0, prob)))

    return surface