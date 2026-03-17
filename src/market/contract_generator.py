from datetime import date

from deep_isobar.core.types import CityProfile


def make_internal_contract_id(
    city_code: str,
    metric: str,
    comparison_operator: str,
    threshold_f: int,
    target_date: date,
) -> str:
    return f"{city_code}_{metric.upper()}_{comparison_operator.upper()}_{threshold_f}_{target_date.strftime('%Y%m%d')}"


def generate_contracts_for_surface(
    city_profile: CityProfile,
    metric: str,
    comparison_operator: str,
    target_date: date,
    probability_surface: dict[int, float],
) -> list[dict]:
    contracts: list[dict] = []
    for threshold_f, model_probability in probability_surface.items():
        contracts.append(
            {
                "contract_id": make_internal_contract_id(
                    city_profile.city_code,
                    metric,
                    comparison_operator,
                    threshold_f,
                    target_date,
                ),
                "city": city_profile.city,
                "threshold_f": threshold_f,
                "metric": metric,
                "comparison_operator": comparison_operator,
                "target_date": target_date,
                "model_probability": model_probability,
            }
        )
    return contracts