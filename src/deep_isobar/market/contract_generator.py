"""Contract Generator — internal contract ID and universe builder for Deep Isobar.

Maps city + metric + threshold + target date combinations into canonical
internal contract identifiers, and generates a list of contract dictionaries
from a full probability surface.

Contract ID format::

    {CITY_CODE}_{METRIC_UPPER}_{OP_UPPER}_{THRESHOLD}_{YYYYMMDD}

    Example: CHI_HIGH_TEMP_F_GE_90_20260704

Typical usage::

    from datetime import date
    from deep_isobar.market.contract_generator import (
        make_internal_contract_id,
        generate_contracts_for_surface,
    )

    cid = make_internal_contract_id("CHI", "high_temp_f", "ge", 90, date(2026, 7, 4))
    # → "CHI_HIGH_TEMP_F_GE_90_20260704"

    contracts = generate_contracts_for_surface(
        city_profile=chi_profile,
        metric="high_temp_f",
        comparison_operator="ge",
        target_date=date(2026, 7, 4),
        probability_surface={88: 0.60, 89: 0.48, 90: 0.35},
    )
"""

from __future__ import annotations

import logging
from datetime import date

from deep_isobar.core.types import CityProfile

logger = logging.getLogger(__name__)


def make_internal_contract_id(
    city_code: str,
    metric: str,
    comparison_operator: str,
    threshold_f: int,
    target_date: date,
) -> str:
    """Build the canonical internal contract identifier.

    Args:
        city_code: Short city code (e.g. ``"CHI"``).
        metric: Metric key, e.g. ``"high_temp_f"`` or ``"low_temp_f"``.
        comparison_operator: ``"ge"`` or ``"le"``.
        threshold_f: Integer threshold temperature in °F.
        target_date: Event target date.

    Returns:
        Canonical contract ID string, e.g.
        ``"CHI_HIGH_TEMP_F_GE_90_20260704"``.

    Raises:
        ValueError: If *comparison_operator* is not ``"ge"`` or ``"le"``.
    """
    op = comparison_operator.lower()
    if op not in {"ge", "le"}:
        raise ValueError(
            f"comparison_operator must be 'ge' or 'le', got '{comparison_operator}'"
        )

    contract_id = (
        f"{city_code.upper()}"
        f"_{metric.upper()}"
        f"_{op.upper()}"
        f"_{threshold_f}"
        f"_{target_date.strftime('%Y%m%d')}"
    )
    logger.debug("make_internal_contract_id → %s", contract_id)
    return contract_id


def generate_contracts_for_surface(
    city_profile: CityProfile,
    metric: str,
    comparison_operator: str,
    target_date: date,
    probability_surface: dict[int, float],
) -> list[dict]:
    """Generate a contract record for each threshold in the probability surface.

    Args:
        city_profile: Profile for the city being traded.
        metric: ``"high_temp_f"`` or ``"low_temp_f"``.
        comparison_operator: ``"ge"`` or ``"le"``.
        target_date: Event target date.
        probability_surface: Mapping of ``{threshold_f: model_probability}``.

    Returns:
        List of dicts, each containing at minimum:
        ``contract_id``, ``city``, ``threshold_f``, ``metric``,
        ``comparison_operator``, ``target_date``, ``model_probability``.

    Raises:
        ValueError: If *probability_surface* is empty or
            *comparison_operator* is invalid.
    """
    if not probability_surface:
        raise ValueError("probability_surface must not be empty")

    contracts: list[dict] = []
    for threshold_f, model_probability in probability_surface.items():
        contract_id = make_internal_contract_id(
            city_profile.city_code,
            metric,
            comparison_operator,
            threshold_f,
            target_date,
        )
        contracts.append(
            {
                "contract_id": contract_id,
                "city": city_profile.city,
                "threshold_f": threshold_f,
                "metric": metric,
                "comparison_operator": comparison_operator,
                "target_date": target_date,
                "model_probability": model_probability,
            }
        )

    logger.info(
        "generate_contracts_for_surface  city=%s metric=%s op=%s target=%s → %d contracts",
        city_profile.city, metric, comparison_operator, target_date, len(contracts),
    )
    return contracts
