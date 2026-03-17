from datetime import date, datetime

from deep_isobar.core.types import ForecastPoint
from deep_isobar.data.city_universe import get_city_profile


VALID_METRICS = {"high_temp_f", "low_temp_f"}


def fetch_model_forecast(
    city: str,
    station_id: str,
    model_name: str,
    run_time_utc: datetime,
    target_date: date,
    metric: str,
) -> ForecastPoint:
    if metric not in VALID_METRICS:
        raise ValueError(f"Invalid metric: {metric}")

    return ForecastPoint(
        city=city,
        station_id=station_id,
        model_name=model_name,
        run_time_utc=run_time_utc,
        target_date=target_date,
        metric=metric,
        forecast_value_f=85.0,
        lead_hours=24,
        source_name=model_name,
    )


def fetch_forecasts_for_city(
    city: str,
    target_date: date,
    metric: str,
    model_names: list[str],
) -> list[ForecastPoint]:
    profile = get_city_profile(city)
    run_time = datetime.utcnow()
    return [
        fetch_model_forecast(
            city=city,
            station_id=profile.station_id,
            model_name=model_name,
            run_time_utc=run_time,
            target_date=target_date,
            metric=metric,
        )
        for model_name in model_names
    ]


def register_forecast_run(
    model_name: str,
    run_time_utc: datetime,
    cycle_label: str,
    source_name: str,
) -> dict:
    return {
        "model_name": model_name,
        "run_time_utc": run_time_utc,
        "cycle_label": cycle_label,
        "source_name": source_name,
        "ingest_status": "complete",
    }