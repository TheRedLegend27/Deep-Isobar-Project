from deep_isobar.core.types import ForecastPoint, ForecastShiftEvent


def compute_forecast_shift(
    previous: ForecastPoint,
    current: ForecastPoint,
    significance_threshold_f: float = 2.0,
) -> ForecastShiftEvent:
    if (
        previous.city != current.city
        or previous.model_name != current.model_name
        or previous.target_date != current.target_date
        or previous.metric != current.metric
    ):
        raise ValueError("Forecast points must refer to the same city/model/date/metric")

    shift = current.forecast_value_f - previous.forecast_value_f
    abs_shift = abs(shift)

    if shift > 0:
        direction = "up"
    elif shift < 0:
        direction = "down"
    else:
        direction = "flat"

    return ForecastShiftEvent(
        city=current.city,
        model_name=current.model_name,
        metric=current.metric,
        previous_run_time_utc=previous.run_time_utc,
        current_run_time_utc=current.run_time_utc,
        target_date=current.target_date,
        previous_value_f=previous.forecast_value_f,
        current_value_f=current.forecast_value_f,
        shift_f=shift,
        absolute_shift_f=abs_shift,
        shift_direction=direction,
        significant_shift_flag=abs_shift >= significance_threshold_f,
    )