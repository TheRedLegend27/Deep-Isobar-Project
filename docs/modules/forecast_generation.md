# Module: Forecast Generation

Purpose

Collect forecast outputs from major weather models to create ensemble temperature predictions.

Forecast Models

Global Forecast System (GFS)
ECMWF Integrated Forecasting System
North American Mesoscale Model (NAM)

Inputs

city
forecast_run_time

Outputs

forecast temperature list

Example

GFS: 84
ECMWF: 87
NAM: 83

Responsibilities

collect model forecasts
normalize units
store forecast history

Edge Cases

missing model run
delayed model output
outlier forecasts

Tasks

Create forecast collector
store model outputs in feature store
