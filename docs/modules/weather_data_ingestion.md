# Module: Weather Data Ingestion

Purpose

Collect historical and live weather observations that match settlement sources used by prediction markets.

Primary Source

National Weather Service station observations.

Secondary Sources

ERA5 reanalysis
Weather Underground (backup validation)

Inputs

station_id
start_date
end_date

Example Stations

KORD – Chicago O'Hare
KJFK – New York JFK
KPHX – Phoenix Sky Harbor

Outputs

Structured dataframe

timestamp
station_id
temperature
humidity
wind_speed

Example

2026-07-04T18:00 | KORD | 86.3

Responsibilities

Fetch historical observations
Fetch daily summaries
Store data in feature store

Edge Cases

missing observations
station downtime
temperature units mismatch

Tasks

Implement NOAA API client
Implement historical downloader
Normalize temperature units
