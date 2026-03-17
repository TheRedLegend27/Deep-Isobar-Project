Project: Deep Isobar

Relevant docs:
- ARCHITECTURE.md
- INTERFACES.md
- MODULE_DEPENDENCIES.md

Follow these rules:

Iterative Development:
Build only the requested module.

Prompt Engineering:
Follow the interfaces exactly as defined in INTERFACES.md.

Error Handling:
Raise clear errors for invalid inputs.

Testing:
Include pytest tests.

Now build:

Build the module `src/deep_isobar/data/weather_ingest.py` for Deep Isobar.

Context:
- This is the first version of weather ingestion.
- For now, build it as a clean MVP with deterministic stub-friendly behavior.
- It must expose:
  - `fetch_station_observations(station_id: str, start_date: date, end_date: date, source_name: str = "NWS") -> pd.DataFrame`
  - `normalize_weather_observations(df: pd.DataFrame) -> pd.DataFrame`
  - `build_daily_settlement_observations(df: pd.DataFrame, city: str, station_id: str, settlement_source: str = "NWS") -> pd.DataFrame`

Requirements:
- Use pandas
- Output canonical columns:
  - `timestamp_utc`
  - `station_id`
  - `source_name`
  - `temperature_f`
- `build_daily_settlement_observations` must output:
  - `city`
  - `station_id`
  - `target_date`
  - `high_temp_f`
  - `low_temp_f`
  - `settlement_source`
- Validate required columns
- Raise `ValueError` on bad input
- Keep network calls optional for now; a stubbed deterministic implementation is fine if clearly marked
- Add logging, docstrings, and type hints

Output requirements:
1. Explanation
2. Python implementation
3. Example usage
4. Unit tests with pytest
5. Edge cases handled

Stop after this module.