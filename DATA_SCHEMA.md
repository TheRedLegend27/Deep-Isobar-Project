# Deep Isobar Data Schema

## Purpose

This document defines the core datasets, tables, parquet schemas, and data contracts used by Deep Isobar.

It exists to ensure:

- consistent module interfaces
- reproducible backtests
- clean feature pipelines
- reliable AI-agent development

Every module in Deep Isobar should read from or write to schemas defined here.

---

# Schema Design Principles

## 1. Canonical Keys

The system should use these canonical identifiers wherever possible:

- `city`
- `station_id`
- `contract_id`
- `market_source`
- `model_name`
- `run_time_utc`
- `target_date`
- `timestamp_utc`

## 2. Time Conventions

All internal timestamps should be stored in:

- UTC
- ISO 8601 where serialized

Examples:

- `2026-03-16T12:00:00Z`
- `2026-07-04`

Rules:

- `timestamp_utc` = time record was observed or ingested
- `run_time_utc` = model initialization time
- `target_date` = local event date being forecast or settled

## 3. Temperature Units

Internal canonical unit:

- Fahrenheit for market-facing probability logic
- Celsius may be preserved in raw ingestion tables if source-native

Rules:

- raw tables may keep source units
- normalized and feature tables should use Fahrenheit unless otherwise noted

## 4. Storage Strategy

Recommended storage:

- raw and feature datasets: Parquet
- metadata and registries: PostgreSQL or SQLite initially
- configs: YAML or JSON

Suggested repo/data layout:

data/
    raw/
    normalized/
    features/
    forecasts/
    markets/
    backtests/
    logs/

---

# Core Entity Relationships

Deep Isobar revolves around these entities:

1. City
2. Station
3. Forecast Run
4. Forecast Value
5. Probability Surface
6. Market Contract
7. Order Book Snapshot
8. Alpha Opportunity
9. Trade
10. Backtest Result

Relationship flow:

City
→ Station
→ Forecast Run
→ Forecast Values
→ Temperature Distribution
→ Probability Surface
→ Market Contract Mapping
→ Alpha Opportunity
→ Trade / Backtest Result

---

# 1. City Registry

## Dataset Name

`city_registry`

## Purpose

Defines the tradable city universe and core metadata.

## Primary Key

`city`

## Columns

| column | type | required | description |
|---|---|---:|---|
| city | string | yes | canonical city name |
| city_code | string | yes | short system code, e.g. CHI, NYC |
| state | string | no | state or region |
| country | string | yes | country code or name |
| timezone | string | yes | IANA timezone, e.g. America/Chicago |
| station_id | string | yes | canonical settlement station |
| settlement_source | string | yes | source used for settlement, e.g. NWS |
| active | boolean | yes | whether city is active for trading |
| notes | string | no | freeform notes |

## Example

| city | city_code | timezone | station_id | settlement_source | active |
|---|---|---|---|---|---|
| Chicago | CHI | America/Chicago | KORD | NWS | true |

---

# 2. City Profile

## Dataset Name

`city_profile`

## Purpose

Stores city-specific modeling parameters.

## Primary Key

`city`

## Columns

| column | type | required | description |
|---|---|---:|---|
| city | string | yes | canonical city name |
| station_id | string | yes | linked settlement station |
| variance_multiplier | float | yes | adjusts ensemble spread |
| mean_bias_correction_f | float | yes | additive city bias correction in °F |
| kde_bandwidth | float | yes | KDE smoothing parameter |
| tail_multiplier | float | yes | tail probability scaling, if used |
| model_weight_gfs | float | no | city-specific weight |
| model_weight_ecmwf | float | no | city-specific weight |
| model_weight_nam | float | no | city-specific weight |
| heat_bias_adjustment_f | float | no | high-temp correction |
| cold_bias_adjustment_f | float | no | low-temp correction |
| updated_at_utc | timestamp | yes | last update time |

## Notes

- weights should sum to 1.0 when used
- null weights mean default weighting logic applies

---

# 3. Data Source Registry

## Dataset Name

`data_source_registry`

## Purpose

Defines authoritative sources used across the system.

## Primary Key

`source_name`

## Columns

| column | type | required | description |
|---|---|---:|---|
| source_name | string | yes | e.g. NWS, GFS, ECMWF, Kalshi |
| source_type | string | yes | observation, forecast_model, market, settlement |
| authority_level | integer | yes | higher means more authoritative |
| url | string | no | source endpoint or docs URL |
| data_format | string | no | json, csv, parquet, api |
| active | boolean | yes | active usage flag |
| notes | string | no | freeform notes |

---

# 4. Raw Weather Observation

## Dataset Name

`raw_weather_observation`

## Purpose

Stores source-native weather observations.

## Grain

One record per source observation timestamp per station.

## Columns

| column | type | required | description |
|---|---|---:|---|
| timestamp_utc | timestamp | yes | observation time |
| station_id | string | yes | weather station ID |
| source_name | string | yes | source, usually NWS |
| temperature_c | float | no | source-native temperature |
| temperature_f | float | no | converted temperature |
| dewpoint_c | float | no | dew point |
| humidity_pct | float | no | relative humidity |
| wind_speed_mps | float | no | wind speed |
| pressure_hpa | float | no | pressure |
| cloud_cover_pct | float | no | cloud cover |
| ingestion_time_utc | timestamp | yes | load time |
| raw_payload_hash | string | no | hash for dedupe/debug |
| quality_flag | string | no | source quality marker |

## Primary Key Recommendation

Composite:

- `timestamp_utc`
- `station_id`
- `source_name`

---

# 5. Daily Settlement Observation

## Dataset Name

`daily_settlement_observation`

## Purpose

Stores official daily highs and lows used for settlement and backtests.

## Grain

One row per city per target date.

## Columns

| column | type | required | description |
|---|---|---:|---|
| city | string | yes | canonical city |
| station_id | string | yes | settlement station |
| target_date | date | yes | local calendar date |
| high_temp_f | float | yes | official daily high |
| low_temp_f | float | yes | official daily low |
| settlement_source | string | yes | usually NWS |
| settlement_report_id | string | no | report identifier if available |
| finalized_at_utc | timestamp | no | final settlement timestamp |
| quality_flag | string | no | settlement quality note |

## Primary Key

- `city`
- `target_date`

---

# 6. Forecast Run Registry

## Dataset Name

`forecast_run_registry`

## Purpose

Tracks forecast runs by model and issue time.

## Grain

One row per model run.

## Columns

| column | type | required | description |
|---|---|---:|---|
| model_name | string | yes | GFS, ECMWF, NAM |
| run_time_utc | timestamp | yes | model initialization time |
| availability_time_utc | timestamp | no | when data became available |
| cycle_label | string | yes | 00z, 06z, 12z, 18z |
| source_name | string | yes | forecast source |
| ingest_status | string | yes | pending, complete, failed |
| ingestion_time_utc | timestamp | no | ingestion time |
| notes | string | no | freeform notes |

## Primary Key

- `model_name`
- `run_time_utc`

---

# 7. Forecast Temperature Point

## Dataset Name

`forecast_temperature_point`

## Purpose

Stores point forecast temperatures by model, city, and target date.

## Grain

One row per model, city, run, target date, and metric.

## Columns

| column | type | required | description |
|---|---|---:|---|
| city | string | yes | canonical city |
| station_id | string | yes | linked station |
| model_name | string | yes | forecast model |
| run_time_utc | timestamp | yes | model run time |
| target_date | date | yes | forecast target date |
| metric | string | yes | high_temp_f or low_temp_f |
| forecast_value_f | float | yes | point forecast value |
| lead_hours | integer | yes | lead time in hours |
| source_name | string | yes | source name |
| ingestion_time_utc | timestamp | yes | load time |

## Primary Key Recommendation

Composite:

- `city`
- `model_name`
- `run_time_utc`
- `target_date`
- `metric`

---

# 8. Forecast Shift Event

## Dataset Name

`forecast_shift_event`

## Purpose

Stores run-to-run forecast changes for each model and city.

## Grain

One row per city, model, target date, metric, and run transition.

## Columns

| column | type | required | description |
|---|---|---:|---|
| city | string | yes | canonical city |
| model_name | string | yes | forecast model |
| metric | string | yes | high_temp_f or low_temp_f |
| previous_run_time_utc | timestamp | yes | old run |
| current_run_time_utc | timestamp | yes | new run |
| target_date | date | yes | forecast target date |
| previous_value_f | float | yes | prior forecast |
| current_value_f | float | yes | new forecast |
| shift_f | float | yes | current minus previous |
| absolute_shift_f | float | yes | absolute value |
| shift_direction | string | yes | up, down, flat |
| significant_shift_flag | boolean | yes | threshold-based marker |
| created_at_utc | timestamp | yes | computation time |

---

# 9. Forecast Error History

## Dataset Name

`forecast_error_history`

## Purpose

Measures realized model error by city and target date.

## Grain

One row per city, model, run, target date, and metric.

## Columns

| column | type | required | description |
|---|---|---:|---|
| city | string | yes | canonical city |
| model_name | string | yes | forecast model |
| run_time_utc | timestamp | yes | model run |
| target_date | date | yes | target date |
| metric | string | yes | high_temp_f or low_temp_f |
| forecast_value_f | float | yes | forecast |
| actual_value_f | float | yes | settled observation |
| error_f | float | yes | forecast minus actual |
| absolute_error_f | float | yes | absolute error |
| squared_error | float | yes | squared error |
| lead_hours | integer | yes | lead time |
| season | string | no | DJF, MAM, JJA, SON |
| created_at_utc | timestamp | yes | computation time |

## Derived Aggregates

Useful aggregate tables:

- `forecast_error_summary_city_model`
- `forecast_error_summary_city_model_season`

---

# 10. Ensemble Forecast Summary

## Dataset Name

`ensemble_forecast_summary`

## Purpose

Stores the aggregated ensemble result for each city and target date.

## Grain

One row per city, target date, metric, and run family.

## Columns

| column | type | required | description |
|---|---|---:|---|
| city | string | yes | canonical city |
| target_date | date | yes | target date |
| metric | string | yes | high_temp_f or low_temp_f |
| ensemble_run_time_utc | timestamp | yes | time ensemble was produced |
| contributing_models | string | yes | serialized model list |
| model_count | integer | yes | number of model inputs |
| ensemble_mean_f | float | yes | weighted mean |
| ensemble_std_f | float | yes | spread |
| variance_multiplier | float | yes | city-adjusted multiplier |
| adjusted_std_f | float | yes | final std after adjustments |
| bias_corrected_mean_f | float | yes | adjusted mean |
| methodology | string | yes | normal, weighted_normal, kde |
| created_at_utc | timestamp | yes | computation time |

---

# 11. KDE Distribution Snapshot

## Dataset Name

`kde_distribution_snapshot`

## Purpose

Stores serialized KDE distribution metadata for reproducibility.

## Grain

One row per city, target date, metric, and distribution build.

## Columns

| column | type | required | description |
|---|---|---:|---|
| city | string | yes | canonical city |
| target_date | date | yes | target date |
| metric | string | yes | high_temp_f or low_temp_f |
| distribution_time_utc | timestamp | yes | generation time |
| source_forecasts_json | string | yes | serialized input values |
| kde_bandwidth | float | yes | bandwidth used |
| min_temp_f | float | yes | support lower bound |
| max_temp_f | float | yes | support upper bound |
| methodology | string | yes | gaussian_kde |
| created_at_utc | timestamp | yes | creation time |

## Note

The full density curve can be recomputed, but this snapshot preserves reproducibility.

---

# 12. Probability Surface

## Dataset Name

`probability_surface`

## Purpose

Stores threshold probabilities across a range of temperatures.

## Grain

One row per city, target date, metric, threshold, and model version.

## Columns

| column | type | required | description |
|---|---|---:|---|
| city | string | yes | canonical city |
| target_date | date | yes | target date |
| metric | string | yes | high_temp_f or low_temp_f |
| threshold_f | integer | yes | threshold temperature |
| model_probability | float | yes | P(temp >= threshold) or P(temp <= threshold) |
| comparison_operator | string | yes | ge or le |
| distribution_method | string | yes | normal or kde |
| distribution_time_utc | timestamp | yes | generation time |
| model_version | string | yes | version tag |
| created_at_utc | timestamp | yes | creation time |

## Primary Key Recommendation

Composite:

- `city`
- `target_date`
- `metric`
- `threshold_f`
- `comparison_operator`
- `model_version`

---

# 13. Contract Universe

## Dataset Name

`contract_universe`

## Purpose

Defines the contract shapes Deep Isobar can map to.

## Grain

One row per logical contract template.

## Columns

| column | type | required | description |
|---|---|---:|---|
| contract_template_id | string | yes | unique template ID |
| market_source | string | yes | Kalshi or Polymarket |
| city | string | yes | city |
| metric | string | yes | high_temp_f or low_temp_f |
| comparison_operator | string | yes | ge or le |
| threshold_f | integer | yes | contract threshold |
| active | boolean | yes | active tracking flag |
| settlement_source | string | yes | NWS, Wunderground, etc. |
| notes | string | no | template notes |

---

# 14. Live Market Contract

## Dataset Name

`live_market_contract`

## Purpose

Stores live exchange contract metadata.

## Grain

One row per exchange contract instance.

## Columns

| column | type | required | description |
|---|---|---:|---|
| contract_id | string | yes | exchange contract ID |
| market_source | string | yes | Kalshi or Polymarket |
| contract_template_id | string | no | linked template |
| exchange_symbol | string | no | exchange symbol |
| city | string | yes | parsed city |
| metric | string | yes | high_temp_f or low_temp_f |
| comparison_operator | string | yes | ge or le |
| threshold_f | integer | yes | threshold |
| target_date | date | yes | event date |
| settlement_source | string | yes | source for settlement |
| listed_at_utc | timestamp | no | listing time |
| expires_at_utc | timestamp | no | expiration time |
| active | boolean | yes | active status |
| raw_title | string | no | exchange title |

---

# 15. Order Book Snapshot

## Dataset Name

`order_book_snapshot`

## Purpose

Stores market depth snapshots for microstructure analysis.

## Grain

One row per contract per snapshot time.

## Columns

| column | type | required | description |
|---|---|---:|---|
| timestamp_utc | timestamp | yes | snapshot time |
| contract_id | string | yes | exchange contract |
| market_source | string | yes | source |
| best_bid | float | no | best bid price |
| best_ask | float | no | best ask price |
| mid_price | float | no | computed midpoint |
| last_trade_price | float | no | last traded price |
| bid_size | float | no | size at best bid |
| ask_size | float | no | size at best ask |
| volume_24h | float | no | 24h volume |
| open_interest | float | no | open interest if available |
| snapshot_latency_ms | integer | no | ingestion latency |
| raw_payload_hash | string | no | dedupe/debug hash |

## Derived Fields

Can be materialized later:

- spread
- spread_pct
- imbalance
- staleness_seconds

---

# 16. Market Microstructure Feature

## Dataset Name

`market_microstructure_feature`

## Purpose

Stores computed market-quality signals.

## Grain

One row per contract per snapshot interval.

## Columns

| column | type | required | description |
|---|---|---:|---|
| timestamp_utc | timestamp | yes | feature time |
| contract_id | string | yes | contract |
| spread | float | no | ask minus bid |
| spread_pct | float | no | normalized spread |
| orderbook_imbalance | float | no | bid/ask imbalance metric |
| liquidity_score | float | no | liquidity quality score |
| staleness_seconds | float | no | time since price update |
| low_liquidity_flag | boolean | yes | liquidity warning |
| stale_market_flag | boolean | yes | stale market warning |
| microstructure_score | float | no | composite score |

---

# 17. Alpha Opportunity

## Dataset Name

`alpha_opportunity`

## Purpose

Stores model-vs-market discrepancies and ranked trade opportunities.

## Grain

One row per contract evaluation event.

## Columns

| column | type | required | description |
|---|---|---:|---|
| timestamp_utc | timestamp | yes | evaluation time |
| contract_id | string | yes | live contract |
| city | string | yes | city |
| target_date | date | yes | target date |
| metric | string | yes | high or low |
| threshold_f | integer | yes | threshold |
| comparison_operator | string | yes | ge or le |
| market_probability | float | yes | implied from market |
| model_probability | float | yes | from surface |
| alpha | float | yes | model minus market |
| absolute_alpha | float | yes | abs(alpha) |
| signal_side | string | yes | BUY, SELL, HOLD |
| confidence_score | float | no | composite confidence |
| tail_opportunity_flag | boolean | yes | tail threshold marker |
| forecast_shift_flag | boolean | yes | linked shift marker |
| stale_market_flag | boolean | yes | linked lag marker |
| microstructure_score | float | no | linked microstructure score |
| rank_score | float | no | ranking score |
| model_version | string | yes | model version |

---

# 18. Risk Decision

## Dataset Name

`risk_decision`

## Purpose

Stores whether a trade opportunity was approved or rejected.

## Grain

One row per opportunity evaluated by risk manager.

## Columns

| column | type | required | description |
|---|---|---:|---|
| timestamp_utc | timestamp | yes | decision time |
| contract_id | string | yes | contract |
| proposed_side | string | yes | BUY, SELL |
| proposed_quantity | float | yes | requested size |
| proposed_price | float | yes | intended price |
| approved_flag | boolean | yes | decision |
| rejection_reason | string | no | reason if rejected |
| current_position | float | no | open size before decision |
| projected_position | float | no | position after fill |
| daily_exposure_before | float | no | exposure before |
| daily_exposure_after | float | no | exposure after |

---

# 19. Trade Execution Log

## Dataset Name

`trade_execution_log`

## Purpose

Stores executed or attempted trades.

## Grain

One row per order attempt or fill event.

## Columns

| column | type | required | description |
|---|---|---:|---|
| timestamp_utc | timestamp | yes | event time |
| trade_id | string | yes | internal trade ID |
| contract_id | string | yes | contract |
| market_source | string | yes | source |
| side | string | yes | BUY or SELL |
| quantity | float | yes | requested quantity |
| price | float | yes | execution or limit price |
| order_type | string | yes | market, limit, simulated |
| execution_status | string | yes | submitted, filled, partial, rejected, canceled |
| fill_quantity | float | no | filled qty |
| avg_fill_price | float | no | average fill price |
| exchange_order_id | string | no | external ID |
| paper_trade_flag | boolean | yes | simulation marker |
| notes | string | no | freeform notes |

---

# 20. Position Snapshot

## Dataset Name

`position_snapshot`

## Purpose

Stores point-in-time exposure by contract and city.

## Grain

One row per contract at snapshot time.

## Columns

| column | type | required | description |
|---|---|---:|---|
| timestamp_utc | timestamp | yes | snapshot time |
| contract_id | string | yes | contract |
| city | string | yes | city |
| target_date | date | yes | target date |
| net_quantity | float | yes | signed position |
| avg_entry_price | float | no | average entry |
| mark_price | float | no | current mark |
| unrealized_pnl | float | no | unrealized profit/loss |
| realized_pnl | float | no | realized PnL to date |
| exposure_notional | float | no | exposure estimate |

---

# 21. Backtest Trade

## Dataset Name

`backtest_trade`

## Purpose

Stores simulated trades generated during backtests.

## Grain

One row per simulated trade.

## Columns

| column | type | required | description |
|---|---|---:|---|
| backtest_id | string | yes | backtest run ID |
| timestamp_utc | timestamp | yes | simulated trade time |
| contract_id | string | yes | contract |
| city | string | yes | city |
| target_date | date | yes | event date |
| side | string | yes | BUY or SELL |
| quantity | float | yes | size |
| simulated_price | float | yes | execution price |
| alpha | float | yes | alpha at trade |
| model_probability | float | yes | model probability |
| market_probability | float | yes | market probability |
| strategy_name | string | yes | strategy identifier |

---

# 22. Backtest Summary

## Dataset Name

`backtest_summary`

## Purpose

Stores aggregated backtest performance.

## Grain

One row per backtest run.

## Columns

| column | type | required | description |
|---|---|---:|---|
| backtest_id | string | yes | unique run ID |
| strategy_name | string | yes | strategy |
| start_date | date | yes | backtest start |
| end_date | date | yes | backtest end |
| total_trades | integer | yes | number of trades |
| win_rate | float | no | proportion of winners |
| gross_pnl | float | no | gross profit/loss |
| net_pnl | float | no | after fees/slippage if modeled |
| max_drawdown | float | no | worst peak-to-trough |
| sharpe_like_score | float | no | optional normalized metric |
| created_at_utc | timestamp | yes | run creation time |
| notes | string | no | freeform notes |

---

# 23. Feature Store Registry

## Dataset Name

`feature_store_registry`

## Purpose

Tracks feature datasets and versions.

## Grain

One row per feature artifact.

## Columns

| column | type | required | description |
|---|---|---:|---|
| feature_name | string | yes | name of feature set |
| version | string | yes | semantic or dated version |
| path | string | yes | parquet path |
| grain_description | string | yes | dataset grain |
| owner_module | string | yes | module that produced it |
| created_at_utc | timestamp | yes | creation time |
| active_flag | boolean | yes | active version |
| schema_hash | string | no | schema fingerprint |
| notes | string | no | freeform notes |

---

# Canonical Feature Names

Use these standardized feature names whenever possible.

## Weather / Forecast Features

- `forecast_mean_f`
- `forecast_std_f`
- `forecast_shift_f`
- `forecast_abs_shift_f`
- `forecast_model_count`
- `forecast_error_mean_f`
- `forecast_error_rmse_f`
- `forecast_volatility_score`

## Distribution Features

- `kde_bandwidth`
- `tail_probability_score`
- `surface_probability_ge`
- `surface_probability_le`

## Market Features

- `market_probability`
- `best_bid`
- `best_ask`
- `spread`
- `spread_pct`
- `liquidity_score`
- `staleness_seconds`
- `microstructure_score`

## Alpha Features

- `alpha`
- `absolute_alpha`
- `confidence_score`
- `rank_score`
- `tail_opportunity_flag`
- `forecast_shift_flag`
- `stale_market_flag`

---

# Recommended Parquet File Layout

Recommended layout by partition:

```text
data/
  raw/
    weather_observation/
      source_name=NWS/
        station_id=KORD/
          year=2026/
    market_orderbook/
      market_source=Kalshi/
        date=2026-07-04/

  forecasts/
    forecast_temperature_point/
      model_name=GFS/
        city=Chicago/
          year=2026/

  features/
    ensemble_forecast_summary/
      city=Chicago/
        year=2026/
    probability_surface/
      city=Chicago/
        metric=high_temp_f/
          year=2026/
    market_microstructure_feature/
      market_source=Kalshi/
        date=2026-07-04/

  backtests/
    backtest_trade/
      backtest_id=bt_20260704_001/