# Deep Isobar Module Dependencies

## Purpose

This document defines how Deep Isobar modules depend on each other.

It exists to ensure:

- modules are built in the correct order
- AI agents know what can be assumed to exist
- interfaces remain stable
- development stays iterative and modular

This file should be used together with:

- `SYSTEM_DESIGN.md`
- `DATA_SCHEMA.md`
- `ARCHITECTURE.md`

---

# Dependency Rules

## 1. Build Small, Build Forward

A module may depend only on:

- core configuration files
- canonical schemas from `DATA_SCHEMA.md`
- modules that have already been completed and tested

A module should not depend on unfinished future modules.

## 2. Prefer One-Way Dependencies

Dependencies should generally flow in one direction:

Data
→ Forecasting
→ Probability
→ Market Data
→ Alpha
→ Risk
→ Execution

Avoid circular dependencies.

## 3. Stable Interfaces First

Before building a dependent module, define:

- required inputs
- required outputs
- expected schema
- error behavior

---

# Dependency Graph Overview

## High-Level Flow

City Universe / Profiles
→ Weather Data Ingestion
→ Forecast Generation
→ Forecast Shift Detection
→ Historical Forecast Error
→ Temperature Ensemble
→ KDE Temperature Distribution
→ Probability Surface
→ Contract Generator
→ Prediction Market API Integration
→ Market Microstructure Scanner
→ Alpha Engine / Market Scanner
→ Risk Manager
→ Trade Execution
→ Backtest Engine

---

# Core Foundation Modules

These modules are foundational and should be assumed by most others.

## 1. Config

File examples:

- `src/config.py`
- `config/*.yaml`

Used by:

- all runtime modules

Dependencies:

- none

Provides:

- paths
- thresholds
- environment settings
- market source toggles
- risk limits

---

## 2. City Universe

File examples:

- `src/data/city_universe.py`
- `config/cities.yaml`
- `config/city_profiles/*.yaml`

Dependencies:

- config

Used by:

- weather ingestion
- forecast generation
- ensemble generation
- contract generator
- backtesting

Provides:

- city registry
- settlement station IDs
- city-specific parameters
- model weights
- KDE bandwidth
- variance multipliers

---

## 3. Data Source Registry

File examples:

- `src/data/data_sources.py`
- `config/data_sources.yaml`

Dependencies:

- config

Used by:

- weather ingestion
- forecast generation
- market API clients
- settlement logic

Provides:

- authoritative source definitions
- source metadata
- source activation flags

---

# Data Layer Modules

## 4. Weather Data Ingestion

File examples:

- `src/data/weather_ingest.py`

Dependencies:

- config
- city universe
- data source registry

Used by:

- settlement observation builder
- historical forecast error
- backtests

Reads/Writes:

- `raw_weather_observation`
- `daily_settlement_observation`

Requires:

- station IDs
- source metadata
- storage paths

Provides:

- normalized weather observations
- official daily highs/lows

---

## 5. Forecast Generation

File examples:

- `src/models/forecast_generation.py`
- `src/models/forecast_collector.py`

Dependencies:

- config
- city universe
- data source registry

Used by:

- forecast shift detection
- ensemble generation
- forecast error tracking

Reads/Writes:

- `forecast_run_registry`
- `forecast_temperature_point`

Requires:

- city list
- source/model definitions
- target dates

Provides:

- point forecasts by city/model/run

---

## 6. Prediction Market API Integration

File examples:

- `src/market/kalshi_client.py`
- `src/market/polymarket_client.py`

Dependencies:

- config
- data source registry
- contract generator (optional at first)

Used by:

- market price normalization
- microstructure scanner
- alpha engine

Reads/Writes:

- `live_market_contract`
- `order_book_snapshot`

Requires:

- market credentials/config
- contract discovery rules

Provides:

- contract metadata
- order book snapshots
- live price data

---

# Forecast Intelligence Modules

## 7. Forecast Shift Detection

File examples:

- `src/models/forecast_shift.py`

Dependencies:

- forecast generation

Used by:

- alpha engine
- market lag detection
- opportunity ranking

Reads/Writes:

- `forecast_temperature_point`
- `forecast_shift_event`

Requires:

- previous and current forecast runs

Provides:

- run-to-run forecast changes
- shift magnitude
- significant shift flags

---

## 8. Historical Forecast Error

File examples:

- `src/models/forecast_error.py`

Dependencies:

- forecast generation
- weather data ingestion
- city universe

Used by:

- ensemble weighting
- city profile tuning
- backtests

Reads/Writes:

- `forecast_temperature_point`
- `daily_settlement_observation`
- `forecast_error_history`

Requires:

- historical forecasts
- settled actual temperatures

Provides:

- model bias
- RMSE
- city-specific error summaries

---

## 9. Forecast Model Run Schedule

File examples:

- `src/models/model_run_schedule.py`

Dependencies:

- config
- forecast generation

Used by:

- scheduler
- forecast ingestion triggers
- forecast shift detection

Requires:

- model cycle times
- timezone settings

Provides:

- expected run schedule
- release monitoring
- trigger windows

---

## 10. Forecast Volatility

File examples:

- `src/models/forecast_volatility.py`

Dependencies:

- forecast generation
- city profiles (optional)

Used by:

- ensemble generation
- KDE distribution
- alpha ranking

Reads/Writes:

- may read `forecast_temperature_point`
- may enrich `ensemble_forecast_summary`

Requires:

- multiple model forecasts

Provides:

- spread metrics
- uncertainty score
- variance estimates

---

# Distribution / Probability Modules

## 11. Temperature Ensemble

File examples:

- `src/models/temperature_ensemble.py`

Dependencies:

- forecast generation
- city universe
- historical forecast error (optional but recommended)
- forecast volatility

Used by:

- KDE temperature distribution
- normal probability engine fallback

Reads/Writes:

- `forecast_temperature_point`
- `city_profile`
- `ensemble_forecast_summary`

Requires:

- model forecasts
- weights/bias adjustments

Provides:

- weighted mean
- adjusted variance
- ensemble summary

---

## 12. KDE Temperature Distribution

File examples:

- `src/models/kde_temperature_distribution.py`

Dependencies:

- temperature ensemble
- city universe

Used by:

- probability surface
- tail alpha detection

Reads/Writes:

- `ensemble_forecast_summary`
- `kde_distribution_snapshot`

Requires:

- ensemble input values
- KDE bandwidth

Provides:

- smoothed temperature distribution
- reproducible distribution snapshot

---

## 13. Probability Surface

File examples:

- `src/models/probability_surface.py`

Dependencies:

- KDE temperature distribution
- temperature ensemble (for fallback)
- city universe

Used by:

- contract generator
- alpha engine
- backtest engine

Reads/Writes:

- `probability_surface`

Requires:

- distribution object
- threshold range

Provides:

- probability by threshold
- tail probabilities
- full threshold map

---

## 14. Distribution Tail Alpha

File examples:

- `src/trading/distribution_tail_alpha.py`

Dependencies:

- probability surface
- market prices

Used by:

- alpha ranking
- opportunity prioritization

Requires:

- threshold probabilities
- market-implied probabilities

Provides:

- tail opportunity flags
- tail-specific alpha signals

---

# Market Intelligence Modules

## 15. Contract Generator

File examples:

- `src/market/contract_generator.py`

Dependencies:

- city universe
- probability surface

Used by:

- market API matching
- alpha engine
- scanner

Reads/Writes:

- `contract_universe`
- may enrich `live_market_contract`

Requires:

- city codes
- metric type
- thresholds
- target date

Provides:

- internal contract IDs
- contract mapping templates

---

## 16. Market Price Adapter / Normalizer

File examples:

- `src/market/market_price_adapter.py`

Dependencies:

- prediction market API integration

Used by:

- alpha engine
- microstructure scanner

Requires:

- bid/ask/last price inputs

Provides:

- normalized market probability
- midpoint pricing
- validation of order books

---

## 17. Market Microstructure Scanner

File examples:

- `src/market/microstructure_scanner.py`

Dependencies:

- prediction market API integration
- market price adapter

Used by:

- alpha engine
- risk manager
- opportunity ranking

Reads/Writes:

- `order_book_snapshot`
- `market_microstructure_feature`

Requires:

- order book snapshots
- recent price history

Provides:

- spread metrics
- liquidity score
- staleness metrics
- microstructure score

---

## 18. Market Lag Detection

File examples:

- `src/market/market_lag_detection.py`

Dependencies:

- forecast shift detection
- prediction market API integration
- market microstructure scanner

Used by:

- alpha engine
- opportunity prioritization

Requires:

- recent forecast shifts
- current market response

Provides:

- stale probability flags
- lag-based opportunity signal

---

## 19. Mispriced Weather Markets

File examples:

- `src/research/mispriced_weather_markets.py`

Dependencies:

- forecast error history
- probability surface
- alpha opportunity history
- microstructure features

Used by:

- city prioritization
- contract prioritization
- scanner targeting

Requires:

- historical results

Provides:

- city scores
- threshold priority lists
- market focus recommendations

---

# Trading Modules

## 20. Alpha Engine

File examples:

- `src/trading/alpha_engine.py`
- `src/trading/alpha_surface.py`

Dependencies:

- probability surface
- contract generator
- market price adapter
- forecast shift detection
- market lag detection
- microstructure scanner
- tail alpha detection

Used by:

- scanner
- risk manager
- backtests

Reads/Writes:

- `alpha_opportunity`

Requires:

- model probability
- market probability
- threshold
- contract mapping

Provides:

- alpha values
- BUY / SELL / HOLD signal
- confidence/rank scores

---

## 21. Market Scanner

File examples:

- `src/market/market_scanner.py`

Dependencies:

- prediction market API integration
- market price adapter
- alpha engine
- contract generator
- mispriced weather markets (optional targeting)

Used by:

- scheduler
- trade execution pipeline

Requires:

- live contracts
- current model outputs

Provides:

- ranked opportunity list
- top candidate trades

---

## 22. Risk Manager

File examples:

- `src/trading/risk_manager.py`

Dependencies:

- alpha engine
- position tracking
- config

Used by:

- trade execution

Reads/Writes:

- `risk_decision`
- `position_snapshot`

Requires:

- proposed trade
- current positions
- exposure limits

Provides:

- approve/reject decision
- rejection reason
- projected exposure

---

## 23. Trade Execution

File examples:

- `src/trading/trade_execution.py`

Dependencies:

- risk manager
- prediction market API integration

Used by:

- live trading loop

Reads/Writes:

- `trade_execution_log`
- `position_snapshot`

Requires:

- approved trade instructions

Provides:

- submitted orders
- execution log
- position updates

---

# Research / Evaluation Modules

## 24. Backtest Engine

File examples:

- `src/research/backtest_engine.py`

Dependencies:

- city universe
- weather data ingestion
- forecast generation
- probability surface
- contract generator
- alpha engine
- risk manager

Used by:

- research workflow
- strategy evaluation

Reads/Writes:

- `backtest_trade`
- `backtest_summary`

Requires:

- historical forecasts
- actual outcomes
- simulated market prices or market history

Provides:

- PnL metrics
- strategy evaluation
- trade simulations

---

# Runtime / Orchestration Modules

## 25. Scheduler / Automation

File examples:

- `src/scheduler.py`

Dependencies:

- forecast model run schedule
- forecast generation
- prediction market API integration
- market scanner
- trade execution

Used by:

- live system runtime

Requires:

- execution intervals
- model run windows
- retry logic

Provides:

- timed execution
- automation loop
- scan orchestration

---

# Dependency Matrix by Module

## City Universe depends on

- config

## Weather Data Ingestion depends on

- config
- city universe
- data source registry

## Forecast Generation depends on

- config
- city universe
- data source registry

## Forecast Shift Detection depends on

- forecast generation

## Historical Forecast Error depends on

- forecast generation
- weather data ingestion
- city universe

## Forecast Model Run Schedule depends on

- config
- forecast generation

## Forecast Volatility depends on

- forecast generation

## Temperature Ensemble depends on

- forecast generation
- city universe
- forecast volatility
- historical forecast error (recommended)

## KDE Temperature Distribution depends on

- temperature ensemble
- city universe

## Probability Surface depends on

- KDE temperature distribution
- temperature ensemble

## Contract Generator depends on

- city universe
- probability surface

## Prediction Market API Integration depends on

- config
- data source registry

## Market Price Adapter depends on

- prediction market API integration

## Market Microstructure Scanner depends on

- prediction market API integration
- market price adapter

## Market Lag Detection depends on

- forecast shift detection
- prediction market API integration
- market microstructure scanner

## Distribution Tail Alpha depends on

- probability surface
- market prices

## Alpha Engine depends on

- probability surface
- contract generator
- market price adapter
- forecast shift detection
- market lag detection
- market microstructure scanner
- distribution tail alpha

## Market Scanner depends on

- prediction market API integration
- market price adapter
- alpha engine
- contract generator

## Risk Manager depends on

- alpha engine
- config
- position tracking

## Trade Execution depends on

- risk manager
- prediction market API integration

## Backtest Engine depends on

- city universe
- weather data ingestion
- forecast generation
- probability surface
- contract generator
- alpha engine
- risk manager

## Scheduler depends on

- forecast model run schedule
- forecast generation
- prediction market API integration
- market scanner
- trade execution

---

# Circular Dependency Warnings

Avoid these patterns:

## Bad Pattern 1

Trade Execution
↔ Risk Manager

Correct pattern:

- risk manager evaluates proposed trade
- trade execution only consumes approved decision

## Bad Pattern 2

Alpha Engine
↔ Market Scanner

Correct pattern:

- alpha engine evaluates one contract/opportunity
- market scanner orchestrates many alpha evaluations

## Bad Pattern 3

Probability Surface
↔ Contract Generator

Correct pattern:

- probability surface produces threshold probabilities
- contract generator maps thresholds to contract IDs

## Bad Pattern 4

Forecast Generation
↔ Historical Forecast Error

Correct pattern:

- forecast generation writes forecasts
- historical forecast error evaluates them later using settled outcomes

---

# Minimal Dependency Set for MVP

For a first working system, only these dependencies are required:

1. config
2. city universe
3. weather data ingestion
4. forecast generation
5. temperature ensemble
6. KDE temperature distribution or normal fallback
7. probability surface
8. contract generator
9. prediction market API integration
10. market price adapter
11. alpha engine
12. market scanner
13. risk manager
14. trade execution
15. scheduler

Everything else is an enhancement.

---

# AI Agent Guidance

When building a module:

1. check this file first
2. confirm dependencies already exist
3. do not import future modules
4. use schemas from `DATA_SCHEMA.md`
5. keep interfaces narrow and testable

If a dependency is missing, build that dependency first instead of mocking the whole system.