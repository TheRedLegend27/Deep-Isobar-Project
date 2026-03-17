# Deep Isobar Build Order

## Purpose

This document defines the recommended implementation sequence for Deep Isobar.

It exists to ensure:

- modules are built in a logical order
- AI agents stay focused
- the system becomes usable early
- later features build on stable foundations

This build order is optimized for:

- Mac development
- GitHub version control
- deployment to Dell PowerEdge servers
- AI-agent iterative coding

---

# Build Philosophy

## 1. Build the Spine First

Deep Isobar should first be able to do one full pass:

forecast
→ probability
→ market price
→ alpha
→ ranked opportunity

before adding sophisticated extras.

## 2. Separate MVP from Enhancements

Do not start with the most advanced features.

Build a working system first, then improve it with:

- city-specific weighting
- KDE upgrades
- forecast lag exploitation
- microstructure scoring

## 3. Every Step Must Be Testable

Each build step should produce:

- working code
- unit tests
- clear inputs and outputs
- no broken placeholder dependencies

---

# Phase 0 — Repository Foundation

Goal:

Create the repo structure and shared docs so AI agents know what to build.

## Build Items

1. `README.md`
2. `ARCHITECTURE.md`
3. `SYSTEM_DESIGN.md`
4. `DATA_SCHEMA.md`
5. `MODULE_DEPENDENCIES.md`
6. `BUILD_ORDER.md`

## Also Create

- `requirements.txt`
- `src/` folder layout
- `tests/` folder layout
- `config/` folder
- `docs/modules/` folder

## Exit Criteria

- repo structure exists
- documentation is committed
- AI agents can be pointed to module docs

---

# Phase 1 — Core Configuration and Universe

Goal:

Define the environment, cities, and source metadata.

## Step 1. Build Config Loader

Recommended files:

- `src/config.py`
- `config/settings.yaml`

Responsibilities:

- paths
- thresholds
- runtime settings
- risk limits
- market source toggles

Why first:

Everything else depends on config.

## Step 2. Build City Universe

Recommended files:

- `src/data/city_universe.py`
- `config/cities.yaml`
- `config/city_profiles/*.yaml`

Responsibilities:

- city registry
- station mapping
- model weights
- variance multipliers
- KDE bandwidth
- settlement source

Why early:

Most downstream modules need city metadata.

## Step 3. Build Data Source Registry

Recommended files:

- `src/data/data_sources.py`
- `config/data_sources.yaml`

Responsibilities:

- source definitions
- source type
- authority level
- source activation flags

## Exit Criteria

- config loads successfully
- city profiles are readable
- source registry works

---

# Phase 2 — Historical and Live Weather Inputs

Goal:

Get weather data into the system.

## Step 4. Build Weather Data Ingestion

Recommended files:

- `src/data/weather_ingest.py`

Responsibilities:

- fetch raw station data
- normalize observations
- produce official daily highs/lows

Schemas:

- `raw_weather_observation`
- `daily_settlement_observation`

Why now:

You need actual observations for later validation and backtests.

## Step 5. Build Forecast Generation

Recommended files:

- `src/models/forecast_generation.py`
- `src/models/forecast_collector.py`

Responsibilities:

- collect model forecasts
- store per-city forecasts
- register forecast runs

Schemas:

- `forecast_run_registry`
- `forecast_temperature_point`

## Step 6. Build Forecast Model Run Schedule

Recommended files:

- `src/models/model_run_schedule.py`

Responsibilities:

- track GFS/ECMWF cycles
- determine scan windows
- trigger ingestion timing

## Exit Criteria

- historical observations can be loaded
- forecast runs can be stored
- model schedules are represented correctly

---

# Phase 3 — Forecast Intelligence

Goal:

Measure changes and quality in forecasts.

## Step 7. Build Forecast Shift Detection

Recommended files:

- `src/models/forecast_shift.py`

Responsibilities:

- compare consecutive runs
- compute shift magnitude
- flag significant changes

Schema:

- `forecast_shift_event`

## Step 8. Build Historical Forecast Error

Recommended files:

- `src/models/forecast_error.py`

Responsibilities:

- compare forecasts to settled outcomes
- compute bias
- compute RMSE by city/model

Schema:

- `forecast_error_history`

## Step 9. Build Forecast Volatility

Recommended files:

- `src/models/forecast_volatility.py`

Responsibilities:

- compute ensemble spread
- estimate uncertainty score

Why here:

It feeds directly into ensemble and distribution logic.

## Exit Criteria

- forecast changes are measurable
- model errors are measurable
- uncertainty metrics exist

---

# Phase 4 — Temperature Modeling Core

Goal:

Turn forecasts into usable temperature distributions and probabilities.

## Step 10. Build Temperature Ensemble

Recommended files:

- `src/models/temperature_ensemble.py`

Responsibilities:

- combine forecasts
- apply city weights
- apply city bias correction
- apply variance adjustment

Schema:

- `ensemble_forecast_summary`

## Step 11. Build Normal Probability Engine (Fallback)

Recommended files:

- `src/models/probability_engine.py`

Responsibilities:

- compute threshold probabilities from mean/std
- serve as simple fallback model

Why before KDE:

You want a simpler working engine first.

## Step 12. Build KDE Temperature Distribution

Recommended files:

- `src/models/kde_temperature_distribution.py`

Responsibilities:

- build KDE distributions
- serialize distribution metadata

Schema:

- `kde_distribution_snapshot`

## Step 13. Build Probability Surface

Recommended files:

- `src/models/probability_surface.py`

Responsibilities:

- compute probabilities for all thresholds
- support high and low contracts
- persist threshold surface

Schema:

- `probability_surface`

## Exit Criteria

- a city forecast can become a full threshold probability map
- both fallback and KDE paths are testable

---

# Phase 5 — Contract and Market Mapping

Goal:

Map model probabilities to actual market contracts.

## Step 14. Build Contract Generator

Recommended files:

- `src/market/contract_generator.py`

Responsibilities:

- create internal contract IDs
- map city + threshold + target date
- support high/low contracts

Schemas:

- `contract_universe`
- `live_market_contract` linkage

## Step 15. Build Prediction Market API Integration

Recommended files:

- `src/market/kalshi_client.py`
- `src/market/polymarket_client.py`

Responsibilities:

- fetch contracts
- fetch order books
- normalize metadata

Schemas:

- `live_market_contract`
- `order_book_snapshot`

## Step 16. Build Market Price Adapter

Recommended files:

- `src/market/market_price_adapter.py`

Responsibilities:

- validate bid/ask
- compute market-implied probabilities
- normalize order book output

## Exit Criteria

- internal contracts can be mapped to live market contracts
- order books can be converted into probabilities

---

# Phase 6 — Alpha Detection MVP

Goal:

Detect mispricing with a working end-to-end scanner.

## Step 17. Build Alpha Engine

Recommended files:

- `src/trading/alpha_engine.py`

Responsibilities:

- compare model probability vs market probability
- return BUY / SELL / HOLD
- calculate alpha and confidence

Schema:

- `alpha_opportunity`

## Step 18. Build Market Scanner

Recommended files:

- `src/market/market_scanner.py`

Responsibilities:

- iterate over live contracts
- evaluate alpha
- rank opportunities

Why this phase matters:

At this point Deep Isobar becomes useful.

## Exit Criteria

- scanner outputs ranked trade opportunities
- system works without microstructure or lag enhancements

---

# Phase 7 — Risk and Execution MVP

Goal:

Allow safe paper-trading and later live execution.

## Step 19. Build Risk Manager

Recommended files:

- `src/trading/risk_manager.py`

Responsibilities:

- position limits
- exposure checks
- approve/reject trades

Schema:

- `risk_decision`

## Step 20. Build Trade Execution

Recommended files:

- `src/trading/trade_execution.py`

Responsibilities:

- log orders
- simulate or send trades
- update positions

Schemas:

- `trade_execution_log`
- `position_snapshot`

## Step 21. Build Scheduler / Automation

Recommended files:

- `src/scheduler.py`

Responsibilities:

- schedule scans
- trigger forecast updates
- run trading loop
- support paper mode first

## Exit Criteria

- paper-trading loop runs
- scheduler repeatedly scans markets
- risk rules prevent unsafe trades

---

# Phase 8 — Research and Backtesting

Goal:

Measure whether the strategy actually works.

## Step 22. Build Backtest Engine

Recommended files:

- `src/research/backtest_engine.py`

Responsibilities:

- replay forecasts and outcomes
- simulate strategy trades
- compute PnL and drawdown

Schemas:

- `backtest_trade`
- `backtest_summary`

Why not earlier:

A backtest engine is most useful once the signal logic is stable.

## Exit Criteria

- historical strategy performance can be evaluated
- alpha thresholds can be tuned

---

# Phase 9 — Alpha Enhancements

Goal:

Add the real edge layers after MVP works.

## Step 23. Build Market Microstructure Scanner

Recommended files:

- `src/market/microstructure_scanner.py`

Responsibilities:

- compute spread
- liquidity score
- price staleness
- microstructure score

Schema:

- `market_microstructure_feature`

## Step 24. Build Market Lag Detection

Recommended files:

- `src/market/market_lag_detection.py`

Responsibilities:

- detect when forecast changes are not reflected in market prices
- flag stale market opportunities

## Step 25. Build Distribution Tail Alpha

Recommended files:

- `src/trading/distribution_tail_alpha.py`

Responsibilities:

- identify extreme-threshold mispricings
- prioritize tail opportunities

## Step 26. Integrate Enhancements into Alpha Ranking

Update:

- `alpha_engine.py`
- `market_scanner.py`

to include:

- forecast shift score
- microstructure score
- tail opportunity score
- lag score

## Exit Criteria

- opportunities are ranked by more than just raw alpha
- system reacts better to forecast shifts and stale markets

---

# Phase 10 — Targeting and Optimization

Goal:

Focus the scanner on the best cities and contracts.

## Step 27. Build Mispriced Weather Markets Analyzer

Recommended files:

- `src/research/mispriced_weather_markets.py`

Responsibilities:

- identify best cities
- identify best thresholds
- score market types by profitability

## Step 28. Refine City Profiles

Update city-specific:

- model weights
- variance multipliers
- KDE bandwidth
- bias correction

## Exit Criteria

- system targets high-value cities and thresholds first
- scanner prioritization improves

---

# Suggested MVP Build Order Summary

If you want the shortest path to a working system, build in this exact order:

1. config
2. city universe
3. data source registry
4. weather data ingestion
5. forecast generation
6. forecast shift detection
7. forecast volatility
8. temperature ensemble
9. normal probability engine
10. KDE distribution
11. probability surface
12. contract generator
13. prediction market API integration
14. market price adapter
15. alpha engine
16. market scanner
17. risk manager
18. trade execution
19. scheduler

That is the MVP spine.

---

# Suggested Enhancement Order Summary

After the MVP works, build in this order:

1. historical forecast error
2. backtest engine
3. market microstructure scanner
4. market lag detection
5. distribution tail alpha
6. mispriced weather markets analyzer
7. city profile refinement

---

# Deliverable Rules for Each Step

For every module built in this order, require the AI agent to produce:

1. explanation
2. implementation
3. example usage
4. unit tests
5. edge cases
6. logging/error handling

Then stop.

Do not ask the agent to build multiple phases at once.

---

# Build Milestones

## Milestone A — Data Ready

Completed when:

- weather ingestion works
- forecast generation works
- city profiles exist

## Milestone B — Probability Ready

Completed when:

- ensemble exists
- KDE or fallback probability engine works
- probability surface exists

## Milestone C — Market Ready

Completed when:

- contracts can be mapped
- live prices can be read
- market probabilities are normalized

## Milestone D — Scanner Ready

Completed when:

- alpha engine works
- scanner ranks opportunities

## Milestone E — Trading Ready

Completed when:

- risk manager works
- paper execution works
- scheduler runs the loop

## Milestone F — Research Ready

Completed when:

- backtests work
- forecast error analytics work
- targeting and enhancement layers are integrated

---

# AI Agent Guidance

If an agent is asked to build a module, it should first check:

1. Is this module in the current phase?
2. Are all dependencies already built?
3. Are the required schemas defined in `DATA_SCHEMA.md`?
4. Is the interface narrow enough to test now?

If the answer to any of these is no, build the missing prerequisite instead.

---

# Recommended First 5 Coding Tasks

To start immediately in Antigravity, use these five tasks first:

1. Build `src/config.py`
2. Build `src/data/city_universe.py`
3. Build `src/data/data_sources.py`
4. Build `src/data/weather_ingest.py`
5. Build `src/models/forecast_generation.py`

That sequence gives you the cleanest launch.