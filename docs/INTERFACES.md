# Deep Isobar Interfaces

## Purpose

This document defines the canonical function and class interfaces for the Deep Isobar MVP.

It exists to ensure:

- modules connect cleanly
- AI agents use stable signatures
- tests can be written consistently
- refactors do not break downstream modules

This file should be used together with:

- `SYSTEM_DESIGN.md`
- `DATA_SCHEMA.md`
- `MODULE_DEPENDENCIES.md`
- `BUILD_ORDER.md`

---

# Interface Rules

## 1. Keep Interfaces Narrow

Functions should do one thing.

Prefer:

- small pure functions
- explicit inputs
- explicit return types
- deterministic outputs where possible

Avoid:

- giant stateful classes without need
- implicit globals
- hidden side effects

## 2. Prefer Typed Inputs and Outputs

Use:

- dataclasses
- pydantic models
- typed dictionaries only when necessary

## 3. Canonical Units

Unless otherwise noted:

- temperature inputs to modeling layers should be Fahrenheit
- probabilities must be floats in `[0.0, 1.0]`
- timestamps should be UTC-aware or stored as UTC strings

## 4. Error Handling

Functions should raise explicit exceptions for invalid input.

Examples:

- `ValueError`
- `FileNotFoundError`
- `RuntimeError`

Do not silently swallow bad data.

---

# Shared Data Models

These models can live in:

- `src/core/types.py`
- `src/core/models.py`

or similar shared module.

---

## CityProfile

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class CityProfile:
    city: str
    city_code: str
    station_id: str
    timezone: str
    settlement_source: str
    variance_multiplier: float
    mean_bias_correction_f: float
    kde_bandwidth: float
    tail_multiplier: float = 1.0
    model_weight_gfs: Optional[float] = None
    model_weight_ecmwf: Optional[float] = None
    model_weight_nam: Optional[float] = None
    heat_bias_adjustment_f: float = 0.0
    cold_bias_adjustment_f: float = 0.0

ForecastPoint
from dataclasses import dataclass
from datetime import datetime, date

@dataclass
class ForecastPoint:
    city: str
    station_id: str
    model_name: str
    run_time_utc: datetime
    target_date: date
    metric: str
    forecast_value_f: float
    lead_hours: int
    source_name: str
ForecastShiftEvent
from dataclasses import dataclass
from datetime import datetime, date

@dataclass
class ForecastShiftEvent:
    city: str
    model_name: str
    metric: str
    previous_run_time_utc: datetime
    current_run_time_utc: datetime
    target_date: date
    previous_value_f: float
    current_value_f: float
    shift_f: float
    absolute_shift_f: float
    shift_direction: str
    significant_shift_flag: bool
EnsembleSummary
from dataclasses import dataclass
from datetime import datetime, date
from typing import List

@dataclass
class EnsembleSummary:
    city: str
    target_date: date
    metric: str
    ensemble_run_time_utc: datetime
    contributing_models: List[str]
    model_count: int
    ensemble_mean_f: float
    ensemble_std_f: float
    variance_multiplier: float
    adjusted_std_f: float
    bias_corrected_mean_f: float
    methodology: str
TemperatureDistributionSnapshot
from dataclasses import dataclass
from datetime import datetime, date
from typing import List

@dataclass
class TemperatureDistributionSnapshot:
    city: str
    target_date: date
    metric: str
    distribution_time_utc: datetime
    source_forecasts_f: List[float]
    kde_bandwidth: float
    min_temp_f: float
    max_temp_f: float
    methodology: str
OrderBookSnapshot
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class OrderBookSnapshot:
    timestamp_utc: datetime
    contract_id: str
    market_source: str
    best_bid: Optional[float]
    best_ask: Optional[float]
    last_trade_price: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    volume_24h: Optional[float] = None
    open_interest: Optional[float] = None
MarketContract
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

@dataclass
class MarketContract:
    contract_id: str
    market_source: str
    city: str
    metric: str
    comparison_operator: str
    threshold_f: int
    target_date: date
    settlement_source: str
    raw_title: Optional[str] = None
    listed_at_utc: Optional[datetime] = None
    expires_at_utc: Optional[datetime] = None
    active: bool = True
TradeSignal
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional

@dataclass
class TradeSignal:
    timestamp_utc: datetime
    contract_id: str
    city: str
    target_date: date
    metric: str
    threshold_f: int
    comparison_operator: str
    market_probability: float
    model_probability: float
    alpha: float
    absolute_alpha: float
    signal_side: str
    confidence_score: float
    tail_opportunity_flag: bool = False
    forecast_shift_flag: bool = False
    stale_market_flag: bool = False
    microstructure_score: Optional[float] = None
    rank_score: Optional[float] = None
    model_version: str = "v1"
RiskDecision
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class RiskDecision:
    timestamp_utc: datetime
    contract_id: str
    proposed_side: str
    proposed_quantity: float
    proposed_price: float
    approved_flag: bool
    rejection_reason: Optional[str] = None
    current_position: Optional[float] = None
    projected_position: Optional[float] = None
    daily_exposure_before: Optional[float] = None
    daily_exposure_after: Optional[float] = None
ExecutedTrade
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class ExecutedTrade:
    timestamp_utc: datetime
    trade_id: str
    contract_id: str
    market_source: str
    side: str
    quantity: float
    price: float
    order_type: str
    execution_status: str
    fill_quantity: Optional[float] = None
    avg_fill_price: Optional[float] = None
    exchange_order_id: Optional[str] = None
    paper_trade_flag: bool = True
    notes: Optional[str] = None
Module Interfaces
1. Config

Suggested file:

src/config.py

Interface
from pathlib import Path
from typing import Any

def get_project_root() -> Path:
    ...

def load_settings(config_path: str | None = None) -> dict[str, Any]:
    ...

def get_setting(key: str, default: Any = None) -> Any:
    ...
Behavior

loads YAML or JSON config

returns dictionary-like settings

raises FileNotFoundError for missing config

raises ValueError for invalid config format

2. City Universe

Suggested file:

src/data/city_universe.py

Interface
from typing import List

def load_city_profiles(config_dir: str | None = None) -> list[CityProfile]:
    ...

def get_city_profile(city: str, config_dir: str | None = None) -> CityProfile:
    ...

def list_active_cities(config_dir: str | None = None) -> list[str]:
    ...
Behavior

loads all city profiles from config

filters inactive cities if needed

raises KeyError if city not found

3. Data Sources

Suggested file:

src/data/data_sources.py

Interface
from typing import Any

def load_data_sources(config_path: str | None = None) -> dict[str, dict[str, Any]]:
    ...

def get_data_source(name: str, config_path: str | None = None) -> dict[str, Any]:
    ...
Behavior

loads source registry

returns metadata for source

raises KeyError if missing

4. Weather Data Ingestion

Suggested file:

src/data/weather_ingest.py

Interface
import pandas as pd
from datetime import date

def fetch_station_observations(
    station_id: str,
    start_date: date,
    end_date: date,
    source_name: str = "NWS",
) -> pd.DataFrame:
    ...

def normalize_weather_observations(df: pd.DataFrame) -> pd.DataFrame:
    ...

def build_daily_settlement_observations(
    df: pd.DataFrame,
    city: str,
    station_id: str,
    settlement_source: str = "NWS",
) -> pd.DataFrame:
    ...
Expected Columns Returned
fetch_station_observations

Returns a dataframe that can be normalized into:

timestamp_utc

station_id

temperature_c and/or temperature_f

normalize_weather_observations

Returns canonical columns such as:

timestamp_utc

station_id

source_name

temperature_f

build_daily_settlement_observations

Returns canonical daily rows:

city

station_id

target_date

high_temp_f

low_temp_f

settlement_source

5. Forecast Generation

Suggested files:

src/models/forecast_generation.py

src/models/forecast_collector.py

Interface
from datetime import date, datetime

def fetch_model_forecast(
    city: str,
    station_id: str,
    model_name: str,
    run_time_utc: datetime,
    target_date: date,
    metric: str,
) -> ForecastPoint:
    ...

def fetch_forecasts_for_city(
    city: str,
    target_date: date,
    metric: str,
    model_names: list[str],
) -> list[ForecastPoint]:
    ...

def register_forecast_run(
    model_name: str,
    run_time_utc: datetime,
    cycle_label: str,
    source_name: str,
) -> dict:
    ...
Behavior

one ForecastPoint per model/city/run/target/metric

should validate metric in:

high_temp_f

low_temp_f

6. Forecast Model Run Schedule

Suggested file:

src/models/model_run_schedule.py

Interface
from datetime import datetime
from typing import List

def get_expected_model_cycles(model_name: str) -> list[str]:
    ...

def is_expected_run_time(model_name: str, run_time_utc: datetime) -> bool:
    ...

def get_next_expected_run_times(
    model_name: str,
    now_utc: datetime,
    count: int = 4,
) -> list[datetime]:
    ...
Behavior

supports GFS and ECMWF initially

returns UTC times

raises ValueError for unsupported models

7. Forecast Shift Detection

Suggested file:

src/models/forecast_shift.py

Interface
def compute_forecast_shift(
    previous: ForecastPoint,
    current: ForecastPoint,
    significance_threshold_f: float = 2.0,
) -> ForecastShiftEvent:
    ...
Behavior

both points must match on:

city

model_name

target_date

metric

computes signed and absolute shift

flags significant shift if abs shift exceeds threshold

8. Historical Forecast Error

Suggested file:

src/models/forecast_error.py

Interface
import pandas as pd

def compute_forecast_error(
    forecast_df: pd.DataFrame,
    actual_df: pd.DataFrame,
) -> pd.DataFrame:
    ...

def summarize_forecast_error_by_city_model(
    error_df: pd.DataFrame,
) -> pd.DataFrame:
    ...
Behavior
compute_forecast_error

Must output columns including:

city

model_name

target_date

metric

forecast_value_f

actual_value_f

error_f

absolute_error_f

squared_error

summarize_forecast_error_by_city_model

Must output aggregates like:

mean error

mean absolute error

RMSE

sample count

9. Forecast Volatility

Suggested file:

src/models/forecast_volatility.py

Interface
def compute_forecast_std(forecast_values_f: list[float]) -> float:
    ...

def compute_forecast_variance(forecast_values_f: list[float]) -> float:
    ...

def compute_forecast_volatility_score(forecast_values_f: list[float]) -> float:
    ...
Behavior

non-empty list required

raises ValueError on empty input

10. Temperature Ensemble

Suggested file:

src/models/temperature_ensemble.py

Interface
from datetime import datetime, date

def build_temperature_ensemble(
    city_profile: CityProfile,
    forecasts: list[ForecastPoint],
    target_date: date,
    metric: str,
    ensemble_run_time_utc: datetime,
) -> EnsembleSummary:
    ...
Behavior

applies model weights if available

otherwise defaults to equal weights

applies mean bias correction

applies variance multiplier

requires at least 1 forecast point

11. Normal Probability Engine

Suggested file:

src/models/probability_engine.py

Interface
def probability_ge_normal(mean_f: float, std_f: float, threshold_f: float) -> float:
    ...

def probability_le_normal(mean_f: float, std_f: float, threshold_f: float) -> float:
    ...
Behavior

returns values in [0, 1]

raises ValueError if std_f <= 0

12. KDE Temperature Distribution

Suggested file:

src/models/kde_temperature_distribution.py

Interface
from scipy.stats import gaussian_kde

def build_kde_distribution(
    forecast_values_f: list[float],
    bandwidth: float | None = None,
) -> gaussian_kde:
    ...

def build_distribution_snapshot(
    city: str,
    target_date,
    metric: str,
    forecast_values_f: list[float],
    kde_bandwidth: float,
    min_temp_f: float,
    max_temp_f: float,
) -> TemperatureDistributionSnapshot:
    ...
Behavior

requires at least 2 points unless special fallback is implemented

raises ValueError for invalid bandwidth or empty input

13. Probability Surface

Suggested file:

src/models/probability_surface.py

Interface
def generate_probability_surface_normal(
    mean_f: float,
    std_f: float,
    min_temp_f: int,
    max_temp_f: int,
    comparison_operator: str = "ge",
) -> dict[int, float]:
    ...

def generate_probability_surface_kde(
    kde_distribution,
    min_temp_f: int,
    max_temp_f: int,
    comparison_operator: str = "ge",
) -> dict[int, float]:
    ...
Behavior

keys are integer thresholds

values are probabilities

valid operators:

ge

le

14. Contract Generator

Suggested file:

src/market/contract_generator.py

Interface
from datetime import date

def make_internal_contract_id(
    city_code: str,
    metric: str,
    comparison_operator: str,
    threshold_f: int,
    target_date: date,
) -> str:
    ...

def generate_contracts_for_surface(
    city_profile: CityProfile,
    metric: str,
    comparison_operator: str,
    target_date: date,
    probability_surface: dict[int, float],
) -> list[dict]:
    ...
Behavior
make_internal_contract_id

Expected format:

{CITY_CODE}_{METRIC}_{OP}_{THRESHOLD}_{YYYYMMDD}

generate_contracts_for_surface

Returns list of dicts containing at minimum:

contract_id

city

threshold_f

metric

comparison_operator

target_date

model_probability

15. Prediction Market API Integration

Suggested files:

src/market/kalshi_client.py

src/market/polymarket_client.py

Interface
def fetch_live_contracts(market_source: str) -> list[MarketContract]:
    ...

def fetch_orderbook_for_contract(
    market_source: str,
    contract_id: str,
) -> OrderBookSnapshot:
    ...
Behavior

supports paper/mock mode

raises RuntimeError on API failure

validates market source

16. Market Price Adapter

Suggested file:

src/market/market_price_adapter.py

Interface
def validate_orderbook(snapshot: OrderBookSnapshot) -> None:
    ...

def compute_mid_price(snapshot: OrderBookSnapshot) -> float:
    ...

def compute_market_probability(snapshot: OrderBookSnapshot) -> float:
    ...
Behavior

bid and ask must each be in [0, 1] if present

best_bid <= best_ask

midpoint is (bid + ask) / 2

17. Market Microstructure Scanner

Suggested file:

src/market/microstructure_scanner.py

Interface
def compute_spread(snapshot: OrderBookSnapshot) -> float:
    ...

def compute_liquidity_score(snapshot: OrderBookSnapshot) -> float:
    ...

def compute_staleness_seconds(
    snapshot: OrderBookSnapshot,
    now_utc,
) -> float:
    ...

def compute_microstructure_score(snapshot: OrderBookSnapshot, now_utc) -> float:
    ...
Behavior

spread may be null if bid/ask missing

score should increase with healthier liquidity or, depending on design, inefficiency

document score direction clearly

18. Market Lag Detection

Suggested file:

src/market/market_lag_detection.py

Interface
def detect_market_lag(
    shift_event: ForecastShiftEvent,
    market_probability_before: float,
    market_probability_after: float,
    min_expected_response: float = 0.05,
) -> bool:
    ...
Behavior

returns True when forecast shift is large but market reaction is small

intended as a simple first-pass lag detector

19. Distribution Tail Alpha

Suggested file:

src/trading/distribution_tail_alpha.py

Interface
def is_tail_threshold(
    ensemble_mean_f: float,
    threshold_f: int,
    adjusted_std_f: float,
    z_score_cutoff: float = 1.5,
) -> bool:
    ...

def compute_tail_alpha_boost(
    alpha: float,
    tail_multiplier: float,
    tail_flag: bool,
) -> float:
    ...
Behavior

identifies whether threshold lies in a distribution tail

optional score boost for ranking, not for raw alpha itself

20. Alpha Engine

Suggested file:

src/trading/alpha_engine.py

Interface
from datetime import datetime

def compute_alpha(model_probability: float, market_probability: float) -> float:
    ...

def classify_signal(alpha: float, threshold: float) -> str:
    ...

def build_trade_signal(
    timestamp_utc: datetime,
    contract: MarketContract,
    model_probability: float,
    market_probability: float,
    signal_threshold: float,
    confidence_score: float,
    tail_opportunity_flag: bool = False,
    forecast_shift_flag: bool = False,
    stale_market_flag: bool = False,
    microstructure_score: float | None = None,
    rank_score: float | None = None,
    model_version: str = "v1",
) -> TradeSignal:
    ...
Behavior
classify_signal

Rules:

alpha > threshold → BUY

alpha < -threshold → SELL

otherwise → HOLD

21. Market Scanner

Suggested file:

src/market/market_scanner.py

Interface
def evaluate_contract_opportunity(
    contract: MarketContract,
    probability_surface: dict[int, float],
    orderbook: OrderBookSnapshot,
    signal_threshold: float,
    timestamp_utc,
) -> TradeSignal:
    ...

def rank_trade_signals(signals: list[TradeSignal]) -> list[TradeSignal]:
    ...
Behavior
evaluate_contract_opportunity

gets threshold probability from surface

gets market probability from orderbook

builds a TradeSignal

rank_trade_signals

Default ranking suggestion:

descending by abs(alpha)

enhancements can incorporate rank_score

22. Risk Manager

Suggested file:

src/trading/risk_manager.py

Interface
def approve_trade(
    signal: TradeSignal,
    proposed_quantity: float,
    proposed_price: float,
    current_position: float,
    max_position_per_contract: float,
    daily_exposure_before: float,
    max_daily_exposure: float,
) -> RiskDecision:
    ...
Behavior

rejects HOLD

rejects if projected position exceeds limits

rejects if daily exposure would exceed max

otherwise approves

23. Trade Execution

Suggested file:

src/trading/trade_execution.py

Interface
def execute_paper_trade(
    signal: TradeSignal,
    quantity: float,
    price: float,
) -> ExecutedTrade:
    ...

def submit_live_trade(
    market_source: str,
    contract_id: str,
    side: str,
    quantity: float,
    price: float,
    order_type: str = "limit",
) -> ExecutedTrade:
    ...
Behavior

paper mode should work without exchange credentials

live mode may raise RuntimeError if API unavailable

every execution returns an ExecutedTrade

24. Backtest Engine

Suggested file:

src/research/backtest_engine.py

Interface
import pandas as pd

def simulate_backtest_trades(
    opportunities_df: pd.DataFrame,
    quantity: float,
) -> pd.DataFrame:
    ...

def summarize_backtest_results(trades_df: pd.DataFrame) -> dict:
    ...
Behavior
simulate_backtest_trades

Should return rows including:

contract_id

side

quantity

simulated_price

realized_pnl

summarize_backtest_results

Should return keys like:

total_trades

win_rate

gross_pnl

max_drawdown

25. Scheduler

Suggested file:

src/scheduler.py

Interface
def run_once(now_utc=None) -> list[TradeSignal]:
    ...

def run_loop(interval_seconds: int = 300) -> None:
    ...
Behavior
run_once

Should orchestrate:

forecast refresh if needed

market refresh

scanner evaluation

return ranked signals

run_loop

Should:

call run_once

sleep

log errors

continue unless fatal configuration issue occurs

Storage Interfaces
Feature Store

Suggested file:

src/data/feature_store.py

Interface
import pandas as pd

def save_dataframe(
    df: pd.DataFrame,
    dataset_name: str,
    partition_cols: list[str] | None = None,
    mode: str = "append",
) -> str:
    ...

def load_dataframe(
    dataset_name: str,
    filters: dict | None = None,
) -> pd.DataFrame:
    ...

def dataset_exists(dataset_name: str) -> bool:
    ...
Behavior

save parquet datasets

return path written

raise ValueError for invalid mode

use schemas from DATA_SCHEMA.md

Logging Interface

Suggested file:

src/core/logging_utils.py

Interface
def get_logger(name: str):
    ...
Behavior

returns module-specific logger

standardizes log formatting

Minimum Test Interfaces

For each MVP module, the AI agent should create at least:

Unit Tests

valid input case

invalid input case

edge case

schema/shape validation case

Example

For compute_alpha:

def test_compute_alpha_positive():
    assert compute_alpha(0.7, 0.5) == 0.2

def test_compute_alpha_negative():
    assert compute_alpha(0.3, 0.5) == -0.2
MVP Interface Checklist

These are the interfaces you should build first in Antigravity:

load_settings

load_city_profiles

fetch_station_observations

build_daily_settlement_observations

fetch_forecasts_for_city

compute_forecast_shift

compute_forecast_std

build_temperature_ensemble

probability_ge_normal

build_kde_distribution

generate_probability_surface_kde

make_internal_contract_id

fetch_live_contracts

fetch_orderbook_for_contract

compute_market_probability

compute_alpha

build_trade_signal

rank_trade_signals

approve_trade

execute_paper_trade

run_once

Build those first, then expand.

AI Agent Guidance

When building from this file:

do not rename interfaces without updating this document

do not widen signatures unnecessarily

do not merge unrelated interfaces into giant classes

prefer small testable functions

add docstrings and type hints

If an implementation needs a helper function, keep it private unless it is intended as a stable interface.