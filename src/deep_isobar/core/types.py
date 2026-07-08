"""Shared data models for Deep Isobar.

Canonical dataclasses used across all modules — forecasting, market data,
trading signals, and risk management.

These are pure data containers with no business logic.  Every field name,
type, and default value matches the spec in ``docs/INTERFACES.md``.

Example usage::

    from datetime import date, datetime
    from deep_isobar.core.types import CityProfile, ForecastPoint

    profile = CityProfile(
        city="New York",
        city_code="NYC",
        station_id="KNYC",
        timezone="America/New_York",
        settlement_source="NWS",
        variance_multiplier=1.2,
        mean_bias_correction_f=0.5,
        kde_bandwidth=0.8,
    )

    point = ForecastPoint(
        city="New York",
        station_id="KNYC",
        model_name="GFS",
        run_time_utc=datetime(2026, 3, 16, 12, 0, 0),
        target_date=date(2026, 3, 17),
        metric="high_temp_f",
        forecast_value_f=72.0,
        lead_hours=24,
        source_name="NOAA",
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Optional


# ── City & Forecast ──────────────────────────────────────────────────────


@dataclass
class CityProfile:
    """Profile for a tradable city in the Deep Isobar universe.

    Contains station metadata, bias-correction parameters, KDE tuning,
    and per-model ensemble weights.
    """

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
    model_weight_icon: Optional[float] = None
    model_weight_gem: Optional[float] = None
    heat_bias_adjustment_f: float = 0.0
    cold_bias_adjustment_f: float = 0.0
    sep_climate_normal_f: Optional[float] = None
    sep_anomaly_trigger_f: Optional[float] = None
    lead_decay_halflife_hours: float = 48.0
    # Multi-city fields — populated from cities.yaml; default to safe no-ops.
    active: bool = True
    # Data-only onboarding: active cities with trade=false get forecasts,
    # EMOS calibration, orderbook collection, and scorecard rows — but the
    # trading session never opens positions.  Flip to true once calibration
    # has matured AND the book's track record justifies expansion.
    trade: bool = True
    kalshi_series: str = ""
    # ACIS settlement station; may differ from METAR station_id (e.g. Dallas:
    # METAR=KDFW, ACIS/Kalshi settlement=CLIDFW).  Falls back to station_id
    # at runtime when empty.
    acis_station_id: str = ""
    nws_lat: float = 0.0
    nws_lon: float = 0.0


@dataclass
class ForecastPoint:
    """A single model forecast for a city/date/metric combination."""

    city: str
    station_id: str
    model_name: str
    run_time_utc: datetime
    target_date: date
    metric: str
    forecast_value_f: float
    lead_hours: int
    source_name: str


@dataclass
class ForecastShiftEvent:
    """Run-to-run change in a model forecast for the same target."""

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


# ── Ensemble & Distribution ──────────────────────────────────────────────


@dataclass
class EnsembleSummary:
    """Summary statistics from a multi-model temperature ensemble."""

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


@dataclass
class TemperatureDistributionSnapshot:
    """Snapshot of a KDE-based temperature distribution."""

    city: str
    target_date: date
    metric: str
    distribution_time_utc: datetime
    source_forecasts_f: List[float]
    kde_bandwidth: float
    min_temp_f: float
    max_temp_f: float
    methodology: str


# ── Market Data ──────────────────────────────────────────────────────────


@dataclass
class OrderBookSnapshot:
    """Point-in-time snapshot of a prediction-market order book."""

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


@dataclass
class MarketContract:
    """Represents a prediction-market contract."""

    contract_id: str
    market_source: str
    city: str
    metric: str
    comparison_operator: str
    threshold_f: int
    target_date: date
    settlement_source: str
    # Kalshi strike fields — populated from the live API; empty string / None
    # for stub contracts and legacy backtesting code that pre-dates this field.
    strike_type: str = ""           # "less" | "greater" | "between"
    floor_strike: Optional[int] = None   # lower boundary in °F (None for "less")
    cap_strike: Optional[int] = None     # upper boundary in °F (None for "greater")
    raw_title: Optional[str] = None
    listed_at_utc: Optional[datetime] = None
    expires_at_utc: Optional[datetime] = None
    active: bool = True


# ── Trading & Execution ──────────────────────────────────────────────────


@dataclass
class TradeSignal:
    """A scored trade opportunity produced by the alpha engine."""

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


@dataclass
class RiskDecision:
    """Risk-gate decision on a proposed trade."""

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


@dataclass
class ExecutedTrade:
    """Record of a trade submitted to or filled on an exchange."""

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
