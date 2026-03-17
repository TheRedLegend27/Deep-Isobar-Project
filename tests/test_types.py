"""Comprehensive tests for deep_isobar.core.types module.

Validates all 10 canonical dataclasses: instantiation with required fields,
default values, optional fields, and edge-case inputs.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from deep_isobar.core.types import (
    CityProfile,
    EnsembleSummary,
    ExecutedTrade,
    ForecastPoint,
    ForecastShiftEvent,
    MarketContract,
    OrderBookSnapshot,
    RiskDecision,
    TemperatureDistributionSnapshot,
    TradeSignal,
)


# ── Helpers ──────────────────────────────────────────────────────────────

NOW = datetime(2026, 3, 16, 12, 0, 0)
TODAY = date(2026, 3, 16)
TOMORROW = date(2026, 3, 17)


# ── CityProfile ─────────────────────────────────────────────────────────


class TestCityProfile:
    def test_instantiation(self):
        p = CityProfile(
            city="New York",
            city_code="NYC",
            station_id="KNYC",
            timezone="America/New_York",
            settlement_source="NWS",
            variance_multiplier=1.2,
            mean_bias_correction_f=0.5,
            kde_bandwidth=0.8,
        )
        assert p.city == "New York"
        assert p.city_code == "NYC"
        assert p.station_id == "KNYC"
        assert p.timezone == "America/New_York"
        assert p.settlement_source == "NWS"
        assert p.variance_multiplier == 1.2
        assert p.mean_bias_correction_f == 0.5
        assert p.kde_bandwidth == 0.8

    def test_defaults(self):
        p = CityProfile(
            city="Chicago",
            city_code="CHI",
            station_id="KORD",
            timezone="America/Chicago",
            settlement_source="NWS",
            variance_multiplier=1.0,
            mean_bias_correction_f=0.0,
            kde_bandwidth=1.0,
        )
        assert p.tail_multiplier == 1.0
        assert p.model_weight_gfs is None
        assert p.model_weight_ecmwf is None
        assert p.model_weight_nam is None
        assert p.heat_bias_adjustment_f == 0.0
        assert p.cold_bias_adjustment_f == 0.0

    def test_optional_weights(self):
        p = CityProfile(
            city="X",
            city_code="X",
            station_id="X",
            timezone="UTC",
            settlement_source="NWS",
            variance_multiplier=1.0,
            mean_bias_correction_f=0.0,
            kde_bandwidth=1.0,
            model_weight_gfs=0.5,
            model_weight_ecmwf=0.3,
            model_weight_nam=0.2,
        )
        assert p.model_weight_gfs == 0.5
        assert p.model_weight_ecmwf == 0.3
        assert p.model_weight_nam == 0.2

    def test_edge_zero_values(self):
        p = CityProfile(
            city="",
            city_code="",
            station_id="",
            timezone="",
            settlement_source="",
            variance_multiplier=0.0,
            mean_bias_correction_f=0.0,
            kde_bandwidth=0.0,
        )
        assert p.city == ""
        assert p.variance_multiplier == 0.0

    def test_negative_bias(self):
        p = CityProfile(
            city="Test",
            city_code="T",
            station_id="T",
            timezone="UTC",
            settlement_source="NWS",
            variance_multiplier=1.0,
            mean_bias_correction_f=-1.5,
            kde_bandwidth=0.5,
            heat_bias_adjustment_f=-0.3,
            cold_bias_adjustment_f=-0.7,
        )
        assert p.mean_bias_correction_f == -1.5
        assert p.heat_bias_adjustment_f == -0.3
        assert p.cold_bias_adjustment_f == -0.7


# ── ForecastPoint ────────────────────────────────────────────────────────


class TestForecastPoint:
    def test_instantiation(self):
        fp = ForecastPoint(
            city="New York",
            station_id="KNYC",
            model_name="GFS",
            run_time_utc=NOW,
            target_date=TOMORROW,
            metric="high_temp_f",
            forecast_value_f=72.0,
            lead_hours=24,
            source_name="NOAA",
        )
        assert fp.city == "New York"
        assert fp.model_name == "GFS"
        assert fp.run_time_utc == NOW
        assert fp.target_date == TOMORROW
        assert fp.metric == "high_temp_f"
        assert fp.forecast_value_f == 72.0
        assert fp.lead_hours == 24

    def test_negative_forecast(self):
        fp = ForecastPoint(
            city="Anchorage",
            station_id="PANC",
            model_name="ECMWF",
            run_time_utc=NOW,
            target_date=TOMORROW,
            metric="low_temp_f",
            forecast_value_f=-15.0,
            lead_hours=48,
            source_name="ECMWF",
        )
        assert fp.forecast_value_f == -15.0

    def test_zero_lead_hours(self):
        fp = ForecastPoint(
            city="X",
            station_id="X",
            model_name="GFS",
            run_time_utc=NOW,
            target_date=TODAY,
            metric="high_temp_f",
            forecast_value_f=70.0,
            lead_hours=0,
            source_name="NOAA",
        )
        assert fp.lead_hours == 0


# ── ForecastShiftEvent ───────────────────────────────────────────────────


class TestForecastShiftEvent:
    def test_instantiation(self):
        prev_run = datetime(2026, 3, 16, 6, 0, 0)
        curr_run = datetime(2026, 3, 16, 12, 0, 0)
        ev = ForecastShiftEvent(
            city="New York",
            model_name="GFS",
            metric="high_temp_f",
            previous_run_time_utc=prev_run,
            current_run_time_utc=curr_run,
            target_date=TOMORROW,
            previous_value_f=70.0,
            current_value_f=73.0,
            shift_f=3.0,
            absolute_shift_f=3.0,
            shift_direction="up",
            significant_shift_flag=True,
        )
        assert ev.city == "New York"
        assert ev.shift_f == 3.0
        assert ev.absolute_shift_f == 3.0
        assert ev.shift_direction == "up"
        assert ev.significant_shift_flag is True

    def test_negative_shift(self):
        ev = ForecastShiftEvent(
            city="Chicago",
            model_name="ECMWF",
            metric="high_temp_f",
            previous_run_time_utc=NOW,
            current_run_time_utc=NOW,
            target_date=TOMORROW,
            previous_value_f=80.0,
            current_value_f=75.0,
            shift_f=-5.0,
            absolute_shift_f=5.0,
            shift_direction="down",
            significant_shift_flag=True,
        )
        assert ev.shift_f == -5.0
        assert ev.shift_direction == "down"

    def test_zero_shift(self):
        ev = ForecastShiftEvent(
            city="X",
            model_name="GFS",
            metric="low_temp_f",
            previous_run_time_utc=NOW,
            current_run_time_utc=NOW,
            target_date=TODAY,
            previous_value_f=50.0,
            current_value_f=50.0,
            shift_f=0.0,
            absolute_shift_f=0.0,
            shift_direction="none",
            significant_shift_flag=False,
        )
        assert ev.shift_f == 0.0
        assert ev.significant_shift_flag is False


# ── EnsembleSummary ──────────────────────────────────────────────────────


class TestEnsembleSummary:
    def test_instantiation(self):
        es = EnsembleSummary(
            city="New York",
            target_date=TOMORROW,
            metric="high_temp_f",
            ensemble_run_time_utc=NOW,
            contributing_models=["GFS", "ECMWF", "NAM"],
            model_count=3,
            ensemble_mean_f=72.5,
            ensemble_std_f=2.3,
            variance_multiplier=1.2,
            adjusted_std_f=2.76,
            bias_corrected_mean_f=73.0,
            methodology="weighted_mean",
        )
        assert es.city == "New York"
        assert es.contributing_models == ["GFS", "ECMWF", "NAM"]
        assert es.model_count == 3
        assert es.ensemble_mean_f == 72.5
        assert es.methodology == "weighted_mean"

    def test_single_model(self):
        es = EnsembleSummary(
            city="X",
            target_date=TODAY,
            metric="low_temp_f",
            ensemble_run_time_utc=NOW,
            contributing_models=["GFS"],
            model_count=1,
            ensemble_mean_f=60.0,
            ensemble_std_f=0.0,
            variance_multiplier=1.0,
            adjusted_std_f=0.0,
            bias_corrected_mean_f=60.0,
            methodology="single_model",
        )
        assert es.model_count == 1
        assert es.contributing_models == ["GFS"]

    def test_empty_models_list(self):
        es = EnsembleSummary(
            city="X",
            target_date=TODAY,
            metric="high_temp_f",
            ensemble_run_time_utc=NOW,
            contributing_models=[],
            model_count=0,
            ensemble_mean_f=0.0,
            ensemble_std_f=0.0,
            variance_multiplier=1.0,
            adjusted_std_f=0.0,
            bias_corrected_mean_f=0.0,
            methodology="none",
        )
        assert es.contributing_models == []
        assert es.model_count == 0


# ── TemperatureDistributionSnapshot ──────────────────────────────────────


class TestTemperatureDistributionSnapshot:
    def test_instantiation(self):
        snap = TemperatureDistributionSnapshot(
            city="New York",
            target_date=TOMORROW,
            metric="high_temp_f",
            distribution_time_utc=NOW,
            source_forecasts_f=[70.0, 72.0, 74.0],
            kde_bandwidth=0.8,
            min_temp_f=60.0,
            max_temp_f=85.0,
            methodology="gaussian_kde",
        )
        assert snap.city == "New York"
        assert snap.source_forecasts_f == [70.0, 72.0, 74.0]
        assert snap.kde_bandwidth == 0.8
        assert snap.min_temp_f == 60.0
        assert snap.max_temp_f == 85.0

    def test_single_forecast_value(self):
        snap = TemperatureDistributionSnapshot(
            city="X",
            target_date=TODAY,
            metric="low_temp_f",
            distribution_time_utc=NOW,
            source_forecasts_f=[65.0],
            kde_bandwidth=1.0,
            min_temp_f=50.0,
            max_temp_f=80.0,
            methodology="fallback",
        )
        assert len(snap.source_forecasts_f) == 1

    def test_negative_temperatures(self):
        snap = TemperatureDistributionSnapshot(
            city="Anchorage",
            target_date=TODAY,
            metric="low_temp_f",
            distribution_time_utc=NOW,
            source_forecasts_f=[-10.0, -5.0, -15.0],
            kde_bandwidth=0.5,
            min_temp_f=-30.0,
            max_temp_f=10.0,
            methodology="gaussian_kde",
        )
        assert snap.min_temp_f == -30.0


# ── OrderBookSnapshot ────────────────────────────────────────────────────


class TestOrderBookSnapshot:
    def test_instantiation(self):
        ob = OrderBookSnapshot(
            timestamp_utc=NOW,
            contract_id="NYC_HIGH_GE_75_20260317",
            market_source="Kalshi",
            best_bid=0.45,
            best_ask=0.50,
        )
        assert ob.contract_id == "NYC_HIGH_GE_75_20260317"
        assert ob.best_bid == 0.45
        assert ob.best_ask == 0.50

    def test_defaults(self):
        ob = OrderBookSnapshot(
            timestamp_utc=NOW,
            contract_id="X",
            market_source="Kalshi",
            best_bid=0.40,
            best_ask=0.60,
        )
        assert ob.last_trade_price is None
        assert ob.bid_size is None
        assert ob.ask_size is None
        assert ob.volume_24h is None
        assert ob.open_interest is None

    def test_all_optional_fields(self):
        ob = OrderBookSnapshot(
            timestamp_utc=NOW,
            contract_id="X",
            market_source="Polymarket",
            best_bid=0.30,
            best_ask=0.35,
            last_trade_price=0.32,
            bid_size=100.0,
            ask_size=200.0,
            volume_24h=5000.0,
            open_interest=12000.0,
        )
        assert ob.last_trade_price == 0.32
        assert ob.volume_24h == 5000.0

    def test_none_bid_ask(self):
        ob = OrderBookSnapshot(
            timestamp_utc=NOW,
            contract_id="X",
            market_source="Kalshi",
            best_bid=None,
            best_ask=None,
        )
        assert ob.best_bid is None
        assert ob.best_ask is None


# ── MarketContract ───────────────────────────────────────────────────────


class TestMarketContract:
    def test_instantiation(self):
        mc = MarketContract(
            contract_id="NYC_HIGH_GE_75_20260317",
            market_source="Kalshi",
            city="New York",
            metric="high_temp_f",
            comparison_operator="ge",
            threshold_f=75,
            target_date=TOMORROW,
            settlement_source="NWS",
        )
        assert mc.contract_id == "NYC_HIGH_GE_75_20260317"
        assert mc.threshold_f == 75
        assert mc.comparison_operator == "ge"

    def test_defaults(self):
        mc = MarketContract(
            contract_id="X",
            market_source="Kalshi",
            city="X",
            metric="high_temp_f",
            comparison_operator="ge",
            threshold_f=70,
            target_date=TODAY,
            settlement_source="NWS",
        )
        assert mc.raw_title is None
        assert mc.listed_at_utc is None
        assert mc.expires_at_utc is None
        assert mc.active is True

    def test_inactive_contract(self):
        mc = MarketContract(
            contract_id="X",
            market_source="Kalshi",
            city="X",
            metric="low_temp_f",
            comparison_operator="le",
            threshold_f=32,
            target_date=TODAY,
            settlement_source="NWS",
            active=False,
        )
        assert mc.active is False

    def test_optional_timestamps(self):
        listed = datetime(2026, 3, 10, 8, 0, 0)
        expires = datetime(2026, 3, 18, 0, 0, 0)
        mc = MarketContract(
            contract_id="X",
            market_source="Polymarket",
            city="Chicago",
            metric="high_temp_f",
            comparison_operator="ge",
            threshold_f=80,
            target_date=TOMORROW,
            settlement_source="NWS",
            raw_title="Will Chicago high temp be >= 80F?",
            listed_at_utc=listed,
            expires_at_utc=expires,
        )
        assert mc.raw_title == "Will Chicago high temp be >= 80F?"
        assert mc.listed_at_utc == listed
        assert mc.expires_at_utc == expires


# ── TradeSignal ──────────────────────────────────────────────────────────


class TestTradeSignal:
    def test_instantiation(self):
        ts = TradeSignal(
            timestamp_utc=NOW,
            contract_id="NYC_HIGH_GE_75_20260317",
            city="New York",
            target_date=TOMORROW,
            metric="high_temp_f",
            threshold_f=75,
            comparison_operator="ge",
            market_probability=0.45,
            model_probability=0.60,
            alpha=0.15,
            absolute_alpha=0.15,
            signal_side="BUY",
            confidence_score=0.82,
        )
        assert ts.alpha == 0.15
        assert ts.signal_side == "BUY"
        assert ts.confidence_score == 0.82

    def test_defaults(self):
        ts = TradeSignal(
            timestamp_utc=NOW,
            contract_id="X",
            city="X",
            target_date=TODAY,
            metric="high_temp_f",
            threshold_f=70,
            comparison_operator="ge",
            market_probability=0.50,
            model_probability=0.50,
            alpha=0.0,
            absolute_alpha=0.0,
            signal_side="HOLD",
            confidence_score=0.0,
        )
        assert ts.tail_opportunity_flag is False
        assert ts.forecast_shift_flag is False
        assert ts.stale_market_flag is False
        assert ts.microstructure_score is None
        assert ts.rank_score is None
        assert ts.model_version == "v1"

    def test_all_flags_set(self):
        ts = TradeSignal(
            timestamp_utc=NOW,
            contract_id="X",
            city="X",
            target_date=TODAY,
            metric="low_temp_f",
            threshold_f=32,
            comparison_operator="le",
            market_probability=0.80,
            model_probability=0.40,
            alpha=-0.40,
            absolute_alpha=0.40,
            signal_side="SELL",
            confidence_score=0.90,
            tail_opportunity_flag=True,
            forecast_shift_flag=True,
            stale_market_flag=True,
            microstructure_score=0.75,
            rank_score=0.95,
            model_version="v2",
        )
        assert ts.tail_opportunity_flag is True
        assert ts.forecast_shift_flag is True
        assert ts.stale_market_flag is True
        assert ts.microstructure_score == 0.75
        assert ts.rank_score == 0.95
        assert ts.model_version == "v2"

    def test_negative_alpha(self):
        ts = TradeSignal(
            timestamp_utc=NOW,
            contract_id="X",
            city="X",
            target_date=TODAY,
            metric="high_temp_f",
            threshold_f=70,
            comparison_operator="ge",
            market_probability=0.60,
            model_probability=0.40,
            alpha=-0.20,
            absolute_alpha=0.20,
            signal_side="SELL",
            confidence_score=0.5,
        )
        assert ts.alpha == -0.20
        assert ts.absolute_alpha == 0.20


# ── RiskDecision ─────────────────────────────────────────────────────────


class TestRiskDecision:
    def test_instantiation_approved(self):
        rd = RiskDecision(
            timestamp_utc=NOW,
            contract_id="NYC_HIGH_GE_75_20260317",
            proposed_side="BUY",
            proposed_quantity=10.0,
            proposed_price=0.45,
            approved_flag=True,
        )
        assert rd.approved_flag is True
        assert rd.proposed_side == "BUY"
        assert rd.proposed_quantity == 10.0

    def test_defaults(self):
        rd = RiskDecision(
            timestamp_utc=NOW,
            contract_id="X",
            proposed_side="BUY",
            proposed_quantity=1.0,
            proposed_price=0.50,
            approved_flag=True,
        )
        assert rd.rejection_reason is None
        assert rd.current_position is None
        assert rd.projected_position is None
        assert rd.daily_exposure_before is None
        assert rd.daily_exposure_after is None

    def test_rejected_with_reason(self):
        rd = RiskDecision(
            timestamp_utc=NOW,
            contract_id="X",
            proposed_side="BUY",
            proposed_quantity=100.0,
            proposed_price=0.90,
            approved_flag=False,
            rejection_reason="max daily exposure exceeded",
            current_position=50.0,
            projected_position=150.0,
            daily_exposure_before=500.0,
            daily_exposure_after=590.0,
        )
        assert rd.approved_flag is False
        assert rd.rejection_reason == "max daily exposure exceeded"
        assert rd.daily_exposure_after == 590.0

    def test_zero_quantity(self):
        rd = RiskDecision(
            timestamp_utc=NOW,
            contract_id="X",
            proposed_side="SELL",
            proposed_quantity=0.0,
            proposed_price=0.0,
            approved_flag=False,
            rejection_reason="zero quantity",
        )
        assert rd.proposed_quantity == 0.0


# ── ExecutedTrade ────────────────────────────────────────────────────────


class TestExecutedTrade:
    def test_instantiation(self):
        et = ExecutedTrade(
            timestamp_utc=NOW,
            trade_id="trade_001",
            contract_id="NYC_HIGH_GE_75_20260317",
            market_source="Kalshi",
            side="BUY",
            quantity=10.0,
            price=0.45,
            order_type="limit",
            execution_status="filled",
        )
        assert et.trade_id == "trade_001"
        assert et.side == "BUY"
        assert et.quantity == 10.0
        assert et.execution_status == "filled"

    def test_defaults(self):
        et = ExecutedTrade(
            timestamp_utc=NOW,
            trade_id="t1",
            contract_id="X",
            market_source="Kalshi",
            side="BUY",
            quantity=1.0,
            price=0.50,
            order_type="market",
            execution_status="pending",
        )
        assert et.fill_quantity is None
        assert et.avg_fill_price is None
        assert et.exchange_order_id is None
        assert et.paper_trade_flag is True
        assert et.notes is None

    def test_live_trade(self):
        et = ExecutedTrade(
            timestamp_utc=NOW,
            trade_id="t2",
            contract_id="X",
            market_source="Kalshi",
            side="SELL",
            quantity=5.0,
            price=0.80,
            order_type="limit",
            execution_status="filled",
            fill_quantity=5.0,
            avg_fill_price=0.79,
            exchange_order_id="EX_12345",
            paper_trade_flag=False,
            notes="real money trade",
        )
        assert et.paper_trade_flag is False
        assert et.fill_quantity == 5.0
        assert et.exchange_order_id == "EX_12345"
        assert et.notes == "real money trade"

    def test_partial_fill(self):
        et = ExecutedTrade(
            timestamp_utc=NOW,
            trade_id="t3",
            contract_id="X",
            market_source="Polymarket",
            side="BUY",
            quantity=10.0,
            price=0.45,
            order_type="limit",
            execution_status="partial",
            fill_quantity=3.0,
            avg_fill_price=0.44,
        )
        assert et.execution_status == "partial"
        assert et.fill_quantity == 3.0
