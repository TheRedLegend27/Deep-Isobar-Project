"""Comprehensive tests for deep_isobar.data.weather_ingest module."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from deep_isobar.data.weather_ingest import (
    build_daily_settlement_observations,
    fetch_station_observations,
    normalize_weather_observations,
)


# ═══════════════════════════════════════════════════════════════════════════
# fetch_station_observations
# ═══════════════════════════════════════════════════════════════════════════


class TestFetchStationObservations:
    """Tests for the deterministic stub fetch function."""

    # ── Happy path ────────────────────────────────────────────────────

    def test_single_day_returns_24_rows(self):
        """A single-day range should produce 24 hourly rows."""
        df = fetch_station_observations("KORD", date(2026, 7, 4), date(2026, 7, 4))
        assert len(df) == 24

    def test_multi_day_returns_correct_row_count(self):
        """Two full days should produce 48 rows."""
        df = fetch_station_observations("KORD", date(2026, 7, 4), date(2026, 7, 5))
        assert len(df) == 48

    def test_output_columns(self):
        """Output must contain the four canonical columns."""
        df = fetch_station_observations("KORD", date(2026, 7, 4), date(2026, 7, 4))
        expected = {"timestamp_utc", "station_id", "source_name", "temperature_f"}
        assert set(df.columns) == expected

    def test_station_id_propagated(self):
        """The station_id argument should appear in every row."""
        df = fetch_station_observations("KJFK", date(2026, 1, 1), date(2026, 1, 1))
        assert (df["station_id"] == "KJFK").all()

    def test_source_name_default(self):
        """Default source_name should be NWS."""
        df = fetch_station_observations("KORD", date(2026, 7, 4), date(2026, 7, 4))
        assert (df["source_name"] == "NWS").all()

    def test_custom_source_name(self):
        """A custom source_name should be propagated."""
        df = fetch_station_observations(
            "KORD", date(2026, 7, 4), date(2026, 7, 4), source_name="ERA5"
        )
        assert (df["source_name"] == "ERA5").all()

    def test_deterministic(self):
        """Two calls with the same args should return identical data."""
        args = ("KORD", date(2026, 7, 4), date(2026, 7, 4))
        df1 = fetch_station_observations(*args)
        df2 = fetch_station_observations(*args)
        pd.testing.assert_frame_equal(df1, df2)

    def test_temperature_values_are_floats(self):
        """Temperature values should be numeric floats."""
        df = fetch_station_observations("KORD", date(2026, 7, 4), date(2026, 7, 4))
        assert df["temperature_f"].dtype in ("float64", "float32")

    # ── Error cases ───────────────────────────────────────────────────

    def test_empty_station_raises(self):
        with pytest.raises(ValueError, match="station_id must not be empty"):
            fetch_station_observations("", date(2026, 7, 4), date(2026, 7, 4))

    def test_blank_station_raises(self):
        with pytest.raises(ValueError, match="station_id must not be empty"):
            fetch_station_observations("   ", date(2026, 7, 4), date(2026, 7, 4))

    def test_non_string_station_raises(self):
        with pytest.raises(ValueError, match="station_id must be a string"):
            fetch_station_observations(123, date(2026, 7, 4), date(2026, 7, 4))  # type: ignore[arg-type]

    def test_reversed_dates_raises(self):
        with pytest.raises(ValueError, match="start_date.*must not be after"):
            fetch_station_observations("KORD", date(2026, 7, 5), date(2026, 7, 4))

    def test_empty_source_name_raises(self):
        with pytest.raises(ValueError, match="source_name must be a non-empty"):
            fetch_station_observations(
                "KORD", date(2026, 7, 4), date(2026, 7, 4), source_name=""
            )


# ═══════════════════════════════════════════════════════════════════════════
# normalize_weather_observations
# ═══════════════════════════════════════════════════════════════════════════


class TestNormalizeWeatherObservations:
    """Tests for the normalization function."""

    @pytest.fixture()
    def raw_df(self) -> pd.DataFrame:
        """Minimal valid raw DataFrame."""
        return pd.DataFrame(
            [
                {
                    "timestamp_utc": "2026-07-04T12:00:00Z",
                    "station_id": "KORD",
                    "source_name": "NWS",
                    "temperature_f": 88.0,
                }
            ]
        )

    # ── Happy path ────────────────────────────────────────────────────

    def test_passthrough(self, raw_df: pd.DataFrame):
        """A well-formed DF should pass through with canonical columns."""
        result = normalize_weather_observations(raw_df)
        assert list(result.columns) == [
            "timestamp_utc",
            "station_id",
            "source_name",
            "temperature_f",
        ]

    def test_timestamp_converted_to_datetime(self, raw_df: pd.DataFrame):
        """timestamp_utc should be a datetime after normalization."""
        result = normalize_weather_observations(raw_df)
        assert pd.api.types.is_datetime64_any_dtype(result["timestamp_utc"])

    def test_celsius_conversion(self):
        """If only temperature_c is present, it should be converted to F."""
        df = pd.DataFrame(
            [
                {
                    "timestamp_utc": "2026-07-04T12:00:00Z",
                    "station_id": "KORD",
                    "source_name": "NWS",
                    "temperature_c": 0.0,
                }
            ]
        )
        result = normalize_weather_observations(df)
        assert result["temperature_f"].iloc[0] == pytest.approx(32.0)

    def test_celsius_conversion_100(self):
        """100°C should convert to 212°F."""
        df = pd.DataFrame(
            [
                {
                    "timestamp_utc": "2026-07-04T12:00:00Z",
                    "station_id": "KORD",
                    "source_name": "NWS",
                    "temperature_c": 100.0,
                }
            ]
        )
        result = normalize_weather_observations(df)
        assert result["temperature_f"].iloc[0] == pytest.approx(212.0)

    def test_extra_columns_dropped(self):
        """Non-canonical columns should not appear in the output."""
        df = pd.DataFrame(
            [
                {
                    "timestamp_utc": "2026-07-04T12:00:00Z",
                    "station_id": "KORD",
                    "source_name": "NWS",
                    "temperature_f": 88.0,
                    "humidity_pct": 45.0,
                }
            ]
        )
        result = normalize_weather_observations(df)
        assert "humidity_pct" not in result.columns

    # ── Error cases ───────────────────────────────────────────────────

    def test_empty_df_raises(self):
        with pytest.raises(ValueError, match="empty DataFrame"):
            normalize_weather_observations(pd.DataFrame())

    def test_missing_columns_raises(self):
        df = pd.DataFrame([{"timestamp_utc": "2026-07-04T12:00:00Z"}])
        with pytest.raises(ValueError, match="Missing required columns"):
            normalize_weather_observations(df)

    def test_no_temperature_column_raises(self):
        """Raises when neither temperature_f nor temperature_c is present."""
        df = pd.DataFrame(
            [
                {
                    "timestamp_utc": "2026-07-04T12:00:00Z",
                    "station_id": "KORD",
                    "source_name": "NWS",
                }
            ]
        )
        with pytest.raises(ValueError, match="temperature_f.*temperature_c"):
            normalize_weather_observations(df)

    def test_non_dataframe_raises(self):
        with pytest.raises(ValueError, match="pandas DataFrame"):
            normalize_weather_observations("not a dataframe")  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════
# build_daily_settlement_observations
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildDailySettlementObservations:
    """Tests for the daily settlement aggregation function."""

    @pytest.fixture()
    def hourly_df(self) -> pd.DataFrame:
        """Two days of hourly observations (simple)."""
        rows = []
        for day in (date(2026, 7, 4), date(2026, 7, 5)):
            for hour in range(24):
                rows.append(
                    {
                        "timestamp_utc": f"{day.isoformat()}T{hour:02d}:00:00Z",
                        "station_id": "KORD",
                        "source_name": "NWS",
                        "temperature_f": 60.0 + hour,  # 60–83
                    }
                )
        return pd.DataFrame(rows)

    # ── Happy path ────────────────────────────────────────────────────

    def test_two_day_output_rows(self, hourly_df: pd.DataFrame):
        result = build_daily_settlement_observations(hourly_df, "Chicago", "KORD")
        assert len(result) == 2

    def test_output_columns(self, hourly_df: pd.DataFrame):
        result = build_daily_settlement_observations(hourly_df, "Chicago", "KORD")
        expected = [
            "city",
            "station_id",
            "target_date",
            "high_temp_f",
            "low_temp_f",
            "settlement_source",
        ]
        assert list(result.columns) == expected

    def test_high_low_values(self, hourly_df: pd.DataFrame):
        """High should be 83 (hour 23) and low should be 60 (hour 0)."""
        result = build_daily_settlement_observations(hourly_df, "Chicago", "KORD")
        row = result.iloc[0]
        assert row["high_temp_f"] == 83.0
        assert row["low_temp_f"] == 60.0

    def test_city_propagated(self, hourly_df: pd.DataFrame):
        result = build_daily_settlement_observations(hourly_df, "New York", "KJFK")
        assert (result["city"] == "New York").all()

    def test_station_propagated(self, hourly_df: pd.DataFrame):
        result = build_daily_settlement_observations(hourly_df, "Chicago", "KORD")
        assert (result["station_id"] == "KORD").all()

    def test_default_settlement_source(self, hourly_df: pd.DataFrame):
        result = build_daily_settlement_observations(hourly_df, "Chicago", "KORD")
        assert (result["settlement_source"] == "NWS").all()

    def test_custom_settlement_source(self, hourly_df: pd.DataFrame):
        result = build_daily_settlement_observations(
            hourly_df, "Chicago", "KORD", settlement_source="ERA5"
        )
        assert (result["settlement_source"] == "ERA5").all()

    def test_single_observation(self):
        """A single row should produce one settlement day with high == low."""
        df = pd.DataFrame(
            [
                {
                    "timestamp_utc": "2026-07-04T15:00:00Z",
                    "station_id": "KORD",
                    "temperature_f": 90.0,
                }
            ]
        )
        result = build_daily_settlement_observations(df, "Chicago", "KORD")
        assert len(result) == 1
        assert result.iloc[0]["high_temp_f"] == 90.0
        assert result.iloc[0]["low_temp_f"] == 90.0

    # ── Integration with fetch ────────────────────────────────────────

    def test_end_to_end_with_fetch(self):
        """Fetch → normalize → settle should produce valid output."""
        raw = fetch_station_observations("KORD", date(2026, 7, 4), date(2026, 7, 5))
        norm = normalize_weather_observations(raw)
        result = build_daily_settlement_observations(norm, "Chicago", "KORD")
        assert len(result) == 2
        for col in ("city", "station_id", "target_date", "high_temp_f", "low_temp_f"):
            assert col in result.columns

    # ── Error cases ───────────────────────────────────────────────────

    def test_empty_df_raises(self):
        with pytest.raises(ValueError, match="empty DataFrame"):
            build_daily_settlement_observations(pd.DataFrame(), "Chicago", "KORD")

    def test_missing_temperature_column_raises(self):
        df = pd.DataFrame(
            [{"timestamp_utc": "2026-07-04T12:00:00Z", "station_id": "KORD"}]
        )
        with pytest.raises(ValueError, match="temperature_f column is required"):
            build_daily_settlement_observations(df, "Chicago", "KORD")

    def test_missing_timestamp_column_raises(self):
        df = pd.DataFrame([{"temperature_f": 90.0}])
        with pytest.raises(ValueError, match="timestamp_utc column is required"):
            build_daily_settlement_observations(df, "Chicago", "KORD")

    def test_empty_city_raises(self):
        df = pd.DataFrame(
            [{"timestamp_utc": "2026-07-04T12:00:00Z", "temperature_f": 90.0}]
        )
        with pytest.raises(ValueError, match="city must be a non-empty"):
            build_daily_settlement_observations(df, "", "KORD")

    def test_blank_station_raises(self):
        df = pd.DataFrame(
            [{"timestamp_utc": "2026-07-04T12:00:00Z", "temperature_f": 90.0}]
        )
        with pytest.raises(ValueError, match="station_id must be a non-empty"):
            build_daily_settlement_observations(df, "Chicago", "   ")

    def test_non_dataframe_raises(self):
        with pytest.raises(ValueError, match="pandas DataFrame"):
            build_daily_settlement_observations(
                [1, 2, 3], "Chicago", "KORD"  # type: ignore[arg-type]
            )
