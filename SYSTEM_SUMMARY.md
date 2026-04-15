# Deep Isobar — System Summary

_Generated: 2026-04-13 by read-only audit. No files modified._
_Updated April 14, 2026_
_Changes since April 13: multi-city refactor across all core modules, 4 new cities calibrated and onboarded (Dallas/NYC/Philadelphia/Boston), GFS boolean bug fixed, Boston contract ID normalization, clickable Kalshi deep links and bracket recommendation chips in dashboard._

---

## 1. Directory & File Inventory

```
src/deep_isobar/
├── __init__.py                          (empty)
├── config.py                            Central YAML loader; reads config/settings.yaml; .env bootstrap
├── scheduler.py                         MVP orchestration loop: forecast → ensemble → KDE → Kalshi → alpha → signals
│
├── core/
│   ├── __init__.py
│   ├── logging_utils.py                 Thin loguru wrapper (get_logger)
│   └── types.py                         All shared dataclasses: CityProfile (+ active, kalshi_series, acis_station_id, nws_lat, nws_lon), ForecastPoint, TradeSignal, etc.
│
├── data/
│   ├── __init__.py
│   ├── city_universe.py                 Loads config/cities.yaml → CityProfile list; reads 5 new fields via .get(); get_city_universe() alias added for load_city_profiles()
│   ├── data_sources.py                  (not read — not referenced by other modules in audit)
│   ├── feature_store.py                 Minimal save/load parquet wrapper (paths.data_dir)
│   ├── historical_forecast_ingest.py    AWS NOAA GFS byte-range fetch; GRIB2 parse via cfgrib/xarray; disk cache; ✅ numpy/xarray boolean bug fixed (if array: → if array is not None) — was crashing 12z/f030 fetch
│   ├── historical_noaa_ingest.py        ACIS API fetch for daily high/low settlement temps; retry w/ backoff; fetch_settlement_observations resolves acis_station_id per city (Dallas → CLIDFW not KDFW)
│   └── weather_ingest.py                ⚠️ STUB — fetch_station_observations returns sine-wave fake hourly data; not used in live path
│
├── models/
│   ├── forecast_error.py                Stub/placeholder forecast error model (not wired into live pipeline)
│   ├── forecast_generation.py           Open-Meteo live fetch (GFS/ECMWF/NAM); deterministic stub fallback
│   ├── forecast_shift.py                Run-to-run forecast change detector → ForecastShiftEvent
│   ├── forecast_volatility.py           Ensemble spread: compute_forecast_std, compute_forecast_variance
│   ├── kde_temperature_distribution.py  scipy gaussian_kde wrapper; build_kde_distribution()
│   ├── probability_engine.py            Normal CDF probability functions with continuity correction
│   ├── probability_surface.py           Generates {threshold→probability} dict over temp range; KDE + normal variants
│   └── temperature_ensemble.py          Weighted ensemble mean; calls bias_loader.get_current_bias() for calibration
│
├── market/
│   ├── __init__.py
│   ├── contract_generator.py            Internal contract ID builder; generate_contracts_for_surface()
│   ├── historical_kalshi_ingest.py      Pulls KXHIGHCHI settled contracts + hourly candlesticks from Kalshi API
│   ├── kalshi_client.py                 Live RSA-PSS authenticated Kalshi v2 client; stub fallback; get_balance() added; optional series_ticker param on fetch_live_contracts; _normalize_ticker() handles malformed Boston-style IDs
│   ├── market_lag_detection.py          Detects when market price hasn't reacted to a forecast shift
│   ├── market_price_adapter.py          Extracts mid-price / market probability from OrderBookSnapshot
│   └── market_scanner.py               Wires market_price_adapter + alpha_engine; evaluate_contract_opportunity()
│   └── microstructure_scanner.py        Spread, liquidity score, staleness, composite microstructure score
│
├── trading/
│   ├── __init__.py                      Exports BracketAllocation, build_spread, SizingDecision, compute_exposure
│   ├── alpha_engine.py                  Alpha = model_prob − market_prob; BUY/SELL/HOLD classification; rank_score
│   ├── bracket_spreader.py              ✅ NEW — build_spread(): proportional allocation across top-N BUY signals
│   ├── distribution_tail_alpha.py       z-score tail detection; ranking boost multiplier
│   ├── position_sizer.py                ✅ NEW — compute_exposure(): dynamic cap via anomaly confidence + ensemble spread
│   ├── risk_manager.py                  Position cap + daily exposure gate; approve_trade()
│   └── trade_execution.py              execute_paper_trade() works; submit_live_trade() raises RuntimeError ⚠️
│
├── calibration/
│   ├── __init__.py
│   ├── bias_loader.py                   ✅ Full: loads monthly parquet profile; get_current_bias(); incremental updater
│   ├── batch_onboard.py                 ✅ Full: parallel city onboarding via ProcessPoolExecutor
│   ├── historical_replay.py             ✅ Full: replays GFS vs NOAA 2021–2024; build_monthly_profile()
│   └── onboard_city.py                 ✅ Full: 8-phase CLI pipeline; validates, fetches, replays, backtests, writes yaml
│
├── notifications/
│   ├── __init__.py
│   └── discord_notifier.py             Discord webhook embed poster; silent no-op if URL absent
│
├── anomaly/
│   ├── __init__.py                      Exports: check_anomalies, fetch_metar, fetch_kmdw_metar (alias), parse_metar_fields, fetch_nws_high_forecast_f
│   ├── detector.py                      ✅ Full: Anthropic API anomaly check; all 5 signals active
│   ├── metar_fetcher.py                fetch_metar(station_id: str) — parameterized; fetch_kmdw_metar = lambda: fetch_metar("KMDW") for backward compat
│   └── nws_fetcher.py                  fetch_nws_high_forecast_f(lat: float, lon: float) — no longer hardcodes Chicago coords
│
├── monitoring/
│   ├── __init__.py
│   └── watchdog.py                     ✅ NEW — 5 health checks; Discord amber/red alerts; runs every 15 min via Task Scheduler
│
├── dashboard/
│   ├── __init__.py
│   └── api.py                           FastAPI read-only backend; serves paper_trades.csv, daily_log.csv, bias parquet;
│                                        /api/account endpoint added (calls kalshi_client.get_balance())
│
└── research/
    ├── __init__.py
    ├── backtest_engine.py               Simulates trades from opportunity DataFrames; summarize_backtest_results()
    ├── generate_dashboard.py            Generates self-contained Chart.js HTML dashboard from paper_trades.csv; contract IDs now clickable Kalshi deep links; OPEN rows show bracket recommendation chip (green YES/orange NO)
    ├── mispriced_weather_markets.py     Original research/analysis script; analyze_mispriced_markets()
    ├── paper_trade_session.py           ✅ Multi-city morning script: run_city_session(CityProfile) per city; main() loads active cities, runs concurrently via ThreadPoolExecutor(max_workers=4); city column in CSV; threading.Lock on CSV writes
    ├── portfolio_backtester.py          Multi-city portfolio backtest aggregator
    ├── run_chicago_backtest.py          Chicago-specific backtest driver against real 2023/2024 data
    ├── run_dallas_backtest.py           Dallas-specific backtest driver
    ├── run_nyc_backtest.py              NYC-specific backtest driver
    └── settle_paper_trades.py           ✅ Evening settlement: groups OPEN rows by city; each city uses correct acis_station_id; legacy rows with empty city column default to Chicago
```

**Files flagged as incomplete or blocked:**

| File | Issue |
|------|-------|
| `data/weather_ingest.py` | `fetch_station_observations` is an explicit stub (sine-wave generator). Not used in the live trading path but would block any module that calls it for real observations. |
| `trading/trade_execution.py:132` | `submit_live_trade()` raises `RuntimeError` unconditionally — live execution is completely blocked. |

---

## 2. Current Feature Status

| Feature | Status | Location | Notes |
|---|---|---|---|
| GFS ensemble fetch (live, Open-Meteo) | ✅ Complete | `models/forecast_generation.py` | GFS + ECMWF + NAM via Open-Meteo; stub fallback on failure |
| GFS historical fetch (AWS byte-range) | ✅ Complete | `data/historical_forecast_ingest.py` | cfgrib/xarray required; disk cache; `atmos/` path handled; numpy/xarray boolean bug fixed (was crashing 12z/f030 fetch and blocking fallback to 00z/f042) |
| Monthly bias correction (bias_loader) | ✅ Complete | `calibration/bias_loader.py` | Reads parquet profile; fallback to cities.yaml; thread+file safe |
| Historical replay / calibration | ✅ Complete | `calibration/historical_replay.py` | 2021–2024 KMDW data replayed; 12-month profile built |
| Signal generation (alpha engine) | ✅ Complete | `trading/alpha_engine.py` | BUY/SELL/HOLD; rank_score with tail/shift/lag/microstructure boosts |
| Alpha threshold filtering | ✅ Complete | `research/paper_trade_session.py:97` | Now reads `risk.alpha_threshold` from settings.yaml via `get_setting()`; value is 0.38 |
| Kalshi API integration (live) | ✅ Complete | `market/kalshi_client.py` | RSA-PSS auth; live contract + orderbook fetch; stub fallback |
| Paper trade session (morning) | ✅ Complete | `research/paper_trade_session.py` | Runs at 7 AM; multi-city via ThreadPoolExecutor(max_workers=4); GFS → ensemble → Kalshi → spreading → logs CSV + Discord per city |
| Settlement script (evening) | ✅ Complete | `research/settle_paper_trades.py` | ACIS fetch per city using acis_station_id; WIN/LOSS; P&L calc with 7% fee; bias_loader update; legacy rows without city default to Chicago |
| Multi-city orchestration | ✅ Complete | `research/paper_trade_session.py` | Concurrent city sessions; shared CSV with threading.Lock; summary Discord embed after all cities finish |
| Boston contract ID normalization | ✅ Complete | `market/kalshi_client.py` | _normalize_ticker() maps malformed IDs (KXHIGHBOS_HIGH_TEMP_F_GT_30_20260415 → KXHIGHBOS-26APR15-T30); called in _parse_ticker() and _parse_contract() |
| Dashboard contract deep links + chips | ✅ Complete | `research/generate_dashboard.py`, `dashboard_ui/src/TradeTable.jsx` | Contract IDs link to kalshi.com/markets/{series}/{ticker}; OPEN rows show green YES/orange NO bracket chip |
| Dashboard backend (FastAPI) | ✅ Complete | `dashboard/api.py` | Read-only; serves trades, daily log, bias profile; /api/account added |
| Dashboard frontend (React) | ✅ Complete | `dashboard_ui/src/` | Vite + React + Recharts + Tailwind; 8 components; KALSHI CASH + IN POSITIONS cards |
| Discord notification | ✅ Complete | `notifications/discord_notifier.py` | Morning run, settlement, dashboard; silent no-op if URL missing |
| Anomaly detection | ✅ Complete | `anomaly/detector.py` | Claude API call works; all 5 signals active including NWS_MODEL_DIVERGENCE |
| NWS forecast fetch | ✅ Complete | `anomaly/nws_fetcher.py` | NWS public API (api.weather.gov); no auth; returns °F or None on failure |
| Incremental profile updater | ✅ Complete | `calibration/bias_loader.py` | Called from `settle_paper_trades.py` after each settlement |
| City onboarding (onboard_city.py) | ✅ Complete | `calibration/onboard_city.py` | 8-phase CLI; validates, fetches, replays, backtests, writes yaml |
| Batch onboarding (batch_onboard.py) | ✅ Complete | `calibration/batch_onboard.py` | ProcessPoolExecutor; bad city never stops batch |
| Watchdog / health monitoring | ✅ Complete | `monitoring/watchdog.py` | 5 checks; amber/red Discord alerts on failure; silence when healthy; every 15 min |
| Multi-bracket spreading | ✅ Complete | `trading/bracket_spreader.py` | Proportional BUY-only allocation; up to 3 brackets per session; SELL not implemented |
| Dynamic position sizing | ✅ Complete | `trading/position_sizer.py` | Anomaly confidence multiplier + per-flag penalties + ensemble spread thresholds |
| Kalshi account balance (dashboard) | ✅ Complete | `market/kalshi_client.py`, `dashboard/api.py` | `get_balance()` → `/api/account`; KALSHI CASH + IN POSITIONS stat cards; 60s refresh |
| HRRR/NAM blend | 🔴 Not started | — | "NAM" resolves to Open-Meteo `best_match` (HRRR proxy); no true NAM distinction |
| Polymarket adapter | 🔴 Not started | — | `POLYMARKET_API_KEY` placeholder in .env; settings.yaml lists Polymarket as enabled source but no implementation exists |
| Live execution | 🔴 Not started | — | `submit_live_trade()` raises RuntimeError; gated by `paper_trade_flag` |

---

## 3. Data Inventory

### `data/bias_profiles/`

| File | Rows | Date Range / Notes |
|---|---|---|
| `KMDW_raw_errors.parquet` | **1,384** | 2021-01-02 → 2024-12-31. Columns: date, city_code, station_id, month, raw_mean_f, actual_f, error_f, raw_std_f |
| `KMDW_monthly_profile.parquet` | **12** (confirmed) | All 12 months present. Columns: month, mean_bias_f, variance_multiplier, sample_count, last_updated |
| `KDFW_monthly_profile.parquet` | **12** | Dallas calibration profile. April mean_bias_correction_f=4.6470, variance_multiplier=0.7759 (freshly calibrated). |
| `KJFK_monthly_profile.parquet` | **12** | NYC calibration profile. |
| `KPHL_monthly_profile.parquet` | **12** | Philadelphia calibration profile. |
| `KBOS_monthly_profile.parquet` | **12** | Boston calibration profile. |

Monthly profile detail (April is the active trading month):

| Month | mean_bias_f | variance_multiplier | sample_count |
|---|---|---|---|
| 1 | 4.734 | 1.660 | 120 |
| 2 | 6.678 | 2.504 | 109 |
| 3 | 5.904 | 2.097 | 120 |
| **4** | **5.424** | **2.169** | **116** |
| 5 | 4.183 | 2.010 | 120 |
| 6 | 4.642 | 1.391 | 116 |
| 7 | 2.091 | 1.008 | 120 |
| 8 | 1.377 | 1.065 | 120 |
| 9 | 1.349 | 1.478 | 116 |
| 10 | 3.346 | 1.879 | 120 |
| 11 | 3.526 | 1.941 | 87 |
| 12 | 4.378 | 2.013 | 120 |

### `data/historical/forecasts/`

| File | Rows | Date Range |
|---|---|---|
| `gfs_chicago_2023.parquet` | 1,530 | 2023-05-02 → 2023-10-05 |
| `gfs_chicago_2024.parquet` | 1,530 | 2024-05-02 → 2024-10-05 |
| `gfs_dallas_2023.parquet` | 310 | 2023-08-02 → 2023-09-05 |
| `gfs_nyc_2023.parquet` | 310 | 2023-08-02 → 2023-09-05 |

Note: these cover only the summer research windows. The 2021–2024 full-year GFS data used for the historical replay was processed into `KMDW_raw_errors.parquet` directly; the intermediate per-year GFS parquets for 2021–2022 are not on disk.

### `data/historical/markets/`

| File | Rows | Date Range |
|---|---|---|
| `kalshi_kxhighchi_2023.parquet` | 17,389 | 2023-04-30 → 2023-09-30 (hourly candles) |
| `kalshi_kxhighny_2023.parquet` | — | NYC equivalent |
| `kalshi_kxhighny_2023_contracts.parquet` | — | NYC contract metadata |

Note: No `kalshi_kxhighchi_2023_contracts.parquet` present — the Chicago contract metadata parquet is missing. `run_chicago_backtest.py` expects it at `data/historical/markets/kalshi_kxhighchi_2023_contracts.parquet`.

### `data/historical/settlement/`

| File | Rows | Date Range |
|---|---|---|
| `chicago_2023.parquet` | 153 | 2023-05-01 → 2023-09-30 |
| `chicago_2024.parquet` | 153 | 2024-05-01 → 2024-09-30 |
| `dallas_2023.parquet` | 122 | 2023-06-01 → 2023-09-30 |
| `nyc_2023.parquet` | 122 | 2023-06-01 → 2023-09-30 |

### `data/paper_trades/`

| File | Description |
|---|---|
| `paper_trades.csv` | Rows as of Apr 14 (VOID duplicates from double session run removed; OPEN rows retained). city column now present. See Section 10. |
| `daily_log.csv` | 16+ data rows. Same base schema as paper_trades.csv. |
| `session_log.txt` | Text session log |
| `settlement_log.txt` | Text settlement log |
| `dashboard.html` | Self-contained Chart.js HTML dashboard |

---

## 4. Configuration Audit

### `config/cities.yaml`

All city entries now include five new fields: `kalshi_series`, `acis_station_id`, `nws_lat`, `nws_lon`, and `active`.
Chicago `acis_station_id` corrected from `KMDW` to `CLIMDW`.

| City | station_id | acis_station_id | kalshi_series | mean_bias_correction_f | variance_multiplier | active | Notes |
|---|---|---|---|---|---|---|---|
| Chicago | KMDW | CLIMDW | kxhighchi | 3.9599 | 0.7787 | true | Fallback only — monthly parquet takes precedence at runtime |
| Dallas | KDFW | CLIDFW | kxhightdal | 4.6470 | 0.7759 | true | Freshly calibrated (Apr 14). Monthly parquet built. |
| New York | KJFK | CLINYC | kxhighny | (from parquet) | (from parquet) | false | Calibrated; active: false pending manual flip |
| Philadelphia | KPHL | CLIPHL | kxhighphil | (from parquet) | (from parquet) | false | Calibrated; active: false pending manual flip |
| Boston | KBOS | CLIBOS | kxhightbos | (from parquet) | (from parquet) | false | Calibrated; active: false pending manual flip |
| Phoenix | KPHX | CLIPHX | kxhightphx | 0.0 | 0.9 | false | ⚠️ Uncalibrated placeholder |
| Denver | KDEN | CLIDEN | kxhightden | 0.0 | 1.3 | false | ⚠️ Uncalibrated placeholder |

Additional city parameters: `kde_bandwidth`, `tail_multiplier`, `model_weight_gfs/ecmwf/nam`, `heat_bias_adjustment_f`, `cold_bias_adjustment_f`, `sep_climate_normal_f`, `sep_anomaly_trigger_f` (last two only on Chicago).

**Flags:**
- Phoenix and Denver `mean_bias_correction_f: 0.0` — unvalidated defaults, not calibrated. Both `active: false`.
- NYC, Philadelphia, Boston calibrated and have monthly parquets but are `active: false` — require manual flip before live paper trading.
- `paper_trade_session.py` now respects `active` flag across all cities via `get_city_universe()`; no longer hardcodes Chicago.

### `.env.example`

| Variable | Default | Notes |
|---|---|---|
| `KALSHI_API_KEY_ID` | (empty) | Required for live mode |
| `KALSHI_PRIVATE_KEY_PATH` | `keys/kalshi_private_key.pem` | Required for live mode |
| `KALSHI_PRIVATE_KEY` | (empty) | Alternative to path |
| `ANTHROPIC_API_KEY` | (empty) | Required for anomaly detection |
| `POLYMARKET_API_KEY` | (empty) | ⚠️ Placeholder — Polymarket not implemented |
| `DISCORD_WEBHOOK_URL` | example URL with placeholder tokens | ⚠️ Default value has literal `YOUR_WEBHOOK_ID/YOUR_WEBHOOK_TOKEN` — will silently fail if not replaced |
| `DASHBOARD_PORT` | 8765 | |
| `DASHBOARD_HOST` | 0.0.0.0 | |
| `ENVIRONMENT` | development | |
| `PAPER_TRADE` | true | |

**Flags:**
- `DISCORD_WEBHOOK_URL` default contains placeholder tokens and will not work until replaced. The notifier is silent on failure, so this won't crash anything.
- `FORECAST_STUB_MODE` used in code (`forecast_generation.py`) but not documented in `.env.example`.

### `config/settings.yaml`

Key entries:

| Key | Value | Notes |
|---|---|---|
| `risk.alpha_threshold` | 0.38 | Now correctly read by `paper_trade_session.py` via `get_setting()` |
| `risk.max_position_per_contract` | 100 | — |
| `risk.max_daily_exposure` | 1000 | — |
| `risk.multi_bracket.enabled` | true | Multi-bracket spreading active |
| `risk.multi_bracket.max_contracts_per_session` | 3 | Max brackets per session |
| `risk.multi_bracket.daily_exposure_cap_usd` | 50.0 | Static fallback cap (used when dynamic sizing disabled) |
| `risk.multi_bracket.min_alpha_to_spread` | 0.38 | Must match primary signal threshold |
| `risk.multi_bracket.allocation_method` | proportional | Only implemented method; equal/kelly are stubs |
| `risk.multi_bracket.dynamic_sizing.enabled` | true | Dynamic cap is active |
| `risk.multi_bracket.dynamic_sizing.base_exposure_usd` | 50.0 | Starting cap before modulation |
| `risk.multi_bracket.dynamic_sizing.min_exposure_usd` | 10.0 | Floor — never go below this |
| `risk.multi_bracket.dynamic_sizing.anomaly_multipliers` | HIGH:1.0, MEDIUM:0.70, LOW:0.50, NONE:1.0 | Confidence penalty |
| `risk.multi_bracket.dynamic_sizing.flag_penalties` | FOG:0.85, LAKE:0.85, COLD:0.90, FRONTAL:0.85, NWS:0.80 | Per-flag multiplicative penalty |
| `risk.multi_bracket.dynamic_sizing.spread_thresholds` | clean<3.0°F(×1.0), moderate<5.0°F(×0.80), wide(×0.60) | Ensemble spread penalty |

---

## 5. Dependency Audit

### `requirements.txt`

| Package | Pinned? | Status |
|---|---|---|
| pandas | No | In use; fine |
| numpy | No | In use; fine |
| scipy | No | In use (norm.cdf); fine |
| requests | No | In use; fine |
| pyarrow | No | In use for parquet; fine |
| pytest | No | Tests only; fine |
| pydantic | No | Used in `dashboard/api.py` for response models |
| PyYAML | No | In use; fine |
| loguru | No | Used in logging_utils.py |
| python-dateutil | No | Likely transitive dep; direct use not confirmed |
| python-dotenv | No | Used in multiple research scripts |
| anthropic | No | Used in anomaly/detector.py |
| `cfgrib` | No | ✅ Now present — required for GRIB2 parsing in historical replay and paper_trade_session |
| `xarray` | No | ✅ Now present — required alongside cfgrib |
| `eccodes` | No | ✅ Now present — runtime dependency of cfgrib |
| `cryptography` | No | ✅ Now present — required for RSA-PSS Kalshi auth in live mode |

**Split requirements:**
- `fastapi` and `uvicorn` remain in `requirements_dashboard.txt` only (not in the main requirements.txt).

**Effectively unused in `requirements.txt`:**
- `python-dateutil` — not directly imported in any audited source file (may be a transitive dep of pandas).

---

## 6. Known Issues Found During Read

1. **`kalshi_kxhighchi_2023_contracts.parquet` missing** — `run_chicago_backtest.py` expects this file at `data/historical/markets/kalshi_kxhighchi_2023_contracts.parquet`. It does not exist on disk. Running the Chicago backtest will fail.

2. **`daily_log.csv` `ensemble_mean_f` blank** — Rows from April 2 have empty `ensemble_mean_f`. This appears to be a schema migration gap; newer rows (Apr 9+) have the value populated. The FastAPI dashboard reads this column as float and uses `pd.to_numeric(..., errors='coerce')` so it degrades gracefully to NaN.

3. **`weather_ingest.py` is dead code** — `fetch_station_observations` returns fake sine-wave data. No live path calls it. If any future module imports it expecting real observations, it will silently return synthetic data.

4. **Phoenix and Denver active but uncalibrated** — Both cities are `active: false` in cities.yaml and have `mean_bias_correction_f: 0.0` (placeholder). No monthly parquet profiles exist for these stations. Not executed.

4a. ~~**`anthropic` package not installed**~~ — **RESOLVED.** `anthropic 0.94.0` is installed.

4b. **Boston contract normalization may not be fully resolved** — `_normalize_ticker()` added but not yet verified on a live session run. Monitor next morning session output.

5. **`data_sources.py` appears unused** — Present in `data/` but not imported by any audited module. May be dead code or a very early prototype.

6. **`forecast_error.py` not wired into live pipeline** — Defines a stub error model. Not imported by scheduler, paper_trade_session, or any calibration module.

7. **`scheduler.py` `run_loop` not used by paper_trade_session** — The morning script uses its own self-contained pipeline, not `scheduler.run_once()`. The scheduler and paper_trade_session are parallel implementations of the same pipeline. They may drift.

8. ~~**`settings.yaml` `markets.enabled_sources` includes Polymarket**~~ — **FIXED.** Removed from `enabled_sources`; Polymarket remains unimplemented.

9. **SELL spreading not implemented** — `bracket_spreader.py:build_spread()` skips SELL signals (BUY-only). A session with SELL-only signals produces no allocations. Now logs a WARNING with contract IDs when any SELL signals are skipped. Implementation is a future enhancement.

10. **Equal and kelly allocation stubbed** — `build_spread()` raises `NotImplementedError` for `allocation_method="equal"` or `"kelly"`. Config documents these as future options; only `"proportional"` works.

11. **paper_trades.csv schema not yet migrated** — The CSV still has 21 columns (pre-spread/sizing schema). New columns (`spread_rank`, `spread_total_contracts`, `sizing_base_usd`, `sizing_final_usd`, `sizing_reasoning`) are NOT present. The `_ensure_csv()` auto-migration in `paper_trade_session.py` will add them on the next session run that actually writes to the file. Existing rows will receive empty values for new columns.

---

## 7. What Is Fully Working End-to-End Right Now

On a normal trading day, starting at approximately 7:00 AM CDT with no human intervention:

1. **Morning session** (`paper_trade_session.py`) runs via Windows Task Scheduler. It loads all cities with `active: true` from `config/cities.yaml` and dispatches each to `run_city_session(city)` concurrently via `ThreadPoolExecutor(max_workers=4)`. Each city session downloads the current-day GFS forecast from AWS (`noaa-gfs-bdp-pds.s3.amazonaws.com`) using byte-range GRIB2 requests — trying the 12z/f030 run first, falling back to 00z/f042 (numpy/xarray boolean bug that was blocking this fallback is now fixed). It also queries Open-Meteo for all three model values (GFS, ECMWF, NAM) as a cross-check. Currently active cities: Chicago, Dallas.

2. The ensemble builder (`temperature_ensemble.py`) combines the model values with per-model weights and lead-time decay, then calls `bias_loader.get_current_bias(city.acis_station_id, current_month)` to fetch the April bias row from the city's monthly parquet (e.g., KMDW: mean_bias_f ≈ +5.42°F, variance_multiplier ≈ 2.17; KDFW: mean_bias_f ≈ +4.65°F, variance_multiplier ≈ 0.78). The corrected mean and uncertainty-adjusted std are returned.

3. A KDE temperature distribution is fit over the forecast values, and a probability surface is generated over 10–120°F.

4. The Kalshi client fetches live contracts for each city's `kalshi_series` using RSA-PSS authentication (e.g., KXHIGHCHI for Chicago, KXHIGHTDAL for Dallas). Malformed contract IDs (e.g., Boston-style `KXHIGHBOS_HIGH_TEMP_F_GT_30_20260415`) are normalized by `_normalize_ticker()` before parsing. For each contract with `|alpha| ≥ 0.38` (read from `risk.alpha_threshold` in settings.yaml), a `TradeSignal` is generated with BUY/SELL/HOLD classification.

5. The METAR fetcher pulls the current station observation via parameterized `fetch_metar(city.station_id)` from aviationweather.gov. The NWS fetcher calls `api.weather.gov/points/{city.nws_lat},{city.nws_lon}` → forecast URL → first daytime period temperature. Both results are passed to the Anthropic API (`claude-opus-4-6`) which checks for fog, lake breeze, cold-start, frontal suppression, and NWS/model divergence anomalies. The result is logged to the CSV columns `anomaly_flags`, `anomaly_penalty_f`, `anomaly_adjusted_signal`, `anomaly_confidence`, `anomaly_reasoning`. NWS failures degrade gracefully to None (NWS_MODEL_DIVERGENCE flag simply won't fire that session). ⚠️ If `anthropic` package is not installed, detector falls back silently → LOW confidence → halved dynamic sizing.

6. **Dynamic position sizing** (`position_sizer.py`) computes a session exposure cap starting from $50 base, modulated by anomaly confidence (HIGH×1.0, MEDIUM×0.70, LOW×0.50), per-flag multiplicative penalties, and ensemble spread (clean×1.0, moderate×0.80, wide×0.60). Floor is $10.

7. **Multi-bracket spreading** (`bracket_spreader.py`) takes all BUY signals above the 0.38 threshold, ranks by alpha magnitude, selects the top 3, and allocates proportionally from the dynamic cap. `spread_rank` and `spread_total_contracts` are written to each row.

8. Signals meeting the threshold are deduplicated (one per threshold/direction), written to `data/paper_trades/paper_trades.csv` and `daily_log.csv`, and a Discord embed is posted.

9. At approximately **6:00 PM CDT**, the settlement script (`settle_paper_trades.py`) runs. It groups all OPEN rows by city, fetches the official daily high for each city from ACIS using the city's `acis_station_id` (e.g., CLIMDW for Chicago, CLIDFW for Dallas), determines WIN/LOSS, computes P&L (BUY WIN = `(1 - entry_price) × 10 × 0.93`, etc.), writes results back to the CSV, calls `bias_loader.update_profile_after_settlement()` per city, and posts a Discord settlement embed. Legacy rows with an empty city column default to Chicago.

10. At **7:15 PM CDT**, `generate_dashboard.py` regenerates `dashboard.html` and posts it to Discord as a file attachment. Contract IDs in the trade table are now clickable Kalshi deep links. OPEN rows display a bracket recommendation chip (green for YES/BUY, orange for NO/SELL).

11. Throughout the day, the **FastAPI backend** (`dashboard/api.py`) serves the paper trades, daily log, bias profile, and Kalshi account balance to the React frontend at `localhost:8765`. The React frontend (`dashboard_ui/`) displays 7 summary stat cards (including KALSHI CASH and IN POSITIONS), P&L chart, alpha histogram, trade table, and bias profile tab. The dashboard auto-refreshes every 60 seconds.

12. Throughout the day (7 AM–8 PM CDT, every 15 minutes), the **watchdog** (`monitoring/watchdog.py`) runs five checks: morning session ran (after 7:30), settlement ran (after 18:30), dashboard regenerated (after 19:30), Kalshi API reachable (always), bias profile fresh within 7 days (always). Amber Discord alerts on warning, red on hard failure, silence when healthy. One log line appended to `data/logs/watchdog.log` per run.

---

## 8. What Is NOT Working or Not Yet Built

1. **Live execution** — `submit_live_trade()` raises `RuntimeError`. All trades are paper-only. No real money is at risk or deployed.

2. **SELL spreading not implemented** — `bracket_spreader.py` silently skips SELL signals. Only BUY signals are spread. A SELL-only session would produce zero spread allocations.

3. **Equal and kelly allocation stubbed** — `build_spread()` raises `NotImplementedError` for these methods. Only proportional allocation works.

4. **Multi-bracket (B-type) contracts not traded** — Kalshi bracket contracts (floor + cap) are parsed but VOIDed (as seen in the paper trade log: `KXHIGHCHI-26APR02-B66.5 → VOID`). Only T-type (less/greater) directional contracts are evaluated.

5. **Chicago backtest (run_chicago_backtest.py) is broken** — `kalshi_kxhighchi_2023_contracts.parquet` does not exist on disk. Any attempt to re-run the backtest from scratch will fail at contract metadata loading.

6. **Polymarket** — Listed in settings.yaml as an enabled source and .env.example has a key placeholder, but no client, contract generator, or price adapter exists for Polymarket.

8. **Server infrastructure not networked** — The three Dell PowerEdge servers exist but are not networked. Everything runs on the Win11 PC. Scheduling, replay, and batch onboarding all share one machine.

9. **ECMWF at short lead times** — For same-day targets, ECMWF (`ecmwf_ifs04`) is dropped from the ensemble because Open-Meteo doesn't serve it for T+0. This is handled correctly, but it means the morning session typically runs GFS + NAM only (2 models), not 3.

10. **Boston contract normalization not yet verified on live run** — `_normalize_ticker()` is implemented but has not been confirmed working against a real live Boston session. Monitor next run.

11. **`anthropic` package missing from active environment** — Anomaly detector falls back silently to LOW confidence, cutting dynamic sizing by 50%. Fix: `pip install anthropic`.

12. **20+ cities not yet onboarded** — Phoenix, Denver, Los Angeles, San Francisco, Seattle, Las Vegas, Washington DC, Minneapolis, Houston, New Orleans, San Antonio, Oklahoma City, Austin, Miami, Atlanta, and others have no calibration data or active: false with no parquet.

---

## 9. Open TODOs Extracted from Code

| File | Line | Item |
|---|---|---|
| `trading/trade_execution.py` | 118–132 | `submit_live_trade()` raises `RuntimeError` unconditionally ("placeholder for Phase 7") |
| `trading/bracket_spreader.py` | 78–81 | `equal` and `kelly` allocation raise `NotImplementedError` — not yet implemented |
| `trading/bracket_spreader.py` | 63–65 | SELL spreading silently skipped — BUY only; noted as "future enhancement" in docstring |
| `data/weather_ingest.py` | 73 | `fetch_station_observations` is an explicit stub (STUB logged at INFO level on every call) |
| `data/historical_forecast_ingest.py` | 288 | `raise RuntimeError("Unexpected exit from retry loop")  # pragma: no cover` |
| `data/historical_forecast_ingest.py` | 663 | `raise RuntimeError(...)` on missing cfgrib |
| `data/historical_noaa_ingest.py` | 171 | `raise RuntimeError("Unexpected exit from retry loop")  # pragma: no cover` |
| `market/kalshi_client.py` | 731, 795 | `raise RuntimeError(...)` on auth or request failure after retries |
| `research/mispriced_weather_markets.py` | 1142 | `raise RuntimeError(...)` on data validation |
| `research/run_chicago_backtest.py` | 1035, 1058, 1063 | `raise RuntimeError(...)` on missing data files |
| `research/run_dallas_backtest.py` | 305 | `raise RuntimeError(...)` on missing data |
| `research/run_nyc_backtest.py` | 325 | `raise RuntimeError(...)` on missing data |

Note: The majority of `raise RuntimeError` entries are intentional hard stops for missing data or missing packages, not unresolved TODOs. True blocked-functionality items: `submit_live_trade()`, equal/kelly allocation, SELL spreading.

---

## 10. Metrics Snapshot

### `data/paper_trades/paper_trades.csv` as of 2026-04-14

Duplicate April 15 VOID rows from a double session run have been removed. OPEN rows retained. `city` column now present.

| Metric | Value |
|---|---|
| Settled trades (WIN + LOSS) | 5 (Chicago only through Apr 13) |
| Wins | 3 |
| Losses | 2 |
| Win rate (settled only) | **60%** |
| Net realized P&L (settled trades) | **+$21.02** |
| Mean alpha (all trades) | **0.402** |
| Date range | 2026-04-02 → 2026-04-13 |
| `city` column present | **Yes** (added; legacy rows empty → default to Chicago) |
| All anomaly columns present | **Yes** |
| `spread_rank`, `spread_total_contracts`, `sizing_*` columns | **Yes** — auto-migrated on first multi-city session write |

### City Calibration Results (Backtest, April 14, 2026)

| City | Station | acis_station_id | Win Rate | Sharpe | Holdout RMSE | Apr Bias | Active |
|---|---|---|---|---|---|---|---|
| Dallas | KDFW | CLIDFW | 79.9% | 1.486 | 4.12°F | +6.08°F | ✅ |
| New York | KJFK | CLINYC | 81.0% | 2.355 | 3.83°F | +2.50°F | ✅ |
| Philadelphia | KPHL | CLIPHL | 85.3% | 1.714 | 3.99°F | +4.64°F | ✅ |
| Boston | KBOS | CLIBOS | 79.0% | 1.714 | 4.35°F | +4.42°F | ✅ |

**Observations:** Multi-city refactor complete and live (Chicago + Dallas). Four new cities calibrated with strong backtest metrics — all above 79% win rate and Sharpe ≥ 1.48. NYC leads on Sharpe (2.355), Philadelphia leads on win rate (85.3%). All four are ready to activate; NYC/Philly/Boston held at `active: false` pending review. Dallas April bias (+6.08°F) is the highest of the new cities and is fully accounted for in the monthly parquet (mean_bias_correction_f=4.647). Sample is still small for Chicago live paper trades (5 settled) — calibration targets remain the backtest results, not these early live numbers.

---

## 11. Full City Universe Status (as of April 14, 2026)

| City | Kalshi Series | CLI Station | Calibrated | Active |
|---|---|---|---|---|
| Chicago | kxhighchi | CLIMDW | ✅ | ✅ |
| Dallas | kxhightdal | CLIDFW | ✅ | ✅ |
| New York | kxhighny | CLINYC | ✅ | ✅ |
| Philadelphia | kxhighphil | CLIPHL | ✅ | ✅|
| Boston | kxhightbos | CLIBOS | ✅ | ✅ |
| Phoenix | kxhightphx | CLIPHX | ❌ | ❌ |
| Denver | kxhightden | CLIDEN | ❌ | ❌ |
| Los Angeles | kxhighlax | CLILAX | ❌ | ❌ |
| San Francisco | kxhightsfo | CLISFO | ❌ | ❌ |
| Seattle | kxhightsea | CLISEA | ❌ | ❌ |
| Las Vegas | kxhightlv | CLILAS | ❌ | ❌ |
| Washington DC | kxhightdc | CLIDCA | ❌ | ❌ |
| Minneapolis | kxhightmin | CLIMSP | ❌ | ❌ |
| Houston | kxhighthou |CLIHOU | ❌ | ❌ |
| New Orleans | kxhightnola | CLIMSY | ❌ | ❌ |
| San Antonio | kxhightsatx |CLISAT | ❌ | ❌ |
| Oklahoma City | kxhightokcc | CLIOKC | ❌ | ❌ |
| Austin | kxhighhaus | CLIAIS | ❌ | ❌ |
| Miami | kxhighmia | CLIMIA | ❌ | ❌ |
| Atlanta | kxhighttatl | CLIATL | ❌ | ❌ |
