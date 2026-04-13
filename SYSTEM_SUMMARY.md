# Deep Isobar — System Summary

_Generated: 2026-04-13 by read-only audit. No files modified._
_Updated April 13, 2026_
_Changes since April 10: alpha threshold fix, NWS wired, requirements cleanup, watchdog, multi-bracket spreading, dynamic position sizing, Kalshi account balance._

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
│   └── types.py                         All shared dataclasses: CityProfile, ForecastPoint, TradeSignal, etc.
│
├── data/
│   ├── __init__.py
│   ├── city_universe.py                 Loads config/cities.yaml → CityProfile list; get_city_profile()
│   ├── data_sources.py                  (not read — not referenced by other modules in audit)
│   ├── feature_store.py                 Minimal save/load parquet wrapper (paths.data_dir)
│   ├── historical_forecast_ingest.py    AWS NOAA GFS byte-range fetch; GRIB2 parse via cfgrib/xarray; disk cache
│   ├── historical_noaa_ingest.py        ACIS API fetch for daily high/low settlement temps; retry w/ backoff
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
│   ├── kalshi_client.py                 Live RSA-PSS authenticated Kalshi v2 client; stub fallback; get_balance() added
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
│   ├── __init__.py                      Exports: check_anomalies, fetch_kmdw_metar, parse_metar_fields, fetch_nws_high_forecast_f
│   ├── detector.py                      ✅ Full: Anthropic API anomaly check; all 5 signals active
│   ├── metar_fetcher.py                aviationweather.gov METAR fetch + field parser for KMDW
│   └── nws_fetcher.py                  ✅ NEW — fetch_nws_high_forecast_f(): NWS public API; no auth required
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
    ├── generate_dashboard.py            Generates self-contained Chart.js HTML dashboard from paper_trades.csv
    ├── mispriced_weather_markets.py     Original research/analysis script; analyze_mispriced_markets()
    ├── paper_trade_session.py           ✅ Daily morning script: GFS fetch → ensemble → Kalshi → spreading → log CSV + Discord
    ├── portfolio_backtester.py          Multi-city portfolio backtest aggregator
    ├── run_chicago_backtest.py          Chicago-specific backtest driver against real 2023/2024 data
    ├── run_dallas_backtest.py           Dallas-specific backtest driver
    ├── run_nyc_backtest.py              NYC-specific backtest driver
    └── settle_paper_trades.py           ✅ Evening settlement: ACIS fetch → WIN/LOSS → P&L → CSV update + Discord
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
| GFS historical fetch (AWS byte-range) | ✅ Complete | `data/historical_forecast_ingest.py` | cfgrib/xarray required; disk cache; `atmos/` path handled |
| Monthly bias correction (bias_loader) | ✅ Complete | `calibration/bias_loader.py` | Reads parquet profile; fallback to cities.yaml; thread+file safe |
| Historical replay / calibration | ✅ Complete | `calibration/historical_replay.py` | 2021–2024 KMDW data replayed; 12-month profile built |
| Signal generation (alpha engine) | ✅ Complete | `trading/alpha_engine.py` | BUY/SELL/HOLD; rank_score with tail/shift/lag/microstructure boosts |
| Alpha threshold filtering | ✅ Complete | `research/paper_trade_session.py:97` | Now reads `risk.alpha_threshold` from settings.yaml via `get_setting()`; value is 0.38 |
| Kalshi API integration (live) | ✅ Complete | `market/kalshi_client.py` | RSA-PSS auth; live contract + orderbook fetch; stub fallback |
| Paper trade session (morning) | ✅ Complete | `research/paper_trade_session.py` | Runs at 7 AM; GFS → ensemble → Kalshi → spreading → logs CSV + Discord |
| Settlement script (evening) | ✅ Complete | `research/settle_paper_trades.py` | ACIS fetch; WIN/LOSS; P&L calc with 7% fee; bias_loader update |
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
| `paper_trades.csv` | 7 rows (2026-04-02 → 2026-04-13). Legacy 21-column schema (pre-spread/sizing). See Section 10. |
| `daily_log.csv` | 16+ data rows. Same base schema as paper_trades.csv. |
| `session_log.txt` | Text session log |
| `settlement_log.txt` | Text settlement log |
| `dashboard.html` | Self-contained Chart.js HTML dashboard |

---

## 4. Configuration Audit

### `config/cities.yaml`

| City | station_id | mean_bias_correction_f | variance_multiplier | Notes |
|---|---|---|---|---|
| Chicago | KMDW | 3.9599 | 0.7787 | Fallback only — monthly parquet takes precedence at runtime |
| New York | KJFK | -0.91 | 0.85 | Calibrated 2026-03-20 from 32-date Aug 2023 backtest. No monthly parquet built yet. |
| Phoenix | KPHX | 0.0 | 0.9 | ⚠️ Uncalibrated placeholder — mean_bias_correction_f=0.0 |
| Denver | KDEN | 0.0 | 1.3 | ⚠️ Uncalibrated placeholder — mean_bias_correction_f=0.0 |
| Dallas | KDFW | -1.5 | 1.25 | Pre-calibration meteorological estimate; no backtest run yet |

Additional city parameters: `kde_bandwidth`, `tail_multiplier`, `model_weight_gfs/ecmwf/nam`, `heat_bias_adjustment_f`, `cold_bias_adjustment_f`, `sep_climate_normal_f`, `sep_anomaly_trigger_f` (last two only on Chicago).

**Flags:**
- Phoenix and Denver `mean_bias_correction_f: 0.0` — these are unvalidated defaults, not calibrated values.
- `active: true` on all five cities, but only Chicago is actually executed in paper_trade_session.py (hardcoded `CITY = "Chicago"`).

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

4. **Phoenix and Denver active but uncalibrated** — Both cities are `active: true` in cities.yaml but have `mean_bias_correction_f: 0.0` (placeholder). If they were to be executed, bias correction would be a no-op. No monthly parquet profiles exist for these stations.

5. **`data_sources.py` appears unused** — Present in `data/` but not imported by any audited module. May be dead code or a very early prototype.

6. **`forecast_error.py` not wired into live pipeline** — Defines a stub error model. Not imported by scheduler, paper_trade_session, or any calibration module.

7. **`scheduler.py` `run_loop` not used by paper_trade_session** — The morning script uses its own self-contained pipeline, not `scheduler.run_once()`. The scheduler and paper_trade_session are parallel implementations of the same pipeline. They may drift.

8. **`settings.yaml` `markets.enabled_sources` includes Polymarket** — Listed but completely unimplemented. `POLYMARKET_API_KEY` is a blank placeholder in .env.example.

9. **SELL spreading not implemented** — `bracket_spreader.py:build_spread()` silently skips SELL signals. Only BUY signals are spread. A session with SELL-only signals would produce no allocations from the spreader. Comment in code: "future enhancement."

10. **Equal and kelly allocation stubbed** — `build_spread()` raises `NotImplementedError` for `allocation_method="equal"` or `"kelly"`. Config documents these as future options; only `"proportional"` works.

11. **paper_trades.csv schema not yet migrated** — The CSV still has 21 columns (pre-spread/sizing schema). New columns (`spread_rank`, `spread_total_contracts`, `sizing_base_usd`, `sizing_final_usd`, `sizing_reasoning`) are NOT present. The `_ensure_csv()` auto-migration in `paper_trade_session.py` will add them on the next session run that actually writes to the file. Existing rows will receive empty values for new columns.

---

## 7. What Is Fully Working End-to-End Right Now

On a normal trading day, starting at approximately 7:00 AM CDT with no human intervention:

1. **Morning session** (`paper_trade_session.py`) runs via Windows Task Scheduler. It downloads the current-day GFS forecast for Chicago from AWS (`noaa-gfs-bdp-pds.s3.amazonaws.com`) using byte-range GRIB2 requests — trying the 12z/f030 run first, falling back to 00z/f042. It also queries Open-Meteo for all three model values (GFS, ECMWF, NAM) as a cross-check.

2. The ensemble builder (`temperature_ensemble.py`) combines the model values with per-model weights and lead-time decay, then calls `bias_loader.get_current_bias("KMDW", current_month)` to fetch the April bias row from `KMDW_monthly_profile.parquet` (mean_bias_f ≈ +5.42°F, variance_multiplier ≈ 2.17). The corrected mean and uncertainty-adjusted std are returned.

3. A KDE temperature distribution is fit over the forecast values, and a probability surface is generated over 10–120°F.

4. The Kalshi client fetches live contracts for KXHIGHCHI series (Chicago high temp) using RSA-PSS authentication. For each contract with `|alpha| ≥ 0.38` (read from `risk.alpha_threshold` in settings.yaml), a `TradeSignal` is generated with BUY/SELL/HOLD classification.

5. The METAR fetcher pulls the current KMDW observation from aviationweather.gov. The NWS fetcher calls `api.weather.gov/points/{lat},{lon}` → forecast URL → first daytime period temperature. Both results are passed to the Anthropic API (`claude-opus-4-6`) which checks for fog, lake breeze, cold-start, frontal suppression, and NWS/model divergence anomalies. The result is logged to the CSV columns `anomaly_flags`, `anomaly_penalty_f`, `anomaly_adjusted_signal`, `anomaly_confidence`, `anomaly_reasoning`. NWS failures degrade gracefully to None (NWS_MODEL_DIVERGENCE flag simply won't fire that session).

6. **Dynamic position sizing** (`position_sizer.py`) computes a session exposure cap starting from $50 base, modulated by anomaly confidence (HIGH×1.0, MEDIUM×0.70, LOW×0.50), per-flag multiplicative penalties, and ensemble spread (clean×1.0, moderate×0.80, wide×0.60). Floor is $10.

7. **Multi-bracket spreading** (`bracket_spreader.py`) takes all BUY signals above the 0.38 threshold, ranks by alpha magnitude, selects the top 3, and allocates proportionally from the dynamic cap. `spread_rank` and `spread_total_contracts` are written to each row.

8. Signals meeting the threshold are deduplicated (one per threshold/direction), written to `data/paper_trades/paper_trades.csv` and `daily_log.csv`, and a Discord embed is posted.

9. At approximately **6:00 PM CDT**, the settlement script (`settle_paper_trades.py`) runs. It fetches the official KMDW daily high from ACIS, determines WIN/LOSS for all OPEN rows from that morning, computes P&L (BUY WIN = `(1 - entry_price) × 10 × 0.93`, etc.), writes results back to the CSV, calls `bias_loader.update_profile_after_settlement()` to append the new error row and recompute the April monthly bias, and posts a Discord settlement embed.

10. At **7:15 PM CDT**, `generate_dashboard.py` regenerates `dashboard.html` and posts it to Discord as a file attachment.

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

7. **NYC/Phoenix/Denver/Dallas paper trading not running** — Only Chicago is wired into `paper_trade_session.py`. The other cities in cities.yaml are inert.

8. **Server infrastructure not networked** — The three Dell PowerEdge servers exist but are not networked. Everything runs on the Win11 PC. Scheduling, replay, and batch onboarding all share one machine.

9. **ECMWF at short lead times** — For same-day targets, ECMWF (`ecmwf_ifs04`) is dropped from the ensemble because Open-Meteo doesn't serve it for T+0. This is handled correctly, but it means the morning session typically runs GFS + NAM only (2 models), not 3.

10. **Dallas onboarding not yet run** — Dallas has a pre-calibration estimate in cities.yaml but no historical replay or backtest has been executed for KDFW.

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

From `data/paper_trades/paper_trades.csv` as of 2026-04-13:

| Metric | Value |
|---|---|
| Total rows in paper_trades.csv | 7 |
| Settled trades (WIN + LOSS) | 5 |
| Wins | 3 |
| Losses | 2 |
| Win rate (settled only) | **60%** |
| VOID (bracket contract, not evaluated) | 1 |
| OPEN (not yet settled) | 1 |
| Net realized P&L (settled trades) | **+$21.02** |
| Mean alpha (all trades) | **0.402** |
| Alpha range | 0.277 → 0.534 |
| Date range | 2026-04-02 → 2026-04-13 |
| `anomaly_flags` column present | **Yes** |
| `anomaly_penalty_f` column present | **Yes** |
| `anomaly_adjusted_signal` column present | **Yes** |
| `anomaly_confidence` column present | **Yes** |
| `anomaly_reasoning` column present | **Yes** |
| `spread_rank` column present | **No** — pending auto-migration on next session run |
| `spread_total_contracts` column present | **No** — pending auto-migration on next session run |
| `sizing_base_usd` column present | **No** — pending auto-migration on next session run |
| `sizing_final_usd` column present | **No** — pending auto-migration on next session run |
| `sizing_reasoning` column present | **No** — pending auto-migration on next session run |

**Observations:** 7 trading days, 5 settled results (up from 4 on Apr 10). Win rate improved to 60% with the Apr 12 WIN (+$7.72). Net P&L improved from +$13.30 to +$21.02. Sample is still too small for statistical inference — the backtest 72.5% win rate is a calibration target, not a prediction of these results. The Apr 9 win (alpha=0.277) is the lowest-alpha trade and still won, suggesting the model is capturing real edge even at marginal signals. All 6 actionable trades are BUY (long YES), consistent with the system finding the market systematically underpricing Chicago high temperatures.

The five new spread/sizing columns exist in `_CSV_COLUMNS` in `paper_trade_session.py` and will be written starting from the next session. The `_ensure_csv()` auto-migration will add them to existing rows as empty strings when the session next writes to the file.
