# Deep Isobar — System Summary

_Generated: 2026-04-13 by read-only audit. No files modified._

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
│   ├── kalshi_client.py                 Live RSA-PSS authenticated Kalshi v2 client; stub fallback
│   ├── market_lag_detection.py          Detects when market price hasn't reacted to a forecast shift
│   ├── market_price_adapter.py          Extracts mid-price / market probability from OrderBookSnapshot
│   └── market_scanner.py               Wires market_price_adapter + alpha_engine; evaluate_contract_opportunity()
│   └── microstructure_scanner.py        Spread, liquidity score, staleness, composite microstructure score
│
├── trading/
│   ├── __init__.py
│   ├── alpha_engine.py                  Alpha = model_prob − market_prob; BUY/SELL/HOLD classification; rank_score
│   ├── distribution_tail_alpha.py       z-score tail detection; ranking boost multiplier
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
│   ├── __init__.py
│   ├── detector.py                      Anthropic API anomaly check (fog, lake breeze, etc.); informational only
│   └── metar_fetcher.py                aviationweather.gov METAR fetch + field parser for KMDW
│
├── dashboard/
│   ├── __init__.py
│   └── api.py                           FastAPI read-only backend; serves paper_trades.csv, daily_log.csv, bias parquet
│
└── research/
    ├── __init__.py
    ├── backtest_engine.py               Simulates trades from opportunity DataFrames; summarize_backtest_results()
    ├── generate_dashboard.py            Generates self-contained Chart.js HTML dashboard from paper_trades.csv
    ├── mispriced_weather_markets.py     Original research/analysis script; analyze_mispriced_markets()
    ├── paper_trade_session.py           ✅ Daily morning script: GFS fetch → ensemble → Kalshi → log CSV + Discord
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
| `research/paper_trade_session.py:339` | `nws_forecast_f=None` — NWS human forecast fetch is unwired (TODO comment). Anomaly detector never receives NWS data, missing one of its five detection signals. |

---

## 2. Current Feature Status

| Feature | Status | Location | Notes |
|---|---|---|---|
| GFS ensemble fetch (live, Open-Meteo) | ✅ Complete | `models/forecast_generation.py` | GFS + ECMWF + NAM via Open-Meteo; stub fallback on failure |
| GFS historical fetch (AWS byte-range) | ✅ Complete | `data/historical_forecast_ingest.py` | cfgrib/xarray required; disk cache; `atmos/` path handled |
| Monthly bias correction (bias_loader) | ✅ Complete | `calibration/bias_loader.py` | Reads parquet profile; fallback to cities.yaml; thread+file safe |
| Historical replay / calibration | ✅ Complete | `calibration/historical_replay.py` | 2021–2024 KMDW data replayed; 12-month profile built |
| Signal generation (alpha engine) | ✅ Complete | `trading/alpha_engine.py` | BUY/SELL/HOLD; rank_score with tail/shift/lag/microstructure boosts |
| Alpha threshold filtering | ✅ Complete | `research/paper_trade_session.py:96` | Hardcoded `SIGNAL_THRESHOLD = 0.25`; settings.yaml has `0.38` ⚠️ |
| Kalshi API integration (live) | ✅ Complete | `market/kalshi_client.py` | RSA-PSS auth; live contract + orderbook fetch; stub fallback |
| Paper trade session (morning) | ✅ Complete | `research/paper_trade_session.py` | Runs at 7 AM; fetches GFS → ensemble → Kalshi → logs CSV + Discord |
| Settlement script (evening) | ✅ Complete | `research/settle_paper_trades.py` | ACIS fetch; WIN/LOSS; P&L calc with 7% fee; bias_loader update |
| Dashboard backend (FastAPI) | ✅ Complete | `dashboard/api.py` | Read-only; serves trades, daily log, bias profile; uvicorn |
| Dashboard frontend (React) | ✅ Complete | `dashboard_ui/src/` | Vite + React + Recharts + Tailwind; 8 components |
| Discord notification | ✅ Complete | `notifications/discord_notifier.py` | Morning run, settlement, dashboard; silent no-op if URL missing |
| Anomaly detection | 🟡 Partial | `anomaly/detector.py` | Claude API call works; `nws_forecast_f` always None (unwired) |
| Incremental profile updater | ✅ Complete | `calibration/bias_loader.py` | Called from `settle_paper_trades.py` after each settlement |
| City onboarding (onboard_city.py) | ✅ Complete | `calibration/onboard_city.py` | 8-phase CLI; validates, fetches, replays, backtests, writes yaml |
| Batch onboarding (batch_onboard.py) | ✅ Complete | `calibration/batch_onboard.py` | ProcessPoolExecutor; bad city never stops batch |
| Multi-bracket spreading | 🔴 Not started | — | Only T-type (less/greater) contracts evaluated; B-type bracket contracts filtered/VOIDed |
| Dynamic position sizing | 🔴 Not started | — | `POSITION_SIZE = 10.0` hardcoded; no alpha-scaled sizing |
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
| `paper_trades.csv` | 7 rows (2026-04-02 → 2026-04-13). Full schema including anomaly columns. See Section 10. |
| `daily_log.csv` | 16 data rows. Same schema as paper_trades.csv. ensemble_mean_f blank in early rows (Apr 2). |
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

Key entries: `risk.alpha_threshold: 0.38`, `risk.max_position_per_contract: 100`, `risk.max_daily_exposure: 1000`. The live paper_trade_session.py uses a **hardcoded** `SIGNAL_THRESHOLD = 0.25`, not the settings.yaml value.

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

**Missing from `requirements.txt`:**

| Package | Where used | Notes |
|---|---|---|
| `cryptography` | `market/kalshi_client.py:152` | Required for RSA-PSS Kalshi auth in live mode; imported lazily |
| `cfgrib` | `data/historical_forecast_ingest.py:141` | Required for GRIB2 parsing in historical replay and paper_trade_session |
| `xarray` | `data/historical_forecast_ingest.py:145` | Required alongside cfgrib |
| `eccodes` | Underpins cfgrib | Runtime dependency of cfgrib |
| `fastapi` | `dashboard/api.py` | In `requirements_dashboard.txt` only |
| `uvicorn` | Dashboard server | In `requirements_dashboard.txt` only |

**Effectively unused in `requirements.txt`:**
- `python-dateutil` — not directly imported in any audited source file (may be a transitive dep of pandas).

---

## 6. Known Issues Found During Read

1. **Alpha threshold split** — `paper_trade_session.py:96` hardcodes `SIGNAL_THRESHOLD = 0.25`. `config/settings.yaml` has `risk.alpha_threshold: 0.38`. README documents `0.38`. The session is operating at a lower threshold than designed. No code reads the settings.yaml value for this decision.

2. **`kalshi_kxhighchi_2023_contracts.parquet` missing** — `run_chicago_backtest.py` expects this file at `data/historical/markets/kalshi_kxhighchi_2023_contracts.parquet`. It does not exist on disk. Running the Chicago backtest will fail.

3. **NWS forecast fetch unwired** — `paper_trade_session.py:339` passes `nws_forecast_f=None` to `check_anomalies()`. The anomaly detector's `NWS_MODEL_DIVERGENCE` flag (described as "meaningful" in the system prompt) can never fire.

4. **`daily_log.csv` `ensemble_mean_f` blank** — Rows from April 2 have empty `ensemble_mean_f`. This appears to be a schema migration gap; newer rows (Apr 9+) have the value populated. The FastAPI dashboard reads this column as float and uses `pd.to_numeric(..., errors='coerce')` so it degrades gracefully to NaN.

5. **`weather_ingest.py` is dead code** — `fetch_station_observations` returns fake sine-wave data. No live path calls it. If any future module imports it expecting real observations, it will silently return synthetic data.

6. **Phoenix and Denver active but uncalibrated** — Both cities are `active: true` in cities.yaml but have `mean_bias_correction_f: 0.0` (placeholder). If they were to be executed, bias correction would be a no-op. No monthly parquet profiles exist for these stations.

7. **`data_sources.py` appears unused** — Present in `data/` but not imported by any audited module. May be dead code or a very early prototype.

8. **`forecast_error.py` not wired into live pipeline** — Defines a stub error model. Not imported by scheduler, paper_trade_session, or any calibration module.

9. **`scheduler.py` `run_loop` not used by paper_trade_session** — The morning script uses its own self-contained pipeline, not `scheduler.run_once()`. The scheduler and paper_trade_session are parallel implementations of the same pipeline. They may drift.

10. **`settings.yaml` `markets.enabled_sources` includes Polymarket** — Listed but completely unimplemented. `POLYMARKET_API_KEY` is a blank placeholder in .env.example.

---

## 7. What Is Fully Working End-to-End Right Now

On a normal trading day, starting at approximately 7:00 AM CDT with no human intervention:

1. **Morning session** (`paper_trade_session.py`) runs via Windows Task Scheduler. It downloads the current-day GFS forecast for Chicago from AWS (`noaa-gfs-bdp-pds.s3.amazonaws.com`) using byte-range GRIB2 requests — trying the 12z/f030 run first, falling back to 00z/f042. It also queries Open-Meteo for all three model values (GFS, ECMWF, NAM) as a cross-check.

2. The ensemble builder (`temperature_ensemble.py`) combines the model values with per-model weights and lead-time decay, then calls `bias_loader.get_current_bias("KMDW", current_month)` to fetch the April bias row from `KMDW_monthly_profile.parquet` (mean_bias_f ≈ +5.42°F, variance_multiplier ≈ 2.17). The corrected mean and uncertainty-adjusted std are returned.

3. A KDE temperature distribution is fit over the forecast values, and a probability surface is generated over 10–120°F.

4. The Kalshi client fetches live contracts for KXHIGHCHI series (Chicago high temp) using RSA-PSS authentication. For each contract with `|alpha| ≥ 0.25`, a `TradeSignal` is generated with BUY/SELL/HOLD classification.

5. The METAR fetcher pulls the current KMDW observation from aviationweather.gov. The Anthropic API (`claude-opus-4-6`) analyzes the METAR for fog, lake breeze, cold-start, and frontal suppression anomalies. The result is logged to the CSV columns `anomaly_flags`, `anomaly_penalty_f`, `anomaly_adjusted_signal`, `anomaly_confidence`, `anomaly_reasoning`.

6. Signals meeting the threshold are deduplicated (one per threshold/direction), written to `data/paper_trades/paper_trades.csv` and `daily_log.csv`, and a Discord embed is posted.

7. At approximately **6:00 PM CDT**, the settlement script (`settle_paper_trades.py`) runs. It fetches the official KMDW daily high from ACIS, determines WIN/LOSS for all OPEN rows from that morning, computes P&L (BUY WIN = `(1 - entry_price) × 10 × 0.93`, etc.), writes results back to the CSV, calls `bias_loader.update_profile_after_settlement()` to append the new error row and recompute the April monthly bias, and posts a Discord settlement embed.

8. At **7:15 PM CDT**, `generate_dashboard.py` regenerates `dashboard.html` and posts it to Discord as a file attachment.

9. Throughout the day, the **FastAPI backend** (`dashboard/api.py`) serves the paper trades, daily log, and bias profile to the React frontend at `localhost:8765`. The React frontend (`dashboard_ui/`) displays summary cards, P&L chart, alpha histogram, trade table, and bias profile tab.

---

## 8. What Is NOT Working or Not Yet Built

1. **Live execution** — `submit_live_trade()` raises `RuntimeError`. All trades are paper-only. No real money is at risk or deployed.

2. **NWS human forecast not fetched** — The `NWS_MODEL_DIVERGENCE` anomaly signal is permanently silent. The anomaly detector has four functional signals, not five as designed.

3. **Multi-bracket (B-type) contracts not traded** — Kalshi bracket contracts (floor + cap) are parsed but VOIDed (as seen in the paper trade log: `KXHIGHCHI-26APR02-B66.5 → VOID`). Only T-type (less/greater) directional contracts are evaluated.

4. **Polymarket** — Listed in settings.yaml as an enabled source and .env.example has a key placeholder, but no client, contract generator, or price adapter exists for Polymarket.

5. **Dynamic position sizing** — `POSITION_SIZE = 10.0` is hardcoded. Position size does not scale with alpha magnitude, confidence, or volatility.

6. **Alpha threshold not read from config** — `paper_trade_session.py` uses `SIGNAL_THRESHOLD = 0.25` regardless of `settings.yaml`. The backtest and README use 0.38. Live sessions are trading at a lower bar than the calibrated system.

7. **Chicago backtest (run_chicago_backtest.py) is broken** — `kalshi_kxhighchi_2023_contracts.parquet` does not exist on disk. Any attempt to re-run the backtest from scratch will fail at contract metadata loading.

8. **NYC/Phoenix/Denver/Dallas paper trading not running** — Only Chicago is wired into `paper_trade_session.py`. The other cities in cities.yaml are inert.

9. **Server infrastructure not networked** — The three Dell PowerEdge servers exist but are not networked. Everything runs on the Win11 PC. Scheduling, replay, and batch onboarding all share one machine.

10. **ECMWF at short lead times** — For same-day targets, ECMWF (`ecmwf_ifs04`) is dropped from the ensemble because Open-Meteo doesn't serve it for T+0. This is handled correctly, but it means the morning session typically runs GFS + NAM only (2 models), not 3.

11. **No log rotation or monitoring** — Logs land in `data/paper_trades/session_log.txt` and `settlement_log.txt` with no rotation. No alerting if the morning session fails silently.

---

## 9. Open TODOs Extracted from Code

| File | Line | Item |
|---|---|---|
| `research/paper_trade_session.py` | 339 | `nws_forecast_f=None,  # TODO: wire NWS forecast fetch` |
| `trading/trade_execution.py` | 118–132 | `submit_live_trade()` raises `RuntimeError` unconditionally ("placeholder for Phase 7") |
| `data/weather_ingest.py` | 73 | `fetch_station_observations` is an explicit stub (STUB logged at INFO level on every call) |
| `data/historical_forecast_ingest.py` | 288 | `raise RuntimeError("Unexpected exit from retry loop")  # pragma: no cover` |
| `data/historical_forecast_ingest.py` | 663 | `raise RuntimeError(...)` on missing cfgrib |
| `data/historical_noaa_ingest.py` | 171 | `raise RuntimeError("Unexpected exit from retry loop")  # pragma: no cover` |
| `market/kalshi_client.py` | 731, 795 | `raise RuntimeError(...)` on auth or request failure after retries |
| `research/mispriced_weather_markets.py` | 1142 | `raise RuntimeError(...)` on data validation |
| `research/run_chicago_backtest.py` | 1035, 1058, 1063 | `raise RuntimeError(...)` on missing data files |
| `research/run_dallas_backtest.py` | 305 | `raise RuntimeError(...)` on missing data |
| `research/run_nyc_backtest.py` | 325 | `raise RuntimeError(...)` on missing data |

Note: The majority of `raise RuntimeError` entries are intentional hard stops for missing data or missing packages, not unresolved TODOs. The only true blocked-functionality TODO is `nws_forecast_f=None` in `paper_trade_session.py`.

---

## 10. Metrics Snapshot

From `data/paper_trades/paper_trades.csv` as of 2026-04-13:

| Metric | Value |
|---|---|
| Total rows in paper_trades.csv | 7 |
| Settled trades (WIN + LOSS) | 4 |
| Wins | 2 |
| Losses | 2 |
| Win rate (settled only) | **50%** |
| VOID (bracket contract, not evaluated) | 1 |
| OPEN (not yet settled) | 2 |
| Net realized P&L (settled trades) | **+$13.30** |
| Mean alpha (all trades) | **0.402** |
| Alpha range | 0.277 → 0.534 |
| Date range | 2026-04-02 → 2026-04-13 |
| `anomaly_flags` column present | **Yes** |
| `anomaly_penalty_f` column present | **Yes** |
| `anomaly_adjusted_signal` column present | **Yes** |
| `anomaly_confidence` column present | **Yes** |
| `anomaly_reasoning` column present | **Yes** |

**Observations:** 7 trading days, 4 settled results. Sample is too small for statistical inference — the backtest 72.5% win rate is a calibration target, not a prediction of these 4 trades. The two losses were both small (−$1.90, −$0.70); the two wins were larger (+$8.37, +$7.53), consistent with positive-expectation asymmetry. Alpha values are all well above the 0.25 threshold (min 0.277), suggesting the signal filter is working. All trades are BUY (long YES), consistent with the system finding the market systematically underpricing Chicago high temperatures.
