# Deep Isobar

Autonomous weather-intelligence and prediction market trading system.

Deep Isobar identifies **mispriced weather contracts** by comparing ensemble weather model probabilities against market-implied probabilities, then executes trades when the edge clears risk thresholds.

**Current MVP:** Chicago daily high temperature (KXHIGHCHI on Kalshi), paper trading only.

---

## How It Works

```
alpha = model_probability − market_probability
```

When a market prices a contract at 52% but the ensemble assigns 31%, the contract is overpriced — sell it. The edge is only real when it's large enough, consistent enough, and the market is liquid enough to exit. Every layer of the pipeline pressure-tests that claim before a dollar is risked.

---

## Backtest Results (Chicago KMDW)

| Metric | Value |
|---|---|
| Holdout win rate | 72.5% |
| Holdout raw Sharpe | 1.944 |
| 2024 OOT RMSE | 3.725°F |
| 2024 OOT transfer score | +0.009 |
| Look-ahead bias | None |
| Fill model | BUY at ask / SELL at bid |
| Fee model | 7% on winning trades only |

---

## Pipeline

```
GFS Forecast Ingestion
        ↓
Temperature Ensemble  ←  Adaptive Bias Calibration (per city, per month)
        ↓
KDE Probability Engine
        ↓
Market Price Adapter  ←  Kalshi Client
        ↓
Alpha Engine  ←  Distribution Tail Alpha
        ↓
Market Microstructure Scanner
        ↓
Risk Manager
        ↓
Trade Execution
```

---

## Calibration

GFS runs warm for Chicago in April due to Lake Michigan (still ~42°F, lake breeze suppresses afternoon highs 5–10°F on east-wind days). A static seasonal constant doesn't capture this.

The **Adaptive Calibration Engine** solves this permanently. Every city gets a monthly bias profile derived from a 5-year historical replay of GFS vs. NOAA actuals:

```
data/bias_profiles/
  KMDW_raw_errors.parquet       # full row-level history
  KMDW_monthly_profile.parquet  # 12-row table: mean_bias_f, variance_multiplier
  KDFW_monthly_profile.parquet
  ...
```

At runtime, `bias_loader.py` reads the current month's row and applies it to the ensemble. After each settlement, the profile updates incrementally. No manual tuning.

**Current status:** `historical_replay.py` is built and running the KMDW backfill (2021–2024). `bias_loader.py` and the remaining calibration modules are next.

---

## Locked Calibration Parameters (KMDW — current)

```yaml
station_id: KMDW
mean_bias_correction_f: 3.1493
variance_multiplier: 0.82
sep_climate_normal_f: 72.7
sep_anomaly_trigger_f: 5.0
sep_heat_bias_adjustment_f: -2.0
alpha_threshold: 0.38
lead_decay_halflife_hours: 48.0
```

---

## Key Architectural Decisions

**Station KMDW not KORD** — Kalshi settles Chicago contracts on Midway Airport. GFS reads ~2°F warmer at KORD. All coordinates are KMDW (41.7868°N, 87.7522°W).

**T-type contracts only** — Bracket (B-type) contracts filtered at parse. Only `less` and `greater` directional contracts evaluated.

**Correct probability formula:**
```
less:    P(actual < cap)   = norm.cdf(cap − 0.5, mean, std)
greater: P(actual > floor) = 1 − norm.cdf(floor + 0.5, mean, std)
```

**Fill price convention** — BUY at ask, SELL at bid. Mid-price fills overstate edge.

**Deduplication** — One signal per (threshold, direction, date). Highest alpha wins; others marked `DEDUP_DROP`.

---

## Source Layout

```
src/
  deep_isobar/
    core/           # Types, logging, scheduler
    data/           # City universe, NOAA ingest, GFS ingest, feature store
    models/         # Ensemble, KDE, probability engine, forecast error,
                    # volatility, shift detection, probability surface
    market/         # Kalshi client, contract generator, market scanner,
                    # microstructure scanner, market lag detection
    trading/        # Alpha engine, tail alpha, risk manager, trade execution
    calibration/    # historical_replay, bias_loader (in progress),
                    # onboard_city, batch_onboard

docs/modules/       # Per-module design specs (25 modules)
tests/
config/
```

---

## Live Paper Trading

**Schedule (Windows Task Scheduler, SYSTEM account):**

| Task | Time | Script |
|---|---|---|
| Morning session | 7:00 AM CDT | `paper_trade_session.py` |
| Settlement | 6:00 PM CDT | `settle_paper_trades.py` |
| Dashboard | 7:15 PM CDT | `generate_dashboard.py` |
 
**Data sources:**

| Source | Notes |
|---|---|
| GFS | `noaa-gfs-bdp-pds.s3.amazonaws.com`, byte-range GRIB2, disk-cached |
| Kalshi live prices | RSA-PSS API key; stub mode if credentials missing |
| NOAA actuals | ACIS API, posts 6–11 PM CDT |

---

## Infrastructure

| Server | Role | Status |
|---|---|---|
| Dell PowerEdge R750 | Primary scheduler, ensemble runner | On-premises, not yet networked |
| Dell PowerEdge R520 #1 | Backtest farm, city onboarding | On-premises, not yet networked |
| Dell PowerEdge R520 #2 | Market data collector, redundant execution | On-premises, not yet networked |

Currently running on a Win11 PC. Servers will be networked via a Win10 NAT gateway (5 Ethernet ports) → Cisco Catalyst. Replay and batch onboarding should run on the R520s under Linux — the Windows cfgrib file-locking issues do not exist on Linux.

---

## Build Phases

**Phase 1 — Calibration Engine (in progress)**
1. Review `KMDW_monthly_profile.parquet` — confirm April `mean_bias_f` is negative
2. Build `bias_loader.py` — runtime loader with fallback to `cities.yaml`
3. Wire into `temperature_ensemble.py` replacing hardcoded April block
4. Build `onboard_city.py` — one-command city pipeline
5. Build `batch_onboard.py` — parallel onboarding for 40 cities

**Phase 2 — City Expansion**
Dallas (KDFW), NYC (KJFK), and up to 40 cities. Each: `python onboard_city.py --city X --station Y --lat Z --lon W --history-years 5`

**Phase 3 — Server Infrastructure**
Network R750/R520s, move scheduling off Win11 PC, add message queue (Redis or RabbitMQ).

**Phase 4 — Live Execution**
Remove `RuntimeError` stub, add hard position limits ($50–100/trade), order status polling, gate behind `paper_trade=False`.

**Phase 5 — Future**
Post-settlement incremental profile updater, ECMWF feed (better than GFS at T+72–120h), Polymarket adapter, ML post-processing after 500+ settled trades.

---

## Data Sources

| Source | Use |
|---|---|
| NOAA GHCND | Historical observed temperatures for error modeling |
| GFS (NOAA) | Operational forecast grids, ingested per model run |
| ERA5 (ECMWF) | Climatology, long-range reanalysis |
| Kalshi API | Live prices, resolved contract history |

**Note:** `markets?status=settled` retains only ~582 recent contracts. As of April 2026, history cuts off around late December 2025. All 2023–2024 Kalshi data has aged out of the API; only the previously-pulled 2023 parquet survives.

---

## Development Rules

- One module at a time
- Every module ships with tests and logging
- Validate inputs at module boundaries
- Never request the full application in one step
