# Deep Isobar

Autonomous weather-intelligence and prediction-market trading system.

Deep Isobar identifies **mispriced daily-high-temperature contracts** by building a
calibrated probabilistic forecast for each settlement station and comparing it
against market-implied probabilities, executing (paper) trades when the edge
clears risk thresholds.

**Current status (July 2026):** paper trading five cities on Kalshi
(Chicago, New York, Dallas, Philadelphia, Boston) with per-station
EMOS-calibrated distributions — while **calibrating and collecting order
books for Kalshi's entire weather universe** (20 cities, daily highs *and*
lows) in data-only mode, ready to flip on when the track record earns it.
Runs unattended on a ten-job daily schedule with pre-trade circuit
breakers (including a wildfire-smoke stand-down), ops-health invariant
alarms, and settlement results cross-verified against the exchange.

**Docs:** [Operations manual](docs/OPERATIONS.md) ·
[Roadmap](docs/ROADMAP.md) · [Server runbook](docs/RUNBOOK_SERVERS.md) ·
[As-of data store](docs/DATA_STORE.md)

```
alpha = model_probability − market_probability
```

When a market prices a contract at 52% and the calibrated distribution assigns
31%, the contract is overpriced — sell it. The edge is only real when it's
large enough, consistent enough, and the market is liquid enough to exit.
Every layer of the pipeline pressure-tests that claim before a dollar is risked.

---

## Quick Start

**Prerequisites:** Python 3.11+ (Node 18+ only for the web dashboard;
ecCodes/cfgrib only for the legacy GRIB2 path — the EMOS pipeline doesn't need it)

```bash
# Python env (uv)
uv venv && uv pip install -e .
#   — or classic pip:  pip install -r requirements.txt

# Credentials (Kalshi key ID + RSA PEM; Discord webhook optional)
cp .env.example .env

# Fit calibration for all active cities (needs network; ~5 min)
make train

# Score today's markets without trading anything
make dry
```

### Daily commands

```bash
make dry        # dry-run: score today's markets, print signals, place nothing
make trade      # live paper-trade session (writes to data/paper_trades/)
make settle     # settle contracts against NOAA observed temps
make scorecard  # calibration + P&L scorecard (also runs nightly at 18:45)
make train      # rebuild EMOS training data + refit all stations
```

### Other useful targets

```bash
make mac-install    # register the launchd agent (see Live Paper Trading)
make mac-status     # agent state + full job schedule + last-run times
make mac-uninstall  # remove the agent
make up / make down # web dashboard dev servers (API :8765, UI :5173)
make logs           # tail the most recent log in data/paper_trades/
make status         # today's date, last CSV row, API health
```

**Windows (no Make):** use the `start_*.bat` equivalents.

---

## Pipeline

```
Forecast fetch (Open-Meteo daily MaxT: GFS · ECMWF · ICON · GEM · NBM)
        +  GEFS-31 / ECMWF-EPS-51 member spread (Open-Meteo Ensemble API)
        ↓
EMOS calibrated distribution        μ = a₀ + Σ aᵢ·modelᵢ
  (per-station, min-CRPS fit)       σ² = c + d·S²   (floor 1.3°F)
        ↓
Probability surface per contract threshold
        ↓
Market Price Adapter  ←  Kalshi client (RSA-PSS; stub mode without creds)
        ↓
Alpha Engine  ←  Distribution Tail Alpha
        ↓
Microstructure scanner → dedup → multi-bracket spreader → position sizing
        ↓
Paper trade log  →  settlement (ACIS)  →  scorecard / dashboard
```

Stations without fitted EMOS params fall back to the legacy pipeline
(18z GFS snapshot + monthly bias profile + 5.5°F std floor). The two paths
are never mixed for one station.

---

## Calibration (EMOS)

Each station gets an **EMOS / non-homogeneous Gaussian regression** fit by
minimum CRPS over a rolling ~90-day window (Gneiting et al. 2005), refit
every morning at 6:15:

- **Mean** — linear in five model forecasts of the *daily max over the local
  settlement day* (max-of-trace, never a fixed-hour snapshot). NBM — NOAA's
  operationally calibrated blend — is both a member and an external sanity
  benchmark: the session warns when |μ − NBM| > 4°F.
- **Variance** — `c + d·S²` where S² is the pooled GEFS/EPS member variance
  (flow-dependent uncertainty). Historical member spread is not retrievable
  from free APIs, so the spread history **accumulates forward** one day per
  run (`data/emos_training/{station}_t1_spread.parquet`); until coverage
  reaches 60% of the window, the fit uses deterministic-member variance and
  the params are marked `spread_source: members`.
- **Floor** — σ ≥ 1.3°F (sensor + integer-settlement discretization), not the
  legacy 5.5°F floor that caused chronic under-confidence.

Artifacts: `data/emos_params/{station}_emos.json`,
`data/emos_training/{station}_t1_training.parquet`.
Legacy monthly bias profiles (`data/bias_profiles/`) remain only as the
fallback path and for stations not yet fitted.

---

## Key Architectural Decisions

**Settlement stations, not city names** — forecasts target the exact station
each venue settles on:

| City | Kalshi settles | Notes |
|---|---|---|
| Chicago | KMDW (Midway) | Polymarket uses KORD — several °F apart on lake-breeze days |
| New York | KNYC (Central Park) | *Not* KJFK — the sea-breeze microclimate is different |
| Dallas | KDFW | Polymarket uses KDAL (~0.5–1°F warmer) |
| Philadelphia | KPHL | Well-behaved, low mesoscale noise |
| Boston | KBOS | Sea-breeze driven; highest forecast variance |

**T-type contracts only** — bracket (B-type) contracts filtered at parse.

**Probability with integer settlement:**
```
less:    P(actual < cap)   = norm.cdf(cap − 0.5, μ, σ)
greater: P(actual > floor) = 1 − norm.cdf(floor + 0.5, μ, σ)
```

**Fill price convention** — BUY at ask, SELL at bid; mid-price fills overstate edge.

**Alpha threshold 0.10** — lowered from 0.38 with the EMOS upgrade; calibrated
edges are ~3–10 points, and 0.10 stays above Kalshi's taker-fee peak (~1.75%).

**Deduplication** — one signal per (threshold, direction, date); highest |alpha| wins.

---

## Live Paper Trading

**Daily schedule** (run by `deep_isobar.supervisor`, configured in
`config/settings.yaml` → `scheduler.jobs`, local machine time):

| Task | Time | Module |
|---|---|---|
| EMOS refit + spread recording (all 20 cities, highs+lows) | 6:15 AM | `calibration.emos_training` |
| Trading session (circuit breakers → Kelly-sized trades) | 10:30 AM | `research.paper_trade_session` |
| Ops-health invariant alarms (dead session, stale params, stub books, …) | 11:15 AM | `ops.health` |
| Intraday lock-in check | 2:00 PM | `research.intraday_check` |
| Settlement (all pending dates) | 6:00 PM | `research.settle_paper_trades` |
| **Exchange cross-verification** of every graded trade | 6:30 PM | `research.verify_settlements` |
| Daily scorecard | 6:45 PM | `research.daily_scorecard` |
| Dashboard | 7:15 PM | `research.generate_dashboard` |
| Orderbook collector (~305 books, all series) | every 10 min, 6–21 | `data.orderbook_collector` |
| Backup with rotation | 9:30 PM | `ops.backup` |

**Runtime registration:**

- **macOS (current runtime)** — launchd agent: `make mac-install`
  (status: `make mac-status`, remove: `make mac-uninstall`). Runs at login,
  restarts on crash, catches up jobs missed while the laptop was asleep.
- **Windows** — Task Scheduler at logon: `start_supervisor.bat`.

> **⚠️ iCloud warning (macOS):** never run this project from an iCloud-synced
> folder (`~/Desktop`, `~/Documents` with sync on). iCloud evicts file
> contents and sets hidden flags — observed 2026-07-02 breaking the venv
> (Python skips hidden `.pth` files) and deleting the working tree out from
> under the runtime. Use a non-synced path such as `~/deep-isobar`; after
> moving, rebuild the venv and rerun `make mac-install` (the agent renders
> absolute paths).

**Day-to-day watch:** the scorecard writes
`data/reports/scorecard_YYYY-MM-DD.md` (and a Discord embed when a webhook is
configured). It shows the day's settled trades, rolling 7d/30d P&L + win rate
+ **Brier edge vs the market** (were our probabilities closer to reality than
the market's — the honest edge measure), and per-station calibration: CRPS,
MAE vs the NBM benchmark, PIT histogram (flat = calibrated), and
ensemble-spread coverage toward the 60% variance-source gate.

**Data sources:**

| Source | Notes |
|---|---|
| Open-Meteo forecast API | Daily MaxT for GFS/ECMWF/ICON/GEM/NBM (EMOS path) |
| Open-Meteo Ensemble API | GEFS 31-member + ECMWF-EPS 51-member daily MaxT |
| Open-Meteo Previous Runs | T+1-lead training pairs (~92-day history) |
| GFS GRIB2 (AWS S3) | Legacy 18z snapshot path only; needs cfgrib |
| Kalshi API | RSA-PSS key; deterministic stub mode without credentials |
| NOAA ACIS | Settlement observations, posts evenings local |

---

## Evaluation

Performance claims are tracked, not assumed:

- `research/forecast_evaluation.py` — holdout comparison of EMOS vs the legacy
  pipeline (CRPS, MAE, modal-bucket probability, PIT).
- The nightly scorecard tracks the same metrics forward-looking, per station,
  plus trading results. Early single-city backtest numbers (Chicago 72.5% win
  rate, Sharpe 1.9) predate EMOS and multi-city correlation and should be
  treated as historical upper bounds, not expectations.

---

## Source Layout

```
src/deep_isobar/
  core/           # Types, logging
  data/           # City universe, NOAA/ACIS ingest, GFS GRIB ingest,
                  # ensemble_ingest (GEFS/EPS members + spread history),
                  # orderbook_collector (market-history recorder)
  models/         # Temperature ensemble, KDE, probability engine/surface
  market/         # Kalshi client (incl. official-result fetch), scanners
  trading/        # Alpha engine, Kelly sizing, bracket spreader,
                  # preflight circuit breakers, risk, execution
  calibration/    # emos (nonneg/ridge/trend options), emos_training
                  # (highs + lows), bias_loader (legacy), onboarding
  research/       # paper_trade_session, settle, verify_settlements,
                  # intraday_check, daily_scorecard, forecast_evaluation,
                  # validate_emos_variants (the holdout race)
  lab/            # Forecast Lab: asof_store (lookahead-proof reads),
                  # gefs_backfill (server fleet worker), maker_fills
  ops/            # backup
  dashboard/      # FastAPI API + auth + audit log + xlsx exports
  monitoring/     # watchdog
  notifications/  # Discord embeds (no-op without webhook)
  supervisor.py   # daily + interval job scheduler with catch-up

deploy/macos/     # launchd plist template (make mac-install)
deploy/server/    # drop-and-use Linux fleet: bootstrap.sh + systemd units
docs/             # OPERATIONS.md, ROADMAP.md, RUNBOOK_SERVERS.md,
                  # DATA_STORE.md, research/ (overnight-agent reports)
tests/            # pytest suite
config/           # settings.yaml (scheduler, risk, emos), cities.yaml
```

---

## Infrastructure

Current runtime: **MacBook** (launchd agent, left on during the day). The
Win11 PC and the Dell PowerEdge servers (R750 + 2×R520, not yet networked)
are future capacity; replay and batch onboarding belong on Linux where the
cfgrib file-locking issues don't exist.

---

## Feature summary

- **Calibrated forecasting** — per-station EMOS (min-CRPS, non-negative
  weights, 1.3°F floor), five models incl. NBM, max/min-of-trace extraction,
  ensemble-member spread accumulating toward flow-dependent variance.
- **Full-universe coverage** — 20 Kalshi cities × highs and lows calibrated
  and tracked; per-city `trade` flag separates learning from trading.
- **Risk & integrity** — fee-adjusted fractional Kelly with same-day
  correlation haircut; pre-trade circuit breakers (stub/bounds/NBM/params/
  sigma/liquidity/smoke); ops-health invariant alarms so failures are never
  silent; settlement cross-verified against the exchange nightly.
- **Evidence loop** — daily scorecard (P&L, Brier-edge-vs-market, CRPS/PIT
  per station); holdout race required before any model change ships.
- **Data assets** — orderbook history recorder (~305 books/10 min), live
  member-spread history, lookahead-proof as-of store, GEFS archive backfill
  worker for the server fleet.
- **Ops** — 10-job supervisor with sleep catch-up, launchd/systemd deploys,
  nightly rotated backups, append-only audit log, xlsx export on API
  endpoints, Discord notifications (optional).

## Roadmap

**Done (June–July 2026):** EMOS calibration + max-of-trace + station fixes ·
NBM + GEFS/EPS members · Kelly sizing + SELL spreading + fee model · circuit
breakers · settlement cross-verification · full-universe onboarding (highs +
lows, data-only) · scorecard/evaluation harness · orderbook collector ·
maker-fill simulator (verdict: hybrid, not maker-only) · non-negative EMOS
weights (won its holdout race) · server fleet package · fixed-LST settlement
windowing (NWS CLI midnight-LST day, year-round) · ops-health invariant
alarms · fully green test suite + golden Kalshi fixtures (real payloads
through parse→probability→settlement) · METAR smoke gate (stands cities down
under wildfire smoke — blocked Chicago on day one) · per-station EMOS
overrides (trend-variance shipped Dallas-only after winning its holdout
rematch) (see [docs/ROADMAP.md](docs/ROADMAP.md) for the full phased plan).

**Next:**
1. **Server day** (docs/RUNBOOK_SERVERS.md) — Tailscale + Proxmox/TrueNAS,
   then the GEFS member backfill → flow-dependent variance gate crosses.
2. **Forecast Lab (Phase 2)** — walk-forward sweeps on backfilled vintages;
   first customers: trend-variance rematch, Boston sea-breeze regime flag.
3. **City/low flips** — gated on 2–3 weeks positive Brier edge + collector
   liquidity evidence (+ midnight-minimum settlement spec for lows).
4. **Intraday repricing** then **live execution** — in that order, each
   gated on the evidence before it (docs/ROADMAP.md).
