# Deep Isobar — Research & Improvement Report

*Daily high-temperature contracts on Kalshi / Polymarket. Prepared June 2026.*

This report answers your eight research questions in order, each with key findings + a concrete recommendation, then closes with a ranked roadmap (impact vs. effort) and a list of things in your current design the literature says are outright wrong.

**The single most important takeaway up front:** your three most-cited symptoms — chronic under-confidence (model maxes at 0.14 where the market sits at 0.45), the Dallas ~8°F cold bias, and needing a *0.38* alpha threshold to trade — are not three problems. They are one problem wearing three hats: **your predictive distribution is mis-specified.** The 5.5°F variance floor and the 18z fixed-hour snapshot are the two root causes, and both are cheap to fix. Almost all the P&L is in Q1 and Q2.

---

## Q1 — Distribution sharpness & calibration

**Findings.** The established best practice for turning a small multi-model ensemble into a calibrated predictive distribution for station daily-max temperature is **Ensemble Model Output Statistics (EMOS), a.k.a. Non-homogeneous Gaussian Regression (NGR)**, introduced by Gneiting et al. (2005) and fit by **minimum-CRPS estimation**. For 2 m temperature you assume a Gaussian predictive distribution, model the **mean** as a linear function of the member forecasts (`μ = a₀ + a₁m₁ + … + a₄m₄`) and the **variance** as a linear function of the ensemble spread (`σ² = c + d·S²`). The CRPS for a Gaussian has a closed form, so the parameters `(a, c, d)` are fit by gradient descent over a rolling/seasonal training window. This simultaneously removes bias *and* corrects dispersion. It is the field-standard and consistently beats raw ensemble spread, especially in the short range and for temperature ([Gneiting et al. 2005, min-CRPS EMOS](https://www.researchgate.net/publication/228930829_Calibrated_Probabilistic_Forecasting_Using_Ensemble_Model_Output_Statistics_and_Minimum_CRPS_Estimation); [spatial NGR for temperature, MWR 2015](https://journals.ametsoc.org/view/journals/mwre/143/3/mwr-d-14-00210.1.xml); [lead-time-continuous postprocessing, QJRMS 2024](https://doi.org/10.1002/qj.4701)).

The "maximize sharpness subject to calibration" paradigm (Gneiting, Balabdaoui & Raftery 2007) is the right objective function for exactly your situation — you want the tightest distribution that still passes a PIT test ([Gneiting et al., calibration & sharpness](https://www.stat.washington.edu/research/reports/2005/tr483.pdf)).

Alternatives, ranked for your case:
- **BMA (Bayesian Model Averaging, Raftery et al. 2005):** a mixture of per-model kernels. Better than EMOS only when models are genuinely distinct or the distribution is multimodal. For 4 correlated NWP models targeting one scalar, it's heavier than needed.
- **Quantile regression / quantile regression forests:** nonparametric, captures the slight left-skew of daily max in some regimes, but needs far more training data than you have per station-month. Revisit once you have years of paired data.
- **EMOS/NGR is the right complexity** for 4 models × 5 stations.

**On the variance floor specifically:** a hard floor of 5.5°F is the bug behind your under-confidence. A calibrated predictive std for a T+1-day station max from a multi-model ensemble is typically **~1.5–2.5°F**, not 5.5. Do not floor at a guessed constant. Let `σ² = c + d·S²` be learned. If you keep a floor at all, set it to the *irreducible* observation + representation error: ASOS sensor (~0.5°F) ⊕ integer-settlement discretization (uniform on 1°F has std 1/√12 ≈ **0.29°F**) ⊕ residual snapshot error. That floors around **1.0–1.5°F**, not 5.5. A market that concentrates 0.45 on the modal 2°F bucket implies a predictive std on the order of ~1.7–2.0°F — which is what a calibrated EMOS will give you, and is why the market currently looks "overconfident" to you when in fact *you* are under-confident.

**Recommendation.** Implement per-station, per-month (or rolling 40-day) EMOS/NGR fit by minimum CRPS. Drop the 5.5°F floor entirely; replace with a learned `c + d·S²` and at most a ~1.3°F hard floor for discretization + sensor error. Validate with a PIT histogram (see Q7): you should see your current ∪-shape flatten. This one change should move your modal-bucket probability from ~0.14 toward the ~0.3–0.45 range and is the highest-ROI item in the whole report.

---

## Q2 — Better raw inputs

**Findings.**

**NBM (National Blend of Models) is the highest-value single addition, and it directly solves two of your problems at once.** It is NOAA's operationally calibrated, bias-corrected blend, and it already publishes **probabilistic daytime-max (MaxT) and nighttime-min (MinT) temperature as percentiles and exceedance values** — i.e. a ready-made calibrated daily-max distribution, not a fixed-hour snapshot. It is free on AWS Open Data, updated hourly, out to ~11 days ([NOAA NBM on AWS Open Data](https://registry.opendata.aws/noaa-nbm/); [NBM MaxT/MinT percentile products](https://vlab.noaa.gov/web/mdl/nbm-versions)). Use it both as an input *and* as an external benchmark for your own EMOS distribution — if your distribution disagrees badly with NBM's percentiles, you're probably the one who's wrong.

**Ensemble members beat deterministic runs for your lead time.** GEFS (31 members) and ECMWF-EPS (51 members) are available free through the **Open-Meteo Ensemble API**, which exposes `temperature_2m_max` as a native daily variable per member ([Open-Meteo Ensemble API](https://open-meteo.com/en/docs/ensemble-api)). The member spread is exactly the **flow-dependent uncertainty signal** you need to replace a static floor; on a calm, well-clustered day the spread is small (sharp distribution), on a frontal-passage day it's wide. The public competitor bot (Q8) already does the crude version of this — counting the fraction of 31 GFS members above the threshold. EMOS on the members is strictly better than member-counting because it corrects bias and recalibrates spread.

**HRRR:** high-resolution, good for mesoscale features (lake/sea breeze), but only runs to ~48 h, so it covers your T+18–36 h window but not longer-dated contracts. Worth adding specifically for short-lead, breeze-prone stations (Chicago, Boston). Medium priority, heavier ingest.

**MOS (GFS MOS MAV/MEX, LAMP):** classic station-specific calibrated guidance including daytime max. Largely **subsumed by NBM** now; low marginal value once you take NBM.

**How pros extract "daily max" (this is the fix for Dallas).** Nobody uses a single fixed-hour point as the daily-max proxy. They either (a) use the model's **native MaxT field** (NBM MaxT, GEFS/ECMWF `temperature_2m_max`, MOS daytime max), or (b) take the **max over the hourly 2 m-temperature trace** across the contract's local settlement day. Your 18z UTC snapshot = **1:00 pm CDT in Dallas**, but Dallas's summer high lands around 4–6 pm local (≈21–23z). An 18z snapshot therefore samples the temperature *hours before the peak* and during the steep part of the heating curve — which produces exactly the systematic cold bias of several degrees you observe. It is not a model bias at all; it's a sampling-time artifact, and your static `mean_bias_f` is papering over a state- and season-dependent error with a constant.

**Recommendation.** (1) Add NBM as both an input and a benchmark — biggest bang. (2) Pull GEFS + ECMWF-EPS members from Open-Meteo and feed member spread into the EMOS variance. (3) **Stop using the 18z point snapshot.** Switch to `max()` of the hourly trace over the local settlement day (you can already pull Open-Meteo hourly), and/or adopt native MaxT fields. This alone should eliminate the Dallas cold bias and shrink your per-station bias terms toward zero. (4) Add HRRR later for short-lead breeze days only.

---

## Q3 — Station-level quirks

**Findings, station by station** (difficulty and what drives busts):

- **KMDW (Chicago Midway) vs KORD (O'Hare):** Kalshi settles Chicago on **Midway**, Polymarket on **O'Hare**. Midway is more urban (denser South Side) and farther inland from the lake's direct NE-wind reach; O'Hare is NW and more exposed. On lake-breeze days (E/SE flow) the two can diverge several degrees, with **Midway usually the warmer station** because the lake-breeze front often stalls before reaching it. This is a *regime-dependent* gradient, not a constant — the highest-edge Chicago days are lake-breeze days, and they're also the days a snapshot proxy fails worst.
- **Central Park (NYC settlement) vs KJFK / KLGA:** Central Park is vegetated — in summer afternoons it runs **cooler than the airports** (canopy + grass), and at night in winter warmer (urban heat retention). JFK is coastal and **sea-breeze-suppressed**; LGA is on the East River, intermediate ([Central Park vs the three airports](https://thestarryeye.typepad.com/weather/2018/07/comparing-central-park-weather-observations-with-nycs-three-airports.html); [NYC urban heat island](https://a816-dohbesp.nyc.gov/IndicatorPublic/data-stories/urban-heat-island/)). **You list KJFK METAR as the NYC forecast input but settlement is Central Park — that is a station mismatch and a real bug** (more in the "wrong" list). JFK's sea breeze is not Central Park's microclimate.
- **KDFW vs KDAL:** Kalshi settles Dallas on **DFW** (the official climate station); Polymarket on **Love Field (DAL)**, which is more urban and typically **~0.5–1°F warmer**.
- **KPHL (Philadelphia):** airport by the Delaware/Schuylkill, urban, comparatively **well-behaved** — fewer mesoscale surprises. Good candidate for clean edge.
- **KBOS (Boston Logan):** sits directly on the harbor → **strongly sea-breeze-driven**, frequent large forecast busts when the sea-breeze front timing is off. Highest difficulty and highest variance of your set.

**Where the edge is.** Edge = (your model is good) AND (market is bad). Sea-breeze stations (BOS, JFK) are where deterministic-model traders misprice most, but they're *also* where your snapshot proxy fails — so net edge is noisy. The cleanest combination is stations where your model can be made genuinely good and the market is still lazy: **Chicago/Midway (lake-breeze regime modeling) and Philadelphia (low mesoscale noise).** Treat Boston as high-variance and size it down until you have a sea-breeze-aware model.

**Recommendation.** Forecast the **exact settlement station's lat/lon** for every city (Central Park, not JFK). Build a simple regime flag (onshore/offshore wind, lake-breeze potential) from the ensemble and let EMOS coefficients differ by regime where you have data. Concentrate capital on Chicago and Philadelphia first; quarantine Boston/JFK until breeze-aware.

---

## Q4 — Intraday alpha

**Findings.** The daily max has a **diurnal "lock-in" structure**: the conditional distribution of the final settle collapses as the observed running max climbs and the afternoon peak passes. Once you're past the local peak (~3–6 pm), the daily max is essentially determined, and brackets should price to ~0/1 — but retail pricing lags. Two windows are systematically most mispriced:
1. **Morning:** overnight prices go stale while fresh 06z/12z model runs land; the market hasn't repriced the new guidance.
2. **Mid-to-late afternoon lock-in:** the running max (from 5-min ASOS) already sits at or just past a bracket boundary and the remaining-day max is near-deterministic, yet the market still prices residual uncertainty.

A correct intraday model is `daily_max = max(observed_running_high_so_far, forecast_of_remaining-day_max | current temp, time-of-day, trend)`. Your 2 pm METAR running-high check is the right idea but under-powered — it's a sanity check, not a repricing engine, and you only act once at 7 am.

**Recommendation.** Move from one shot/day to **event-driven repricing**: re-run on each model cycle (00/06/12/18z) and on 5-min ASOS updates, recomputing the conditional max distribution given obs-so-far. The lock-in window is where you'll find the most reliable, lowest-variance fills. Crucially, **post these as resting limit (maker) orders** — both venues charge takers but not makers (Q5), so an intraday strategy that crosses the spread bleeds fees, while one that provides liquidity trades for free.

---

## Q5 — Cross-venue structure & fees

**Findings.** The paired stations (KORD−KMDW, KLGA−Central Park, KDAL−KDFW) make cross-venue trades **correlated-station spreads, not arbitrage.** Typical daily-max offsets: KMDW slightly warmer than KORD on average (~0–1°F, but several °F on lake-breeze days); KDAL ~0.5–1°F warmer than KDFW; Central Park cooler than LGA on summer afternoons, seasonally variable. The risk is the **standard deviation of the daily difference** — on the order of ~1.5–2.5°F for these pairs — which is large relative to a 1–2°F bracket. So there is real **basis risk**; a "spread" can settle in opposite brackets on the two venues.

**Fees (model both — your assumption that Polymarket is free is now outdated):**
- **Kalshi:** taker fee ≈ `0.07 × C × P × (1−P)` per contract, peaking at **1.75% of contract value at P = 0.5** and tapering to ~0 at the extremes; **maker orders are free**; no settlement, membership, or ACH fees ([Kalshi fee schedule](https://kalshi.com/fee-schedule); [Kalshi fees guide 2026](https://www.predictionhunt.com/blog/kalshi-fees-complete-guide-2026)).
- **Polymarket (changed in 2026):** now charges **taker fees by category**; **weather uses the 0.05 coefficient, peaking at $1.25 per contract-unit at P = 0.5**, symmetric around 50% and decreasing toward the extremes. **Makers are never charged** and are paid via a maker-rebate program ([Polymarket fees](https://docs.polymarket.com/trading/fees); [Polymarket trading fees help](https://help.polymarket.com/en/articles/13364478-trading-fees)).

**Recommendation.** Do **not** treat cross-venue as riskless arbitrage. Model each leg as a directional bet on its own station and size the *pair* against the joint distribution of (station A, station B) given a shared airmass — the legs are highly but not perfectly correlated, and the residual is your basis risk. Only put it on when *both* legs are independently +EV after fees, not because the spread "should" close. Given both venues are maker-free, your default execution should be resting limit orders on both; pure taker spread trades rarely survive ~1.75% (Kalshi) + ~$1.25/contract (Polymarket weather) + ~2°F basis std.

---

## Q6 — Sizing & risk

**Findings.** For a binary contract bought at price `q` (pays 1), with your model probability `p`, the Kelly fraction is

```
f* = (p − q) / (1 − q)
```

(net odds `b = (1−q)/q`). The literature on **parameter uncertainty** (Baker & McHale 2013, *Optimal Betting Under Parameter Uncertainty*) shows raw Kelly overbets out-of-sample because it treats your *estimated* `p` as the truth; the optimal correction is **bet shrinkage that increases with the variance of your probability estimate** ([Baker & McHale 2013](https://ideas.repec.org/a/inm/ordeca/v10y2013i3p189-199.html); [why fractional Kelly under uncertainty](https://matthewdowney.github.io/uncertainty-kelly-criterion-optimal-bet-size.html)). Fractional Kelly (¼–½) is the standard robust proxy; thin edges are where estimation error hurts most, so shrink hardest there. The public competitor bot uses **0.15 Kelly** — conservative but defensible.

**Same-airmass correlation is the risk your design under-handles.** Kelly assumes independent bets. Your five cities on a single day are driven by overlapping synoptic airmasses — a warm bust hits Chicago, NYC, Philly and Boston together. Summing independent per-city Kelly fractions therefore *overbets* the day. Your flat `$50/day` cap blunts this but isn't principled.

**Recommendation.** Use **fractional Kelly at ~0.25**, and additionally **shrink each bet by your calibration confidence** (e.g. scale by `1 − Var(p̂)`-style factor, or simply discount more on stations/regimes where your PIT is worst). Replace the flat daily cap with a **correlation-aware total-exposure limit**: estimate the average pairwise correlation of same-day outcomes and haircut the summed Kelly fractions accordingly (a quick approximation is to size the *portfolio* as if you had `N_eff = N / (1 + (N−1)ρ̄)` independent bets). And implement **SELL sizing** — you're currently leaving half the book untouched (see "wrong" list).

---

## Q7 — Evaluation

**Findings.** Track the proper-scoring trio plus calibration diagnostics:
- **CRPS** — one number combining sharpness and calibration; your primary optimization and tracking metric.
- **Brier score with Murphy decomposition** — reliability (calibration), resolution (discrimination), uncertainty. Watching reliability vs. resolution separately tells you *whether* edge decay is a calibration drift or a loss of signal.
- **Reliability/calibration diagram** — predicted vs. observed frequency.
- **PIT histogram** — the fastest read on your exact failure mode: a **∪-shape = under-dispersion** (your current state — distribution too narrow… wait, you're *over*-floored, so your PIT is ∩-shaped/over-dispersed); a **∩-shape = over-dispersion**; a sloped/triangular shape = bias ([PIT/rank histogram interpretation](https://ar5iv.labs.arxiv.org/html/1310.0236); [calibration & sharpness](https://hal.science/hal-00363242/document)). With a 5.5°F floor your distribution is too *wide* relative to truth, which is over-dispersion → expect a ∩ (hump-shaped) PIT now, flattening after EMOS.

**Sample size for win rate.** Win rate is a weak, odds-blind metric. Distinguishing 72.5% from a 50% coin is easy, but distinguishing 72.5% from, say, 65% (the difference between "great" and "fine") is hard: with `SE ≈ √(p(1−p)/n)`, a ±5% 95% CI on p = 0.725 needs **~320 trades**; a ±3% CI needs **~885**. And those must be *near-independent* — your same-day correlated trades inflate the effective sample much less than the raw count, so plan on **several hundred genuinely independent trading days** before the 72.5% is trustworthy. Your backtest is also Chicago-only, GFS-only, in-sample-ish; the Sharpe 1.94 is almost certainly optimistic once within-day correlation and multi-city/multi-model reality are included.

**Recommendation.** Build a daily scorecard logging CRPS, Brier + decomposition, reliability-diagram points, and a rolling PIT histogram, **sliced by station and month** to catch regime-specific decay fast. Track realized P&L and CRPS *against a baseline* (e.g. NBM percentiles, or "trade at market") rather than win rate. Treat any single-config backtest number as an upper bound.

---

## Q8 — Competition scan

**Findings.** Kalshi weather markets (the **KXHIGH** series: KXHIGHNY, KXHIGHCHI, etc.) attract weather-desk pros, quant hobbyists, and a growing set of bots. The most directly comparable public artifact is the open-source **suislanchez/polymarket-kalshi-weather-bot** — it trades the KXHIGH series and Polymarket using a **31-member GFS ensemble from Open-Meteo, member-count = probability, fractional Kelly (0.15), Brier-score calibration tracking**, with NWS for settlement ([repo](https://github.com/suislanchez/polymarket-kalshi-weather-bot)). Commercial bot vendors advertise "**164 ensemble members across 4 forecast systems**" for the same markets ([Kalshi trading bots guide 2026](https://www.botforkalshi.com/blog/kalshi-trading-bots-complete-guide); [Kalshi weather markets help](https://help.kalshi.com/en/articles/13823837-weather-markets)). The recurring edge thesis in these writeups is explicit: *"most Kalshi weather traders don't systematically process raw model data."*

**Implication for you.** Owning a multi-model ensemble is now **table stakes**, not edge — the public bots already do member-counting. Your differentiation has to come from where they're weak: **calibration quality (EMOS vs. naive member-counting), correct daily-max extraction, station-microclimate/regime modeling, and intraday repricing with maker execution.** Those are precisely the items in this report. The competitive window for "I have an ensemble and they don't" is closing; the window for "my distribution is actually calibrated and I reprice intraday" is open.

---

## Top-5 ranked roadmap (impact vs. effort)

| # | Action | Impact | Effort | Why it's ranked here |
|---|--------|--------|--------|----------------------|
| **1** | **Kill the 5.5°F floor; fit per-station-month EMOS/NGR variance by min-CRPS** | ★★★★★ | ★★ | Directly fixes the under-confidence (0.14 → ~0.3–0.45 modal bucket) and lets you trade real edges instead of 0.38 monsters. Closed-form Gaussian CRPS makes the fit easy. |
| **2** | **Replace the 18z snapshot with max-of-hourly-trace + native MaxT (NBM/GEFS)** | ★★★★★ | ★★ | Eliminates the Dallas ~8°F cold bias at the source and removes a state-dependent error you're currently hiding in a static constant. |
| **3** | **Add NBM (input + calibrated benchmark) and GEFS/ECMWF-EPS members via Open-Meteo** | ★★★★ | ★★ | Free, low-latency, gives you a professional calibrated daily-max distribution to check yourself against and flow-dependent spread for EMOS variance. |
| **4** | **Intraday event-driven repricing (5-min ASOS + late runs + lock-in), maker orders only** | ★★★★ | ★★★ | The morning-staleness and afternoon lock-in windows are where the market is most wrong; maker execution makes it fee-free. |
| **5** | **Evaluation harness (CRPS/Brier-decomp/PIT/reliability) + ¼-Kelly with correlation haircut + SELL sizing** | ★★★ | ★★ | Protects 1–4 from silently breaking, sizes correlated same-day bets correctly, and unlocks the half of the book you currently ignore. |

Fix-the-distribution (1–3) before anything else — they're cheap and they're where the money is. Items 4–5 compound on top of a calibrated model; running them on the *current* mis-specified distribution would just trade noise faster.

---

## Things in your current design the literature says are outright wrong

1. **The 5.5°F std floor.** Post-ensemble it is the single biggest error. A calibrated T+1 station-max std is ~1.5–2.5°F; flooring at 5.5 forces chronic over-dispersion and is the direct cause of your under-confidence. Replace with learned EMOS variance + at most a ~1.3°F discretization/sensor floor.
2. **The 18z fixed-hour snapshot as a daily-max proxy.** Wrong methodology. It samples Dallas hours before its afternoon peak → the ~8°F cold bias. Pros use native MaxT or max-of-trace. Your static bias term is masking a state-dependent error rather than fixing it.
3. **Forecasting KJFK to settle Central Park (NYC station mismatch).** JFK's coastal sea-breeze microclimate is not Central Park's. Forecast the exact settlement station's lat/lon for every city.
4. **A single static `mean_bias_f` absorbing both model bias and snapshot-vs-max error.** These are different errors — one roughly constant, one strongly seasonal/diurnal/state-dependent. Conflating them guarantees residual bias in some regimes. Fix the snapshot at the source (item 2) and let EMOS handle the rest.
5. **The |alpha| ≥ 0.38 trade threshold.** With a calibrated distribution you will almost never see a *legitimate* 38-point edge; when you do, it usually means the *market* is right and your model is broken. The need for such a huge threshold is itself a symptom of the mis-specified distribution. After EMOS, real edges are ~3–10 points and the threshold should drop accordingly.
6. **BUY-only / no SELL spreading.** You're systematically ignoring half the opportunity set. Overpriced longshot brackets (the market's fat modal bucket spillover) are often the *cleaner* edge, and selling them is frequently better risk-adjusted than buying.
7. **Treating cross-venue as arbitrage and assuming Polymarket is cheap/free.** It's basis-risk spread trading with ~2°F residual std, and as of 2026 Polymarket charges a weather taker fee (0.05 coeff, ~$1.25 peak at P=0.5). Model both fee schedules and trade the pair only when both legs are independently +EV.
