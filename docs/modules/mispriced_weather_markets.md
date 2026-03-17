# Module: Mispriced Weather Prediction Markets

Purpose

Identify cities and contract types where prediction markets consistently misprice probabilities.

Weather markets are most inefficient in:

• cities with volatile temperature swings
• cities where forecast errors are common
• extreme temperature thresholds
• late-day forecast shifts

Deep Isobar prioritizes these locations.

---

# High Alpha Cities

These cities historically produce larger forecast uncertainty and distribution tails.

Chicago

station: KORD

Reasons

• lake breeze effects
• cold front variability
• large temperature swings

Typical contracts

CHI_HIGH_GE_85
CHI_HIGH_GE_90

---

Denver

station: KDEN

Reasons

• altitude effects
• fast-moving weather fronts
• strong diurnal swings

Typical contracts

DEN_HIGH_GE_80
DEN_HIGH_GE_85

---

Phoenix

station: KPHX

Reasons

• extremely tight distributions
• small forecast shifts change probabilities significantly

Typical contracts

PHX_HIGH_GE_105
PHX_HIGH_GE_110

---

Dallas

station: KDFW

Reasons

• thunderstorm boundaries
• rapid heat spikes

Typical contracts

DAL_HIGH_GE_90
DAL_HIGH_GE_95

---

Atlanta

station: KATL

Reasons

• summer convection uncertainty
• cloud cover shifts

Typical contracts

ATL_HIGH_GE_90
ATL_HIGH_GE_95

---

# High Alpha Contract Types

Prediction markets struggle with **tail probabilities**.

Best opportunities occur at:

temperature ≥ extreme thresholds

Example

CHI_HIGH_GE_90

Market often overestimates these probabilities.

---

# Tail Mispricing

Markets often assume near-normal distributions.

Real weather distributions often have:

• skew
• fat tails
• multimodal peaks

This creates large pricing errors in:

≥ 90°F
≥ 95°F
≤ 20°F

---

# Forecast Shift Alpha

Prediction markets update slowly after forecast changes.

Example

GFS forecast shift

84°F → 90°F

Markets may take minutes or hours to adjust.

Deep Isobar monitors these shifts continuously.

---

# Market Inefficiency Patterns

Common inefficiencies include:

slow market updates after model runs

large bid-ask spreads

low liquidity contracts

These create trading opportunities.

---

# City Prioritization Score

Cities are ranked by:

forecast volatility
forecast error history
market liquidity

Example score

Chicago

volatility score: 8.4
market liquidity score: 6.2

priority rank: HIGH

---

# Outputs

List of prioritized markets

Example

CHI_HIGH_GE_90
DEN_HIGH_GE_85
PHX_HIGH_GE_110

---

# Responsibilities

Identify profitable cities

Identify profitable contract thresholds

Update rankings periodically

---

# Tasks

build city scoring system

build contract priority list

feed results into Market Microstructure Scanner
