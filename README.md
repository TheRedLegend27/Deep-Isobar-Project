# Deep Isobar

Deep Isobar is an autonomous weather-intelligence and prediction market trading system.

The system identifies **mispriced weather prediction contracts** by comparing statistical weather model probabilities against market-implied probabilities.

Primary markets include contracts from:

* Kalshi
* Polymarket

The initial strategy focuses on **daily high/low temperature markets** because they:

• occur every day
• exist across many cities
• are easier to model statistically
• produce large datasets for backtesting

---

# Strategy

Deep Isobar detects **alpha**.

alpha = model_probability − market_probability

Example:

Model probability: 0.31
Market probability: 0.52

Alpha = -0.21

Trade:

SELL

---

# Core Pipeline

Forecast Models
↓
Temperature Ensemble
↓
Probability Engine
↓
Market Price Adapter
↓
Alpha Engine
↓
Market Scanner
↓
Trade Execution
↓
Risk Management

---

# Infrastructure

Development Machine
Mac

Deployment Servers

Compute Node
Dell PowerEdge R750

Responsibilities:

• forecast models
• probability engine
• market scanner
• trade execution

Data Node
Dell PowerEdge R520

Responsibilities:

• feature store
• historical datasets
• backtesting

---

# Project Structure

deep-isobar/

docs/modules
src
tests

---

# Development Rules

Iterative development only.

Agents must:

• implement one module at a time
• include tests
• include logging
• validate inputs

Never request the entire application in one step.

---

# Data Sources

Weather data:

NOAA historical observations
ERA5 reanalysis

Market data:

Kalshi
Polymarket

---

# Goal

Build an autonomous system capable of:

• detecting mispriced weather markets
• executing trades automatically
• managing risk
