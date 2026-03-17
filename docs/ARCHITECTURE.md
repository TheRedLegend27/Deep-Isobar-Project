# Deep Isobar Architecture

Deep Isobar follows a modular quantitative trading architecture.

---

# System Layers

Data Layer
Feature Layer
Research Layer
Signal Layer
Execution Layer

---

# Data Layer

Sources:

NOAA temperature observations
ERA5 historical datasets
Prediction market price feeds

Outputs stored in:

Parquet datasets
Feature Store

---

# Feature Layer

Transforms raw data into structured signals.

Examples:

forecast_high_mean
forecast_high_std
market_probability
alpha

---

# Research Layer

Used for strategy development.

Tools:

pandas
scikit-learn
PyTorch

Research questions:

Do markets misprice temperature probabilities?
How does forecast uncertainty affect pricing?

---

# Signal Layer

Generates trading signals.

alpha = model_probability − market_probability

Rules:

alpha > threshold → BUY
alpha < −threshold → SELL
otherwise → HOLD

---

# Execution Layer

Handles interaction with exchanges.

Responsibilities:

submit orders
track fills
manage positions
enforce risk limits

---

# Server Deployment

Data Node — PowerEdge R520

runs:

feature store
backtests
historical datasets

Compute Node — PowerEdge R750

runs:

ensemble forecasts
probability engine
market scanner
trade execution

---

# System Pipeline

Weather Forecast Models
↓
Temperature Ensemble
↓
Temperature Probability Engine
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
