# Module: Alpha Engine

Purpose

Detect mispricing between model and market probabilities.

---

# Formula

alpha = model_probability − market_probability

---

# Trading Rules

alpha > threshold → BUY
alpha < −threshold → SELL
otherwise → HOLD

---

# Inputs

contract
model_probability
market_probability

---

# Outputs

TradeSignal object

contract
side
alpha
confidence

---

# Tests

buy signal
sell signal
hold signal
