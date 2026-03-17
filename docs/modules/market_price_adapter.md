# Module: Market Price Adapter

Purpose

Convert order book prices into implied market probability.

---

# Inputs

bid price
ask price

Example

bid = 0.51
ask = 0.54

---

# Output

market_probability

Example

0.525

---

# Calculation

market_probability = (bid + ask) / 2

---

# Edge Cases

bid greater than ask
prices outside 0-1 range
missing market data

---

# Tests

mid price calculation
invalid price rejection
