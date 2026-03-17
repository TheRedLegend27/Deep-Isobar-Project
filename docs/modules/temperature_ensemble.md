# Module: Temperature Ensemble

Purpose

Combine multiple forecast models into a single forecast distribution.

---

# Inputs

List of forecast temperatures

Example

[84, 86, 83, 82]

---

# Outputs

EnsembleForecast object

mean
standard deviation
model count

---

# Example Output

mean = 83.75
std_dev = 1.71
models = 4

---

# Edge Cases

empty input list
missing model forecasts

---

# Tests

mean calculation
empty forecast error
