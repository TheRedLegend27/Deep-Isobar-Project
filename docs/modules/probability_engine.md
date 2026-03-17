# Module: Temperature Probability Engine

Purpose

Convert forecast distribution into contract probabilities.

---

# Model

Normal distribution

Inputs

mean temperature
standard deviation
threshold

Example

mean = 84
std = 2
threshold = 85

Output

P(temp ≥ 85)

---

# Example Output

P(high ≥ 85) = 0.31

---

# Edge Cases

zero standard deviation
extreme thresholds
invalid input types

---

# Tests

probability bounds
distribution behavior
