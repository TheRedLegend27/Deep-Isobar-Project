# Module: Feature Store

Purpose

Store structured datasets used by models, backtests, and trading engines.

Storage format:

Parquet files

---

# Responsibilities

save_feature
load_feature
list_features

---

# Inputs

pandas DataFrame

---

# Outputs

Parquet dataset stored in feature_store directory.

---

# Example

timestamp | city | forecast_high_mean | forecast_high_std

2026-07-04 | Chicago | 84.3 | 2.1

---

# Edge Cases

invalid dataframe input
missing file
corrupt parquet file

---

# Tests

save feature
load feature
invalid data input
