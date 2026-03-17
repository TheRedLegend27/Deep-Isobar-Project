Project: Deep Isobar

Relevant docs:
- ARCHITECTURE.md
- INTERFACES.md
- MODULE_DEPENDENCIES.md

Follow these rules:

Iterative Development:
Build only the requested module.

Prompt Engineering:
Follow the interfaces exactly as defined in INTERFACES.md.

Error Handling:
Raise clear errors for invalid inputs.

Testing:
Include pytest tests.

Now build:

Build the module `src/deep_isobar/core/types.py` for Deep Isobar.

Context:
- This module contains shared dataclasses used across the system.
- Implement these dataclasses exactly:
  - `CityProfile`
  - `ForecastPoint`
  - `ForecastShiftEvent`
  - `EnsembleSummary`
  - `TemperatureDistributionSnapshot`
  - `OrderBookSnapshot`
  - `MarketContract`
  - `TradeSignal`
  - `RiskDecision`
  - `ExecutedTrade`

Requirements:
- Use Python dataclasses
- Add type hints for every field
- Use `datetime.date` and `datetime.datetime` where appropriate
- Keep these as pure data containers with no business logic
- Match field names from `INTERFACES.md`
- Do not add extra fields unless necessary for correctness
- Keep the module import-safe and dependency-light

Output requirements:
1. Explanation
2. Python implementation
3. Example usage
4. Minimal tests that instantiate each dataclass
5. Edge cases handled

Stop after this module.