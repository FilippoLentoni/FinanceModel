# Native crypto approaches in the consolidated comparison

`finplan_model.forward.crypto.crypto_adapter` is a weights-only adapter for the
six existing native crypto approaches: cash, BTC buy-and-hold, equal-weight
buy-and-hold, minimum variance, mean variance and CVaR. Each must have its own
isolated portfolio with the comparison experiment's starting capital, rather
than reusing the legacy USD 100 books. Their cohort is explicitly BTC-USD/ETH-USD,
hourly and 24/7, separate from equity and fixed-universe RL cohorts.

The external adapter map is `{'crypto_native': crypto_adapter}`. The persisted
shared source supplies `native_crypto_dataset`, the actual platform
`api/crypto-dataset-response`, and `investable_universe=['BTC-USD','ETH-USD']`.
The model never retrieves provider data. Normal forward runs reject synthetic
source datasets; only an explicitly labelled `trigger=controlled_replay` permits
test fixtures. Gaps, late availability, stale data, unsupported holdings and
missing cash fail visibly before a solver executes. The cash control needs no
price input. News is not consumed by these six strategies.

Before each solver executes, the storage callback receives its actual ordered
hourly closes, return scenarios, mean vector, sample covariance plus the native
1e-10 diagonal regularizer, current holdings/weights, complete solver settings,
source provenance and bounded fallback configuration. Units and ordering are
included. Legacy setting names `lookback_days` and `horizon_sessions` mean hours
inside this native crypto policy; the adapter does not manufacture daily bars.
The actual classical solver, objective and scalar fallback match the legacy
implementation. Outputs are allocations, not calibrated value forecasts.

Buy-and-hold controls allocate until actual invested positions exist, then return
an explicit no-rebalance hold with their current weights and residual cash.
They do not buy repeatedly to erase drift. No historical hourly candle is used
as a forward execution price by this adapter. A later eligible actual quote must
reach the common point evaluator and authoritative paper ledger separately.
The caller must enforce the declared hourly opportunities, actual quote
availability/freshness and 24/7 venue rules.

The common experiment execution policy must be declared and versioned. Legacy
crypto `rebalance` fees are 60 bps of the slippage-adjusted notional with 10 bps
slippage: effective reference-notional costs are 70.06 bps for buys and 69.94 bps
for sells. A common evaluator configured with independent 60 bps fee and 10 bps
spread/slippage is a different, explicit comparison policy; it must not be
labelled the exact legacy fill protocol. This adapter preserves decision math,
not a second authoritative ledger or a claim of identical execution costs.

Validation: `uv run pytest tests/unit/test_forward_crypto.py
 tests/unit/test_crypto_research.py` covers the six boundaries, native numerical
parity, exact capture before invocation, failed-save prevention, missing/future/
stale/gapped/synthetic input rejection, buy-and-hold drift, fallback and the
normal hourly external-adapter wrapper. Fixtures are controlled tests only.
