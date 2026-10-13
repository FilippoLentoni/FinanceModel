# Proposal

## Why

The deployed policy requires holdings and snapshot identifiers in every request, so a simple request for today's portfolio recommendation cannot invoke it automatically. Beta needs a saved paper book that can be valued from approved market data and passed to the existing frozen policy.

## What Changes

- Add saved-paper mode to the existing read-only recommendation operation while retaining explicit experiment inputs.
- Resolve the default research portfolio and newest approved snapshot through same-environment platform interfaces.
- Mark saved share quantities and cash to completed closes, preserve historical high watermark, and run the selected policy without retraining.
- Return indicative share changes and current-state provenance alongside the existing target weights and dollar changes.
- Deploy beta through the existing pipeline; keep production selection and Gamma/prod workloads unchanged.

## Capabilities

### New Capabilities

- `saved-portfolio-inference`: automatic saved-paper state resolution, valuation, frozen inference and indicative share recommendations.

### Modified Capabilities

None. Main specs have not yet been synchronized; this extends the completed selected-strategy serving change.

## Impact

FinanceModel serving orchestration, platform HTTP client, scoped inference IAM, contracts 1.3.0 dependency and focused serving tests. FinancialPlanning owns paper state and initialization; FinanceAgent and FinanceLambdasTool expose the existing MCP operation. This change does not execute proposed trades or implement missing diagnostic workers.
