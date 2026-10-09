# Tasks

## 1. Saved inference inputs

- [x] 1.1 Pin contracts 1.3.0 and add scoped same-environment plan/state/snapshot clients and permissions; verify pin checks and IAM/client unit tests.
- [x] 1.2 Resolve saved defaults and completed snapshot dates while retaining explicit inputs; verify empty, ambiguous, missing and future input tests and document the request modes.

## 2. Valuation and trades

- [x] 2.1 Value saved quantities on raw closes and reconstruct interim high watermark; verify adjusted/raw separation, invalid states and missing-history tests.
- [x] 2.2 Enrich frozen-policy recommendations with indicative shares, cash and state provenance; verify arithmetic, repeated read-only results and contract validation, and document limitations.

## 3. Beta integration

- [ ] 3.1 Deploy the reviewed beta artifact and verify natural hosted-agent requests invoke the existing MCP tool against initialized paper state, with Gamma/prod unchanged and incremental spend below $2.
