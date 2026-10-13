# Proposal

## Why

The 37-value PPO state omits cross-asset dependence and temporal information. Existing research presets combine changes, preventing controlled attribution.

## What Changes

Implement backward-compatible state feature groups, controlled released ablations, point-in-time news records and eligibility, matched research experiments, reproducible result/report artifacts and beta pipeline gates.

## Capabilities

### New Capabilities
- `ppo-state-research`: Implement backward-compatible state feature groups, controlled released ablations, point-in-time news records and eligibility, matched research experiments, reproducible result/report artifacts and beta pipeline gates.

### Modified Capabilities
None; older incomplete change packages remain separately tracked.

## Impact

Owned implementation, configuration, tests, documentation and beta release in FinanceModel. Cross-repository integration: Platform budget → Model profiles/results → Tools catalog → Agent skills; the existing four-pipeline ownership is preserved. The USD 100 ceiling is a total project cap, not an incremental grant.
