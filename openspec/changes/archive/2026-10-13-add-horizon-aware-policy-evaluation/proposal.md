# Proposal

## Why
Daily paper losses and frozen-allocation returns do not evaluate a sequential policy over its intended horizon. Preserve the evaluation protocol when recommendations are issued and replay the exact frozen strategy with consistent execution and controls before proposing model improvements.

## What Changes
- Record objective, primary and diagnostic session windows, control settings and cost assumptions with new decisions.
- Add sequential frozen-policy/optimizer replay to decision evaluation, preserving existing accounting fields and explicitly distinguishing immature horizons, unavailable evidence and retrospective legacy protocols.
- Compare return, drawdown, volatility, turnover and objective-relevant diagnostics with unchanged holdings, equal weight and traditional optimizers on the same saved price path.
- Treat underperformance as evidence for review, never proof of optimality or a model defect; retain immutable input, actor and implementation references.

## Capabilities

### New Capabilities
- `decision-horizon-evaluation`: Reproducible objective-aware sequential decision evaluation using saved market evidence.

### Modified Capabilities
None.

## Impact
FinanceModel decision analysis, recommendation provenance, common simulator initial-book support, deterministic offline tests and beta verification. No live execution, policy activation, paid training or additional data ingestion.
