# Design

## Context
Existing decision analysis preserves paper accounting but compares frozen allocations. Frozen actors and optimizer inputs exist; the common simulator starts in cash only.

## Goals / Non-Goals
Goals: deterministic replay from original holdings with objective-aware fixed windows and cautious interpretation through the existing evaluator.
Non-goals: forecasts, broker execution, activation, paid training, causal news inference or proof of statistical significance.

## Decisions
Persist versioned evaluation contracts in provenance. Use 1/5/21-session diagnostics and recorded episode length or optimizer horizon as primary; legacy fallback is retrospective.
Extend the common simulator with validated starting shares and cash. Use the same execution settings for all controls. Freeze actor, configuration, optimizer settings and first input evidence; never read the current advisory pin.
Bound replay to maximum protocol horizon (252 sessions). Use raw bars for execution and point-in-time adjusted closes for actor/estimator features. Refuse unaccounted forward corporate actions and incompatible initial history.
Compute discounted reward only if frozen gamma exists. Preserve NAV, fills and sequential targets in existing immutable analysis artifacts.

## Risks / Trade-offs
Saved provider history can be revised → verify original inputs and disclose historical-vintage limitations.
Overlapping windows are dependent → report descriptive evidence without statistical proof.
Legacy protocols lack predeclaration → mark retrospective controls explicitly.

## Migration Plan
Backward-compatible response extension; beta deployment and live MCP checks owned by root coordinator. Roll back prior Model artifact if needed. No strategy switch.
