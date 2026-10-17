# Design

## Context
The news adapter and PPO feature group exist, but sensitivity output does not measure realized returns. Existing saved-book lifecycle APIs are available and immutable decision evidence already exists.

## Goals / Non-Goals
Goals: runnable local frozen-corpus experiment, actual costed forward accounting, explicit vintage blockers, separate caller-bound small-paper initialization. Non-goals: new training jobs, paid LLM evaluations, live trades, or presenting fixtures as performance evidence.

## Decisions
Use a strict JSON frozen study manifest with decision-time price vintages, completed bars, optional initial quantities and immutable news corpus. Shared inputs and identical capital/cost settings feed both arms; target policies use a common price-only holding state to isolate news effects while each execution book retains its own realized holdings/costs. Fills occur only at next observed open after the decision. Retrospective vintage or coverage gaps return an explicit ineligible report and prospective protocol. Preserve original availability dates and transaction costs. Reuse existing lifecycle APIs to create a separately named $100 paper portfolio with immutable local identity receipt; never infer an existing portfolio ID or mutate it.

## Risks / Trade-offs
Sparse news → no historical performance claim. Sensitivity coefficients are uncalibrated → label as hypothesis. Separate policy/common holding inputs versus divergent execution holdings → report this controlled convention explicitly. Local identity receipts → operator must preserve receipt for retry and audit.

## Migration Plan
Publish additive code without contract changes. Run fixture verification, then root runs real frozen evidence and beta lifecycle creation. Existing hosted APIs and production strategies are unchanged.
