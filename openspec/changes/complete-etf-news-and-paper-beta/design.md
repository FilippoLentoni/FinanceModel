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


## Strict completion update
The user requires actual hosted optimizer and PPO consumption. The saved USD100 profile therefore requires fresh licensed news in numerical solver inputs, with formation time equal to actual issuance unless explicitly historical. Existing compatible-state inference must consume the same versioned inputs; training validates frozen coverage and training rights before creating a learner. Current research-only forward metadata cannot train a policy. A valid compatible source artifact and hosted source/observation evidence are mandatory; tasks remain open until these exist. Public-domain historical archives need separately audited historical availability and explicit no-event coverage, not backdated retrieval.

## Selected daily universe (October 17, 2026)
The user selected GOOGL, AMZN, NVDA, MSFT, AAPL, META, TSLA and the S&P 500.
Represent the index using VOO, not an orderable index ticker. Collect eight
separate symbol-scoped news searches per trading day and persist raw/curated
records centrally. Preserve the $10 cumulative data limit and current authorized
expiry. Observed expected cost is $0.088/day, $0.88 for ten sessions; quotes
remain authoritative. Keep old receipts and their original symbol scope.
The existing API permits at most five symbols per response; capture the eight
assets in two four-symbol batches. This does not constitute an eight-asset PPO
training integration. Aligned approved price ingestion for all eight assets,
explicit daily collection/no-news/failure status, compatible multi-batch PPO
state integration and deployed recommendation evidence remain required. Do not
claim an old two-asset profile or policy covers the new investment universe.
