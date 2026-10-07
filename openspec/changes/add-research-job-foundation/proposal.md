# Proposal

## Why

FinanceModel owns research compute for the planning platform, but nothing yet defines how experiments run, what inputs they see, how strategies are compared fairly or how results reach the platform without bypassing its validation. Without one common evaluator and job interface, every strategy family (controls, classical optimizers, and later RL, LLM swarm and Jev) would be measured under different data, costs and timing, and FinanceLambdasTool would have no stable contract for `submit_experiment`, `get_job_status` and `get_experiment_result`. This change builds the CPU-only, fixture-backed foundation (FinanceModel phase 1) that all later strategy families plug into. It must stay within the CPU research allocation (USD 7 by default) of the USD 50 total AWS budget (user decision 2026-10-07).

## What Changes

- Add **research workspace storage** owned by FinanceModel (per environment, encrypted, with retention), separate from platform storage. It reads approved platform input snapshots **read-only** through trusted artifact references, and verifies each snapshot's manifest checksum before use.
- Define the **initial instrument**: S&P 500 exposure through a tracking ETF daily series (for example SPY), kept distinct from the index level and the constituent universe; completed daily observations only, no intraday. Phase 1 uses fixtures and a mock provider. Provider decided 2026-10-07: the platform's ingestion Lambda or container retrieves SPY daily data with `yfinance` (pinned, with the `exchange_calendars` XNYS calendar). FinanceModel reads ONLY approved platform snapshots, never calls yfinance and has no yfinance dependency; provider library version and retrieval timestamp flow into dataset lineage; no retrieved market data is committed to the public repo (fixtures stay synthetic, per Yahoo's personal and research use terms) and CI uses the mock provider.
- Add **evaluation dataset preparation and versioning**: deterministic, content-addressed datasets derived from `input_snapshot_id`s plus a preparation `configuration_id`, with point-in-time availability enforcement, chronological train/validation/test splits, walk-forward folds with an embargo, an untouched holdout whose access is logged, and a prospective paper period.
- Add a **common evaluator and paper execution simulator**: one implementation of point-in-time inputs, portfolio constraints, fees, execution timing, liquidity/slippage and rebalancing, used by every strategy family. It never places live orders.
- Add **control strategies** (cash, buy-and-hold, equal-weight) and **classical optimizers** (minimum-variance, mean-variance, scenario-CVaR), each run as a CPU SageMaker Processing Job.
- Add the **experiment job interface** consumed by FinanceLambdasTool: submit (mints `run_id`), status, result, cancel and list, with idempotency, a job lifecycle, per-job time limits, a per-environment concurrency lease, failure and cancellation semantics, and `completion_status` reported separately from `solution_status`.
- Add **cost guardrails**: a pre-flight cost estimate and a per-category budget check before every paid job (CPU research jobs draw on the `cpu_research` category, default USD 7; the `gpu` category, default USD 25, is reserved for the next change), an explicit human approval gate for paid jobs above the auto-approve threshold, `BUDGET_EXCEEDED` refusals, and cost-allocation tags carrying `run_id`. AWS Budgets alerts and the 100% deny action stay owned by FinancialPlanning; FinanceModel reads their state.
- Add **run-output staging**: production-candidate runs hand a validated-shape output bundle to the platform's staging area. FinanceModel never creates plan versions, publications or executions.
- Add a **model registry** that mints `model_version` for each strategy implementation (code plus container digest plus parameter schema).
- Add **benchmark reporting** that keeps portfolio performance, model accuracy (where applicable) and compute cost in separate sections.
- Add the **FinanceModel deployment pipeline scope**: the repo pipeline deploys job definitions, containers, the job API and the registry, never individual experiments. No model fitting in CodeBuild or Lambda.
- **Out of scope:** RL, the Qwen3.6-27B agent swarm, TypeSafe Jev and GPU compute (see change `add-learning-and-llm-strategies`), real market-data runs before the platform's yfinance ingestion is deployed and SPY snapshots are approved (contracts OQ-5 resolved 2026-10-07), any direct market-data provider call from FinanceModel, and live trading, Coinbase, AgentCore payments and wallet spending (never part of FinanceModel).

## Capabilities

### New Capabilities

- `research-workspace`: FinanceModel-owned research storage, read-only consumption of approved platform snapshots, and isolation from authoritative plan state.
- `research-datasets`: deterministic dataset preparation and versioning, point-in-time availability, chronological splits, walk-forward folds, untouched holdout and prospective paper periods.
- `paper-execution-simulator`: the common evaluator and paper execution simulator (constraints, fees, execution timing, liquidity, rebalancing) shared by all strategy families.
- `baseline-strategies`: cash, buy-and-hold and equal-weight controls, and minimum-variance, mean-variance and scenario-CVaR optimizers.
- `experiment-job-interface`: submit/status/result/cancel/list operations, `run_id` minting, idempotency, job lifecycle and result semantics for consumers.
- `job-execution-controls`: time limits, concurrency lease, cancellation, failure handling, pre-flight cost checks and paid-job approval.
- `run-output-staging`: handoff of production-candidate run outputs to the platform staging area for platform-side validation.
- `model-registry`: `model_version` minting and registration of strategy implementations and their artifacts.
- `benchmark-reporting`: comparable benchmark reports that separate portfolio performance, model accuracy and compute cost.
- `job-deployment-pipeline`: what the FinanceModel pipeline deploys and promotes, and the ban on training in CodeBuild and Lambda.

### Modified Capabilities

None. No specs exist yet in this repository.

## Impact

- **FinanceModel:** new job-control API, job containers (one CPU image for data preparation, evaluation and baselines), SageMaker Processing Job definitions, research storage, run metadata, lease and registry stores, and the per-repo pipeline. Implementation follows the shared contracts in FinancialPlanning change `establish-cross-repo-contracts` (identifiers, `finplan-contracts` package, error envelope, SSM naming, pipeline standard).
- **FinanceLambdasTool:** consumes `/finplan/<env>/financemodel/api/job-endpoint` for `submit_experiment`, `get_job_status` and `get_experiment_result`. Until this change is released in an environment, those tools return `DEPENDENCY_UNAVAILABLE`.
- **FinancialPlanning:** must grant the per-environment FinanceModel job role read access to approved snapshots and write-only access to the run-staging prefix, and must validate and commit staged bundles. Its ingestion must record provider lineage (yfinance library version, retrieval timestamp, XNYS calendar version) and quality flags in approved snapshots (design.md FM-A5). Contract gaps are listed in design.md.
- **AWS (us-east-2):** per-environment CPU Processing Jobs only (observed `ml.m5.xlarge` processing quota is 1, shared by beta, gamma and prod, which all live in one account by user decision). No GPU, no endpoints, no always-on compute.
- **Cost:** phase 1 jobs run on synthetic fixtures with short time limits. USD 50 is the total AWS budget for the whole project. Default allocation (configurable in SSM): platform and serverless infrastructure plus storage USD 8, CPU research jobs USD 7, Bedrock explanation calls USD 5, GPU USD 25, reserve USD 5. Every paid FinanceModel job passes a pre-flight estimate against its category; AWS Budgets alerts at 50/80/100% with a deny action at 100%.
