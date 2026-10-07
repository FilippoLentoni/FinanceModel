# Design

## Context

See proposal.md (Why) for motivation and `specs/` for requirements. FinanceModel is the second repo in the integration order (platform, then model service, then tool wrappers, then agent). All shared rules come from FinancialPlanning change `establish-cross-repo-contracts` and are referenced, not redefined here: identifier formats and minting (`run_id` and `model_version` are minted by FinanceModel), the `finplan-contracts` package (pinned by exact version and digest), the error envelope and registered codes, `completion_status` versus `solution_status`, trusted artifact references, SSM naming `/finplan/<env>/financemodel/<category>/<name>`, the release manifest, the pipeline standard and the isolation rules (ENV-03, ENV-04, ENV-05, ENV-14).

### Observed facts (read-only discovery, 2026-10-07)

- Region us-east-2. No ECR repositories exist there yet. No SageMaker endpoints, models or endpoint configs exist in any region.
- SageMaker quotas in us-east-2 (partial): `ml.m5.xlarge` processing **1**, training 15. GPU processing quotas are 0 (only relevant to the next change).
- One AWS account in us-east-2 holds beta, gamma and prod (user decision 2026-10-07, contracts OQ-1 resolved). Environments are isolated by naming, tags, IAM permission boundaries with environment-tag denies, and separate buckets, tables and endpoints per environment. Multi-account is only a possible future migration. **Account-level SageMaker quotas are therefore shared by beta, gamma and prod.**
- USD 50 is the **total AWS budget** for everything (user decision 2026-10-07, contracts OQ-7 resolved). Default allocation, configurable in SSM: platform/serverless infrastructure plus storage USD 8, CPU research jobs USD 7, Bedrock explanation calls USD 5, GPU (Qwen3.6-27B and RL GPU work) USD 25, reserve USD 5. AWS Budgets alerts at 50/80/100% with a deny action at 100%.
- Pipeline bootstrap uses the user's existing authenticated AWS CLI session and reuses an existing GitHub CodeConnection referenced through SSM (contracts OQ-11 and OQ-2, non-blocking). The bootstrap still creates scoped pipeline, deploy and service roles.
- Pipeline bootstrap is approved in principle by the user (decision 2026-10-07, item 11). It is not an approval blocker: it runs once, only after the bootstrap IaC is implemented, and at run time the exact stacks and a cost estimate are shown before it runs within that approval. Nothing is deployed during spec work.
- Market-data provider decided (user decision 2026-10-07, contracts OQ-5 resolved): the platform's phase 2 daily provider adapter uses the Python library `yfinance` (Yahoo Finance; unofficial, not affiliated with Yahoo, no API key, rate-limited and can break on upstream changes; pinned version) for SPY daily completed OHLCV, adjusted close, dividends and splits, with the `exchange_calendars` XNYS calendar (pinned). It runs inside the FinancialPlanning ingestion Lambda or container; FinanceModel never calls it.

### Assumptions (unverified; revisit when the related question closes)

- FM-A1. Python is the job and API language, and AWS CDK is the IaC (contracts A2, A3).
- FM-A2. Processing quotas for small CPU types other than `ml.m5.xlarge` (for example `ml.t3.medium`) are unknown. Phase 1 plans for a single CPU slot.
- FM-A3. The platform exposes a snapshot-resolve operation that returns trusted artifact references and an approval status, and grants the FinanceModel job role read access to approved snapshots (contracts D1 and D5). Not yet specified in the platform change; see CONTRACT GAP list.
- FM-A5. Approved real ETF snapshots carry provenance fields for the provider adapter, pinned library version, retrieval timestamp, exchange calendar and version, the XNYS session list for the covered range, and quality flags for empty or partial responses. To confirm against the platform ingestion spec and contracts snapshot schema before task 3.7; if a field is missing, FinanceModel raises it as a contract gap rather than calling the provider.
- FM-A4. Open-source solver libraries (for example CVXPY with an open-source solver) fit the optimizer needs. Chosen at implementation time; no commercial solver.

## Goals / Non-Goals

**Goals:**
- One evaluator and simulator that every strategy family, including later RL, LLM and Jev strategies, must use.
- A job interface stable enough that FinanceLambdasTool can integrate once, with fixture-backed jobs in every environment.
- Near-zero phase 1 cost: short CPU jobs on synthetic data, no idle compute, and every paid job estimated and gated.

**Non-Goals:**
- GPU jobs, RL training, LLM serving and Jev (change `add-learning-and-llm-strategies`).
- Running on real market data before the platform's yfinance ingestion is deployed and the platform approves real SPY snapshots (provider decided 2026-10-07, contracts OQ-5 resolved).
- Calling any market-data provider from FinanceModel. FinanceModel has no yfinance dependency.
- Any live order routing. The simulator is paper-only.
- Owning budget alarms. AWS Budgets at the USD 50 cap (alerts at 50/80/100%, deny action at 100%) belong to FinancialPlanning (contracts D1, "Shared budget alarms"); FinanceModel only reads its category allocations and enforces pre-flight checks.

## Decisions

### D1. Job control plane: Lambda API plus DynamoDB plus SageMaker state-change events

- An IAM-authenticated API (API Gateway with SigV4, backed by a small Lambda) implements `submit_job`, `get_job_status`, `get_job_result`, `cancel_job`, `list_jobs` and the human `approve_run` operation. The Lambda only validates, records, estimates cost, takes leases and calls `CreateProcessingJob` or `StopProcessingJob`. It never runs strategy code (spec job-deployment-pipeline).
- A FinanceModel-owned DynamoDB table holds run records, append-only transition events, idempotency records (key `(principal, environment, operation, idempotency_key)` and request hash per contracts D2), the queue and leases. It is research run metadata, not authoritative plan state.
- EventBridge rules for SageMaker job state changes feed a handler that applies transitions with conditional writes, so late or duplicate events after a terminal state are ignored.
- A scheduled dispatcher (EventBridge Scheduler, every minute, only while runs are queued) moves `queued` runs to `starting` when a lease is free.
- **Alternatives:** Step Functions per run gives built-in waits and retries, but adds per-transition cost and splits state between two stores; it can be adopted later behind the same API. SageMaker Pipelines per experiment is too heavy for single backtests and still needs a submission API. Direct `CreateProcessingJob` by FinanceLambdasTool was rejected: it would put job-definition ownership and `run_id` minting outside FinanceModel (contracts D1).

### D2. Lifecycle and outcome mapping

```
submit ─► awaiting_approval ─(approve)─► queued ─(lease)─► starting ─► running ─► terminal
            │ (expire/deny → cancelled)    │ (cancel → cancelled)        │ (cancel) ─► stopping ─► cancelled
SageMaker Completed → succeeded (+ solution_status from result) | Failed → failed | Stopped by MaxRuntime → timed_out
```

- `solution_status` comes from the job's own result file, never from the SageMaker status. An infeasible optimizer run is `succeeded` plus `infeasible` (contracts CS-07).
- Retries: up to 2 retries for start-time throttling only. Container failures are not retried automatically, because reruns cost money.

### D3. Concurrency: per-environment lease plus quota-aware requeue

- The lease is a conditional-write item per `(environment, instance_class)` with `max_holders` from configuration (`/finplan/<env>/financemodel/config/lease-limits`), a TTL and a heartbeat refreshed by the state-change handler.
- Because quotas are account-wide and shared across environments in the single account, a per-environment lease cannot guarantee a free slot. A `ResourceLimitExceeded` at start therefore re-queues the run with exponential backoff up to a maximum wait (spec job-execution-controls). An account-level shared lease would break the "no shared resources across environments" rule (contracts ENV-01). This is recorded as a CONTRACT GAP rather than solved by a shared resource.
- Interim defaults (configuration, not facts): CPU `max_holders` = 1 per environment. Queue depth = 10.

### D4. Cost guardrails

- Prices are not stored in the repo. An operator writes per-instance-type hourly prices from current AWS pricing to `/finplan/<env>/financemodel/config/instance-prices` (JSON), and the date retrieved is recorded. A missing price blocks the job (`PRECONDITION_FAILED`).
- The estimate is the upper bound `price × max_runtime × instance_count + storage estimate`. Each job type declares a budget category. The account-level allocation map at `/finplan/shared/financialplanning/config/budget-allocation` (JSON, category → USD, written by the FinancialPlanning bootstrap; contracts D4/D11, which retired the per-repo `budget-allocation` key) holds the per-category caps; FinanceModel reads it and never writes it; defaults from the user decision of 2026-10-07: `cpu_research` 7 (all CPU research jobs, including dataset preparation, backtests, benchmarks, RL CPU training and weight staging in the next change) and `gpu` 25 (used only by the next change). The other categories of the USD 50 total (infrastructure 8, Bedrock 5, reserve 5) are not FinanceModel's and are not spent by its jobs. The remaining budget of a category is its cap minus the sum of estimated cost of non-cancelled-before-start runs in that category, reconciled with actual cost (Cost Explorer by `run_id` tag, which arrives with a delay) when available.
- The user does not split the budget between environments: one shared allocation map applies to beta, gamma and prod. Because run records are per environment, the per-environment check cannot see the other environments' spend; the account-wide bound is the FinancialPlanning AWS Budgets deny action at 100% of USD 50, surfaced through `budget-state` (below). Pipeline test allowances are counted in `cpu_research`.
- The pre-flight check also reads the platform's `/finplan/shared/financialplanning/config/budget-state`. While the project cap action is active, every paid submission fails with `BUDGET_EXCEEDED`. FinanceModel publishes its job-submission and dispatcher role names at `/finplan/<env>/financemodel/config/budget-enforced-role-names` for the platform's budget action (contracts D4).
- Approval: the auto-approve threshold per environment (`/finplan/<env>/financemodel/config/auto-approve-usd`) defaults to **0**, so every paid job waits for approval until the user sets a value (OPEN QUESTION FM-OQ-3). Pipeline test runs use a pre-approved per-release test allowance the user records at bootstrap, counted against `cpu_research`. `approve_run` is restricted to a human approver role. Agent, tool-wrapper and pipeline roles are explicitly denied.

### D5. Containers and job types

- One CPU image (`financemodel-cpu`) with entry points `prepare_dataset`, `run_backtest` (one strategy over walk-forward folds), `run_benchmark` (controls plus selected strategies over one dataset, sequentially in one job to use a single slot) and `report`. Running the whole benchmark in one job is deliberate under a processing quota of 1.
- Default instance `ml.m5.xlarge` (quota observed as 1). Phase 1 ceilings (configuration): 30 minutes per job, 1 instance.
- The simulator, evaluator and strategies are plain libraries inside the image, so later families (RL, swarm, Jev) import the same evaluator. The evaluator version is the image digest plus an `evaluator_version` semver recorded in results.

### D6. Datasets and splits

- Initial instrument (user decision 2026-10-07, contracts OQ-5 resolved): S&P 500 exposure through a tracking ETF daily series (SPY by default). The ETF series, the S&P 500 index level and the constituent universe are distinct datasets with distinct identifiers; one is never substituted for another. Only completed daily observations are used in phases 1 and 2; no intraday. The ETF ticker is configuration in the preparation `configuration_id`.
- Data provider (RESOLVED 2026-10-07): `yfinance` is the phase 2 daily provider adapter, owned and run by the FinancialPlanning ingestion Lambda or container, behind the platform's provider adapter interface (a fallback such as Stooq via `pandas-datareader` can be added there without contract changes). FinanceModel reads ONLY approved platform snapshots by `input_snapshot_id` and NEVER calls yfinance or any other provider directly; yfinance is not a FinanceModel dependency, and a build-stage dependency and import check enforces this (spec research-datasets, "Market data only through approved platform snapshots").
- Lineage: the platform snapshot provenance records the provider adapter (`yfinance`), the pinned library version, the retrieval timestamp and the exchange calendar (`exchange_calendars` XNYS, pinned version). FinanceModel copies these into the dataset lineage record and refuses real snapshots that lack them. Empty or partial provider responses arrive as platform quality flags; FinanceModel records them and excludes the dates or fails, per configuration, and never fills them silently. Trading-day arithmetic (embargo lengths, fold boundaries) uses the XNYS session list carried with the snapshot, so FinanceModel needs no calendar dependency for real data; synthetic fixtures carry their own session list.
- Yahoo terms caveat: Yahoo Finance data is for personal and research use. Retrieved market data is never committed to the public FinanceModel repository; fixtures stay synthetic (marked `synthetic: true`). Derived outputs (datasets, run results, reports, staged bundles) stay in FinanceModel research storage or the platform staging area, and reports carry aggregate metrics, not raw retrieved series.
- CI and build-stage tests use only the mock provider and synthetic fixtures. Any optional live-provider test is the platform's (rate-limited, shape-only assertions); FinanceModel has none.
- Phase 1 uses synthetic fixtures shaped like the ETF daily series and a mock provider that serves them, so the pipeline is complete before real snapshots exist.

- Dataset identity is the tuple `(sorted input_snapshot_ids, preparation configuration_id)` plus the manifest checksum. Datasets are stored under a research key derived from that hash and referenced in results by trusted artifact reference. No new platform identifier is introduced (see CONTRACT GAP on artifact IDs).
- Each observation keeps `available_at`, so point-in-time joins are an `as-of` filter at decision time.
- Splits and folds are computed once at preparation and stored in the dataset record. Holdout reads go through a single accessor that checks run purpose and logs every read to the run table.

### D7. Simulator conventions

- Default timing `next_open` (configurable `next_close`). Fees: proportional bps plus fixed per trade. Spread: half-spread per side from configuration. Liquidity: participation cap as a fraction of volume at execution, remainder cancelled by default. Constraint policy: `project` by default (Euclidean projection onto the constraint set), with every projection recorded. All defaults are configuration values to confirm with the user, not market facts.
- Accounting reconciliation runs each step with an absolute tolerance from configuration.

### D8. Staging handoff

- `production_candidate` bundles are written to the platform staging area referenced by `/finplan/<env>/financialplanning/config/run-staging-ref`, under `<run_id>/`, data files first and the manifest last.
- Resolved by the cross-repo review (contracts D10): the platform's `manifest.json`, written last, is the only completion marker. The platform notices a bundle only through its explicit accept call (`POST /v1/plans/{plan_id}/staged-outputs/{run_id}/accept`), made by a platform-side caller. FinanceModel never triggers acceptance. The job-API role reads the outcome from `GET /v1/staged-outputs/{run_id}` to link it in the run result.
- `production_candidate` runs are submitted by platform-side principals that FinanceModel grants that purpose (configuration). FinanceLambdasTool tool roles are never granted it.

### D9. Model registry

- A FinanceModel-owned DynamoDB registry table with a uniqueness item on `(strategy, image_digest, param_schema_version, artifact_checksum)` so registration is idempotent. Phase 1 registers the controls and classical optimizers at deploy time (post-deploy step) with their image digest.
- SageMaker Model Registry was considered. It adds value for trained artifacts in phase 2 and may mirror this registry then, but `model_version` stays the contract identifier.

### D10. Pipeline

- Follows contracts D6: build stage builds the image once, runs unit and contract tests with fixtures and a SageMaker-blocking test harness, synthesizes the stacks and records digests. Beta and gamma run fixture integration tests through the job API (one short CPU job each, covered by the pre-approved test allowance). Prod smoke uses `list_jobs` and a dry-run submission only.

## Risks / Trade-offs

- [Processing quota of 1 shared by all environments in the single account] → one-job benchmarks, quota-aware requeue, small queue; request a quota increase only if the user approves the cost (FM-OQ-2).
- [Pre-flight estimates use operator-entered prices that can go stale] → record the price retrieval date and refuse prices older than a configured age.
- [Approval threshold 0 slows iteration] → explicit user decision (FM-OQ-3); the safe default protects the USD 50 cap.
- [Cost Explorer lag] → budget checks use estimates (upper bounds) until actuals arrive, so they over-count rather than under-count.
- [Simulator defaults (timing, fees, liquidity) shape results] → every result records the full simulation configuration, and comparisons across different settings are refused.
- [Synthetic data hides real-data problems] → results carry `synthetic: true`. Real-data runs wait for approved SPY snapshots from the platform's yfinance ingestion (FM-OQ-4 resolved; deployment dependency only).
- [yfinance is unofficial and can break or return partial data upstream] → handled on the platform side (pinned version, retries with backoff, quality flags, fallback adapter possible); FinanceModel only sees approved snapshots, records the quality flags and library version in lineage, and never fills gaps silently.
- [Yahoo terms restrict redistribution] → no retrieved data in the public repo, synthetic fixtures only, derived outputs kept in research or staging storage.
- [Per-environment budget checks cannot see other environments' spend in the single account] → AWS Budgets deny at 100% of USD 50 via `budget-state`, upper-bound estimates, approval for paid jobs, small pipeline allowances.

## Migration Plan

1. Prerequisites: contracts 1.0.0 published; the FinanceModel pipeline bootstrapped (approved in principle 2026-10-07; runs once after the bootstrap IaC is implemented, with the exact stacks and cost estimate shown at run time) with the user's authenticated AWS CLI session and the reused GitHub CodeConnection (repo access verified by a source-stage dry run; the user extends the GitHub App installation only if that fails); the budget allocation document written with the default categories; platform phase 1 deployed in beta with the snapshot-resolve operation, the staging reference and the grants for the FinanceModel job role.
2. Deploy through the pipeline: storage, tables, roles, image, job API, registry seeding. Beta, then gamma, then approval, then prod.
3. FinanceLambdasTool switches `submit_experiment`, `get_job_status` and `get_experiment_result` from `DEPENDENCY_UNAVAILABLE` to the published job endpoint in each environment where this release exists.
4. Rollback: redeploy `previous_release_id` (contracts D6). Run records and registry entries are immutable and remain. In-flight runs at rollback are cancelled through the API before the job definitions change.

## Open Questions

Items marked BLOCKER stop the named step until resolved. Contract-level items (OQ-n) are in the contracts change.

| ID | Question | Blocks | Resolved by | Interim |
|---|---|---|---|---|
| FM-OQ-1 | FinanceModel share of the USD 50 ceiling | None | RESOLVED 2026-10-07: USD 50 is the total AWS budget, not split by environment. FinanceModel categories `cpu_research` USD 7 and `gpu` USD 25 (next change), read from the shared `/finplan/shared/financialplanning/config/budget-allocation`; AWS Budgets 50/80/100% alerts and deny at 100% (FinancialPlanning) | One shared allocation map for all environments |
| FM-OQ-2 | CPU processing quotas for small instance types, and whether to request an increase above 1 | None (performance only) | Service Quotas read, user approval | One CPU slot per account |
| FM-OQ-3 | Auto-approve threshold per environment | None | User decision | 0 (all paid jobs need approval) |
| FM-OQ-4 | Initial instrument universe and data provider (contracts OQ-5) | None (real-data runs depend only on the platform's yfinance ingestion being deployed and SPY snapshots approved) | RESOLVED 2026-10-07: instrument is S&P 500 exposure via a tracking ETF daily series (SPY), distinct from index level and constituents, daily completed observations only. Provider is `yfinance` (pinned) in the platform ingestion Lambda or container with `exchange_calendars` XNYS; lineage records library version and retrieval timestamp; no retrieved data in public repos; CI uses the mock provider. FinanceModel reads approved snapshots only and has no yfinance dependency (D6) | Synthetic ETF-shaped fixtures and mock provider until approved real snapshots exist |
| FM-OQ-5 | Default simulator conventions (timing, fee bps, spread, participation cap, rebalance frequency) | None (configuration) | User review | Defaults in D7, labeled as configuration |
| FM-OQ-6 | How the human approver approves runs (CLI script with approver role vs. a website action) | None for beta tests | User decision | CLI script using the approver role |

## Contract gaps (status after the cross-repo review, 2026-10-07)

These gaps against FinancialPlanning change `establish-cross-repo-contracts` were resolved there (D1, D4, D10) unless noted. The original wording is kept for traceability.

- CG-1: resolved (job lifecycle `state`, `purpose`, `dry_run`, cost-estimate block in contracts 1.0.0).
- CG-2: resolved (D1 rows for the job control plane).
- CG-3: resolved (quotas are an external limit; per-environment lease plus quota-aware requeue, no shared lease; ENV-16). The single account is now the decided topology, not an interim.
- CG-4: resolved (snapshot `status` in contracts 1.0.0; platform `GET /v1/snapshots/{id}` and the approved-snapshot read grant in `add-platform-foundation`).
- CG-5: resolved (staged-output manifest schema; manifest-last; explicit platform accept call).
- CG-6: resolved (`research_dataset` trusted-reference kind with a FinanceModel-issued `artifact_id`; no new platform identifier).
- CG-7: resolved (USD 50 total recorded, tag keys and `budget-allocation` key registered; allocation values decided 2026-10-07, FM-OQ-1).


- CG-1. The job status schema does not enumerate non-terminal states (`awaiting_approval`, `queued`, `starting`, `running`, `stopping`), run `purpose`, `dry_run` or the cost estimate fields. FinanceModel proposes these as an additive minor.
- CG-2. The ownership matrix (contracts D1) has no rows for the FinanceModel run/idempotency/lease table, registry table, job-API Lambda and API, the state-change rules, the dispatcher schedule or the approver role. The OWN-01 "resource missing from the matrix" check would fail the FinanceModel build until they are added.
- CG-3. Account-level SageMaker quotas are shared by all environments in the single account, but ENV-01 forbids shared resources. D3 avoids a shared lease. A contract decision is needed on whether an account-level capacity lease is an allowed `shared` resource.
- CG-4. The contracts do not define the platform snapshot-resolve operation, the snapshot approval status, or the snapshot read grant to the FinanceModel job role (only the D1 text).
- CG-5. The staged run-output bundle schema is not in the contract coverage list, and the staging handoff trigger (S3 event vs. platform API call) is undefined.
- CG-6. There is no identifier or `artifact_id` format for derived research artifacts such as datasets. FinanceModel uses content checksums inside trusted artifact references.
- CG-7. The brief sets a USD 50 total ceiling, but contracts OQ-7 is still open. No SSM key exists for per-repo budget allocation, and the cost-allocation tag keys are not yet listed.
