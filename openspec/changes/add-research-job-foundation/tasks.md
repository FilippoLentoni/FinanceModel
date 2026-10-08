# Tasks

Scope: FinanceModel phase 1. CPU-only, fixture-backed, paper-only. Shared rules come from FinancialPlanning change `establish-cross-repo-contracts`; tasks pin `finplan-contracts` and do not redefine identifiers, error codes or SSM naming. Tasks marked BLOCKED name the open question or contract gap that stops them (design.md). No task runs a paid AWS job except the pipeline's pre-approved fixture test jobs.

## 1. Repository scaffolding and contract pin

- [x] 1.1 Create the repo layout (`src/financemodel/{simulator,evaluator,strategies,datasets,jobs,api,registry,reporting}`, `containers/cpu/`, `infra/`, `tests/{unit,contract,integration,smoke}`) and verify that `pytest --collect-only` and `cdk synth` run on an empty skeleton
- [x] 1.2 Pin `finplan-contracts` by exact version and digest and wire the consumer conformance runner, copied-`$id` check, identifier/secret leak scan and live-permission scan into the build; verify the build fails on a planted copied `$id` and on a planted 12-digit account ID (BLOCKED by contracts 1.0.0 publication; until then pin a beta 0.x in beta only)
  - Done 2026-10-08: pinned to **1.0.0** by version and wheel SHA-256 (the platform's byte-identical build, vendored; the platform's build publishes the same bytes to CodeArtifact after its bootstrap re-run, and switching the source to the registry is task 12.7). Verified locally: `check_contracts_pin.py --env beta|gamma|prod` pass, the planted copied `$id`, planted account ID, tampered wheel and version range all fail (`tests/unit/test_contracts_pin_and_gates.py`).
- [x] 1.3 Add a test harness that blocks any SageMaker or AWS network call in unit tests; verify with a test that attempts `CreateProcessingJob` and is blocked (DEP-02)

## 2. Paper execution simulator and evaluator

- [x] 2.1 Implement the strategy interface (point-in-time view plus holdings in, target weights out) and reject order-shaped outputs; verify SIM-02 unit tests
- [x] 2.2 Implement constraint validation with `project` and `reject` policies and record every projection or rejection; verify SIM-03 unit tests, including the 60%-over-40%-cap and turnover-reject scenarios
- [x] 2.3 Implement execution timing (`next_open`, `next_close`) and forbid same-bar fills; verify SIM-04 with a fixture where same-bar fills would change the result
- [x] 2.4 Implement the fee, spread and slippage models and report gross, net and cost components separately; verify SIM-05 against hand-computed fixture values
- [x] 2.5 Implement the liquidity participation cap with carry-over and cancel modes; verify SIM-06 unit tests
- [x] 2.6 Implement per-step accounting reconciliation with configured tolerance; verify SIM-09 with a fixture that injects a break and fails the run with `INTERNAL`
- [x] 2.7 Add the evaluator version record and the comparison guard that refuses mixed settings; verify SIM-01 (mismatched fee models refused) and SIM-08 (replay of stored outputs gives identical trades and metrics)
- [x] 2.8 Reject live execution venues in simulation configuration; verify SIM-07 returns `OPERATION_NOT_PERMITTED` and that the live-permission scan passes
- [x] 2.9 Document simulator conventions and every configurable default in `docs/simulator.md`, labeled as configuration pending FM-OQ-5; verify the documented example configuration validates

## 3. Research datasets

- [x] 3.1 Implement dataset preparation from synthetic snapshot fixtures with `available_at` on each observation and a SHA-256 manifest; verify DS-01 (identical manifest on repeat; reuse instead of rewrite) and DS-08 (`synthetic: true` propagates)
- [x] 3.2 Implement the dataset lineage record (snapshots with checksums, `configuration_id`, image digest, contract version, domain and schema version); verify DS-02 contract test against fixture lineage
- [x] 3.3 Implement the as-of feature join and look-ahead check; verify DS-03 with late-arriving and look-ahead fixtures, and that `intraday_partial` observations are never used as completed daily bars
- [x] 3.4 Implement chronological splits with embargo and walk-forward fold generation (expanding and rolling); verify DS-04 and DS-05 unit tests (overlap rejected, 20-day embargo, fold boundaries recorded)
- [x] 3.5 Implement the holdout accessor (purpose check, access log, frozen-candidate requirement) and prospective-period snapshot filter; verify DS-06 (tuning read refused, reuse flagged) and DS-07 (pre-freeze snapshot rejected)
- [x] 3.6 Implement the initial-instrument definition (S&P 500 tracking ETF daily series, ticker in `configuration_id`, distinct from index level and constituent universe, completed daily observations only) with synthetic ETF-shaped fixtures and a mock provider; verify DS-09 (index-level snapshot rejected), DS-10 (intraday rejected; non-XNYS-session date rejected) and DS-11 (real-data request without an approved real snapshot gives `DEPENDENCY_UNAVAILABLE`; CI makes no market-data network request) unit tests
- [x] 3.7 Add the build-stage provider guard: a dependency and import check that fails if `yfinance` or any other market-data provider client appears in FinanceModel dependency files, images or source, and a fixture check that fails on any fixture without the `synthetic: true` marker; verify DS-12 (planted `yfinance` dependency and planted provider import both fail the build) and DS-13 (planted non-synthetic price file fails the build) unit tests
- [x] 3.8 Copy snapshot provider provenance (adapter `yfinance`, pinned library version, retrieval timestamp, `exchange_calendars` XNYS version and session list) and quality flags into the dataset lineage record, refuse real snapshots missing them, and apply the configured exclude-or-fail policy for flagged dates; verify DS-12 lineage and quality-flag scenarios as unit and contract tests on synthetic snapshots that mimic the platform provenance fields (field names confirmed against the platform snapshot schema, FM-A5)
- [x] 3.9 Ensure reports and results from real-data datasets are written only to research storage or platform staging with aggregate metrics and no raw retrieved series; verify DS-13 report scenario unit test
- [x] 3.9a Implement the shared outbound payload check for Yahoo-derived data (research purposes only; reject runs of consecutive numbers above the configured limit, values equal to raw price or volume observations of the source dataset, tables and attachments; log rejections without payload values); verify DS-14 unit tests (raw-series payload rejected with no network call, descriptor payload passes) (user decision 15d, 2026-10-07)
- [ ] 3.10 Run dataset preparation in beta on a short range of approved SPY snapshots produced by the platform's yfinance ingestion (FinanceModel reads snapshots only); verify DS-09 and DS-12 integration-beta (depends on the platform ingestion release and an approved SPY snapshot in beta; no longer blocked by a provider choice)

## 4. Baseline strategies

- [x] 4.1 Implement `cash`, `buy_and_hold` and `equal_weight`, and auto-add missing controls to benchmarks; verify BASE-01 unit tests
- [x] 4.2 Implement minimum-variance with a configurable point-in-time covariance estimator; verify BASE-02 (no observation after t is used)
- [x] 4.3 Implement mean-variance with configurable risk aversion and return estimator; verify BASE-03 (variance non-increasing as risk aversion rises, on fixtures)
- [x] 4.4 Implement scenario-CVaR with seeded, point-in-time scenario generation; verify BASE-04 (same seed gives same scenarios and weights)
- [x] 4.5 Map solver outcomes to `solution_status` with the hold-current-weights fallback; verify BASE-05 contract tests for the infeasible, partial-infeasible and optimal cases against contract fixtures (CS-07)
- [x] 4.6 Reject GPU instance types for baseline job types; verify BASE-06 unit test

## 5. Benchmark reporting

- [x] 5.1 Implement the report schema with separate portfolio, accuracy, training-reward and cost sections and refuse blended scores; verify REP-01 unit tests
- [x] 5.2 Implement the portfolio metrics per fold, holdout and aggregate; verify REP-02 against hand-computed fixture values
- [x] 5.3 Implement period and `synthetic` labeling, the holdout access log attachment and seed dispersion; verify REP-04 and REP-05 unit tests
- [x] 5.3a Store the holdout metrics record (net-of-costs cumulative return, maximum drawdown, `dataset_id`, holdout bounds, simulation configuration and cost-model identity, `evaluator_version`) for every holdout evaluation; verify REP-07 (record equals the report's holdout column; missing comparability field fails) (user decision 15c, 2026-10-07)
- [x] 5.4 Implement the cost section (estimated vs. actual, pending state); verify REP-03 with a fixture run that lacks billing data
- [x] 5.5 Make report generation deterministic with a checksum and narrative kept separate; verify REP-06 (regenerated report has the same checksum)

## 6. CPU job container

- [ ] 6.1 Build the `financemodel-cpu` image with entry points `prepare_dataset`, `run_backtest`, `run_benchmark` and `report`; verify a local container run on fixtures (no AWS) produces a result that validates against the contract result schema
- [x] 6.2 Implement snapshot resolution through the platform API and checksum verification inside the job; verify WS-04 with a corrupted fixture (fails before strategy code) and WS-03 with a mocked platform response (beta needs the platform release with `GET /v1/snapshots/{id}` and snapshot `status`, platform tasks 4.8 and 6.11)
- [x] 6.3 Write job results (with `completion_status`, `solution_status`, `model_version`, evaluator version, dataset checksum and artifact references) and never storage paths in responses; verify JOB-06 contract tests

## 7. Job interface and control plane

- [x] 7.1 Implement `submit_job` validation, `configuration_id` computation, `run_id` minting and `dry_run`; verify JOB-02, JOB-03 and JOB-08 unit and contract tests (caller `run_id` rejected, `s3://` rejected, unsupported contract major, dry run creates nothing)
- [x] 7.2 Implement idempotency records for `submit_job` and `cancel_job`; verify JOB-04 (retry returns the same `run_id`; changed body gives `IDEMPOTENCY_KEY_REUSED`)
- [x] 7.3 Implement the lifecycle state machine with append-only events and conditional writes, and the SageMaker state-change handler; verify JOB-05 unit tests including the late event after `cancelled`
- [x] 7.4 Implement run purpose authorization; verify JOB-07 (production candidate without grant returns `FORBIDDEN`)
- [x] 7.5 Implement `DEPENDENCY_UNAVAILABLE` for job types not deployed in an environment; verify JOB-09 unit test
- [x] 7.6 Document the job interface for FinanceLambdasTool in `docs/job-interface.md` (operations, states, errors, examples validated against the pinned contracts); verify the examples validate, using the contracts 1.0.0 job lifecycle, purpose, dry-run and cost-estimate fields

## 8. Execution controls and cost guardrails

- [x] 8.1 Implement per-job-type default and maximum runtime and the SageMaker stopping condition; verify CTL-01 unit tests (above-ceiling request rejected; `MaxRuntimeExceeded` maps to `timed_out`)
- [x] 8.2 Implement the per-environment lease with heartbeat, expiry and safe reclaim, and the dispatcher; verify CTL-02 unit tests (second job queued, stale lease reclaimed only after terminal confirmation)
- [x] 8.3 Implement the bounded queue and quota-aware requeue with backoff; verify CTL-03 (`RATE_LIMITED`) and CTL-04 (mocked `ResourceLimitExceeded` re-queues, then `DEPENDENCY_UNAVAILABLE` after the max wait)
- [x] 8.4 Implement cancellation in every non-terminal state and failure handling with bounded start retries; verify CTL-05 and CTL-06 unit tests (crash gives `failed` plus `INTERNAL` plus `correlation_id`; lease released; nothing staged)
- [x] 8.5 Implement the pre-flight estimate from configured prices and the budget check; verify CTL-07 (`BUDGET_EXCEEDED` with estimate and remaining budget; missing price gives `PRECONDITION_FAILED`)
- [x] 8.6 Implement `awaiting_approval`, the approver-only `approve_run` and approval expiry; verify CTL-08 unit tests and an IAM policy simulation showing tool-wrapper, agent and pipeline roles are denied
- [x] 8.7 Apply cost-allocation tags (environment, repo, `run_id`) to every job; verify CTL-09 unit test on the generated request (tag keys `project`, `owner-repo`, `environment`, `logical-role`, `run-id` from contracts D10)
- [x] 8.8 Implement per-category allocation (`cpu_research` default USD 7, `gpu` default USD 25, read from the shared `/finplan/shared/financialplanning/config/budget-allocation`), the job-type category declaration and the `budget-state` deny check; verify CTL-10 unit tests (CPU category exhausted while GPU has funds; deny action active; missing category gives `PRECONDITION_FAILED`)

## 9. Run-output staging and model registry

- [x] 9.1 Implement bundle writing for `production_candidate` runs only (data files first, manifest last; no separate marker) and run state `staged`; verify RST-01, RST-02 and RST-03 unit and contract tests (research run stages nothing; schema failure stages nothing; missing manifest = incomplete) against the contracts 1.0.0 staged-output manifest schema
- [x] 9.2 Implement the registry with idempotent `model_version` minting, immutable records and lifecycle events; verify REG-01 and REG-02 unit tests
- [x] 9.3 Implement the lineage-verification read used by the platform and include `model_version` in all results; verify REG-03 and REG-04 contract tests
- [x] 9.4 Link platform validation outcomes to run results without claiming a `plan_version_id`; read from the platform `GET /v1/staged-outputs/{run_id}` route; verify RST-05 with a mocked platform `invalid` outcome

## 10. Infrastructure, IAM and pipeline

- [x] 10.1 Define per-environment research storage (encryption, public-access block, tags, lifecycle that keeps referenced artifacts); verify WS-01 and WS-02 with synth-template assertions
- [x] 10.2 Define job, API, dispatcher and approver roles with the contracts' permission boundaries: snapshot read-only, staging write-only, deny on platform metadata, publication and execution APIs, deny on other environments; verify WS-03, WS-05 and RST-04 with IAM policy simulation unit tests (ENV-03, ENV-04, ENV-05)
  - Done 2026-10-08 with contracts 1.0.0: the job-execution role (research boundary) and the approver role now synthesize in every environment; `tests/unit/infra/test_iam_simulation.py` (15 simulations, including the dispatcher-schedule arming grants) and the `boundaries` post gate pass.
- [x] 10.3 Define the job API, run/lease/idempotency and registry tables, state-change rules and dispatcher schedule; verify the OWN-01 ownership check passes against the contracts D1 rows for the job control plane
  - Done 2026-10-08 with contracts 1.0.0: the job API, job-execution role, approver role and registry bucket policy are always synthesized; the `ownership` post gate reports zero problems on all 8 templates (beta, gamma, prod storage and control, the store and the tooling stack).
- [x] 10.4 Publish the release manifest and `/finplan/<env>/financemodel/{api/job-endpoint, model/registry-ref, job/*, config/budget-enforced-role-names}` references, and read `budget-state` in the pre-flight check; verify DEP-04 and ENV-06/ENV-07 synth assertions
- [ ] 10.5 Define the pipeline per the standard (image built once, digest-pinned job definitions, beta and gamma fixture integration tests, approval, prod dry-run smoke) in the single account, sourcing `FilippoLentoni/FinanceModel` (main) through the reused GitHub CodeConnection referenced from SSM; bootstrap (approved in principle 2026-10-07; runs only after this IaC is implemented, showing the exact stacks and a cost estimate at run time) runs once with the user's authenticated AWS CLI session and creates the scoped pipeline, deploy and service roles; verify DEP-01, DEP-03 and DEP-05 with the contract pipeline-structure check and a source-stage dry run (if the dry run cannot access the repo, the user extends the GitHub App installation)
- [ ] 10.6 Write `docs/operations.md` (prices, budget categories and their defaults, approval procedure, cancellation, rollback with in-flight cancellation); verify it contains no account identifiers (leak scan) and that the documented approval command works against beta

## 11. Integration checks (beta and gamma)

- [ ] 11.1 In beta, submit a fixture-backed `run_benchmark` (controls plus the three optimizers) through the job API under the pre-approved test allowance, then check status transitions, result schema, report sections and cost record; verify JOB-01, JOB-05, JOB-06, REP-01 and CTL-09 integration-beta (BLOCKED by the platform beta release)
- [ ] 11.2 In beta, submit, then cancel, a run, and submit a duplicate with the same idempotency key; verify CTL-05 and JOB-04 integration-beta (one SageMaker job only; cancelled state)
- [ ] 11.3 In beta, run a `production_candidate` fixture run and confirm the platform sees a complete bundle and the registry lineage check passes; the acceptance call is made by the platform test harness, not by FinanceModel; verify RST-02, RST-03 and REG-03 integration-beta (BLOCKED by the platform beta release)
- [ ] 11.4 In gamma, repeat 11.1 and run the isolation suite (gamma job role cannot read prod storage, cannot write plan or publication state, cannot write other runs' staging keys); verify WS-01, WS-05, RST-04 and ENV-04 gamma
- [ ] 11.5 After prod promotion, run the smoke test (`list_jobs` plus dry run, no SageMaker job) and confirm digest equality across manifests; verify DEP-05 and DEP-03 smoke

## 12. Contracts 1.0.0 adoption and first-deploy readiness (2026-10-08)

- [x] 12.1 Vendor the contracts 1.0.0 wheel (sha256 `1f644d7a...88ca`), remove 0.2.2, re-pin `contracts-pin.json`, `pyproject.toml` and `uv.lock`, serve contract major 1 only, and drop the beta-only promotion stop; verify `uv sync --locked`, `check_contracts_pin.py --env beta|gamma|prod`, `PromotionCheck` passing in every environment and a planted 0.x pin still refused outside beta
- [x] 12.2 Remove the contract-gap switch (`infra/stacks/contract_gaps.py`) and synthesize the job API, job-execution role, registry bucket policy and CodeBuild log groups unconditionally; make the job API outputs required by `PublishRelease` and the deployed suites; verify OWN-01 with zero problems on every template and the synth tests
- [x] 12.3 Publish the account-level pipeline and build role names at `/finplan/shared/financemodel/config/budget-enforced-role-names` from the bootstrap (contracts D16) and stop repeating them in the per-environment lists; verify the value against the contract key and that the release writes nothing shared
- [x] 12.4 Arm the dispatcher schedule only while runs are queued, active or awaiting an approval deadline (design D1): deployed `DISABLED`, armed by submit/approve/state events, re-planned by every tick, race-guarded; verify unit tests (plan, arming, disarming, race, failures), the moto Scheduler wiring test and the IAM simulation (own schedule only)
- [x] 12.5 Give the four FinanceModel CodeBuild projects explicit 30-day log groups (`add_log_group`, as FinancialPlanning); verify the synth test and the ownership check of the tooling template
- [x] 12.6 Document the operator parameters (`instance-prices`, `integration-snapshot-id`, `auto-approve-usd`, `lease-limits`, `production-candidate-principals`) and have the bootstrap write absent or stale `instance-prices` from the AWS Price List API (`scripts/instance_prices.py`) and report a missing `integration-snapshot-id` without writing it; verify the bootstrap and price unit tests and the docs test
- [ ] 12.7 Switch the contract pin's source from the vendored wheel to the CodeArtifact registry (`/finplan/shared/financialplanning/contract/registry-ref`) once the FinancialPlanning bootstrap re-run and its next build have published 1.0.0; verify the registry digest equals the pinned digest
- [ ] 12.8 Human steps before the first pipeline run: run the FinanceModel bootstrap (10.5), write `/finplan/{beta,gamma}/financemodel/config/integration-snapshot-id` (an approved synthetic platform snapshot; since user decision 26 this is an optional override, real or synthetic: without it the suite uses the snapshot the platform's suite publishes, a real SPY snapshot in phase 2), then re-run the FinancialPlanning bootstrap after the first beta deploy so its budget action covers the FinanceModel roles

## Requirement-to-test mapping

Test types: unit, contract (schemas and conformance with shared fixtures), integration-beta, gamma, smoke.

| Spec | Requirement | Test ID | Type |
|---|---|---|---|
| research-workspace | FinanceModel-owned research storage per environment | WS-01 | unit (synth) + gamma (cross-env deny) |
| research-workspace | Research artifact retention | WS-02 | unit (lifecycle rules in synth) |
| research-workspace | Read-only access to approved platform snapshots | WS-03 | unit (policy simulation) + contract + integration-beta |
| research-workspace | Snapshot integrity verification | WS-04 | unit |
| research-workspace | No authoritative plan state in FinanceModel | WS-05 | unit (policy simulation) + gamma |
| research-datasets | Deterministic, content-addressed datasets | DS-01 | unit |
| research-datasets | Dataset lineage record | DS-02 | contract |
| research-datasets | Point-in-time availability | DS-03 | unit |
| research-datasets | Chronological train, validation and test splits | DS-04 | unit |
| research-datasets | Walk-forward evaluation folds | DS-05 | unit |
| research-datasets | Untouched holdout with logged access | DS-06 | unit + integration-beta |
| research-datasets | Prospective paper period | DS-07 | unit |
| research-datasets | Synthetic fixture datasets for phase 1 | DS-08 | unit + contract |
| research-datasets | Initial instrument definition | DS-09 | unit + integration-beta (after approved SPY snapshots exist in beta) |
| research-datasets | Completed daily observations only | DS-10 | unit |
| research-datasets | Mock provider until approved real snapshots exist | DS-11 | unit + integration-beta |
| research-datasets | Market data only through approved platform snapshots | DS-12 | unit (dependency and import check) + contract + integration-beta |
| research-datasets | Yahoo-derived data leaves FinanceModel only as research descriptors | DS-14 | unit (outbound payload check) |
| research-datasets | No retrieved market data in the public repository | DS-13 | unit (fixture check) |
| paper-execution-simulator | One evaluator for all strategy families | SIM-01 | unit |
| paper-execution-simulator | Strategy interface | SIM-02 | unit |
| paper-execution-simulator | Portfolio constraints enforced by the simulator | SIM-03 | unit |
| paper-execution-simulator | Execution timing | SIM-04 | unit |
| paper-execution-simulator | Fees and transaction costs | SIM-05 | unit |
| paper-execution-simulator | Liquidity and slippage model | SIM-06 | unit |
| paper-execution-simulator | Paper only, never live | SIM-07 | unit + unit (live-permission scan) |
| paper-execution-simulator | Deterministic simulation | SIM-08 | unit |
| paper-execution-simulator | Accounting reconciliation | SIM-09 | unit |
| baseline-strategies | Control strategies | BASE-01 | unit + integration-beta |
| baseline-strategies | Minimum-variance optimizer | BASE-02 | unit |
| baseline-strategies | Mean-variance optimizer | BASE-03 | unit |
| baseline-strategies | Scenario-CVaR optimizer | BASE-04 | unit |
| baseline-strategies | Optimizer outcome reporting | BASE-05 | contract |
| baseline-strategies | CPU-only execution | BASE-06 | unit |
| experiment-job-interface | Published job interface reference | JOB-01 | unit (synth) + integration-beta |
| experiment-job-interface | Asynchronous submission mints run_id | JOB-02 | unit + contract |
| experiment-job-interface | Submission validation | JOB-03 | unit + contract |
| experiment-job-interface | Idempotent submission | JOB-04 | unit + integration-beta |
| experiment-job-interface | Job lifecycle states | JOB-05 | unit + integration-beta |
| experiment-job-interface | Result retrieval | JOB-06 | contract + integration-beta |
| experiment-job-interface | Run purpose | JOB-07 | unit (policy simulation) |
| experiment-job-interface | Dry-run submission | JOB-08 | unit + smoke |
| experiment-job-interface | Fixture-backed availability | JOB-09 | unit + integration-beta |
| job-execution-controls | Per-job time limits | CTL-01 | unit |
| job-execution-controls | Concurrency lease per environment | CTL-02 | unit + integration-beta |
| job-execution-controls | Bounded queue | CTL-03 | unit |
| job-execution-controls | Account quota exhaustion is retryable | CTL-04 | unit (mocked SageMaker) |
| job-execution-controls | Cancellation | CTL-05 | unit + integration-beta |
| job-execution-controls | Failure semantics | CTL-06 | unit |
| job-execution-controls | Pre-flight cost estimate and budget check | CTL-07 | unit + integration-beta |
| job-execution-controls | Human approval for paid jobs | CTL-08 | unit (policy simulation) + integration-beta |
| job-execution-controls | Cost attribution tags | CTL-09 | unit + integration-beta |
| job-execution-controls | Category budget allocation | CTL-10 | unit + integration-beta |
| run-output-staging | Staging only for production-candidate runs | RST-01 | unit |
| run-output-staging | Staged bundle content | RST-02 | contract + integration-beta |
| run-output-staging | Manifest written last as the completion marker | RST-03 | unit + integration-beta |
| run-output-staging | Write-only staging access | RST-04 | unit (policy simulation) + gamma |
| run-output-staging | Platform validation decides commitment | RST-05 | unit + integration-beta |
| model-registry | model_version minting | REG-01 | unit |
| model-registry | Immutable registry records | REG-02 | unit |
| model-registry | Published registry reference | REG-03 | contract + integration-beta |
| model-registry | Results reference model_version | REG-04 | contract |
| benchmark-reporting | Separate reporting sections | REP-01 | unit + integration-beta |
| benchmark-reporting | Portfolio performance metrics | REP-02 | unit |
| benchmark-reporting | Holdout metrics record for promotion | REP-07 | unit + integration-beta |
| benchmark-reporting | Compute cost reporting | REP-03 | unit + integration-beta |
| benchmark-reporting | Period labeling | REP-04 | unit |
| benchmark-reporting | Variability disclosure | REP-05 | unit |
| benchmark-reporting | Deterministic report generation | REP-06 | unit |
| job-deployment-pipeline | Pipeline deploys infrastructure, not experiments | DEP-01 | integration-beta (ENV-14) |
| job-deployment-pipeline | No model fitting in CodeBuild or Lambda | DEP-02 | unit |
| job-deployment-pipeline | Standard stages and immutable images | DEP-03 | unit (pipeline check) + smoke (digest equality) |
| job-deployment-pipeline | Release manifest and published references | DEP-04 | unit (synth) + integration-beta |
| job-deployment-pipeline | Environment test stages | DEP-05 | integration-beta + gamma + smoke |

## Workflow follow-up

- FinanceLambdasTool switches its model-backed tools from `DEPENDENCY_UNAVAILABLE` to this job interface in each environment where this release exists.
- Contract gaps CG-1 to CG-7 are resolved (allocation values decided 2026-10-07). The data provider (FM-OQ-4) was RESOLVED 2026-10-07 (yfinance in the platform ingestion; FinanceModel reads approved snapshots only). Real-data task 3.10 depends only on the platform ingestion release and approved SPY snapshots in beta, and on confirming the snapshot provenance fields (FM-A5).
- Pipeline bootstrap is approved in principle (2026-10-07); it runs once after the bootstrap IaC exists, with the exact stacks and cost estimate shown at run time. It is not an open approval blocker.
- Archive this change after the gamma integration checks pass and the prod smoke succeeds.
- Data parity (user decision 26): gamma and prod accept real phase 2 platform snapshots exactly as beta does (same code paths, `instrument.data_source` `platform_snapshots` in every `config/<env>.json`); prod also accepts still-synthetic snapshots while its platform is in transition. Test-created records stay synthetic; synthetic market-data fixtures remain for offline and unit tests only.
