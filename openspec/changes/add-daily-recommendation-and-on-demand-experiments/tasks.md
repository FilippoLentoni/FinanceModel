# Tasks

Test IDs are defined in the mapping table at the end. CI never starts SageMaker. **Deployed** tests use the deployed beta or gamma job API and SageMaker.

## 1. Contract pin and configuration

- [x] 1.1 Pin `finplan-contracts` 1.1.0 by version and digest once published. Verify that the pin check passes and that the conformance suite (job kind, production-strategy document, `bias_disclosures`) runs in consumer mode.
- [x] 1.2 Add the `daily_recommendation` kind configuration (`ml.m5.xlarge`, 1 instance, 1800 s, `cpu_research`) and the build cost check. Verify DRJ-05 unit tests (a threshold below the estimate fails the build; the estimate is at most USD 0.15).

## 2. Production-strategy setting

- [x] 2.1 Implement get/set/clear with the caller allow-list, confirmation and idempotency. Verify PSS-01 unit tests.
- [x] 2.2 Implement registry validation (M3). Verify PSS-02 unit tests: one per failing rule, plus a valid selection citing the evidence `run_id`.
- [x] 2.3 Add the selection role as the only SSM writer, with audit events. Verify the PSS-03 IAM policy simulation (tool, agent and platform roles denied) and an audit unit test.

## 3. Daily recommendation job

- [x] 3.1 Add the kind to `submit_job` with the caller restriction, server-side strategy resolution, no-strategy refusal and dataset check. Verify DRJ-01 to DRJ-03 unit tests.
- [x] 3.2 Implement `run_daily_recommendation` (dataset from the snapshot, common evaluator, staging with six weights and disclosures). Verify DRJ-04 container unit tests on synthetic universe fixtures and staged-manifest contract validation.

## 4. Universe input for existing experiments

- [x] 4.1 Add `equity-etf-daily` dataset preparation (`adj_close` basis, cash assumption) for `backtest` and `benchmark`. Verify UNV-01 unit tests on synthetic fixtures.
- [x] 4.2 Add the mandatory bias and cash sections to reports and result summaries, failing closed. Verify UNV-02 unit tests (deterministic report checksum; a missing disclosure fails the run).
- [x] 4.3 Enforce the trigger-role kind restriction and the no-schedule template check. Verify UNV-03 unit tests and a negative template fixture.
- [x] 4.4 (beta finding, design M5) Carry the compact strategy comparison (`payload.benchmark`: every evaluated strategy's metrics with units, final and average weights, evaluation window, risk-free assumption) in `run_backtest` and `run_benchmark` results. Verify `tests/unit/jobs/test_job_container.py` (comparison rows, weights summing to 1 with cash, turnover/cost units matching the simulator summary, weight cap, contract-valid `get_experiment_result` response).

## Implementation notes (2026-10-08, verified locally)

- 1.1: pinned 1.1.0 by version and wheel SHA-256 (`scripts/check_contracts_pin.py --repin --from <wheel>`, then `uv sync`); the release manifest declares the pinned `contract_version` (1.1.0). The contract `daily-recommendation` submission fixture runs in the consumer conformance suite (`tests/contract/test_job_interface_contract.py`).
- 1.2: `daily_recommendation` in `config/<env>.json` (one `ml.m5.xlarge`, 1800 s, `cpu_research`, its own `auto_approve_usd` 0.15). The config check (`daily_cost_problems`, run by the `config` build gate) fails when the planning-bound estimate exceeds USD 0.15 or the kind's auto-approve ceiling (`tests/unit/control/test_daily_and_strategy.py::test_drj05_build_cost_check`).
- 2.x: `PUT`/`GET /v1/production-strategy` served by the `strategy-selection` Lambda (`finplan_model.control.selection`, `production_strategy`). The document follows `core/v1/production-strategy.json` (`selected_by`/`selected_at`; the contract has no `configuration_id`). Confirmation travels as `confirmed_by_user` next to the contract request fields. Audit: append-only `AUDIT#production-strategy` items plus a log line.
- 3.x: the trigger role is the only `daily_recommendation` submitter (handler plus resource policy); the strategy is resolved and re-validated at submission and frozen on the run (`production_strategy`), and `plan_id` becomes the run spec's staging target. A control strategy (for example `buy_and_hold`) stages its rule-based allocation as `feasible`.
- 4.x (2026-10-08 follow-up): the user-initiated `model_selection` kind (decision 27; change
  `add-learning-and-llm-strategies` task group 10) uses the same universe input and carries the
  same mandatory `payload.bias_section`, failing closed. Verified by
  `tests/unit/selection/test_model_selection_job.py::test_container_run_writes_a_contract_result_with_every_section`.
  It is never scheduled, and the daily trigger role gets `FORBIDDEN` for it.
- 4.x: `adj_close` basis for universe snapshots, the `zero_nominal` cash assumption, the "Hindsight and survivorship bias" section in results (`payload.bias_section`), run artifacts and `build_report` (checksummed), failing closed without disclosures.

## 5. Deployed verification

- [ ] 5.1 **DRJ-06 (beta, deployed):** with the key absent, call the deployed `submit_job(daily_recommendation)` as the platform trigger principal. Verify `no_production_strategy`, and that SageMaker shows no new job for the env tag.
- [ ] 5.2 **UNV-04 (beta, deployed, real job):** as the operator on behalf of the test user, `dry_run` and then run a universe benchmark through the existing `submit_job`. Verify:
  - success within 1800 s with cost tags;
  - `get_job_result` includes the bias section;
  - the downloaded report hashes to its reference checksum.
- [ ] 5.3 **PSS-04 (beta, deployed):**
  1. Select an unevaluated strategy and verify `no_evaluation_evidence`.
  2. Select `buy_and_hold` with confirmation, and verify the SSM document and the audit event.
  3. Let the platform trigger run (FinancialPlanning DLY-08), and verify one run with the frozen triple, a staged bundle with disclosures, and runtime of at most 1800 s.
  4. Clear the key.
- [ ] 5.4 Repeat 5.1–5.3 in gamma. Prod smoke (non-mutating): `get_production_strategy` and job-API health only.

## Requirement-to-test mapping

| Capability | Requirement | Test ID | Type |
|---|---|---|---|
| daily-recommendation-job | Only the platform trigger submits daily recommendations | DRJ-01 | unit + policy sim |
| daily-recommendation-job | Runs only the configured production strategy | DRJ-02, PSS-04 | unit + **beta/gamma deployed** |
| daily-recommendation-job | No strategy means no run | DRJ-03, DRJ-06 | unit + **beta/gamma deployed** |
| daily-recommendation-job | Staged recommendation with bias disclosures | DRJ-04, PSS-04 | unit + contract + **deployed** |
| daily-recommendation-job | Daily job cost cap | DRJ-05, PSS-04 | unit (build check) + **deployed** |
| production-strategy-setting | Setting is an explicit user action | PSS-01 | unit + **prod read-only smoke** |
| production-strategy-setting | Validated against the strategy registry | PSS-02, PSS-04 | unit + **beta/gamma deployed** |
| production-strategy-setting | Single writer and audit | PSS-03 | policy sim + unit |
| universe-research-input | Universe snapshots accepted by existing experiment kinds | UNV-01, UNV-04 | unit + **beta/gamma deployed** |
| universe-research-input | Reports flag hindsight and survivorship bias | UNV-02, UNV-04 | unit + **deployed** |
| universe-research-input | Experiments are never scheduled | UNV-03 | unit (handler + template check) |

## Workflow follow-up

- Archive after the gamma deployed tests and the prod smoke pass.
