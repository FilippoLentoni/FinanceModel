# Tasks

Scope: FinanceModel phase 2 (contracts phase 3). Depends on `add-research-job-foundation` being deployed in the target environment. Every GPU or paid run named below requires a pre-flight estimate and explicit human approval. Budget (decision 2026-10-07): GPU runs draw on the `gpu` category (USD 25), CPU runs (RL, staging, Jev driver) on `cpu_research` (USD 7), within the USD 50 total AWS budget; TypeSafe Jev charges are tracked separately outside AWS. Tasks marked BLOCKED name the open question (design.md) that stops them. Pipeline stages never start GPU jobs.

## 1. RL environment and specification

- [x] 1.1 Implement the versioned environment specification schema (state features, action space and transform, reward terms and coefficients, episode, decision frequency) and include it in `configuration_id`; verify RL-01 unit tests (coefficient change gives a new `configuration_id`; undeclared feature fails)
- [x] 1.2 Implement the gym-style environment on top of the common simulator and dataset views, with rewards from simulator outputs; verify RL-02 by comparing a scripted policy's environment metrics with the evaluator's metrics for the same weights
- [ ] 1.3 Implement the softmax action transform for PPO and SAC, and the discrete action-set schema for DQN; verify RL-03 (weights sum to 1 with cash) and RL-04 (continuous DQN rejected) unit tests
- [x] 1.4 Document the default environment specification in `docs/rl-environment.md`, labeled as configuration pending user review; verify that the documented specification validates

## 2. RL training and evaluation

- [ ] 2.1 Build the `financemodel-rl` CPU training image (learners plus the shared simulator) and the Training Job entry point; verify a local container run on fixtures with a fixed tiny step count produces a policy artifact and checksum, with no AWS calls
- [ ] 2.2 Add the RL job types to the job interface (one Training Job per configuration and seed, CPU lease, time limits, budget check); verify JOB contract tests for the new job types and RL-07 unit tests on the generated `CreateTrainingJob` request (CPU instance, stopping condition, tags)
- [ ] 2.3 Implement seed sweeps (default 5, minimum 3), validation-only selection, and registration of each trained policy with checksum, seed and `configuration_id`; verify RL-05 (test-metric selection rule refused) and RL-07 (registry entry per policy)
- [ ] 2.4 Report shaped rewards only in the training-reward section, and flag out-of-configuration evaluations as ineligible for promotion; verify RL-06 and RL-08 unit tests
- [x] 2.5 Guard the build stage against learner runs beyond the fixture step count; verify the RL-07 build-guard test fails a planted long training test
- [ ] 2.6 Run one approved PPO and SAC seed sweep on synthetic data in beta through the job interface; verify RL-02, RL-05 and RL-07 integration-beta (report shows all seeds and dispersion; cost records present in `cpu_research`)

Implementation notes for groups 1 and 2 (2026-10-08, verified locally on synthetic data):

- **1.1** `finplan_model.rl.spec.EnvSpec` (`finplan-rl-env/1`). The policy configuration covers the
  specification, the algorithm hyperparameters, the simulation configuration id, the training range
  and the instruments. Verified by:
  - `tests/unit/rl/test_rl_env.py::test_rl01_reward_coefficient_change_gives_a_new_configuration_id`;
  - `::test_rl01_undeclared_or_unknown_features_fail_validation`.
- **1.2** `finplan_model.rl.env.PortfolioEnv` (gymnasium) replicates the simulator's accounting on
  aligned arrays: execution timing, fees, spread, slippage, the liquidity cap, cash scaling and
  `apply_constraints`. Rewards come from the simulated net values. Verified by:
  - `::test_rl02_environment_nav_matches_the_common_evaluator_for_the_same_policy`;
  - `::test_rl02_scripted_buy_and_hold_parity_with_costs_and_cash_interest` (NAV within 1e-6 of
    the capital).
- **1.3** Partly done. The softmax transform and RL-03 are verified
  (`::test_rl03_softmax_transform_gives_long_only_weights_summing_to_one_with_cash`). The DQN
  discrete action-set schema and RL-04 are **not implemented**: `dqn` is refused as an unknown
  algorithm.
- **1.4** `docs/rl-environment.md`, labeled configuration pending user review. Verified by
  `::test_documented_default_specification_validates_and_is_the_configured_one`.
- **2.1** Partly done. There is no separate `financemodel-rl` image: the `financemodel-cpu` image
  installs the locked `rl` extra (task 10.3). The local run of the job entry point on fixtures
  produces policy artifacts with checksums and makes no AWS call
  (`tests/unit/selection/test_model_selection_job.py::test_local_container_run_produces_policy_artifacts_without_aws`).
  The docker build itself is verified only in the pipeline build stage.
- **2.2** Not done as written. RL trains inside the single `model_selection` Training job
  (task 10.2, design D1a), not in one Training job per configuration and seed.
- **2.3** Partly done:
  - Done: seed sweeps (default 5, minimum 3), validation-only selection and RL-05
    (`tests/unit/selection/test_model_selection.py::test_rl05_selection_rules_referencing_test_or_holdout_are_refused`).
  - Done: each policy is recorded with its checksum, seed and `configuration_id` in the result and as
    an `rl_policy` artifact.
  - **Not done:** minting a registry `model_version` per policy. The job role may write only the
    registry's `runs/*`.
- **2.4** Partly done. RL-06 is verified
  (`::test_rl06_training_rewards_are_a_separate_section`). The RL-08 out-of-configuration flag is
  not implemented, because policies are evaluated only in the run that trained them.
- **2.5** `FINPLAN_LEARNER_STEP_LIMIT` (512) is set by the offline harness, and learner runs above
  it fail with `OPERATION_NOT_PERMITTED`. Verified by
  `tests/unit/rl/test_rl_training.py::test_rl07_learner_runs_beyond_the_fixture_step_count_are_refused_in_the_build`.

## 3. Qwen3.6-27B weight staging

- [ ] 3.1 Implement the pinned-identity check (model ID plus revision from configuration; anything else rejected); verify WST-01 unit tests
- [ ] 3.2 Implement the staging job (download, per-file SHA-256, manifest after files, status record last, skip when complete, resume when partial, license file retained); verify WST-03 and WST-05 with a local fake model repository of small files (interrupted, resumed and already-staged cases)
- [ ] 3.3 Implement the pre-inference verification step; verify WST-04 with a tampered fixture file (fails with `PRECONDITION_FAILED` before load)
- [ ] 3.4 Add ownership and leak checks that fail on references to another project's storage or roles; verify WST-02 with a planted reference
- [ ] 3.5 Add storage cost to the staging estimate; verify WST-06 unit test
- [ ] 3.6 Run staging once in the chosen environment after approval and verify the manifest independently with a second verification-only run; verify WST-03 and WST-04 integration-beta (fresh download per the FM2-OQ-1 interim unless the user supplies a copy; cost in `cpu_research`)

## 4. Inference lifecycle, mode A (default)

- [ ] 4.1 Build the GPU runner image on the AWS vLLM DLC referenced from `/finplan/<env>/financemodel/config/vllm-image`, with offline mode and tensor parallelism from configuration; verify the image build and a CPU stub-model run of the runner in the build stage (no GPU)
- [ ] 4.2 Add the mode A job type (Training Job on `ml.g6.12xlarge`, network isolation, weights as an input channel, readiness and first-token timing); verify INF-02 and INF-07 unit tests on the generated request and INF-04 unit tests for the timing record
- [ ] 4.3 Add the GPU lease class (limit 1), the `gpu` budget category on every GPU job type and mandatory approval showing the estimate; verify INF-06 and INF-08 unit tests (second GPU run queued; below-threshold GPU run still awaits approval; estimate above the remaining `gpu` budget refused)
- [ ] 4.4 Block external LLM calls from swarm jobs (network isolation plus IAM deny); verify INF-01 with an IAM policy simulation and a unit test of the client allow-list
- [ ] 4.5 Implement the per-run cost record (startup, inference and teardown seconds, tokens, estimated vs. actual); verify INF-09 unit tests
- [ ] 4.6 Run one approved mode A pilot on synthetic data with a short runtime cap, then record startup durations, tokens per decision and cost; verify INF-02, INF-04 and INF-09 integration-beta (needs task 3.6 and explicit user approval of the estimate)

## 5. Inference lifecycle, mode B (feature-flagged, evaluated)

- [ ] 5.1 Implement endpoint provisioning (model, configuration and endpoint per run), health-wait with configured startup and download timeouts, and teardown on every terminal path; verify INF-03 and INF-04 with mocked SageMaker tests (success, failure, cancel and timeout all delete the endpoint)
- [ ] 5.2 Implement the CPU driver job that calls the endpoint with IAM only, and the endpoint invoke policy restricted to the driver role; verify INF-07 with IAM policy simulation
- [ ] 5.3 Implement the orphan sweeper (scheduled, tag-based, deletes or stops and alerts); verify INF-05 with mocked resources
- [ ] 5.4 Run one approved mode B trial, record measured startup time and cost, and write the adopt-or-reject decision in `docs/llm-serving.md`; verify INF-03, INF-04 and INF-05 integration-beta (BLOCKED by FM2-OQ-5 approval of the trial)

## 6. Agent swarm strategy

- [ ] 6.1 Implement the swarm configuration (fixed ordered roles, versioned prompt templates, output schemas, turn limits, arbitration policy) included in `configuration_id`; verify SWM-01 (role added mid-run refused) and SWM-02 (arbitration recorded) unit tests with a stub model
- [ ] 6.2 Implement structured-output parsing with bounded retry and hold fallback, and the invalid-output counter; verify SWM-03 with malformed stub responses
- [ ] 6.3 Implement point-in-time prompt construction from the dataset view only; verify SWM-04 with a look-ahead fixture
- [ ] 6.4 Implement JSONL message logging sealed by checksum at run end; verify SWM-05 (one decision fully reconstructable from logs) and that logs are immutable afterwards
- [ ] 6.5 Record sampling settings and seeds, and report repeats as seeds; verify SWM-06 unit test
- [ ] 6.6 Implement `leakage_risk` labeling from the configured training-data cutoff (unknown means all historical results are flagged) and the prospective-only claim rule; verify SWM-07 and SWM-08 unit tests (cutoff value BLOCKED by FM2-OQ-3; interim "unknown")
- [ ] 6.7 Document roles, arbitration, logging and the leakage limitation in `docs/agent-swarm.md`; verify the leak scan passes and that the documented configuration validates
- [ ] 6.8 Run the swarm strategy within the mode A pilot (task 4.6) and produce a benchmark report next to the controls; verify SWM-02, SWM-05 and SWM-07 integration-beta (runs within the approved task 4.6 pilot)

## 7. TypeSafe Jev strategy (phase 2, approval-gated)

- [ ] 7.1 Implement the Jev enable flag and mandatory approval (flag off returns `DEPENDENCY_UNAVAILABLE`; every run waits in `awaiting_approval` showing AWS and TypeSafe estimates); verify JEV-01 unit tests
- [ ] 7.2 Implement the `choice` question builder (options exactly `buy`, `hold`, `sell`; versioned template in `configuration_id`), response validation, argmax decision with tie-break to `hold`, and invalid/tie counters; verify JEV-02 and JEV-03 unit tests with mocked responses (missing option probability, exact tie)
- [ ] 7.3 Implement the text and bucketed state encoder from point-in-time features with decision-time bucket edges and the token-budget check; verify JEV-04 unit tests (look-ahead bucket edge refused; oversized state not sent; state built from a synthetic snapshot marked as real-provider-derived contains no raw series)
- [ ] 7.3a Apply the foundation's shared outbound payload check (`add-research-job-foundation` task 3.9a) as the pre-send guard on every Jev request, and add the research-only purpose rule for Yahoo-derived datasets (the check rejects runs of more than the configured number of consecutive numbers, any value equal to a raw price or volume observation of the source dataset, tables or attachments; refuse `production_candidate` Jev runs on Yahoo-derived data); verify JEV-17 unit tests (raw-series payload rejected with no network call, bucketed descriptors pass, `production_candidate` refused with `FORBIDDEN`) (FM2-OQ-12 RESOLVED 2026-10-07)
- [ ] 7.4 Implement the TypeSafe API client (bearer auth from the secret named `finplan/shared/financemodel/jev-api-key`, client-side rate limit, 429 retry-after, 529 and network backoff, 401 and 422 no-retry mapping, response cache by request hash, authorization redaction in logs) and the job-role policy that reads only that secret; verify JEV-10 with a fault-injecting mock and JEV-11 with an IAM policy simulation plus the leak scan on a planted key
- [ ] 7.5 Record the returned `model` per decision and flag `model_drift`; verify JEV-05 unit tests (two model values in one run flagged and refused for promotion)
- [ ] 7.6 Implement label definition and the chronological development/calibration/policy/test ranges with horizon embargo and holdout access rule; verify JEV-06 and JEV-07 unit tests
- [ ] 7.7 Implement calibration (default multinomial temperature scaling, isotonic alternative) and the raw-versus-calibrated reliability table, Brier score and ECE; verify JEV-08 with fixture probabilities
- [ ] 7.8 Implement the deterministic sizing policy (`w_max`, `p_threshold` selected on the policy range only) into the simulator, and replay from cached responses; verify JEV-09 unit tests (worked example gives 0.32; replay makes no API call and gives identical trades)
- [ ] 7.9 Implement the TypeSafe cost record (`external_billing: typesafe`, token counts, configured price), exclusion from AWS categories, the AWS job in `cpu_research`, and the per-run token cap; verify JEV-12 unit tests
- [ ] 7.10 Implement availability-cutoff and `leakage_risk` labeling (training cutoff "unknown" flags all historical periods); verify JEV-13 unit test (cutoff value pending FM2-OQ-10; interim "unknown")
- [ ] 7.11 Report classification and portfolio metrics in separate sections; verify JEV-14 with a fixture where accuracy is high and net return is negative
- [ ] 7.12 Build the TypeSafe API mock (recorded fixture responses, fault injection) used by build, beta and gamma tests, and an egress guard that fails a pipeline stage on any request to the real API host; verify JEV-15 in every stage
- [ ] 7.13 Fill `docs/jev-identification.md` with the verified facts and their evidence (vendor documentation, `GET /v1/models` observation, version aliases, pricing, rate limits, numeric-precision weakness) and no other claims; verify JEV-16 by review
- [ ] 7.14 With user approval, send one test request with an exact model ID to resolve FM2-OQ-11 and read through the TypeSafe vendor terms (the Yahoo-data part of FM2-OQ-12 is RESOLVED 2026-10-07: research only, descriptors only); record the outcome in the identification record (uses the secret already stored by the user; no CI involvement)
- [ ] 7.15 Run one approved small Jev evaluation in beta against the real API on fixture data, then on the SPY dataset prepared from approved platform snapshots (platform yfinance ingestion; FinanceModel reads snapshots only); verify JEV-02, JEV-04 (no raw retrieved series in requests), JEV-17 (payload guard active, run purpose `research` or `holdout_evaluation`), JEV-05, JEV-08, JEV-09 and JEV-12 integration-beta (real-data part depends on approved SPY snapshots in beta, foundation task 3.10; FM-OQ-4 and FM2-OQ-12 are RESOLVED 2026-10-07)

## 8. Strategy promotion

- [ ] 8.1 Implement candidate registration without job execution; verify PRO-01 unit test
- [ ] 8.2 Implement the versioned criteria store writable only by the approver role; verify PRO-02 with IAM policy simulation (agent and tool roles denied)
- [ ] 8.3 Implement evaluation cycles with budget and run limits and the unevaluated-candidate report; verify PRO-03 unit tests
- [ ] 8.4 Implement the deterministic promotion check for criteria v1 (incumbent selection with `buy_and_hold` fallback, comparability check, R1 strict net-of-costs return, R2 drawdown not worse, result record with checksum) and store criteria v1 as version 1 in the criteria store; verify PRO-06 and PRO-07 unit tests with the worked fixtures (pass; higher return but worse drawdown; equal return; gross-beats-net-loses; evaluator-version mismatch → `not_comparable`; `model_drift` → `ineligible`; re-run gives an identical checksum) (FM2-OQ-7 RESOLVED 2026-10-07)
- [ ] 8.5 Implement promotion with recorded user approval (approval only on a `pass` result, approver identity and time, registry status only) and the limitations section; verify PRO-04 and PRO-05 unit tests (approval of a failing result refused; promotion changes no plan or risk preference)
- [ ] 8.6 Run one promotion cycle in beta on synthetic data under criteria v1 with the user's approval; verify PRO-03, PRO-04, PRO-06 and PRO-07 integration-beta

## 9. Pipeline and environment integration

- [ ] 9.1 Add the RL, staging, mode A, mode B (flagged) and Jev (enable flag, egress to the TypeSafe API, read access to the one named secret) job types, GPU lease class, sweeper and criteria store to IaC, and publish them under `/finplan/<env>/financemodel/job/*`; verify synth assertions and the OWN-01 ownership check against the contracts D1 rows
- [ ] 9.2 Extend beta and gamma pipeline tests with CPU stub runs of every new job type (no GPU; Jev against the mock only); verify integration-beta and gamma pass, no GPU job appears in the stage's SageMaker calls, and no request reaches the TypeSafe API host
- [ ] 9.3 In gamma, run isolation checks for the new roles (no prod storage, no external LLM egress from swarm roles, Jev role limited to the TypeSafe API and its one secret, no endpoint invoke by other principals); verify INF-01, INF-07, WST-02 and JEV-11 gamma
- [ ] 9.4 Prod smoke: dry-run submissions for each new job type (validation and estimate only; Jev dry run returns the AWS and TypeSafe estimates without calling the API) and an orphan-sweeper dry run; verify the smoke passes with no SageMaker job or endpoint created

## 10. Offline model selection (decision 27, 2026-10-08)

- [x] 10.1 Implement the model-selection protocol (`finplan_model.selection.protocol`). It covers
  2025-2026 data only, train 2025, validation 2026H1, test 2026-07-01..latest, the three
  families, the traditional and RL grids, at least 3 seeds and validation-only selection. It is held
  in `config/<env>.json`, validated by the config gate and frozen into each run at submission.
  Verify the protocol unit tests: the decision 27 defaults; selection on test or holdout gives
  `OPERATION_NOT_PERMITTED`; overlapping splits, fewer than 3 seeds and bad grids give
  `VALIDATION_FAILED`; the same protocol is used in every environment.
- [x] 10.2 Add the `model_selection` job kind as **one CPU SageMaker Training job** (design D1a):
  - control-plane support for `sagemaker_job: training` (`CreateTrainingJob`, `Describe`, `Stop`,
    the Training state-change rule, and IAM on `training-job/fm-<env>-*`);
  - the 3000 s cap and the build cost check (USD 0.28/h planning bound x 3000 s <= USD 0.25);
  - submission validation (strategy `model_selection`, purpose `research` or `holdout_evaluation`,
    not the daily trigger).

  Verify `tests/unit/selection/test_model_selection_job.py`: the request validates against the
  SageMaker API model; it uses the CPU instance, stopping condition, cost tags and disabled
  profiler; the estimate is under USD 0.25; cancel uses `StopTrainingJob`; `MaxRuntimeExceeded`
  gives `timed_out`. Also verify the synth test of the Training state-change rule and the IAM unit
  test.
- [x] 10.3 Add CPU torch, stable-baselines3 and gymnasium as the locked `rl` extra:
  - exact pins, the PyTorch CPU index and hashes in `uv.lock`;
  - the job image installs `--extra rl`; the Lambda bundle does not;
  - the image import check covers the learners and requires `torch.version.cuda is None`.

  Verify `tests/unit/jobs/test_container_image.py`: the lock and hash test, the no-CUDA test, and
  that the control plane imports without the learners. Also run the import check locally with the
  locked environment.
- [x] 10.4 Report the following:
  - the test comparison as `payload.benchmark`;
  - per-split comparisons with family, params, seed, selected and incumbent;
  - the traditional tuning tables and the RL grid with checkpoints;
  - every seed's train, validation and test metrics with their mean, standard deviation, minimum
    and maximum;
  - a separate training-reward section, the frozen selection record, and the criteria v1 promotion
    check against the incumbent (`buy_and_hold` fallback);
  - the test-reuse count, the compute and cost record and the policy artifacts;
  - the mandatory bias section and caveats, including "about 250 training days is thin for RL" and
    hindsight and survivorship.

  Verify `tests/unit/selection/test_model_selection.py` and the job result contract validation
  (`test_container_run_writes_a_contract_result_with_every_section`).
- [ ] 10.5 Deployed: the beta and gamma suites dry-run `model_selection`, which is validated and
  estimated at or below USD 0.25 with nothing recorded
  (`tests/integration/test_deployed_environment.py::test_model_selection_kind_is_deployed_and_estimated_under_the_auto_approve_threshold`).
- [ ] 10.6 Run one `model_selection` in beta on the approved research-universe snapshot, at the
  user's request. Verify: the run succeeds within 3000 s; all 5 seeds per RL algorithm are reported;
  the cost record is in `cpu_research`; the test is evaluated once; and the result is reviewed with
  the user before any production-strategy selection.

## Requirement-to-test mapping

Test types: unit, contract, integration-beta, gamma, smoke.

| Spec | Requirement | Test ID | Type |
|---|---|---|---|
| rl-strategies | Versioned state, action and reward specification | RL-01 | unit |
| rl-strategies | Simulator-backed environment | RL-02 | unit + integration-beta |
| rl-strategies | Continuous allocation algorithms | RL-03 | unit |
| rl-strategies | DQN only with a discrete action design | RL-04 | unit |
| rl-strategies | Multiple seeds | RL-05 | unit + integration-beta |
| rl-strategies | Shaped reward reported separately | RL-06 | unit |
| rl-strategies | Training as SageMaker Training Jobs | RL-07 | unit + contract + integration-beta + smoke (dry run) |
| rl-strategies | Retraining on reward or environment changes | RL-08 | unit |
| llm-weight-staging | Pinned checkpoint identity | WST-01 | unit |
| llm-weight-staging | FinanceModel-owned staged copy | WST-02 | unit (ownership and leak checks) + gamma |
| llm-weight-staging | Idempotent, checksum-manifested staging | WST-03 | unit + integration-beta |
| llm-weight-staging | Verification before inference | WST-04 | unit + integration-beta |
| llm-weight-staging | License retained | WST-05 | unit |
| llm-weight-staging | Staging cost gated | WST-06 | unit |
| llm-inference-lifecycle | Self-hosted inference only | INF-01 | unit (policy simulation) + gamma |
| llm-inference-lifecycle | Serving mode A, batch runner in a Training Job | INF-02 | unit + integration-beta |
| llm-inference-lifecycle | Serving mode B, short-lived endpoint | INF-03 | unit (mocked) + integration-beta |
| llm-inference-lifecycle | Readiness and startup measurement | INF-04 | unit + integration-beta |
| llm-inference-lifecycle | Orphan detection | INF-05 | unit (mocked) + integration-beta + smoke (dry run) |
| llm-inference-lifecycle | GPU concurrency lease | INF-06 | unit |
| llm-inference-lifecycle | Private access | INF-07 | unit (policy simulation) + gamma |
| llm-inference-lifecycle | GPU runs always need approval and a budget check | INF-08 | unit |
| llm-inference-lifecycle | Per-run cost record | INF-09 | unit + integration-beta |
| agent-swarm-strategy | Fixed agent roles | SWM-01 | unit |
| agent-swarm-strategy | Declared arbitration policy | SWM-02 | unit + integration-beta |
| agent-swarm-strategy | Structured outputs validated by deterministic code | SWM-03 | unit |
| agent-swarm-strategy | Point-in-time inputs only | SWM-04 | unit |
| agent-swarm-strategy | Complete prompt and message logging | SWM-05 | unit + integration-beta |
| agent-swarm-strategy | Reproducibility settings | SWM-06 | unit |
| agent-swarm-strategy | Pretraining-leakage limitation reported | SWM-07 | unit + integration-beta |
| agent-swarm-strategy | Prospective evaluation for LLM claims | SWM-08 | unit |
| jev-classification-strategy | Approval-gated phase 2 enablement | JEV-01 | unit + smoke (dry run) |
| jev-classification-strategy | Single choice question per decision | JEV-02 | unit + integration-beta |
| jev-classification-strategy | Deterministic decision and tie-break | JEV-03 | unit |
| jev-classification-strategy | Text-encoded point-in-time features | JEV-04 | unit |
| jev-classification-strategy | Yahoo-derived data sent to TypeSafe for research only | JEV-17 | unit (payload guard, purpose rule) + integration-beta |
| jev-classification-strategy | Response model identity recorded | JEV-05 | unit + integration-beta |
| jev-classification-strategy | Explicit label definition | JEV-06 | unit |
| jev-classification-strategy | Separate development, calibration, policy and test ranges | JEV-07 | unit |
| jev-classification-strategy | Calibration on FinanceModel splits | JEV-08 | unit + integration-beta |
| jev-classification-strategy | Deterministic probability-to-size mapping | JEV-09 | unit + integration-beta |
| jev-classification-strategy | TypeSafe API client with rate limits and retries | JEV-10 | unit (mocked API with fault injection) |
| jev-classification-strategy | API key by secret reference only | JEV-11 | unit (policy simulation + leak scan) + gamma |
| jev-classification-strategy | TypeSafe cost tracked outside the AWS budget | JEV-12 | unit + integration-beta |
| jev-classification-strategy | Availability cutoff and leakage flag | JEV-13 | unit |
| jev-classification-strategy | Accuracy is not profitability | JEV-14 | unit |
| jev-classification-strategy | Mocked Jev API in CI | JEV-15 | unit + integration-beta + gamma (egress guard) |
| jev-classification-strategy | No fabricated model details | JEV-16 | review checklist (documentation) |
| strategy-promotion | Candidate registration | PRO-01 | unit |
| strategy-promotion | Versioned, approved promotion criteria | PRO-02 | unit (policy simulation) |
| strategy-promotion | Budgeted evaluation | PRO-03 | unit + integration-beta |
| strategy-promotion | Deterministic promotion check (criteria v1) | PRO-06 | unit (worked fixtures, checksum stability) + integration-beta |
| strategy-promotion | Comparable evaluation inputs | PRO-07 | unit (`not_comparable`, `ineligible`, `buy_and_hold` fallback) + integration-beta |
| strategy-promotion | Promotion requires recorded user approval | PRO-04 | unit + integration-beta |
| strategy-promotion | No guaranteed-improvement claims | PRO-05 | unit |

## Workflow follow-up

- CG-8 is a planned phase 2 contract minor; CG-9 and CG-10 are resolved (CG-10 by the 2026-10-07 allocation).
- Archive after the beta and gamma checks pass. If the real-data Jev evaluation (7.15) is still waiting for approved SPY snapshots at archive time, split it into a follow-up change.
- Remaining blockers: mode B trial approval (FM2-OQ-5, mode B only). Promotion criteria v1 (FM2-OQ-7) and Yahoo-derived data to TypeSafe (FM2-OQ-12, research only, descriptors only) were RESOLVED 2026-10-07. GPU budget (FM2-OQ-2), Jev identity (FM2-OQ-4) and the data provider (foundation FM-OQ-4: yfinance in the platform ingestion; FinanceModel reads approved snapshots only and has no yfinance dependency) were resolved on 2026-10-07. Real-data runs depend on approved SPY snapshots in beta, not on an open decision.
