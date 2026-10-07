# Design

## Context

See proposal.md (Why) and `specs/` for requirements. This change depends on `add-research-job-foundation` (simulator, evaluator, datasets, job interface, execution controls, registry, reporting) and on FinancialPlanning change `establish-cross-repo-contracts` (identifiers, `finplan-contracts`, error envelope, SSM naming, pipeline standard, isolation). Nothing here redefines those.

### Observed facts (read-only discovery, 2026-10-07)

- A previous Qwen3.6-27B deployment exists in us-east-2, owned by **another project's** CloudFormation stack. FinanceModel must not adopt, read from or modify its bucket, role or stack.
- Model: Hugging Face `Qwen/Qwen3.6-27B`, revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, Apache-2.0, architecture `Qwen3_5ForConditionalGeneration` (multimodal-capable, bf16), 31 files, 51.8 GiB.
- That project staged weights once with a CPU Processing Job: idempotent, per-file SHA-256 `MANIFEST.sha256` written last, then a `STAGED.json` status file.
- That project serves **offline vLLM inside SageMaker Training Jobs** (batch runner, not an endpoint): AWS vLLM DLC `vllm:0.29.0-gpu-py312-cu130-ubuntu24.04-sagemaker-v1.2`, one `ml.g6.12xlarge` (4x L4), 200 GB volume, `HF_HUB_OFFLINE=1`, temperature 0, tensor parallelism from a job file. Last Qwen run: Completed, 4754 s (a generic classification benchmark, not a finance swarm).
- No SageMaker endpoints, models or endpoint configs exist in any region.
- Quotas in us-east-2: `ml.g6.12xlarge` training **1**, processing **0**, endpoint **1**. `ml.m5.xlarge` training 15, processing 1.
- TypeSafe Jev: identified 2026-10-07 (contracts OQ-4 resolved) as TypeSafe AI's external "System One" decision API; details in D7. It is a vendor API, not an AWS resource.
- Budget (user decision 2026-10-07): **USD 50 total AWS budget**, not split by environment. Default allocation, configurable in SSM: infrastructure plus storage USD 8, CPU research jobs USD 7, Bedrock explanation calls USD 5, GPU (Qwen3.6-27B and RL GPU work) USD 25, reserve USD 5. AWS Budgets (FinancialPlanning) alerts at 50/80/100% with a deny action at 100%. Jev usage is billed by TypeSafe from prepaid credits and is outside this budget.
- One AWS account in us-east-2 holds beta, gamma and prod (user decision 2026-10-07).
- Market-data provider (user decision 2026-10-07, contracts OQ-5 and foundation FM-OQ-4 RESOLVED): the FinancialPlanning ingestion Lambda or container retrieves the SPY daily series with `yfinance` (unofficial Yahoo Finance library, no API key, pinned version) and the `exchange_calendars` XNYS calendar; snapshot lineage records the library version and retrieval timestamp. Every strategy family in this change (RL, swarm, Jev) consumes real data only through foundation datasets prepared from approved platform snapshots. FinanceModel never calls yfinance and has no yfinance dependency (foundation spec research-datasets). CI uses the mock provider and synthetic fixtures only.

### Assumptions (unverified)

- FM2-A1. The same DLC tag (or a newer AWS vLLM DLC tag) can serve as a SageMaker real-time endpoint container. **To verify** in the mode B trial; the observed use is Training Jobs only.
- FM2-A2. The public Hugging Face revision can be downloaded without a token from a SageMaker CPU job that has internet egress. To verify in the staging dry run.
- FM2-A3. Stable-Baselines3-style PPO/SAC on CPU is adequate for small-universe allocation; GPU is not needed for RL in this phase.
- FM2-A4. The Qwen3.6-27B model card states, or does not state, a training-data cutoff. Not yet checked (FM2-OQ-3).

## Goals / Non-Goals

**Goals:**
- Fair comparison: every learned and LLM strategy runs through the same simulator, datasets and reports as the baselines.
- Bounded spend: no idle GPU, one GPU run at a time, every GPU run estimated and human-approved within the USD 25 `gpu` allocation; CPU work within `cpu_research` (USD 7); TypeSafe spend tracked separately.
- Reuse the proven serving pattern without touching the other project's resources.

**Non-Goals:**
- Fine-tuning or RL-from-feedback on Qwen3.6-27B.
- A persistent inference service for other consumers. The explanation agent in FinanceAgent uses Amazon Bedrock (decision 2026-10-07), not this model and not Jev.
- Fine-tuning Jev (not offered) or relying on vendor calibration claims.

## Decisions

### D1. RL on CPU Training Jobs

- PPO and SAC run as SageMaker Training Jobs on CPU (`ml.m5.xlarge`, observed training quota 15), one job per `(configuration_id, seed)`, so a seed sweep is several small jobs within the CPU lease limits. GPU was rejected: small MLP policies on daily data gain little from a GPU, and the GPU quota of 1 is reserved for Qwen.
- Default environment specification (configuration, to confirm with the user): state = trailing window of per-instrument log returns and rolling volatility at decision time, plus current weights and cash. Action = real vector of size n+1 mapped by softmax to long-only weights including cash. Reward = log of net portfolio value change from the simulator, minus a turnover penalty, minus a drawdown-increment penalty, each with a configured coefficient. Episode = one training fold. Decision frequency = the benchmark's rebalance frequency.
- Seeds: default 5, minimum 3. Selection on validation folds by a declared metric. Test and holdout are read only by `holdout_evaluation` runs.
- DQN: rejected as a default. Allowed only with a declared finite action set (for example "hold" or "move k% from asset i to asset j").
- Build-stage tests run the environment step, reward and action transform on fixtures with a stub policy and a fixed tiny step count. No learner runs in CodeBuild.

### D2. Weight staging: own copy, discovered pattern

- A FinanceModel CPU Processing Job (`ml.m5.xlarge`, attached volume sized above the 51.8 GiB checkpoint plus headroom) downloads the pinned revision from Hugging Face into FinanceModel research storage, writes a per-file SHA-256 manifest after all files, then a staged-status record naming model ID, revision and license. It is idempotent: an existing complete manifest that matches the revision skips the download.
- **Alternatives:** (1) copying from the other project's bucket. Rejected as a FinanceModel action because it means reading another project's resource. If the user wants to avoid a second download, the user may copy the files into FinanceModel staging themselves, and the staging job's verification step then validates them (FM2-OQ-1). (2) Pulling from Hugging Face at every inference start. Rejected: it repeats a 52 GiB transfer on paid GPU time, and the GPU job needs network isolation.
- The staging job needs internet egress. Inference jobs do not and run network-isolated.

### D3. Serving: mode A by default, mode B measured before adoption

| | Mode A: offline vLLM in a Training Job | Mode B: short-lived endpoint + CPU driver |
|---|---|---|
| Evidence | Observed working in this account (4754 s run) | Not observed; DLC endpoint compatibility unverified (FM2-A1) |
| Quota | g6.12xlarge training 1 | g6.12xlarge endpoint 1, plus an `ml.m5.xlarge` processing slot for the driver (quota 1, shared with CPU jobs) |
| Idle cost | None. The swarm loop runs inside the job | Billed from endpoint creation to deletion, including driver gaps |
| Lifecycle | Job start to job end. SageMaker stops it at max runtime | Create model, config and endpoint, health wait, run, delete all three. Orphan sweeper |
| Startup | Instance plus image pull plus loading 52 GiB from S3 inputs (measure) | Same plus endpoint health checks (measure). Startup and data-download timeouts must be set above the measured load time |
| Network | Network isolation on the job | Container isolation, IAM-only invoke |

- **Decision:** mode A is the default and the only mode in the first GPU milestone. Mode B is built behind a feature flag and adopted only after one approved, measured trial shows a benefit (for example many short interactive runs) that outweighs its idle and driver cost. Pause/resume is never offered. "Stopping" an endpoint always means deleting it and recreating it next time.
- GPU Processing Jobs are not used: quota is 0. Requesting an increase is possible but adds nothing over mode A for this workload (FM2-OQ-6).
- The image reference (DLC repository and tag) comes from configuration (`/finplan/<env>/financemodel/config/vllm-image`). It is not written in the repo because it embeds an AWS account ID.

### D4. GPU lease and gating

- GPU lease class `gpu-g6-12xlarge` with `max_holders` 1 per environment. Because the account quota of 1 is shared across environments (CG-3, resolved in contracts D10 as an external limit), quota-aware requeue also applies.
- Every GPU run: pre-flight estimate (`price × max_runtime`, including startup, from the configured price table) checked against the remaining `gpu` category (default USD 25, from the shared `/finplan/shared/financialplanning/config/budget-allocation`; foundation CTL-10), mandatory `awaiting_approval` with the estimate shown to the user regardless of threshold, then lease. A per-run maximum runtime is required. The first milestone uses a small pilot with a short cap so one run cannot exhaust the allocation. RL CPU training, weight staging and Jev driver jobs use `cpu_research` (USD 7, shared with the foundation).
- Pipeline tests never start GPU jobs. Beta and gamma test mode A wiring with the CPU fallback image and a stub model.

### D5. Swarm design

- Default fixed roles (configuration to confirm, FM2-OQ-8): `market_analyst` (summarizes point-in-time features), `risk_analyst` (risk and drawdown view), `allocator` (proposes target weights as JSON), `critic` (checks the proposal against constraints and the analysts' views; one revision round allowed), `arbiter` (chooses the original or the revised proposal). Arbitration policy `arbiter_selects`, with a deterministic alternative `median_of_proposals` for ablation.
- Output parsing uses JSON schemas with vLLM guided decoding where the DLC version supports it; otherwise strict parsing with a bounded retry, then fallback to hold.
- Logs: one JSONL record per message (`run_id`, decision time, role, prompt, response, sampling parameters, token counts, model revision) written to research storage and sealed with a checksum at run end.
- Decision frequency is kept low (for example monthly rebalances on a small universe) so token volume and GPU time fit the budget. The pilot measures tokens per decision to size later runs.
- Whether a "thinking" or reasoning mode is used is a recorded configuration value, not assumed (FM2-OQ-9).

### D6. Leakage reporting

- Each swarm and Jev result carries `leakage_risk` per period: true for any date before the model's documented training-data cutoff, and for all historical dates if the cutoff is unknown (currently the case for both Qwen3.6-27B and Jev). Prospective paper results after the freeze are the only results presented as leakage-free.

### D7. TypeSafe Jev strategy (identified 2026-10-07; phase 2, approval-gated)

Identification (user-verified from the vendor's official documentation and a live `GET /v1/models` call, recorded in `docs/jev-identification.md`; only these facts are used):

- Maker TypeSafe AI, "System One" decision model: text in, typed decisions with probabilities out. Calibration is a vendor claim (RLCD training) that we do not rely on. No prose generation, no fine-tuning.
- API base `https://api.typesafe.ai`, `Authorization: Bearer <key>`. `POST /v1/systemone` with `state`, `model` and `questions`. Question types: `noul` (yes/no probability), `choice` (up to 255 options, per-option probabilities plus a confidence) and `score` (2 to 10 ordinal levels). `GET /v1/models` lists models. Errors: 401, 422, 429 (with retry-after), 529. Official SDKs exist for Python (`typesafe_sdk`) and JavaScript.
- Versions: documentation names `jev-1.13.0` as current; `/v1/models` exposes the aliases `jev-latest` and `jev-preview`. Context 64k tokens (32k for state plus questions). Rate limit about 80 requests per second (dynamic). Price about USD 0.042 per million input tokens, output free, billed by TypeSafe from prepaid credits.
- Vendor-reported weakness: numeric precision.

Design:

- **Execution.** A CPU job type `jev_backtest` (`ml.m5.xlarge`, budget category `cpu_research`) runs the decision loop through the shared simulator and calls the TypeSafe API over HTTPS. Unlike swarm jobs it needs internet egress, restricted by policy to the TypeSafe API host where the network setup allows it. The job role may read only the secret named `finplan/shared/financemodel/jev-api-key` (us-east-2); the name, or the SSM pointer `/finplan/shared/financemodel/secret-ref/jev-api-key` registered in contracts D4, is the only reference in repo files. The key is never logged, written to results or passed in job arguments.
- **Question.** One `choice` question per instrument and decision date with options exactly `buy`, `hold`, `sell`. One question yields one probability vector, so buy and sell can never both be selected. The question text and option order are versioned templates in the `configuration_id`.
- **State encoding.** Point-in-time features are rendered into the `state` text as words and buckets (for example "20-day return: strongly positive (top decile of the trailing year)"), not raw floats. Bucket edges are computed from data available at decision time and are part of the `configuration_id`. Requests are checked against the 32k state-plus-question token budget before sending.
- **Model identity.** The request `model` is configuration. Default `jev-latest` until an exact ID such as `jev-1.13.0` is confirmed to be accepted by the API (FM2-OQ-11); then the exact ID is pinned. Every response's returned `model` field is stored per decision; a run whose responses report more than one model value is flagged `model_drift` and is ineligible for promotion.
- **Labels and splits.** Jev is not trained by us. Labels (forward return over a configured horizon with buy/sell thresholds) are used only for calibration, policy selection and evaluation. Chronological ranges: template development, calibration, policy (threshold and sizing) selection, test; each separated by an embargo of at least the label horizon. The holdout is read only by `holdout_evaluation` runs.
- **Calibration.** Raw per-option probabilities are calibrated on the calibration range with a configured method (default multinomial temperature scaling; isotonic one-vs-rest as an alternative) and evaluated on the test range with a reliability table, Brier score and expected calibration error. Vendor calibration claims are not reported as results.
- **Sizing policy.** Deterministic: decision = argmax of calibrated probabilities with a fixed tie-break to `hold`. For `buy`, target weight = `w_max × clip((p_buy − p_threshold) / (1 − p_threshold), 0, 1)`; for `sell` (long-only), target weight 0; for `hold`, current weight. `w_max` and `p_threshold` are selected on the policy range only and recorded. Targets then pass the simulator's constraint policy; trades come only from the simulator. The same inputs and responses always give the same weights.
- **Rate limits and retries.** A client-side token bucket stays below a configured rate (default well under the documented 80 requests per second). 429 waits for retry-after; 529 and network errors retry with exponential backoff and jitter up to a configured count. 401 fails the run with `DEPENDENCY_UNAVAILABLE` (credential problem, not retried); 422 fails the run with `INTERNAL` (our request is malformed). Responses are cached by request hash in research storage so a resumed or replayed run makes no repeat calls.
- **Cost.** Two estimates at approval: the AWS CPU job estimate (in `cpu_research`) and the TypeSafe estimate (input tokens × configured price, label `external_billing: typesafe`). The TypeSafe amount is recorded and reported separately and never counted against the USD 50 AWS budget. A configured per-run TypeSafe token cap stops the run when reached.
- **Approval.** Jev job types are deployed in phase 2 with an enable flag in configuration. Every Jev run waits in `awaiting_approval` regardless of the auto-approve threshold. With the flag off, submissions return `DEPENDENCY_UNAVAILABLE`.
- **CI.** Build, beta and gamma pipeline tests use a mocked TypeSafe API (recorded fixture responses and fault injection for 429, 529, 401, 422). No pipeline stage calls the real API or reads the secret value.
- **Market data sent to the vendor.** On real data the `state` carries only bucketed descriptors derived from approved snapshots, never raw retrieved series. Yahoo Finance data (via the platform's yfinance ingestion) is for personal and research use, so whether derived descriptors may be sent to TypeSafe is part of the terms review before the first real-data Jev run (FM2-OQ-12). Request and response caches, swarm prompt and message logs, and reports derived from real data stay in research storage and are never committed to the public repository.
- **Leakage.** The Jev training-data cutoff is not documented (FM2-OQ-10). Every historical result is labeled `leakage_risk: true`; only prospective paper results after the configuration freeze are presented as leakage-free.

### D8. Promotion loop

- Candidate records live in the FinanceModel registry (status `candidate`). Criteria documents are versioned in research storage, written only by the approver role. Cycles have a budget, a run limit and a report. Promotion sets registry status `promoted` with approver identity. Publishing any plan stays a platform action outside FinanceModel.

## Risks / Trade-offs

- [One GPU run can consume much of the USD 25 GPU allocation] → pilot-first, mandatory approval with estimate, short runtime caps, no endpoints by default, orphan sweeper, AWS Budgets deny at 100% of USD 50.
- [Pretraining leakage makes historical LLM results optimistic] → `leakage_risk` labels and prospective-only claims.
- [GPU non-determinism] → temperature 0, recorded seeds, repeated runs reported as seeds.
- [Shared account quota across environments] → GPU work runs in one environment at a time. Only beta and gamma run CPU stubs in tests.
- [Weight staging egress and storage cost] → one-time staging, storage cost in the estimate, optional user-made copy (FM2-OQ-1).
- [RL overfits to few folds] → walk-forward folds, multiple seeds, validation-only selection, holdout once.
- [Jev is an external vendor API whose aliases can change behind the same name] → record the returned `model` per response, flag `model_drift`, pin an exact ID once accepted (FM2-OQ-11), cache responses for replay.
- [Jev numeric weakness] → text and bucketed features; calibration on our own splits.
- [yfinance is unofficial and Yahoo terms are personal or research use] → FinanceModel consumes only approved platform snapshots, keeps real-data-derived artifacts out of the public repo, sends only derived descriptors to TypeSafe after the FM2-OQ-12 review.
- [Jev API key exposure] → Secrets Manager by name only, job role scoped to that secret, leak scan, key never logged.
- [TypeSafe credits run out mid-run] → per-run token cap and estimate at approval; a 401 or credit error fails the run without retries.

## Migration Plan

1. Prerequisites: foundation change deployed in the target environment, with the `gpu` (USD 25) and `cpu_research` (USD 7) categories in the allocation document. Price table entries for the GPU and CPU types. For Jev: the secret `finplan/shared/financemodel/jev-api-key` is present in us-east-2 (stored by the user) and the Jev price is configured.
2. Deploy RL job types; run an approved CPU seed sweep on synthetic data in beta.
3. Deploy weight staging. Run it once (approved), verify the manifest.
4. Deploy mode A swarm job type. Run one approved pilot on synthetic data with a short cap, and record startup, tokens per decision and cost.
5. Optional: one approved mode B trial with measured startup, then decide (FM2-OQ-5).
6. Jev: deploy the job type with the enable flag on in beta; run one approved small evaluation on synthetic or fixture data against the real API, then an approved evaluation on the SPY dataset once the platform's yfinance ingestion has produced approved snapshots in beta (provider decided 2026-10-07) and the terms review (FM2-OQ-12) covers sending derived descriptors.
7. Rollback: redeploy the previous release. Staged weights, logs and registry entries remain (immutable). Orphan sweeper runs after rollback.

## Open Questions

| ID | Question | Blocks | Resolved by | Interim |
|---|---|---|---|---|
| FM2-OQ-1 | Stage a fresh Hugging Face download, or let the user copy the files into FinanceModel staging from their other project? | None (either path is verified by the same manifest check) | User decision | Fresh download by the FinanceModel staging job |
| FM2-OQ-2 | GPU budget within the USD 50 ceiling (per run and in total) | None | RESOLVED 2026-10-07: `gpu` category USD 25 of the USD 50 total; every GPU run needs explicit user approval with a cost estimate (current SageMaker pricing at approval time) | n/a |
| FM2-OQ-3 | Qwen3.6-27B training-data cutoff | None (reporting falls back to "unknown") | Model card review | All historical results `leakage_risk: true` |
| FM2-OQ-4 | TypeSafe Jev identity, version, license, interface (contracts OQ-4) | None | RESOLVED 2026-10-07: TypeSafe AI "System One" API, `choice` question buy/hold/sell, key in Secrets Manager by name; see D7. Jev moves to phase 2, enabled behind per-run approval | n/a |
| FM2-OQ-5 | Does the vLLM DLC work as an endpoint, and what is the measured startup time? | BLOCKER for mode B adoption only | One approved trial | Mode A only |
| FM2-OQ-6 | Request a GPU Processing quota increase? | None | User decision | Not requested; mode A uses Training Jobs |
| FM2-OQ-7 | Content of the promotion criteria | **BLOCKER** for any promotion | User approval of criteria v1 | Evaluation allowed; promotion disabled |
| FM2-OQ-8 | Confirm swarm roles, arbitration policy and decision frequency | None (configuration) | User review | Defaults in D5 |
| FM2-OQ-9 | Use the model's reasoning or thinking mode, if any? | None (configuration) | Model card review plus pilot | Recorded per run; default off |
| FM2-OQ-10 | Jev training-data cutoff | None (reporting falls back to "unknown") | Vendor documentation or vendor statement | All historical Jev results `leakage_risk: true` |
| FM2-OQ-11 | Does the API accept an exact model ID (for example `jev-1.13.0`) instead of an alias? | None | One approved test request | Request `jev-latest`; record the returned `model` per response |
| FM2-OQ-12 | Jev license and terms of use for research backtests; also whether bucketed descriptors derived from Yahoo Finance (yfinance) data may be sent to TypeSafe | None for mocked CI; review before the first real-API run (and before the first real-data Jev run for the Yahoo part) | User review of vendor terms and Yahoo terms | Mocked API only until reviewed; fixture data only for real-API runs until the Yahoo part is reviewed |

## Contract gaps (status after the cross-repo review, 2026-10-07)

- CG-8: scheduled as a phase 2 contract minor (contracts D10). Until it ships, the flags are carried in the FinanceModel result payload extension and reported in benchmark reports.
- CG-9: resolved (contracts D1 rows for staged weights, sweeper and criteria store).
- CG-10: resolved (USD 50 total; `gpu` USD 25 and `cpu_research` USD 7 allocations decided 2026-10-07).

Original wording:

- CG-8. The finance result payload has no `leakage_risk` per period or `out_of_configuration` flag. FinanceModel proposes them as an additive minor.
- CG-9. The ownership matrix (contracts D1) lists "Self-hosted Qwen3.6-27B serving lifecycle and concurrency lease" as internal, but not the staged-weights storage, the orphan-sweeper schedule or the promotion-criteria store. They need rows to pass OWN-01 (extends CG-2).
- CG-10. Contracts OQ-7 (cost ceilings) is open, while the brief sets USD 50 total. GPU allocation needs a recorded per-repo split (extends CG-7).
