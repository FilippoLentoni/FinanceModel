# FinanceModel job interface (for FinanceLambdasTool and platform callers)

Task 7.6 of `add-research-job-foundation`. Specs: `experiment-job-interface`,
`job-execution-controls`. The examples below are validated against the pinned contract package by
`tests/unit/control/test_job_interface_docs.py`; when an example and the contract disagree, the
contract wins and the test fails.

> **Contract version.** FinanceModel pins `finplan-contracts` **1.0.0** (the first stable release,
> allowed in beta, gamma and prod; see [foundation.md](foundation.md)). The job lifecycle states,
> `purpose`, `dry_run` and the cost-estimate block used here are unchanged from 0.2.2.
> Served contract majors are listed in `config/<env>.json` (`served_contract_majors`, now `[1]`;
> FinanceModel was never deployed on 0.x, so no major-0 records or callers exist); a request
> declaring any other major gets `UNSUPPORTED_CONTRACT_VERSION`.

## Endpoint and authentication

- Resolve the endpoint at deploy or run time from `/finplan/<env>/financemodel/api/job-endpoint`.
  Never use a literal. The release manifest `/finplan/<env>/financemodel/release/manifest` says
  whether FinanceModel is released in the environment; until it is, the experiment tools return
  `DEPENDENCY_UNAVAILABLE`.
- API Gateway REST API, **IAM (SigV4) auth**, service `execute-api`. The API resource policy allows
  only the granted invoker roles. Any other principal gets `403` from API Gateway.
- Optional request header `X-Correlation-Id` (8 to 128 characters `[A-Za-z0-9_.:-]`). It is
  echoed in the `X-Correlation-Id` response header and in every error envelope.

| Operation | Method and path | Success | Who may call |
|---|---|---|---|
| `submit_job` | `POST /v1/jobs` | `202` (`200` for a dry run or an idempotent replay) | granted invokers (`production_candidate` only for granted platform principals) |
| `list_jobs` | `GET /v1/jobs?state=&page_size=&next_token=` | `200` | granted invokers |
| `get_job_status` | `GET /v1/jobs/{run_id}` | `200` | granted invokers |
| `get_job_result` | `GET /v1/jobs/{run_id}/result` | `200` | granted invokers |
| `cancel_job` | `POST /v1/jobs/{run_id}/cancel` | `200` | the submitting principal or the approver |
| `approve_run` | `POST /v1/jobs/{run_id}/approve` | `200` | the human approver role **only**. Tool, agent and pipeline roles are denied by the resource policy and by the handler. |

## Job types (phase 1)

| `job_type` | What runs | Default / max runtime | Budget category |
|---|---|---|---|
| `prepare_dataset` | dataset preparation from an approved snapshot | 600 s / 1800 s | `cpu_research` |
| `run_backtest` | one strategy over the evaluation window | 900 s / 1800 s | `cpu_research` |
| `run_benchmark` | the strategy plus the `cash`, `buy_and_hold` and `equal_weight` controls, in one job | 1200 s / 1800 s | `cpu_research` |
| `report` | benchmark report | 300 s / 900 s | `cpu_research` |
| `daily_recommendation` | the production strategy (from `config/production-strategy`) on an approved `equity-etf-daily` snapshot; stages one recommendation | 1800 s / 1800 s | `cpu_research` |
| `model_selection` | decision 27 offline model selection: controls, traditional optimizers and PPO/SAC (5 seeds) on one snapshot, tuned on train/validation, one test evaluation (a SageMaker **Training** job; [model-selection.md](model-selection.md)) | 3000 s / 3000 s | `cpu_research` |

All jobs are CPU jobs on `ml.m5.xlarge` (one instance); all but `model_selection` run as SageMaker
Processing jobs. They run on whatever approved
platform snapshot is named: real phase 2 (yfinance) snapshots or synthetic ones, identically in
beta, gamma and prod (data parity, user decision 26; prod may still serve synthetic snapshots
during the transition). Synthetic fixtures remain for offline and unit tests. The contract
fixtures' `fixture_optimizer` is **not** a FinanceModel job type, so submitting it gives
`VALIDATION_FAILED` (`/job_type`). Job types announced for later changes (`rl_train`,
`rl_evaluate`, `rl_weight_staging`, `swarm_mode_a`, `swarm_mode_b`, `jev_backtest`) give
`DEPENDENCY_UNAVAILABLE` with `retryable` false until they are deployed. A configured job type
whose job definition is not published in the environment gives the same error.

### Strategy comparison in results (`payload.benchmark`)

`run_backtest` and `run_benchmark` results carry `payload.benchmark`, so callers of
`get_job_result` / `get_experiment_result` see the comparison without reading the research
artifact: `primary_strategy`, `evaluation_window` (`start`, `end`, `sessions`), `periods_per_year`,
`base_currency`, `initial_capital`, `risk_free` (`annual_rate`, `source`), a `units` legend, and one
`strategies[]` row per evaluated strategy (configured strategy first, then `cash`, `buy_and_hold`,
`equal_weight`) with `role` (`optimizer` or `control`), `solution_status`, `metrics`
(`total_return`, `cagr`, `ann_volatility`, `sharpe`, `max_drawdown`, `turnover`,
`transaction_cost`, `transaction_cost_fraction`), `final_weights`, `final_cash_weight`,
`average_weights` and `average_cash_weight`. Units: returns, volatility and drawdown are fractions
(`max_drawdown` positive, 0.12 = -12 %); `turnover` is cumulative traded notional (buys + sells)
divided by the initial capital (19.6 = 19.6x the starting capital traded); `transaction_cost` is
fees + spread + slippage in `base_currency` (USD 392 on USD 100,000 = 0.39 %); weights are
fractions of NAV at the session close (final = last session, average = mean over the window).
Weight maps are capped at 40 instruments (`weights_truncated_to`). The contract run-result payload
is an open object, so this is contract-valid under 1.1.0; `payload.performance` is unchanged.

### Research universe and the daily recommendation (contracts 1.1.0)

- `run_backtest` and `run_benchmark` accept approved `finance/equity-etf-daily/research-universe`
  snapshots (VOO, GOOGL, NFLX, AAPL, NVDA plus cash). Returns use `adj_close`; cash follows the
  snapshot's declared assumption (`zero_nominal`). Results and reports carry a "Hindsight and
  survivorship bias" section (`payload.bias_section`) that reproduces the snapshot's
  `bias_disclosures` unchanged and states the cash assumption. A universe snapshot without
  disclosures gives `VALIDATION_FAILED`. These runs are user-initiated only; FinanceModel defines no
  schedule.
- `daily_recommendation` is accepted only from `finplan-<env>-financialplanning-daily-trigger-step-role`,
  with `purpose: production_candidate` and `plan_id`. Other callers get `FORBIDDEN`, and the trigger
  role gets `FORBIDDEN` for any other kind. The strategy comes from
  `/finplan/<env>/financemodel/config/production-strategy`. With no strategy selected the call fails
  with `PRECONDITION_FAILED` (`no_production_strategy`, or `strategy_not_eligible`) before anything
  is recorded or started. A request naming another strategy gets `VALIDATION_FAILED`. The run is
  idempotent by key, checked against the budget (`BUDGET_EXCEEDED`), uses one `ml.m5.xlarge` with a
  1800 s cap, and is auto-approved up to its own USD 0.15 ceiling. A succeeded run stages weights for
  every instrument plus cash, with the snapshot's `bias_disclosures` in the staged manifest.

### Offline model selection (`model_selection`, decision 27)

- Submitted on request only (agent or API; never scheduled; the daily trigger gets `FORBIDDEN`),
  with `purpose` `research` or `holdout_evaluation` and `configuration.payload.strategy`
  `model_selection` (anything else gives `VALIDATION_FAILED`). The payload names the universe and the
  shared simulation settings (fees, constraints, monthly rebalance); the protocol (splits, grids,
  seeds, RL hyperparameters, selection rule) is configuration (`config/<env>.json`
  `job_types.model_selection.protocol`), frozen into the run at submission and never taken from the
  request.
- One CPU SageMaker Training job (`ml.m5.xlarge`, 3000 s cap). Its upper-bound estimate (about
  USD 0.20 at the current price) is under the beta/gamma auto-approve threshold (USD 0.25); in prod a
  human approves. The build check keeps the planning-bound estimate at or below USD 0.25.
- The result carries `payload.benchmark` (the **test** comparison) and `payload.model_selection`:
  per-split comparisons (`train`, `validation`, `test`) with `family`, `params`, `selected` and
  `incumbent` per row, the traditional tuning tables, the RL grid with every seed's train, validation
  and test metrics and their mean, standard deviation, minimum and maximum, a separate
  `training_reward` section, the frozen selection record (`selection_checksum`), the promotion check
  (criteria v1 against the incumbent), caveats and the compute record. Artifacts: the evidence JSON
  and every trained policy (`rl_policy`, `application/zip`).

### Production strategy (`GET` / `PUT /v1/production-strategy`)

`PUT` takes `core/v1/tools/production-strategy-request.json` (`action` `get`, `set` or `clear`)
plus `confirmed_by_user` (required, `true`, for `set` and `clear`) and an optional `on_behalf_of`.
It answers with `core/v1/tools/production-strategy-response.json`. Only the tool plan-writer role and the
platform operator roles may `set` or `clear`. `set` checks that the strategy is registered,
deployed, not retired, supports `equity-etf-daily`, has a succeeded research backtest or benchmark
on that dataset in the environment and, for learning families, is `promoted`. A failure gives
`VALIDATION_FAILED` with `details.rule` (for example `no_evaluation_evidence`). A missing
confirmation gives `PRECONDITION_FAILED` `confirmation_required`. The stored document follows
`core/v1/production-strategy.json`. Every change is audited (old value, new value, user, channel).

Strategies accepted in `configuration.payload.strategy` for `run_backtest` and `run_benchmark` are
`cash`, `buy_and_hold`, `equal_weight`, `min_variance`, `mean_variance` and `scenario_cvar`,
plus any strategy the image registers. Any other name gives `VALIDATION_FAILED`.

## Run lifecycle

```
submit ─► awaiting_approval ─(approve)─► queued ─(lease)─► starting ─► running ─► succeeded | failed | timed_out
            │ (expiry / cancel)            │ (cancel)          │ (quota, throttling: back to queued)
            └──────────► cancelled ◄───────┘                   └─(cancel)─► stopping ─► cancelled
```

- `state` is a non-terminal state or the terminal `completion_status`. Terminal runs never change
  state. A late SageMaker event after a terminal state is logged and ignored.
- `transitions` lists every state with its timestamp (and a `reason` for automatic moves such as
  `approved`, `start_retry`, `account_quota_exhausted` or `approval_expired`).
- `elapsed_seconds` is the runtime since `running`.
- `completion_status` is separate from `solution_status`. An infeasible or unbounded optimizer run
  is `succeeded` with `solution_status` `infeasible` / `unbounded`, never an error. A crash is
  `failed` with an `error` envelope and no `solution_status`. `cancelled` and `timed_out` results
  always have `artifacts_complete` false. Their partial outputs are never staged.
- Extra status fields: `job_type`, `max_runtime_seconds`, `wait_reason` (while waiting, for example
  `awaiting_human_approval`, `account_quota_exhausted`, `start_throttled`), `cancel_reason`
  (`cancelled_by_caller`, `approval_expired`) and `solution_status` (when succeeded).
- Responses never contain storage locations, bucket names, keys or principal ARNs. The approval
  records the approver's **role name**.

## Errors

Errors are contract error envelopes. The HTTP status mapping is the same as the platform API's.

| Code | HTTP | When (job interface) | `retryable` |
|---|---|---|---|
| `VALIDATION_FAILED` | 400 | schema, caller `run_id`, storage location, unknown job type or strategy, runtime above ceiling, `configuration_id` mismatch | false |
| `INVALID_IDENTIFIER` | 400 | malformed `run_id` in the path | false |
| `UNSUPPORTED_CONTRACT_VERSION` | 400 | contract major not served (`details.served_contract_majors`) | false |
| `UNAUTHORIZED` | 401 | no caller identity | false |
| `FORBIDDEN` | 403 | `production_candidate` without grant, approve by a non-approver, cancel by a third party | false |
| `OPERATION_NOT_PERMITTED` | 403 | not returned today: experiment configurations cannot select an execution venue (simulation is paper only) | false |
| `BUDGET_EXCEEDED` | 403 | estimate above the remaining category allocation, or the project budget deny action active | false |
| `NOT_FOUND` | 404 | unknown run, unknown snapshot | false |
| `CONFLICT` | 409 | the run changed concurrently while cancelling or approving; retry | false |
| `IMMUTABLE_RECORD` | 409 | not returned by the job interface (registered for completeness) | false |
| `IDEMPOTENCY_KEY_REUSED` | 422 | same key with a different body | false |
| `PRECONDITION_FAILED` | 422 | result of a non-terminal run (`details.state`), unapproved snapshot, missing or stale price, job type without budget category, approval expired | false |
| `RATE_LIMITED` | 429 | queue full | true |
| `INTERNAL` | 500 | unexpected failure; a crashed job's `error.code` | false |
| `DEPENDENCY_UNAVAILABLE` | 503 | job type not deployed (`retryable` false), runtime setting unreadable (`retryable` true) | as stated |

## Idempotency

`submit_job` (non-dry-run) and `cancel_job` require `idempotency_key` (`[A-Za-z0-9_-]{1,128}`).
The scope is (caller role, environment, operation, key). An assumed-role session is normalized to
its role, so a retry from a new session replays. Records are kept at least 7 days (8 configured).
The same key and the same body hash (RFC 8785) return the original response and start no second
job. The same key with a different body gives `IDEMPOTENCY_KEY_REUSED`. Dry runs are neither
looked up nor recorded, so a dry run and the real submission may share a key, as the
FinanceLambdasTool flow does.

## Cost estimate, budget and approval

- The estimate is the upper bound `price per instance-hour × max runtime × instance count +
  storage estimate`. Prices come from `/finplan/<env>/financemodel/config/instance-prices`. A
  missing or stale price gives `PRECONDITION_FAILED`.
- Every job type declares a budget category. Phase 1 uses `cpu_research`, default USD 7, from
  `/finplan/shared/financialplanning/config/budget-allocation`. The remaining allocation is the cap
  minus this environment's runs in that category (actual cost when billed, else the estimate; runs
  that ended before their job started count 0).
- While `/finplan/shared/financialplanning/config/budget-state` reports the 100% deny action, every
  paid submission (dry runs included) gives `BUDGET_EXCEEDED`.
- A run whose estimate is above `/finplan/<env>/financemodel/config/auto-approve-usd` (default 0,
  so every paid job) waits in `awaiting_approval` until the human approver approves it. GPU runs
  always wait. Approval expires after 72 hours (configuration), and the run then ends `cancelled`
  with `cancel_reason` `approval_expired`. Tools never approve.

## Examples

### Submit (dry run first, then the run)

<!-- example: job-submission -->
```json
{
  "domain": "finance",
  "domain_schema_version": "1.0",
  "job_type": "run_backtest",
  "purpose": "research",
  "dry_run": true,
  "input_snapshot_id": "snap_01KDVDNAZ83BAMMYCEGWF33DPM",
  "configuration": {
    "domain": "finance",
    "domain_schema_version": "1.0",
    "payload": {
      "strategy": "equal_weight",
      "objective": "backtest",
      "universe": ["SPY", "CASH"],
      "rebalance_frequency": "monthly",
      "constraints": {"long_only": true, "max_weight": 1.0},
      "fees": {"transaction_cost_bps": 1}
    },
    "synthetic": true
  },
  "evaluation_window": {"start": "2026-01-05", "end": "2026-03-27"},
  "idempotency_key": "lt_0a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f9",
  "contract_version": "1.0.0",
  "synthetic": true
}
```

Dry-run response (`200`). No `run_id` is minted, no run is recorded and no lease is taken:

<!-- example: tools/submit-experiment-response -->
```json
{
  "run_id": null,
  "configuration_id": "cfg_9eda1821d7a3f8b5c965f4375058e544a3aba022644d6a94d9e446a63100a5f0",
  "state": null,
  "dry_run": true,
  "cost_estimate": {
    "estimated_usd_upper_bound": 0.0725,
    "price_retrieved_at": "2026-01-01T00:00:00Z",
    "remaining_allocation_usd": 6.9275,
    "budget_category": "cpu_research",
    "synthetic": true
  },
  "message": "Dry run: the request is valid and estimated; no run was recorded and no job started.",
  "synthetic": true
}
```

The same body with `"dry_run": false` returns `202`:

<!-- example: tools/submit-experiment-response -->
```json
{
  "run_id": "run_01KE6P4YM01RKH6SAGMJHTCQ0C",
  "configuration_id": "cfg_9eda1821d7a3f8b5c965f4375058e544a3aba022644d6a94d9e446a63100a5f0",
  "state": "awaiting_approval",
  "dry_run": false,
  "cost_estimate": {
    "estimated_usd_upper_bound": 0.0725,
    "price_retrieved_at": "2026-01-01T00:00:00Z",
    "remaining_allocation_usd": 6.9275,
    "budget_category": "cpu_research",
    "synthetic": true
  },
  "message": "A human approver must approve this run before it starts.",
  "synthetic": true
}
```

### Status of a running job

<!-- example: job-status -->
```json
{
  "run_id": "run_01KE6P4YM01RKH6SAGMJHTCQ0C",
  "state": "running",
  "purpose": "research",
  "dry_run": false,
  "compute_class": "cpu",
  "job_type": "run_backtest",
  "max_runtime_seconds": 900,
  "cost_estimate": {
    "estimated_usd_upper_bound": 0.0725,
    "price_retrieved_at": "2026-01-01T00:00:00Z",
    "remaining_allocation_usd": 6.9275,
    "budget_category": "cpu_research",
    "synthetic": true
  },
  "approval": {"approved_by": "finplan-beta-financemodel-approver-role", "approved_at": "2026-01-05T09:05:00Z", "approved_estimate_usd": 0.0725},
  "transitions": [
    {"state": "awaiting_approval", "at": "2026-01-05T09:00:00Z"},
    {"state": "queued", "at": "2026-01-05T09:05:00Z", "reason": "approved"},
    {"state": "starting", "at": "2026-01-05T09:06:00Z", "reason": "lease_acquired"},
    {"state": "running", "at": "2026-01-05T09:08:00Z", "reason": "sagemaker_in_progress"}
  ],
  "elapsed_seconds": 120,
  "configuration_id": "cfg_9eda1821d7a3f8b5c965f4375058e544a3aba022644d6a94d9e446a63100a5f0",
  "input_snapshot_id": "snap_01KDVDNAZ83BAMMYCEGWF33DPM",
  "domain": "finance",
  "submitted_at": "2026-01-05T09:00:00Z",
  "updated_at": "2026-01-05T09:08:00Z",
  "synthetic": true
}
```

### Result of a succeeded backtest

<!-- example: job-result -->
```json
{
  "run_id": "run_01KE6P4YM01RKH6SAGMJHTCQ0C",
  "completion_status": "succeeded",
  "solution_status": "optimal",
  "artifacts": [
    {"artifact_id": "run_artifact_94643d4f46a325d686ee612d9152d14853aaf0c9", "owner": "financemodel", "kind": "run_artifact", "checksum": "sha256:94643d4f46a325d686ee612d9152d14853aaf0c939d939714d4b0030ef770770", "content_type": "application/json", "size_bytes": 48211, "domain": "finance", "synthetic": true}
  ],
  "artifacts_complete": true,
  "configuration_id": "cfg_9eda1821d7a3f8b5c965f4375058e544a3aba022644d6a94d9e446a63100a5f0",
  "input_snapshot_id": "snap_01KDVDNAZ83BAMMYCEGWF33DPM",
  "evaluator_version": "1.0.0",
  "dataset_checksum": "sha256:6198fbe991c7dd2b8fbfbb06763b0286c9d12cc17ae4033f35314d195e7b02f2",
  "domain": "finance",
  "domain_schema_version": "1.0",
  "payload": {
    "performance": {"net_cumulative_return": 0.0123, "gross_cumulative_return": 0.0131, "annualized_volatility": 0.081, "sharpe_ratio": 0.62, "max_drawdown": 0.021, "turnover": 1.02, "total_transaction_costs": 18.4},
    "accuracy": {},
    "compute_cost": {"estimated_usd": 0.0725, "instance_seconds": 312},
    "synthetic": true
  },
  "completed_at": "2026-01-05T09:13:12Z",
  "synthetic": true
}
```

### Result of a crashed job

<!-- example: job-result -->
```json
{
  "run_id": "run_01KE6P4YM01RKH6SAGMJHTCQ0C",
  "completion_status": "failed",
  "error": {"code": "INTERNAL", "message": "job container exited abnormally", "retryable": false, "details": {"exit": "abnormal"}, "correlation_id": "corr-crash-test-0001", "contract_version": "1.0.0", "synthetic": true},
  "artifacts": [],
  "artifacts_complete": false,
  "configuration_id": "cfg_9eda1821d7a3f8b5c965f4375058e544a3aba022644d6a94d9e446a63100a5f0",
  "input_snapshot_id": "snap_01KDVDNAZ83BAMMYCEGWF33DPM",
  "domain": "finance",
  "domain_schema_version": "1.0",
  "completed_at": "2026-01-05T09:10:00Z",
  "synthetic": true
}
```

### Errors

Budget refusal (`403`):

<!-- example: error -->
```json
{
  "code": "BUDGET_EXCEEDED",
  "message": "estimated cost exceeds the remaining cpu_research allocation",
  "retryable": false,
  "details": {"budget_category": "cpu_research", "estimated_usd_upper_bound": 0.0725, "remaining_allocation_usd": 0.05},
  "correlation_id": "corr-http-test-0001",
  "contract_version": "1.0.0"
}
```

Result requested too early (`422`):

<!-- example: error -->
```json
{
  "code": "PRECONDITION_FAILED",
  "message": "the run has not finished; no result yet",
  "retryable": false,
  "details": {"reason": "run_not_terminal", "state": "queued"},
  "correlation_id": "corr-http-test-0001",
  "contract_version": "1.0.0"
}
```

Unsupported contract major (`400`):

<!-- example: error -->
```json
{
  "code": "UNSUPPORTED_CONTRACT_VERSION",
  "message": "the declared contract major is not served",
  "retryable": false,
  "details": {"served_contract_majors": [0]},
  "correlation_id": "corr-http-test-0001",
  "contract_version": "1.0.0"
}
```

### Cancel and list

`POST /v1/jobs/{run_id}/cancel` with body `{"idempotency_key": "cancel-0001"}` returns the job
status: `cancelled` immediately before the job starts, `stopping` while SageMaker stops a started
job (then `cancelled`), and the unchanged terminal state for a finished run.

`GET /v1/jobs?state=queued&page_size=20` returns `{"jobs": [<job-status>...], "next_token": "<opaque>"}`.
`next_token` is opaque and is only present when more runs exist.

## Beta advisory interfaces

`recommend_portfolio`: `GET /v1/recommendations` is a compatibility route for small requests.
MCP serving invokes the dedicated strategy Lambda directly with a longer bounded deadline.
`activate_advisory`: `PUT /v1/advisory-policy` pins a succeeded `prepare_policy` export after
explicit operator intent; production selection is unchanged. `performance_evidence`:
`GET /v1/performance-evidence` returns a bounded published-allocation comparison when an
account ledger is available. This is a hold-baseline summary, not the full CPU evidence workflow.
See `selected-strategy-serving.md` for request inputs, provenance and limitations.
