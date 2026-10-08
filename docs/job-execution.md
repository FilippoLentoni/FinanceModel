# Job container, control plane and execution controls

Task groups 6, 7 and 8 of `add-research-job-foundation`. Specs: `experiment-job-interface`,
`job-execution-controls`, `research-workspace` (snapshot reads inside jobs) and
`job-deployment-pipeline` (the job API runs no strategy code). Consumers should read
[job-interface.md](job-interface.md). This page covers how it works, the deployment wiring the
infrastructure task group (10) needs, and operations.

## Components

| Piece | Module | Runs in |
|---|---|---|
| Job API: submit, status, result, cancel, list, approve | `finplan_model.control.api` / `.service` | Lambda `finplan_model.control.handlers.api_handler` behind API Gateway (IAM auth) |
| Dispatcher: approval expiry, reconcile, stale-lease reclaim, start queued runs | `JobService.dispatch` | Lambda `...handlers.dispatcher_handler` on an EventBridge Scheduler schedule armed only while runs are queued, active or awaiting an approval deadline (design D1, `finplan_model.control.wakeup`), also invoked asynchronously with the run ID after a run is queued or approved |
| State-change handler | `JobService.handle_sagemaker_event` | Lambda `...handlers.state_change_handler` on EventBridge `aws.sagemaker` / `SageMaker Processing Job State Change` |
| Run store: runs, append-only events, idempotency, leases | `finplan_model.control.store` (`DynamoRunStore`) | one DynamoDB table per environment |
| Job container `financemodel-cpu` | `finplan_model.jobs` (`python -m finplan_model.jobs <job_type>`) | SageMaker Processing Job (`ml.m5.xlarge`) |

The Lambdas only validate, record, estimate, take leases and call `CreateProcessingJob`,
`StopProcessingJob` and `DescribeProcessingJob`. Strategy code runs only in the container.

## Run hand-off

1. At start, the dispatcher writes the **run spec** to research storage at
   `runs/<run_id>/spec.json`. The spec holds the submission, the minted ids, the approved runtime,
   the cost estimate and the full simulation configuration (environment defaults plus the
   experiment's fees, constraints and rebalance frequency). It then starts the job. The container
   environment carries only identifiers: `FINPLAN_ENVIRONMENT`, `FINPLAN_RUN_ID`,
   `FINPLAN_JOB_TYPE`, `FINPLAN_CORRELATION_ID`, `FINPLAN_RUN_SPEC_SHA256` and
   `FINPLAN_IMAGE_DIGEST`. No storage path or secret is passed.
2. The container refuses a spec whose SHA-256 differs. It resolves the approved snapshot through
   the platform API by `input_snapshot_id` (endpoint from
   `/finplan/<env>/financialplanning/api/plan-endpoint`, SigV4 with the job role). It recomputes
   every artifact checksum against the manifest **before any strategy code runs** (WS-03, WS-04).
   It runs the job type through the common evaluator, stores artifacts as trusted references, and
   writes `runs/<run_id>/job-result.json` **last**.
3. On `Completed`, `Failed` or `Stopped`, the state-change handler (or the dispatcher's reconcile,
   if an event is lost) reads the job result. `solution_status` comes from that file, never from
   SageMaker. The handler then records the terminal state and releases the lease.

Exit codes: `0` succeeded (any `solution_status`, including `infeasible`); `1` failed, after the
container writes a failed result with the contract error envelope and the run's
`correlation_id`; `2` usage error. A crash without a result file gives `INTERNAL`
("job container exited abnormally"). Container failures are never retried automatically.

### Job types in the image

| Entry point | Behaviour |
|---|---|
| `prepare_dataset` | verified snapshot -> point-in-time market data -> dataset descriptor artifact (`research_dataset`). `solution_status` `not_applicable`. It delegates to `finplan_model.datasets.prepare_dataset_job(inputs)` when that hook exists. |
| `run_backtest` | one strategy (`finplan_model.strategies.build_strategy`, bound to the simulation constraint set) over the evaluation window |
| `run_benchmark` | the strategy plus the `cash`, `buy_and_hold` and `equal_weight` controls, sequentially in one job (a single processing slot). The comparison guard refuses mixed settings. It adds a report artifact when `finplan_model.reporting.benchmark_report(results, ctx=)` exists. |
| `report` | delegates to `finplan_model.reporting.report_job(inputs)`. Without that hook it fails with `DEPENDENCY_UNAVAILABLE`. |

Other task groups and later changes can replace or add a handler with
`finplan_model.jobs.handlers.register_handler(job_type, fn)`, where `fn(JobInputs) -> job-result`.

### Building and running the image

```
docker build -f container/Dockerfile -t financemodel-cpu .
docker run --rm -v "$PWD/.local-run:/work" financemodel-cpu run_backtest \
    --local /work --run-id <run_id> --fixture-snapshot /work/payload.json
```

- The build context is the repository root. `container/Dockerfile.dockerignore` is an allow-list:
  `pyproject.toml`, `uv.lock`, `README.md`, the vendored contract wheel, `src/finplan_model/` and
  `config/*.json`.
- Dependencies come from `uv.lock` only: `uv sync --frozen --no-dev`, with no dev or test packages.
- The image runs as a non-root user (uid 10001). `FINPLAN_CONFIG_DIR` points at the bundled
  config.
- The pipeline passes digest-pinned base images (`--build-arg PYTHON_IMAGE=...@sha256:<digest>`,
  `UV_IMAGE`), builds once and promotes by digest (DEP-03).
- Docker is not installed on the development host. The Dockerfile's `uv sync` steps were
  reproduced offline into a scratch virtual environment (`UV_OFFLINE=1`, `--no-editable`). Then
  `python -m finplan_model.jobs --help` ran from it, and `moto` and `aws_cdk` were absent as
  expected. The unit suite runs the entry point itself in a subprocess on fixtures
  (`tests/unit/jobs/test_job_container.py`). The real `docker build` test runs wherever docker
  exists.

## Lifecycle, leases and the queue

- States and allowed transitions are in `control/states.py` (the state enum comes from the pinned
  contract). Every change is a conditional write on the run's `revision` plus an append-only event
  item (`EVT#<seq>`, created only if absent), in one DynamoDB transaction. Late or duplicate
  SageMaker events after a terminal state are logged (`late_event_ignored`) and change nothing.
- **Lease** (CTL-02): slots `LEASE#<env>#<class>` / `SLOT#<n>` for `n < max_holders`, where
  `max_holders` comes from `/finplan/<env>/financemodel/config/lease-limits` (default `{"cpu": 1,
  "gpu": 0}`). Each slot has a TTL of 900 s; heartbeats come from the dispatcher (every tick while
  the job is `InProgress`) and from state-change events. An expired slot is reclaimed only after
  `DescribeProcessingJob` confirms the holder's job is terminal or absent. If SageMaker cannot be
  reached, the lease is kept (`lease_stale_unconfirmed`). If the job is alive, the lease is renewed
  (`lease_stale_holder_alive`). Every reclaim is logged (`lease_reclaimed`).
- **Queue** (CTL-03): at most `queue.max_depth` (10) runs wait (`awaiting_approval` plus
  `queued`). Beyond that, `RATE_LIMITED`, retryable. The queue is FIFO by submission time.
- **Quota** (CTL-04): `ResourceLimitExceeded` at start (the account-level processing quota is
  shared by beta, gamma and prod) releases the lease and re-queues the run with `wait_reason`
  `account_quota_exhausted` and exponential backoff (60 s doubling, capped at 900 s). After
  `quota_max_wait_seconds` (3600) the run ends `failed` with `DEPENDENCY_UNAVAILABLE`.
- **Start retries** (CTL-06): throttling or 5xx at start re-queues the run (`start_retry`
  recorded) up to `start_retry_limit` (2). After that it is `failed` with `DEPENDENCY_UNAVAILABLE`.
- **Time limits** (CTL-01): `StoppingCondition.MaxRuntimeInSeconds` equals the approved runtime,
  which is never above the job type's ceiling. A stop by the stopping condition ends `timed_out`
  with `artifacts_complete` false; partial outputs are never staged.
- **Cancellation** (CTL-05): `awaiting_approval` and `queued` runs (and `starting` runs without a
  job) end `cancelled` immediately. Started runs go to `stopping`; the handler calls
  `StopProcessingJob` and ends the run `cancelled` once SageMaker reports `Stopped`. If a cancel
  races with job creation, the dispatcher stops the job it just created.
- **Images**: a job definition without a digest-pinned image never starts
  (`PRECONDITION_FAILED` `image_not_digest_pinned`).

## Cost guardrails

- The estimate (CTL-07) is `usd_per_hour[instance_type] × max_runtime_seconds / 3600 ×
  instance_count + cost.storage_estimate_usd`, rounded up to 1e-6 USD. Prices come from
  `/finplan/<env>/financemodel/config/instance-prices`:
  `{"retrieved_at": "<RFC 3339>", "currency": "USD", "usd_per_hour": {"ml.m5.xlarge": <price>}}`.
  The operator writes it from current AWS pricing; no price is in this repository. A missing price,
  a missing date, or a date older than `cost.price_max_age_days` (30) gives `PRECONDITION_FAILED`.
- Budget check (CTL-10): `finplan_contracts.budget.preflight` against the job type's
  `budget_category`, the shared allocation `/finplan/shared/financialplanning/config/budget-allocation`
  (contract defaults when absent: `cpu_research` 7, `gpu` 25) and this environment's runs in that
  category. A run counts its actual cost when billed, otherwise its estimate. Runs cancelled or
  failed before their job started count 0. `budget-state` `enforced` refuses every paid submission,
  every approval, and every start of an already-queued run with `BUDGET_EXCEEDED`.
- Approval (CTL-08): an estimate above `auto-approve-usd` (default 0), and every GPU run, waits in
  `awaiting_approval`. Only the approver role `finplan-<env>-financemodel-approver-role`, named in
  `/finplan/<env>/financemodel/config/approver-role-ref`, may approve. It may never approve its
  own submission. Tool-wrapper, agent, pipeline and FinanceModel service roles are denied twice:
  by the API resource policy (`control/policies.py`, checked with the contract IAM evaluator in
  tests) and by name in the handler. Unapproved runs end `cancelled` (`approval_expired`) after
  `approval.approval_window_hours` (72).
- Tags (CTL-09): every processing job carries `project`, `owner-repo`, `environment`,
  `logical-role` (`research-job`) and `run-id`. The run record stores `estimated_cost_usd` and
  `actual_cost_usd` (null until billing data is reconciled).

### Approving a run (FM-OQ-6 interim: CLI)

```
AWS_PROFILE=<approver-profile> uv run python scripts/approve_run.py --env beta <run_id>
```

The CLI resolves the job endpoint from SSM and shows the state, job type, purpose, estimate,
category, remaining allocation and price date. It asks for confirmation, then signs
`POST /v1/jobs/<run_id>/approve` with SigV4 and the displayed estimate as `approved_estimate_usd`.
`--show` only displays. Running it against beta needs the deployed API (task 10.6).

## Wiring needed from the infrastructure task group (10.2, 10.3, 10.4)

- **Table**: `control.store.TABLE_SPEC`. Keys `pk`/`sk` (strings); GSIs `by_state`
  (`gsi1pk`/`gsi1sk`) and `by_submitted` (`gsi2pk`/`gsi2sk`), projection ALL; TTL attribute
  `ttl`; on-demand billing, point-in-time recovery, no stream. Logical role `runs-table`.
- **Lambdas** (Python 3.12, the `src/finplan_model` package plus `config/`): environment
  `FINPLAN_ENVIRONMENT`, `FINPLAN_RUNS_TABLE`, optional `FINPLAN_DISPATCHER_FUNCTION`,
  `FINPLAN_DISPATCH_SCHEDULE` and `FINPLAN_CONFIG_DIR`. The handlers are listed above.
- **EventBridge**: the dispatcher schedule (deployed `DISABLED`, `rate(1 minute)`) targets the
  dispatcher; the control plane arms it with `rate(1 minute)` while runs are queued or active, with
  one `at(...)` wake-up at the earliest approval deadline while runs only await approval, and
  returns it to the deployed state when nothing is pending. The rule `{"source":
  ["aws.sagemaker"], "detail-type": ["SageMaker Processing Job State Change"], "detail":
  {"ProcessingJobName": [{"prefix": "fm-<env>-"}]}}` targets the state-change handler. Processing
  job names are `fm-<env>-run-<ulid lowercase>-a<attempt>`.
- **Job API role** (`job-api`), and the same for the dispatcher and the state handler:
  - DynamoDB `GetItem`, `PutItem`, `UpdateItem`, `DeleteItem`, `Query` and `TransactWriteItems`
    on the table and its indexes;
  - `sagemaker:CreateProcessingJob`, `StopProcessingJob`, `DescribeProcessingJob` and
    `AddTags` on `processing-job/fm-<env>-*`, with the request tag `environment=<env>`;
  - `iam:PassRole` on the job role, only to `sagemaker.amazonaws.com`;
  - `ssm:GetParameter` on `/finplan/<env>/*` and `/finplan/shared/*`;
  - `s3:PutObject` and `GetObject` on `runs/*` in the research bucket;
  - `execute-api:Invoke` on the platform's `GET /v1/snapshots/*`, for the submission-time snapshot
    check;
  - `lambda:InvokeFunction` on the dispatcher, for the API role only;
  - `scheduler:GetSchedule` and `UpdateSchedule` on the dispatcher schedule, and `iam:PassRole` of
    the schedule role to `scheduler.amazonaws.com` only (arming, all three roles).

  Publish the API and dispatcher role names in `config/budget-enforced-role-names`.
- **Job role** (`job`, research boundary): `s3:GetObject` on `runs/*/spec.json`; `s3:PutObject`
  on `runs/*/job-result.json` and on the artifact and dataset prefixes; read of the
  research-storage and plan-endpoint parameters; `execute-api:Invoke` on the platform snapshot
  routes. Nothing else.
- **API**: REST API with `AWS_IAM` auth on the routes in `control.api.ROUTES`, the resource policy
  `control.policies.job_api_resource_policy(env, invoker_role_patterns=[...])`, and the approver
  role's identity policy `approver_identity_policy(<api arn>/*)`. Use the gateway responses from
  `control.policies.gateway_responses(contract_version)` so that refusals by API Gateway itself
  are contract envelopes too: `UNAUTHORIZED` 401 for a missing or invalid signature, `FORBIDDEN`
  403 for a resource-policy deny.
- **SSM written by the deploy**: `api/job-endpoint`; `job/<job-type-kebab>` with
  `{"job_type", "image_uri": "<repo>@sha256:<digest>", "deployed": true}` per deployed job type
  (`control.settings.job_definition_parameter`); `job/job-role-ref`; `config/approver-role-ref`.
  The operator writes `config/instance-prices`, `config/auto-approve-usd`, `config/lease-limits`
  and `config/production-candidate-principals` (a JSON list of role names).

## Not done here (owned elsewhere or blocked)

- The CDK stacks and the IAM policy simulation of the roles are in task group 10
  ([pipeline.md](pipeline.md), `tests/unit/infra/test_iam_simulation.py`). The resource-policy
  simulation for approve is in `tests/unit/control/test_execution_controls.py`.
- Actual billed cost reconciliation (Cost Explorer by `run-id` tag). `actual_cost_usd` stays null,
  and budget sums use the upper-bound estimate (design D4, "over-count rather than under-count").
- The `model_version` resolver (registry, task 9.2) plugs into `ServiceDeps.model_version_resolver`
  and the run spec. Staging links (task 9.4) plug into `ServiceDeps.result_hooks`.
- Integration in beta and gamma (tasks 11.x) is blocked on the deployed stacks and the platform
  beta release.
