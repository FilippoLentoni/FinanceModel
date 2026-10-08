# Infrastructure, IAM and pipeline

Task group 10 of `add-research-job-foundation`. Specs: job-deployment-pipeline (DEP-01 to DEP-05),
research-workspace (WS-01 to WS-05), run-output-staging (RST-04), model-registry (REG-03). Contracts:
D1 (ownership), D4 (SSM, budget), D5 (isolation), D6 (pipeline standard), D10, D13. One-time
bootstrap: [bootstrap.md](bootstrap.md). The patterns mirror the deployed FinancialPlanning
platform (`FinancialPlanning/docs/pipeline.md`).

## Stacks

| Stack | Module | Contents |
|---|---|---|
| `finplan-<env>-financemodel-storage` | `infra/stacks/storage.py` | Research storage `finplan-<env>-financemodel-research-workspace-<account-id>`, model registry `finplan-<env>-financemodel-model-registry-<account-id>`, job control table `finplan-<env>-financemodel-job-control` |
| `finplan-<env>-financemodel-control` | `infra/stacks/control.py` | Lambdas `job-api-handler`, `job-dispatcher`, `job-state-handler`, `job-registry-lookup`; their roles and 30-day log groups; the SageMaker state-change rule; the dispatcher schedule (EventBridge Scheduler, deployed `DISABLED` and armed only while runs are pending, design D1); the job API, the approver role and the job-execution role |
| `finplan-shared-financemodel-pipeline-store` | `infra/stacks/tooling.py` | Pipeline store (bootstrap only) |
| `finplan-shared-financemodel-tooling` | `infra/stacks/tooling.py`, `pipeline.py` | CPU image repository, the pipeline, its roles, its four CodeBuild projects and their explicit 30-day log groups (bootstrap only) |

Names come from `infra/stacks/naming.py` and IAM documents from `infra/stacks/policies.py`. Both are
shared by the stacks, the release step and the tests. Every role carries its environment's
permission boundary (`ModelStack`), except the job-execution role, which carries the research
boundary (ENV-04). No account ID, ARN or endpoint appears in a file. Bucket names and ARNs are built
from `${AWS::AccountId}` at deploy time.

The two environment stacks share no exports: names are deterministic, and the control stack only
depends on the storage stack for deploy order.

### Research storage (WS-01, WS-02)

- **Protection.** SSE-S3, Block Public Access (anonymous requests are denied), TLS only, bucket
  owner enforced, versioned.
- **Bucket policy.** It denies every principal of another environment (`finplan-<other>-*`). It
  limits object access to `finplan-<env>-financemodel-*` principals.
- **Lifecycle.** Only `scratch/` expires, after `retention.scratch_days` (14 in beta, 30 in gamma,
  90 in prod). That prefix holds unreferenced intermediate output, such as a cancelled run's
  temporary features.
  - Results and registered model versions reference only content-addressed `artifacts/` objects,
    by checksum. These never expire, and neither do `runs/` and `dataset-catalog/`.
  - Noncurrent versions and incomplete multipart uploads are cleaned up.
- **Where its name is published.** Only `config/research-storage-ref` (release step).

### Model registry storage (REG-01, REG-02)

The registry is an S3 bucket of write-once records. Every `PutObject` carries `If-None-Match: *`,
and nothing is ever deleted ([staging-and-registry.md](staging-and-registry.md)). Its bucket policy
(TLS only, other environments denied, `PutObject` without `If-None-Match` denied, deletes denied)
enforces this for every principal; the writers' identity policies repeat it.

Design D9 named a DynamoDB table, but the pinned matrix row `model-registry` lists
`AWS::S3::Bucket`, not `AWS::DynamoDB::Table`. S3 conditional writes give the same
create-if-absent semantics.

### Job control table

It implements `finplan_model.control.store.TABLE_SPEC`: `pk`/`sk`, the GSIs `by_state` and
`by_submitted`, TTL `ttl`, on-demand billing, point-in-time recovery and no stream. Its resource
policy denies data access to anyone but `finplan-<env>-financemodel-*` principals. It lists no
stream action, because DynamoDB table resource policies reject stream actions (a lesson of the
platform deploy). In prod it has deletion protection.

### Control plane

The control plane follows the wiring list in [job-execution.md](job-execution.md):

- **Lambda functions.** Python 3.12, `arm64`, one code bundle (`scripts/lambda_bundle.py`,
  `aarch64-manylinux_2_28` wheels from `uv.lock`, as in the platform; the bundle gate refuses a
  non-arm64 shared object). Environment variables:
  - `FINPLAN_ENVIRONMENT`;
  - `FINPLAN_RUNS_TABLE`;
  - `FINPLAN_DISPATCHER_FUNCTION` (API only);
  - `FINPLAN_DISPATCH_SCHEDULE` (API, dispatcher, state handler: the schedule they arm and disarm);
  - `FINPLAN_CONFIG_DIR=/var/task/config`.
- **Dispatcher schedule (design D1: only while runs are queued).** EventBridge Scheduler
  `finplan-<env>-financemodel-job-dispatcher` is deployed `DISABLED` with `rate(1 minute)`
  (`finplan_model.control.wakeup`):
  - a run `queued`, `starting`, `running` or `stopping` → `rate(1 minute)`, enabled (start queued
    runs, lease heartbeats, reconcile, stale-lease reclaim);
  - only runs `awaiting_approval` → one `at(...)` wake-up 60 s after the earliest approval deadline
    (it expires the approval);
  - nothing pending → back to the deployed state (`rate(1 minute)`, `DISABLED`), so an idle
    environment makes no invocations and does not drift from its template.

  `submit_job` and `approve_run` arm it (they only ever strengthen it) and kick the dispatcher with
  the run ID; the state handler re-arms the tick for a still-active run (a deploy that resets the
  schedule cannot strand a running job); each tick sets it to what is still pending, re-reading once
  before disarming so a racing submission re-arms it, and reading a kicked run ID consistently.
  The three roles may `scheduler:GetSchedule`/`UpdateSchedule` that one schedule and pass its role
  to `scheduler.amazonaws.com` only. The account's Lambda concurrency limit is 10, so the
  dispatcher has no reserved concurrency; correctness relies on the conditional writes instead.
- **State changes.** The rule matches `aws.sagemaker` / `SageMaker Processing Job State Change`
  with `ProcessingJobName` prefix `fm-<env>-`.
- **Job API.** A REST API defined by an OpenAPI body:
  - `AWS_IAM` (SigV4) on every route;
  - Lambda proxy integrations;
  - contract gateway responses (`UNAUTHORIZED` 401, `FORBIDDEN` 403, `RATE_LIMITED` 429);
  - the resource policy `job_api_resource_policy`. Its invokers are FinanceLambdasTool roles, the
    platform plan API handler (lineage lookups), platform operators (production candidates) and
    the FinanceModel stage role;
  - stage `api`, throttled to 5 requests per second (burst 10).

  The approve route admits only `finplan-<env>-financemodel-approver-role`. No `finplan-*` service
  role may assume that role.
- **Published endpoint.** The job API is published at `api/job-endpoint`. The registry reference
  `model/registry-ref` is `<job-endpoint>/v1/registry`, and the platform calls
  `GET /v1/registry/lineage/{run_id}?model_version=...` on it.

## IAM (WS-03, WS-05, RST-04, ENV-03 to ENV-05)

Policy simulation: `tests/unit/infra/test_iam_simulation.py` evaluates the role documents with the
contract evaluator and the boundary each role carries.

| Role | Allowed | Explicitly denied |
|---|---|---|
| `job-execution` (research boundary) | Its run spec (read), result, artifacts, dataset catalog and scratch in research storage. Run lineage in the registry (write-once). Approved snapshots (read; the platform bucket policy enforces `approved`). `PutObject` under `staging/run_*/` of its environment's staging area. The platform key through S3 only. `GET /v1/snapshots/*` on the platform API. Its own config parameters. Image pull. Processing-job logs and metrics | Raw, curated, plan and report platform storage. Snapshot writes. Staging reads, lists and deletes, and any staging put without `If-None-Match` (no overwrite of any run's key). Platform metadata tables. Plan, publication, execution and accept API writes. Registry overwrites and deletes. Other environments (boundary). Live-financial actions (boundary) |
| `job-api-handler` | Control table. `CreateProcessingJob`/`AddTags` on `processing-job/fm-<env>-*` only with request tags `environment=<env>` and `owner-repo=financemodel`. Stop and describe its jobs. `iam:PassRole` of the job role to `sagemaker.amazonaws.com` only. SSM reads of its environment and `shared`. Run hand-off documents. Registry writes (write-once). Platform `GET /v1/snapshots/*` and `GET /v1/staged-outputs/*`. Invoke the dispatcher | Platform metadata. Plan, publication and execution writes. Registry overwrites and deletes |
| `job-dispatcher`, `job-state-handler` | As the API handler, without the dispatcher invoke | As above |
| `job-api-handler`, `job-dispatcher`, `job-state-handler` (all three) | `scheduler:GetSchedule`/`UpdateSchedule` on `schedule/default/finplan-<env>-financemodel-job-dispatcher`; `iam:PassRole` of `finplan-<env>-financemodel-job-dispatcher-schedule-role` to `scheduler.amazonaws.com` | Other schedules, schedule create/delete, other environments |
| `job-registry-lookup` | Read the registry | Everything else (implicit) |
| `job-dispatcher-schedule` | Invoke the dispatcher | Everything else |
| `approver` | `GET /v1/jobs*`, `POST /v1/jobs/*/approve`, `POST /v1/jobs/*/cancel` | Everything else |
| Deploy, execution, stage and build roles | See [bootstrap.md](bootstrap.md) | Other environments (boundary). Platform SSM segments. Plan writes |

**Limitation (RST-04).** One job role serves every run, so IAM cannot scope a write to "this run's
prefix only". Instead, every staging put must carry `If-None-Match`, so an existing object (of any
run) can never be overwritten, and reads, lists and deletes are denied by IAM and by the platform's
bucket policy. The staging writer only ever writes under its own run's prefix
(`finplan_model.staging.S3StagingStore`). An injected extra file under another run's prefix
would make the platform reject that bundle (it refuses unlisted files). The run could therefore be
disrupted, but its bundle could never be silently changed.

## Pipeline (DEP-01 to DEP-03, DEP-05)

`finplan-shared-financemodel-pipeline` (CodePipeline V2, CodeBuild, `infra/stacks/pipeline.py`):

| Stage | Actions |
|---|---|
| Source | CodeConnections source on `FilippoLentoni/FinanceModel`, `main`. The reused connection is read from `/finplan/shared/financemodel/config/codeconnection-ref` (a CloudFormation SSM parameter type, never an ARN) |
| Build | `scripts/build_stage.py` (privileged x86 CodeBuild): pre gates; Lambda bundle; release-mode `cdk synth` once; post gates; `financemodel-cpu` image built once, base images pinned by digest, tagged with the `release_id`, pushed, digest read back; `release-info.json` with the `artifact_digest` over the assembly and the image digest; assets published; BuildOutput stored in the ledger |
| Beta | `DeployStorage`, `DeployControl` (scoped deploy and execution roles); `PublishRelease`; `IntegrationBetaTests` |
| Gamma | `PromotionCheck` (contract pin allowed in the environment — 1.0.0 is, a 0.x pin never leaves beta — and an image with a digest); deploys; `PublishRelease`; `GammaTests` (with the isolation test) |
| Approval | `ApproveProd` (Manual) |
| Prod | `PromotionCheck`; deploys; `PublishRelease` (approver and time from `ApproveProd`); `SmokeTests` (`list_jobs` and the contract envelopes, one dispatcher tick, digest equality with the gamma manifest from the ledger; nothing recorded, no SageMaker job) |

- **Experiments are never pipeline executions** (DEP-01). Runs are submitted at run time through
  the job API.
- **No model fitting in CodeBuild or Lambda** (DEP-02):
  - build-stage tests run under `tests/harness.py`, which blocks every SageMaker call;
  - the Lambdas only validate, record, estimate and start jobs;
  - strategy code runs only in the job image.
- **Images by digest** (DEP-03). `job/<job-type>` holds `<repo>@sha256:<digest>`, never a tag.
  Every environment publishes the same digest, and the manifests carry the same `artifact_digest`.
- **Artifact-only promotion.** Every action after Build reads only `BuildOutput`; no later stage
  synthesizes. Rollback starts the pipeline with `rollback_to_release_id=rel_...`: the Build stage
  re-emits the stored BuildOutput (digest re-verified) and the manifests record `rolled_back_from`.
  The rollback window for images is the newest 10 release images.
- **Stage suites** (`scripts/stage_runner.py tests`) run `tests/integration` (beta, gamma) or
  `tests/smoke` (prod) with `FINPLAN_TARGET_ENV` set. That switches off the offline harness so
  the stage role's real credentials are used (`tests/unit` and `tests/contract` re-install it
  unconditionally). A suite that executes zero tests fails the stage. The beta and gamma suites
  run, as the stage role, against the deployed environment (endpoints from SSM, SigV4):
  - real credentials, the release manifest and every published reference;
  - one synchronous dispatcher tick (the Lambda bundle imports; SSM, the table and the schedule
    answer);
  - through the job API (always deployed; a missing endpoint fails the suite): `list_jobs`, `VALIDATION_FAILED` and
    `NOT_FOUND` envelopes, and one tiny synthetic-labeled `run_backtest` on the approved snapshot
    named in `/finplan/<env>/financemodel/config/integration-snapshot-id` or, without that
    override, the platform-published one (real in phase 2, user decision 26) (dry run, submit, idempotent
    replay, `IDEMPOTENCY_KEY_REUSED`, status, cancel, result; `tests/integration/job_suite.py`,
    proven offline by `tests/unit/control/test_deployed_job_suite_double.py`). With the default
    auto-approve threshold 0 the run is cancelled in `awaiting_approval`, so no SageMaker job runs;
    a final synchronous tick returns the dispatcher schedule to what is still pending;
  - gamma: the isolation denials towards prod.

### Build gates (`scripts/build_gates.py`)

The pre gates are listed in [foundation.md](foundation.md).

| Post gate | Check |
|---|---|
| `ownership` | `finplan_contracts.ownership` per template: zero problems against the pinned matrix |
| `boundaries` | Boundary on every role (ENV-18), shared-resource rule (ENV-16) |
| `live-perm-scan-synth` | No live-financial permission (ENV-05) |
| `pipeline-structure` | `pipeline_check` (ENV-09) and `check_deploy_roles` (ENV-12) |
| `cost` | Cost tags on every taggable resource (CDK log-group helpers are attributed to their function). No always-on or provisioned compute, no API cache, no GPU type |
| `lambda-bundle` | Every Lambda code asset is a complete bundle built for `aarch64-manylinux_2_28` with no non-arm64 shared object; never a source-only package |

The `unit` gate runs pytest with `FINPLAN_RELEASE_BUILD=0`, so the suite's offline synths package
the source tree even inside the release build.

To run them locally:

```sh
uv run python scripts/build_gates.py --stage pre --skip-unit
uv run python scripts/synth.py --release --out cdk.out      # builds the bundle, release-mode synth
uv run python scripts/build_gates.py --stage post --assembly cdk.out
uvx cfn-lint --regions us-east-2 -- cdk.out/*.template.json cdk.out/assembly-*/*.template.json
```

`cfn-lint` reports 0 errors (warnings only).

## Release manifest and references (DEP-04, ENV-06, ENV-07)

`PublishRelease` (`scripts/release.py`) reads the stack outputs and seeds the registry with the
six baselines for this release's image digest (idempotent). It then writes, as the `pipeline`
writer bound to the environment (`finplan_contracts.ssm.check_write`):

| Parameter | Value |
|---|---|
| `config/research-storage-ref`, `config/registry-storage-ref` | Bucket names (read by the job, the control plane and the registry lookup) |
| `job/job-api-role-ref` | Job API handler role (the platform grants it `GET /v1/staged-outputs/*`) |
| `job/<job-type>` per deployed job type | `{"job_type", "image_uri": "<repo>@sha256:<digest>", "image_digest", "release_id", "deployed": true}` |
| `job/job-role-ref` | Job-execution role |
| `api/job-endpoint`, `model/registry-ref`, `config/approver-role-ref` | Job API, registry lineage route, approver role |
| `config/budget-enforced-role-names` | This environment's role names for the platform's budget action (control-plane roles, job-execution role, its deploy, execution and stage roles). The account-level pipeline and build roles are in `/finplan/shared/financemodel/config/budget-enforced-role-names`, written by the bootstrap (contracts D16) |
| `release/manifest`, `release/current-release-id` | Contract `release-manifest`. `outputs` names every parameter above, so it lists the job endpoint, the registry reference and the deployed job types. It also records the pinned contract version and digest. Prod adds `approved_by` and `approved_at` |

Every one of these outputs is required: a deploy without the job API or the job role fails
`PublishRelease`. The manifest is also copied to `releases/<release_id>/manifests/<env>.json` in the store.
FinanceModel IaC declares no `AWS::SSM::Parameter`: the FinanceModel matrix rows list none, and the
release step publishes the references instead.

## Contract rows (contracts 1.0.0)

Contracts 1.0.0 (FinancialPlanning design D16) added exactly the rows FinanceModel needed, so every
resource is synthesized and the ownership gate reports zero problems on every template. The former
switch `infra/stacks/contract_gaps.py` (and its context key) is gone.

| Pair (type, logical role) | Matrix row | Resource |
|---|---|---|
| `AWS::ApiGateway::Deployment`, `AWS::ApiGateway::Stage` / `job-api` | `job-interface` | The job REST API's deployment and stage `api` |
| `AWS::IAM::Role` / `job-execution-role` | `sagemaker-job-definitions` | `finplan-<env>-financemodel-job-execution-role` (research boundary) |
| `AWS::S3::BucketPolicy` / `model-registry` | `model-registry` | The registry bucket policy |
| `AWS::Logs::LogGroup` / `pipeline-build-project` | `pipeline-financemodel` | `/aws/codebuild/<project>` of the build project and the three stage projects: 30 days, deleted with the stack (`infra.stacks.tooling.add_log_group`, as in FinancialPlanning) |

Remaining contract gaps:

- **`plan_id` in submissions.** `core/v1/job-submission.json` has no `plan_id` or
  `parent_plan_version_id` (`additionalProperties: false`), but the staged-output manifest
  requires `plan_id`. Production candidates stage only when the run spec carries a `staging`
  target.
- **No SSM parameter type in FinanceModel rows.** Every reference is published by the release step
  (or the bootstrap), not by CloudFormation.

## Cost notes

There is no always-on compute and no prices in the repository. The idle costs are:

- the dispatcher: **no invocation while idle** (the schedule is disarmed). While runs are queued or
  active it ticks once a minute (one Lambda and one Scheduler invocation, a few DynamoDB queries);
  a run waiting for approval costs one wake-up at its approval deadline;
- CloudWatch Logs: every Lambda and CodeBuild log group keeps 30 days;
- S3 and ECR storage. ECR keeps at most 10 release images, and their layers are shared;
- DynamoDB point-in-time recovery on kilobyte-sized tables.

Every paid job goes through the control plane's pre-flight estimate and the `cpu_research`
category ([job-execution.md](job-execution.md)).
