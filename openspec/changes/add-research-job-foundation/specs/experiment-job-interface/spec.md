# Spec Delta

## Purpose

Defines the asynchronous job interface that FinanceModel publishes and FinanceLambdasTool (and the platform, for production-candidate runs) consumes to submit experiments, track their status, fetch results and cancel them.

## ADDED Requirements

### Requirement: Published job interface reference
FinanceModel SHALL expose the operations `submit_job`, `get_job_status`, `get_job_result`, `cancel_job` and `list_jobs` per environment, authenticated with IAM, and publish the reference at `/finplan/<env>/financemodel/api/job-endpoint`. Requests and responses MUST validate against the job schemas in the pinned `finplan-contracts` version.

#### Scenario: Consumer resolves the endpoint
- **WHEN** FinanceLambdasTool deploys to gamma
- **THEN** it reads the job endpoint from `/finplan/gamma/financemodel/api/job-endpoint` and never from a literal

#### Scenario: Unauthorized caller
- **WHEN** a principal without the granted invoke permission calls `submit_job`
- **THEN** the call fails with `UNAUTHORIZED` or `FORBIDDEN`

### Requirement: Asynchronous submission mints run_id
`submit_job` SHALL validate the request, compute the `configuration_id`, mint a `run_id`, record the run and return immediately with `run_id`, `configuration_id` and the initial status. It MUST NOT wait for job completion.

#### Scenario: Valid submission
- **WHEN** a valid backtest request is submitted
- **THEN** the response contains a `run_` prefixed ULID `run_id`, the computed `configuration_id` and status `queued` or `awaiting_approval`, and returns within the API timeout

#### Scenario: Caller supplies run_id
- **WHEN** a request includes its own `run_id`
- **THEN** it is rejected with `VALIDATION_FAILED`

### Requirement: Submission validation
`submit_job` SHALL reject requests with an unsupported contract major, an unregistered domain or `domain_schema_version`, an unknown job type or strategy, a caller-supplied storage path, a runtime above the job type's ceiling, or a purpose not permitted for the caller. Rejections MUST create no run and start no job.

#### Scenario: Storage path supplied
- **WHEN** a request contains an `s3://` URI as input or output location
- **THEN** it fails with `VALIDATION_FAILED` and no run is recorded

#### Scenario: Unsupported contract major
- **WHEN** the request declares a contract major the interface does not serve
- **THEN** it fails with `UNSUPPORTED_CONTRACT_VERSION` listing served majors

### Requirement: Idempotent submission
`submit_job` and `cancel_job` SHALL require an `idempotency_key` scoped to caller principal, environment and operation and retained at least 7 days. A repeat with the same key and request hash MUST return the original response. The same key with a different body MUST fail with `IDEMPOTENCY_KEY_REUSED`.

#### Scenario: Retry after timeout
- **WHEN** a caller retries `submit_job` with the same key and body after a network timeout
- **THEN** the original `run_id` is returned and only one SageMaker job ever starts

#### Scenario: Key reused with a different body
- **WHEN** the same key is sent with a different strategy
- **THEN** the call fails with `IDEMPOTENCY_KEY_REUSED`, `retryable` false

### Requirement: Job lifecycle states
A run SHALL move through non-terminal states `awaiting_approval`, `queued`, `starting`, `running` and `stopping`, and end in exactly one terminal `completion_status` (`succeeded`, `failed`, `cancelled`, `timed_out`). Each transition MUST be recorded as an append-only event with a timestamp. Terminal runs MUST NOT change state.

#### Scenario: Status query
- **WHEN** a caller queries a running job
- **THEN** the response contains `run_id`, current state, timestamps of each transition and elapsed runtime, and no storage locations

#### Scenario: Late event after terminal state
- **WHEN** a SageMaker state-change event arrives for a run already `cancelled`
- **THEN** the run state is not changed and the event is logged

### Requirement: Result retrieval
`get_job_result` SHALL return, for terminal runs, `completion_status`, `solution_status` where applicable, a compact metrics summary, and trusted artifact references with checksums. For non-terminal runs it MUST return `PRECONDITION_FAILED` with the current state.

#### Scenario: Succeeded run
- **WHEN** a succeeded backtest result is requested
- **THEN** the response includes summary metrics, evaluator version, dataset checksum, `model_version`, and artifact references, never bucket names or keys

#### Scenario: Failed run
- **WHEN** a failed run's result is requested
- **THEN** `completion_status` is `failed`, `solution_status` is absent and `error` is a contract error envelope

### Requirement: Run purpose
Each submission SHALL declare a purpose: `research`, `tuning`, `holdout_evaluation` or `production_candidate`. Only `production_candidate` runs MAY write to the platform staging area, and only principals granted that purpose MAY submit them.

#### Scenario: Tool wrapper submits production candidate without grant
- **WHEN** a caller lacking the production-candidate grant submits with that purpose
- **THEN** it fails with `FORBIDDEN`

### Requirement: Dry-run submission
`submit_job` SHALL accept `dry_run: true`, which performs all validation and the pre-flight cost estimate and returns the computed `configuration_id` and estimate, but MUST NOT mint a `run_id`, record a run, take a lease or start a job.

#### Scenario: Dry run
- **WHEN** a caller submits a valid request with `dry_run: true`
- **THEN** the response contains the `configuration_id` and cost estimate, and no run or SageMaker job exists afterwards

### Requirement: Fixture-backed availability
In phase 1, the interface SHALL accept jobs that run on synthetic-fixture datasets in every environment. Job types not deployed in an environment MUST return `DEPENDENCY_UNAVAILABLE`.

#### Scenario: Unavailable job type
- **WHEN** a caller submits an RL training job before that job type is deployed
- **THEN** the call fails with `DEPENDENCY_UNAVAILABLE`, `retryable` false
