# Spec Delta

## Purpose

Defines the operational controls around every FinanceModel job: time limits, concurrency leases, cancellation, failure handling, pre-flight cost estimation, budget enforcement and human approval of paid jobs.

## ADDED Requirements

### Requirement: Per-job time limits
Every job type SHALL have a configured default and maximum runtime, and every SageMaker job MUST be started with a stopping condition no greater than the run's approved runtime. A job that reaches its limit MUST end with `completion_status` `timed_out`.

#### Scenario: Runtime exceeded
- **WHEN** a backtest reaches its maximum runtime
- **THEN** SageMaker stops it, the run ends `timed_out`, and partial outputs are marked incomplete and never staged

#### Scenario: Requested runtime above ceiling
- **WHEN** a request asks for more than the job type's maximum runtime
- **THEN** it is rejected with `VALIDATION_FAILED`

### Requirement: Concurrency lease per environment
FinanceModel SHALL limit concurrent jobs per environment and per instance class with a lease that has a holder `run_id`, an expiry and a renewal heartbeat. A run MUST hold a lease before its SageMaker job starts, and runs without a lease MUST stay `queued`.

#### Scenario: Lease unavailable
- **WHEN** the CPU lease limit is 1 and a second job is submitted while one runs
- **THEN** the second run stays `queued` and starts after the first releases its lease

#### Scenario: Stale lease
- **WHEN** a lease holder stops renewing and the lease expires
- **THEN** the lease is reclaimed only after the holder's SageMaker job is confirmed terminal, and the reclaim is logged

### Requirement: Bounded queue
The queue per environment SHALL have a configured maximum depth. A submission that would exceed it MUST fail with `RATE_LIMITED`, `retryable` true, and create no run.

#### Scenario: Queue full
- **WHEN** the queue already holds the maximum number of queued runs
- **THEN** a new submission fails with `RATE_LIMITED` and no run is recorded

### Requirement: Account quota exhaustion is retryable
If SageMaker rejects a job start because an account-level instance quota is exhausted, the run SHALL return to `queued` with backoff instead of failing, up to a configured maximum wait, after which it ends `failed` with `DEPENDENCY_UNAVAILABLE`.

#### Scenario: Quota shared with another environment
- **WHEN** a beta job holds the only processing slot for an instance type and a gamma job tries to start
- **THEN** the gamma run is re-queued with backoff and its status shows the quota wait reason

### Requirement: Cancellation
`cancel_job` SHALL cancel a run in any non-terminal state. Before the SageMaker job starts, cancellation MUST be immediate. After it starts, the run MUST enter `stopping`, request the job stop, and end `cancelled` once SageMaker confirms it. Cancelling a terminal run MUST return its terminal state unchanged.

#### Scenario: Cancel running job
- **WHEN** a caller cancels a running backtest
- **THEN** the run goes to `stopping`, then `cancelled`, the lease is released, and no staging output is written

#### Scenario: Cancel finished job
- **WHEN** a caller cancels a run that already `succeeded`
- **THEN** the response reports `succeeded` and nothing changes

### Requirement: Failure semantics
A container that exits abnormally, a SageMaker internal error, or a failed integrity or reconciliation check SHALL end the run `failed` with a contract error envelope and a `correlation_id` that appears in the job logs. Failed runs MUST release their lease, MUST NOT be auto-retried more than the configured retry count, and MUST NOT write staging output.

#### Scenario: Container crash
- **WHEN** the job container exits with a non-zero code
- **THEN** the run is `failed`, `solution_status` is absent, and `error.code` is `INTERNAL`

#### Scenario: Transient start failure
- **WHEN** the job start fails with a throttling error and retries remain
- **THEN** the run is re-queued and the retry is recorded

### Requirement: Pre-flight cost estimate and budget check
Before any paid job starts, FinanceModel SHALL compute an upper-bound cost estimate (instance-hour price from configuration times maximum runtime times instance count, plus storage) and compare it with the remaining budget allocated to FinanceModel in that environment. A job whose estimate exceeds the remaining budget MUST NOT start and MUST fail with `BUDGET_EXCEEDED`.

#### Scenario: Budget exhausted
- **WHEN** the estimate is larger than the remaining FinanceModel budget
- **THEN** submission fails with `BUDGET_EXCEEDED`, `retryable` false, and the response shows the estimate and remaining budget

#### Scenario: Missing price configuration
- **WHEN** no price is configured for the requested instance type
- **THEN** the job does not start and fails with `PRECONDITION_FAILED`

### Requirement: Category budget allocation
Every paid job type SHALL declare one budget category, and the pre-flight check MUST compare the estimate with the remaining budget of that category. Category caps MUST come from the SSM allocation document, with defaults `cpu_research` USD 7 and `gpu` USD 25 within the USD 50 total AWS budget. While `budget-state` reports the AWS Budgets deny action active, every paid submission MUST fail with `BUDGET_EXCEEDED`.

#### Scenario: CPU category exhausted
- **WHEN** a CPU backtest estimate exceeds the remaining `cpu_research` budget while the `gpu` category still has funds
- **THEN** the submission fails with `BUDGET_EXCEEDED` naming the `cpu_research` category, and GPU funds are not used

#### Scenario: Project cap reached
- **WHEN** `budget-state` shows the 100% deny action active
- **THEN** every paid submission fails with `BUDGET_EXCEEDED`, `retryable` false, regardless of category balances

#### Scenario: Job type without category
- **WHEN** a paid job type has no declared budget category
- **THEN** submission fails with `PRECONDITION_FAILED` and no run starts

### Requirement: Human approval for paid jobs
A run whose estimate exceeds the environment's configured auto-approve threshold SHALL wait in `awaiting_approval` until a principal holding the approver role records an approval with identity and timestamp. Approval MUST NOT be possible from the submitting principal or any agent or tool role. GPU jobs MUST always require approval.

#### Scenario: Agent tries to approve
- **WHEN** the tool-wrapper role calls the approve operation
- **THEN** the call fails with `FORBIDDEN` and the run stays `awaiting_approval`

#### Scenario: Approval expires
- **WHEN** no approval arrives within the configured approval window
- **THEN** the run ends `cancelled` with reason `approval_expired`

### Requirement: Cost attribution tags
Every SageMaker job SHALL carry the contract package's cost-allocation tags, including environment, owning repo and `run_id`, and the run record MUST store estimated cost and, when available, the actual billed cost separately.

#### Scenario: Tagged job
- **WHEN** a job starts
- **THEN** its SageMaker resource has the environment, repo and `run_id` tags
