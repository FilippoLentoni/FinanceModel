# Spec Delta

## Purpose

Defines how FinanceModel runs self-hosted Qwen3.6-27B inference on short-lived SageMaker GPU compute, with an explicit provision, readiness, run and teardown lifecycle, a GPU concurrency lease, private access and per-run cost records.

## ADDED Requirements

### Requirement: Self-hosted inference only
The swarm strategy SHALL generate tokens only with the staged Qwen3.6-27B weights on FinanceModel-run SageMaker compute. Calls to paid per-token LLM providers MUST NOT be possible from swarm jobs.

#### Scenario: External LLM call
- **WHEN** swarm code attempts to reach an external LLM API
- **THEN** the job's network and IAM configuration block it, and the job fails

### Requirement: Serving mode A, batch runner in a Training Job
FinanceModel SHALL support running vLLM in offline mode inside a SageMaker Training Job on `ml.g6.12xlarge` with the AWS vLLM DLC image (image reference from configuration), with the swarm loop running in the same job, no network model download, and tensor parallelism from configuration.

#### Scenario: Batch swarm run
- **WHEN** an approved swarm backtest runs in mode A
- **THEN** one Training Job loads the verified weights, runs every decision step of the backtest, writes logs and results, and terminates

### Requirement: Serving mode B, short-lived endpoint
FinanceModel SHALL support an endpoint mode that creates a SageMaker model, endpoint configuration and endpoint for one run, waits for readiness, drives the run from a CPU job, and deletes all three when the run ends. Endpoints MUST NOT be paused or reused across runs.

#### Scenario: Teardown after success
- **WHEN** a mode B run finishes
- **THEN** the endpoint, endpoint configuration and model are deleted and the deletion is confirmed in the run record

#### Scenario: Teardown after failure
- **WHEN** the driver job fails or is cancelled
- **THEN** teardown still runs and the run record shows the endpoint deleted

### Requirement: Readiness and startup measurement
Each run SHALL record the time from provisioning request to first successful health check and first generated token, in both modes. A run that is not ready within the configured startup timeout MUST tear down and end `failed` with `DEPENDENCY_UNAVAILABLE`.

#### Scenario: Startup timeout
- **WHEN** the endpoint is not healthy within the startup timeout
- **THEN** all created resources are deleted and the run ends `failed`

#### Scenario: Startup time recorded
- **WHEN** a run becomes ready
- **THEN** the provisioning, weight-load and first-token durations are stored in the run record and the cost report

### Requirement: Orphan detection
A scheduled check SHALL find any FinanceModel-tagged GPU endpoint or job whose run is terminal or has exceeded its maximum runtime, delete or stop it, and alert. No endpoint MUST outlive its run's maximum runtime plus a configured grace period.

#### Scenario: Orphaned endpoint
- **WHEN** an endpoint's run is `cancelled` but the endpoint still exists
- **THEN** the orphan check deletes it and records the event

### Requirement: GPU concurrency lease
GPU runs SHALL require a GPU lease class whose limit does not exceed the observed account quota (1 for `ml.g6.12xlarge` training and endpoint use). A run without the lease MUST stay `queued`.

#### Scenario: Second GPU run
- **WHEN** a GPU run holds the lease and another GPU run is approved
- **THEN** the second run stays `queued` until the first releases the lease

### Requirement: Private access
Mode B endpoints SHALL be invocable only by the run's driver role through IAM, with no public URL. Model containers MUST have network isolation or VPC egress restrictions that prevent outbound internet access during inference.

#### Scenario: Other principal invokes endpoint
- **WHEN** a principal other than the driver role calls the endpoint
- **THEN** the call is denied

### Requirement: GPU runs always need approval and a budget check
Every GPU run (staging excluded, which is CPU) SHALL pass the pre-flight cost estimate against the `gpu` budget category (default USD 25 of the USD 50 total AWS budget, from configuration) and wait in `awaiting_approval` for explicit user approval that shows the estimate, regardless of the auto-approve threshold. The estimate MUST use the configured maximum runtime, including startup.

#### Scenario: GPU run under threshold
- **WHEN** a GPU run's estimate is below the auto-approve threshold
- **THEN** it still waits for explicit approval

#### Scenario: GPU allocation insufficient
- **WHEN** a GPU run's estimate exceeds the remaining `gpu` category budget
- **THEN** submission fails with `BUDGET_EXCEEDED` naming the `gpu` category, and the `cpu_research` category is not used

### Requirement: Per-run cost record
Each GPU run SHALL record instance type, instance count, billed seconds split into startup, inference and teardown, input and output token counts, the estimated cost and, when available, the actual cost. Costs MUST be labeled as managed SageMaker charges.

#### Scenario: Cost record complete
- **WHEN** a GPU run ends in any terminal state
- **THEN** its cost record contains each listed field, with actual cost marked pending until billing data arrives
