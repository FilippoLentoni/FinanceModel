# Spec Delta

## Purpose

Defines what the FinanceModel CodePipeline builds, tests and promotes (job definitions, containers, the job API and the registry) and keeps experiments and model fitting out of the pipeline.

## ADDED Requirements

### Requirement: Pipeline deploys infrastructure, not experiments
The FinanceModel pipeline SHALL deploy job definitions, container images, the job interface, the lease, run and registry stores, research storage and IAM roles. Individual experiments MUST be submitted at runtime through the job interface and MUST NOT require a pipeline execution.

#### Scenario: New experiment configuration
- **WHEN** a researcher submits a backtest with a new configuration
- **THEN** a run starts through the job interface and no pipeline execution is triggered

### Requirement: No model fitting in CodeBuild or Lambda
CodeBuild projects and Lambda functions SHALL NOT fit, train or tune models or run backtests on non-fixture data. Build-stage tests MUST use tiny deterministic synthetic fixtures and MUST NOT call SageMaker.

#### Scenario: Build-stage test calls SageMaker
- **WHEN** a unit test attempts a SageMaker API call during the build stage
- **THEN** the call is blocked by the test harness and the build fails

#### Scenario: Job API Lambda
- **WHEN** the job-interface Lambda handles a submission
- **THEN** it only validates, records and starts a SageMaker job, and never runs strategy code

### Requirement: Standard stages and immutable images
The pipeline SHALL follow the shared stage order (source, build and test, beta, gamma, approval, prod) and promote the container images by digest built once in the build stage. Job definitions in every environment MUST reference images by digest, not by mutable tag.

#### Scenario: Digest equality
- **WHEN** release `rel_X` reaches prod
- **THEN** the job definitions in beta, gamma and prod reference the same image digests and the manifests record the same `artifact_digest`

### Requirement: Release manifest and published references
Each deployment SHALL publish the FinanceModel release manifest and the references `/finplan/<env>/financemodel/api/job-endpoint`, `/finplan/<env>/financemodel/model/registry-ref` and `/finplan/<env>/financemodel/job/*` for the deployed job types, recording the pinned `finplan-contracts` version.

#### Scenario: Manifest after beta deploy
- **WHEN** FinanceModel deploys to beta
- **THEN** `/finplan/beta/financemodel/release/manifest` lists the job endpoint, registry reference, deployed job types and pinned contract version

### Requirement: Environment test stages
The beta stage SHALL run an integration test that submits a fixture-backed baseline job through the job interface and checks status, result and cancellation. The gamma stage SHALL repeat it and run the isolation tests. The prod stage SHALL run a smoke test that only checks the interface and lease state and starts no paid job.

#### Scenario: Prod smoke
- **WHEN** the prod smoke test runs
- **THEN** it calls `list_jobs` and a validation-only submission, and no SageMaker job is started
