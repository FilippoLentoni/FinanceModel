# Spec Delta

## Purpose

Defines how FinanceModel production-candidate runs hand their outputs to the platform's run-output staging area so that only the platform validates them and commits plan versions.

## ADDED Requirements

### Requirement: Staging only for production-candidate runs
Only runs with purpose `production_candidate` that end `succeeded` with `solution_status` `optimal`, `feasible` or `no_effect` SHALL write a staged output bundle. Research, tuning and holdout runs MUST write only to research storage.

#### Scenario: Research run finishes
- **WHEN** a `research` run succeeds
- **THEN** nothing is written to the platform staging area

#### Scenario: Infeasible production candidate
- **WHEN** a `production_candidate` run ends with `solution_status` `infeasible`
- **THEN** no bundle is staged and the result says so

### Requirement: Staged bundle content
A staged bundle SHALL validate against the staged-output manifest schema (`core/v1/staged-output-manifest.json` plus its finance payload) in the pinned contract package. Its manifest MUST carry `run_id`, `model_version`, `configuration_id`, `input_snapshot_id`, `plan_id`, optional `parent_plan_version_id`, `completion_status`, `solution_status`, `evaluator_version`, `contract_version` and the SHA-256 of every bundle file.

#### Scenario: Bundle fails schema validation
- **WHEN** the produced output does not validate against the contract schema
- **THEN** the run ends `failed` with `VALIDATION_FAILED` and nothing is staged

### Requirement: Manifest written last as the completion marker
A bundle SHALL be written under `<run-staging-ref>/<run_id>/`, with all data files first and the manifest written last. The manifest is the only completion marker; no separate marker file is written. Consumers MUST treat a bundle without a manifest as incomplete.

#### Scenario: Job dies mid-write
- **WHEN** a job stops after writing data files but before the manifest
- **THEN** the bundle has no manifest, the run ends `failed`, and a platform acceptance request for the run fails with `PRECONDITION_FAILED` `staged_output_incomplete`

### Requirement: Write-only staging access
The job role SHALL have write-only access to the staging prefix granted by the platform, published at `/finplan/<env>/financialplanning/config/run-staging-ref`. It MUST NOT read, overwrite or delete other runs' bundles, and MUST NOT write plan versions.

#### Scenario: Overwrite attempt
- **WHEN** a job attempts to write under another run's staging key
- **THEN** the write is denied

### Requirement: Platform validation decides commitment
FinanceModel SHALL report a staged run as `staged`, not as a plan version, and MUST NOT claim a `plan_version_id` until the platform returns one. FinanceModel MUST NOT trigger acceptance itself: acceptance is the explicit platform operation invoked by a platform-side caller. The run result MUST link to the platform's acceptance outcome, read from the platform staged-output outcome operation, when available.

#### Scenario: Platform rejects bundle
- **WHEN** the platform marks a staged bundle `invalid` after accounting reconciliation
- **THEN** the run result shows `staged` with platform validation `invalid` and no `plan_version_id`
