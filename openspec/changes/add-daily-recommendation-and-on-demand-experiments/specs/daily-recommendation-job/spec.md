# Spec Delta

## Purpose

Defines the `daily_recommendation` job kind. Submitted by the platform after daily ingestion, it runs only the user-selected production strategy on the approved research-universe snapshot and stages one recommendation within a fixed CPU cost cap.

## ADDED Requirements

### Requirement: Only the platform trigger submits daily recommendations
`submit_job` for `daily_recommendation` SHALL be accepted only from the same environment's platform trigger principal, with purpose `production_candidate` and an approved `equity-etf-daily` snapshot. Other callers MUST receive `FORBIDDEN`, and other datasets `VALIDATION_FAILED`.

#### Scenario: Tool role submits
- **WHEN** the FinanceLambdasTool `submitter` role submits `daily_recommendation`
- **THEN** the call fails with `FORBIDDEN` and no run exists

### Requirement: Runs only the configured production strategy
The job SHALL resolve `strategy_id`, `model_version` and `configuration_id` from the production-strategy key at submission, re-validate them against the registry, and record them on the run. A request naming a different strategy or configuration MUST fail with `VALIDATION_FAILED`.

#### Scenario: Override attempt
- **WHEN** the request names `mean_variance` while the key holds `min_variance`
- **THEN** the call fails with `VALIDATION_FAILED` naming `strategy_id`

### Requirement: No strategy means no run
If the key is absent, empty, unparseable or no longer passes registry validation, `submit_job` SHALL fail with `PRECONDITION_FAILED` (`no_production_strategy` or `strategy_not_eligible`). It MUST mint no `run_id` and start no SageMaker job.

#### Scenario: Key cleared
- **WHEN** the user cleared the strategy and the trigger still submits
- **THEN** the call fails with `no_production_strategy`, and the SageMaker job count is unchanged

### Requirement: Staged recommendation with bias disclosures
A succeeded run with a committable solution SHALL stage, through the existing run-output staging, target weights for every universe instrument including cash. The manifest MUST carry the snapshot's `bias_disclosures`.

#### Scenario: Successful run
- **WHEN** the run solves `optimal`
- **THEN** the staged manifest lists six weights summing to 1, the resolved strategy lineage and both disclosures

### Requirement: Daily job cost cap
The job SHALL run on one `ml.m5.xlarge` with a 1800 s maximum runtime and budget category `cpu_research`, under the existing pre-flight check. The build MUST fail if its upper-bound estimate exceeds USD 0.15 or the environment's auto-approve threshold.

#### Scenario: Runtime exceeded
- **WHEN** the job is still running at 1800 s
- **THEN** it is stopped, the run ends `timed_out`, and nothing is staged
