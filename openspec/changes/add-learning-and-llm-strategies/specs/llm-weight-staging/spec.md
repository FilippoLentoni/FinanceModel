# Spec Delta

## Purpose

Defines how FinanceModel obtains and verifies its own copy of the pinned Qwen3.6-27B checkpoint, so that every swarm run uses exactly the requested model without depending on resources owned by another project.

## ADDED Requirements

### Requirement: Pinned checkpoint identity
The swarm strategy SHALL use only Hugging Face `Qwen/Qwen3.6-27B` at the revision recorded in configuration (initially `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, Apache-2.0). Any other model ID or revision MUST be rejected, and substituting a different Qwen release MUST NOT be possible through configuration alone.

#### Scenario: Different revision configured
- **WHEN** a run configuration names a different revision or a different Qwen model ID
- **THEN** submission fails with `VALIDATION_FAILED`

### Requirement: FinanceModel-owned staged copy
Weights SHALL be staged into FinanceModel research storage by a FinanceModel CPU job using the FinanceModel job role. FinanceModel MUST NOT read from, write to, or change policies on any bucket or role owned by another project, including the previous Qwen deployment.

#### Scenario: Reference to external project storage
- **WHEN** a configuration or IaC template references storage or a role from another project
- **THEN** the ownership and leak checks fail the build

### Requirement: Idempotent, checksum-manifested staging
The staging job SHALL download every file of the pinned revision with offline-ready layout, write a SHA-256 checksum per file, write the manifest after all files, and write a staged-status record last. A rerun with a complete, matching manifest MUST perform no download.

#### Scenario: Interrupted staging
- **WHEN** staging stops after half the files
- **THEN** no staged-status record exists, inference jobs refuse the prefix, and a rerun completes only the missing or mismatched files

#### Scenario: Already staged
- **WHEN** staging runs again and the manifest and status record match the pinned revision
- **THEN** it exits `succeeded` without downloading

### Requirement: Verification before inference
Every inference job SHALL verify that the staged-status record names the pinned revision and that the per-file checksums match the manifest before loading weights. A mismatch MUST fail the job before GPU inference starts.

#### Scenario: Tampered file
- **WHEN** one weight file's checksum differs from the manifest
- **THEN** the job fails with `PRECONDITION_FAILED` and no tokens are generated

### Requirement: License retained
The staged copy SHALL include the checkpoint's license file, and the run record MUST record the license identifier.

#### Scenario: License recorded
- **WHEN** a swarm run starts
- **THEN** its record includes the model ID, revision and license identifier

### Requirement: Staging cost gated
Staging SHALL pass the pre-flight cost check and human approval like any paid job, and its estimate MUST include ongoing storage of the staged weights.

#### Scenario: Storage cost in estimate
- **WHEN** a staging run is estimated
- **THEN** the estimate shows job cost and monthly storage cost for the staged size separately
