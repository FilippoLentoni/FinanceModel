# Spec Delta

## Purpose

Defines FinanceModel's per-environment research storage and how research and job compute read approved platform input snapshots without ever changing authoritative plan state.

## ADDED Requirements

### Requirement: FinanceModel-owned research storage per environment
FinanceModel SHALL own one research storage area per environment (beta, gamma, prod), encrypted at rest, tagged with environment and owning repo, and blocked from public access. It holds prepared datasets, run artifacts, logs and reports. Its location MUST be published only as configuration under `/finplan/<env>/financemodel/config/*` and MUST NOT appear literally in any repository file.

#### Scenario: Separate storage per environment
- **WHEN** the FinanceModel pipeline deploys to beta, gamma and prod
- **THEN** three distinct research storage areas exist, each tagged with its environment, and the gamma job role is denied access to the prod area

#### Scenario: Public access attempt
- **WHEN** an anonymous principal requests any object in research storage
- **THEN** the request is denied

### Requirement: Research artifact retention
Research storage SHALL apply a lifecycle policy that keeps run artifacts referenced by a registered `model_version` or a staged production bundle, and expires unreferenced intermediate artifacts after a configurable retention period. Expiry MUST NOT remove any artifact a recorded result still references by checksum.

#### Scenario: Unreferenced scratch output expires
- **WHEN** a temporary feature file from a cancelled run exceeds the configured retention period
- **THEN** it is deleted by the lifecycle policy and no result record references it

#### Scenario: Referenced artifact retained
- **WHEN** an artifact is referenced by a registered `model_version`
- **THEN** it is not expired regardless of age

### Requirement: Read-only access to approved platform snapshots
FinanceModel jobs SHALL read platform input data only by `input_snapshot_id`, resolved through the platform API into trusted artifact references, and only for snapshots the platform marks approved. Job roles MUST have read-only access to the approved-snapshot prefix and no access to raw or curated platform storage.

#### Scenario: Approved snapshot resolved
- **WHEN** a job is submitted with an approved `input_snapshot_id`
- **THEN** the job resolves it through the platform API and reads the referenced artifacts without receiving any caller-supplied storage path

#### Scenario: Unapproved snapshot requested
- **WHEN** a submission names a snapshot that is not approved or does not exist
- **THEN** submission fails with `PRECONDITION_FAILED` or `NOT_FOUND` and no job starts

#### Scenario: Write to snapshot storage attempted
- **WHEN** job code attempts to write or delete an object under the snapshot prefix
- **THEN** IAM denies the request

### Requirement: Snapshot integrity verification
Before using a snapshot, a job SHALL recompute the SHA-256 checksums of the artifacts it reads and compare them with the snapshot's manifest checksum. A mismatch MUST fail the job before any strategy code runs.

#### Scenario: Checksum mismatch
- **WHEN** a downloaded snapshot artifact's checksum differs from the manifest
- **THEN** the job ends with `completion_status` `failed` and error code `PRECONDITION_FAILED`, and no result or staging output is written

### Requirement: No authoritative plan state in FinanceModel
FinanceModel SHALL NOT store authoritative portfolio, plan, plan-version, publication or execution records, and job roles MUST be denied writes to platform metadata and to the plan, publication and execution APIs. FinanceModel MAY keep non-authoritative copies keyed by immutable platform identifiers.

#### Scenario: Job attempts to create a plan version
- **WHEN** a job role calls the platform create-plan-version operation
- **THEN** the call is denied and the attempt is logged

#### Scenario: Cached snapshot metadata
- **WHEN** FinanceModel caches snapshot metadata for reuse
- **THEN** the cache is keyed by `input_snapshot_id` and checksum, and is never served as platform state to other repositories
