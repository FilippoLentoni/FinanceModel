# Spec Delta

## Purpose

Defines the FinanceModel model registry that mints `model_version` for every strategy implementation or trained artifact and lets the platform and tools verify the lineage of run outputs.

## ADDED Requirements

### Requirement: model_version minting
The registry SHALL mint `model_version` (`mv_` plus ULID) for each distinct strategy implementation, identified by strategy name, container image digest, parameter schema version and, for trained models, the artifact checksum. Registering an identical combination MUST return the existing `model_version`.

#### Scenario: Same implementation registered twice
- **WHEN** the same strategy, container digest and parameter schema are registered again
- **THEN** the existing `model_version` is returned and no new record is created

#### Scenario: New container digest
- **WHEN** the minimum-variance strategy is rebuilt with a new container digest
- **THEN** a new `model_version` is minted

### Requirement: Immutable registry records
Registry records SHALL be immutable except for a lifecycle status (`registered`, `candidate`, `promoted`, `retired`), and every status change MUST be an append-only event with actor and time.

#### Scenario: Edit attempt
- **WHEN** a caller tries to change the container digest of an existing `model_version`
- **THEN** it fails with `IMMUTABLE_RECORD`

### Requirement: Published registry reference
FinanceModel SHALL publish the per-environment registry reference at `/finplan/<env>/financemodel/model/registry-ref` and SHALL let the platform verify that a `model_version` and `run_id` pair exists and matches a staged bundle.

#### Scenario: Platform verifies lineage
- **WHEN** the platform validates a staged bundle naming `mv_X` and `run_Y`
- **THEN** the registry confirms `run_Y` used `mv_X` or returns `NOT_FOUND`

### Requirement: Results reference model_version
Every run result SHALL include the `model_version` of the strategy that produced it.

#### Scenario: Result lookup
- **WHEN** a benchmark result is retrieved
- **THEN** each strategy row includes its `model_version`
