## ADDED Requirements

### Requirement: Durable bounded recursive improvement

The system SHALL persist immutable cycles and lineage-linked iterations, freeze portfolio identity and iteration bounds, use evidence and literature as untrusted hypothesis inputs, serialize launches, and require budgeted asynchronous evaluation before proposing a strategy update.

#### Scenario: Resume a bounded cycle

- **WHEN** a cycle is resumed after its experiment finishes
- **THEN** the next immutable iteration SHALL record the measured result, stop or select a server-approved candidate, and leave holdings and strategy activation unchanged

#### Scenario: Dry run and exhausted bounds

- **WHEN** a user requests a dry run or a budget, iteration or evidence bound is exhausted
- **THEN** no paid job SHALL start and the response SHALL record the specific stopping or approval condition

### Requirement: Daily bounded benchmark protocol

Every released research family and its controls SHALL decide daily. Generated Qwen/Jev previews SHALL freeze 22 aligned completed sessions for 21 decisions, retaining feature warmup and disclosing limited-pilot scope. Explicit window, alignment, warmup and 32-decision bounds SHALL be checked before inference and against available snapshots at submission. Cost and approval bounds SHALL remain unchanged.

#### Scenario: A daily pilot is evaluated

- **WHEN** an approved Qwen or Jev pilot runs
- **THEN** the strategy and controls SHALL share the exact daily window and costs, record the actual decision count and retained warmup, and exclude out-of-window forecast labels

#### Scenario: An incompatible or oversized preview is submitted

- **WHEN** a frozen monthly preview or a window above the decision cap is submitted
- **THEN** no inference SHALL occur and an incompatible frozen preview SHALL require a fresh review without silently changing its approved configuration

### Requirement: Identified self-hosted Qwen benchmark

The system SHALL use the discovered exact Qwen3.6-27B checkpoint in a run-scoped offline vLLM batch, verify staged manifests, log fixed-role messages, arbitrate typed target weights and evaluate through the shared simulator. GPU work SHALL require approval, a concurrency lease and a hard runtime; no always-on endpoint SHALL exist.

#### Scenario: Failed readiness or invalid allocation

- **WHEN** weights fail verification or model readiness fails
- **THEN** the batch SHALL fail before decisions and release run-scoped compute

#### Scenario: Invalid role response

- **WHEN** the allocator or arbiter produces invalid target weights after its bounded retry
- **THEN** the strategy SHALL hold existing positions and record the invalid response in its evidence

### Requirement: Identified Jev benchmark

The system SHALL use the verified TypeSafe System One API with mutually exclusive buy/hold/sell choice questions, bucketed point-in-time descriptors, constrained deterministic position sizing, token/call bounds, cached responses and recorded returned model. Historical leakage, classification calibration and net portfolio performance SHALL be reported separately.

#### Scenario: Provider faults and payload checks

- **WHEN** raw numerical series, an invalid probability vector or a permanent provider error is encountered
- **THEN** the benchmark SHALL refuse or fail safely without uncontrolled retries or leaking authorization values
