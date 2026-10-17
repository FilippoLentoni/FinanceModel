# Spec Delta

## Purpose

Serves frozen portfolio strategies on user request using the same state, constraints and model semantics as offline evaluation.

## ADDED Requirements

### Requirement: Reproducible frozen selection
The serving service SHALL run only an explicitly pinned strategy from a succeeded offline experiment, with its selected parameters, universe, constraints and source checksums. It MUST support existing controls/classical optimizers and exported PPO/SAC actors through the same request interface.

#### Scenario: Classical strategy replaces PPO
- **WHEN** a minimum-variance configuration from the source run is pinned
- **THEN** the existing recommendation operation runs minimum variance with that frozen configuration

### Requirement: Inference parity and point-in-time inputs
Policy inference SHALL preserve the training observation order, scaling, action transform, ensemble aggregation and constraint policy. It MUST use completed observations at or before the decision date and reject incomplete history, unsupported holdings, missing selection or corrupt artifacts.

#### Scenario: Action parity
- **WHEN** the exported actor and the training implementation receive the same observation
- **THEN** their deterministic actions match within a declared numerical tolerance

### Requirement: Bounded inference without training
An on-demand request SHALL produce target weights within the serving deadline without launching training or changing selection. Research policies SHALL be served as beta advisory/paper recommendations and MUST NOT bypass production promotion.

#### Scenario: Inference request
- **WHEN** the agent requests a recommendation from a pinned research PPO policy
- **THEN** the service returns advisory weights with provenance and creates no training job or production selection

### Requirement: Preserve evidence semantics
This serving extension MUST NOT replace full strategy replay or explanation evidence jobs with an allocation-hold accounting comparison. Recommendations MUST report a return forecast as unavailable when the selected strategy publishes none.

#### Scenario: Policy has no forecast
- **WHEN** a PPO allocation is returned without a forecast artifact
- **THEN** no expected-return claim is generated from its reward or critic
