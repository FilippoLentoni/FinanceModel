# Spec Delta

## Purpose

Enable the hosted agent to request a frozen portfolio policy recommendation using a saved paper portfolio and approved completed market observations without requiring repeated holdings input.

## ADDED Requirements

### Requirement: Saved paper recommendation defaults
The service SHALL resolve an empty recommendation request to the same-environment research portfolio's saved paper state and newest approved research-universe snapshot. Explicit snapshot and decision date SHALL be provided together. Supplied holdings SHALL retain explicit experiment behavior and SHALL NOT be combined with a saved portfolio identifier.

#### Scenario: Natural recommendation request
- **WHEN** the existing recommendation operation receives an empty request in beta
- **THEN** it reads the configured research plan, saved portfolio state and newest approved snapshot and returns a recommendation with their identifiers and dates.

#### Scenario: Missing saved book
- **WHEN** the resolved portfolio has no initialized paper state
- **THEN** the service returns a typed precondition error and creates no state.

### Requirement: Completed close valuation
The service SHALL value saved quantities and cash using aligned unadjusted completed closes. It SHALL derive current weights and a high watermark covering every completed marked session since the saved book date, retain adjusted observations for actor features, and reject invalid positions, incomplete history, future dates or state newer than the decision date.

#### Scenario: Interim peak exceeds current value
- **WHEN** the saved book marked value rises above its stored peak and subsequently falls
- **THEN** the policy receives drawdown relative to that interim peak and current weights marked at the decision close.

#### Scenario: Adjusted and actual closes differ
- **WHEN** adjusted feature prices differ from raw closes
- **THEN** quantities and dollar values use actual closes while frozen actor features retain the training price basis.

### Requirement: Reproducible share recommendations
The response SHALL retain the frozen strategy, constraints and ensemble aggregation, include current, target and signed delta quantities for each instrument, and identify price dates, cash, portfolio value, state revision and policy provenance. Quantities SHALL be indicative fractional shares before execution costs, with no calibrated forecast invented.

#### Scenario: Target reduces a position
- **WHEN** the selected policy assigns a lower target value to an existing holding
- **THEN** the response contains a sell decision with the corresponding negative quantity delta and its reference close.

### Requirement: Read-only beta isolation
The recommendation operation SHALL NOT change paper holdings, high watermark, policy selection or research runs, and SHALL NOT train or execute trades. It SHALL use scoped same-environment read permissions. Deployment SHALL be confined to beta, preserve Gamma/prod manifests and stay within the authorized experimentation budget.

#### Scenario: Repeated request
- **WHEN** identical requests are made against unchanged saved state, policy and snapshot
- **THEN** the results match and the state revision and selected policy remain unchanged.
