# Spec Delta

## Purpose

Provide reproducible classical portfolio advice for auditable beta portfolio decisions, explanations and controlled research.

## ADDED Requirements

### Requirement: Independent optimizer recommendation
The system SHALL support minimum variance, mean variance and scenario CVaR using approved completed market data and saved or explicitly supplied paper holdings. It MUST preserve the selected PPO policy.

#### Scenario: Classical recommendation
- **WHEN** a user requests a traditional recommendation
- **THEN** the result contains all stock and cash targets, indicative share changes, source data and solver settings.

### Requirement: Immutable issued advisory plans
Each issued advisory plan SHALL store exact solve inputs, output and provenance under an immutable checksum-bound analysis ID. Retrieval and listing MUST support later explanation without treating recommendations as executed trades.

#### Scenario: Repeated recommendation
- **WHEN** the same request is retried
- **THEN** the same input identity is returned or a verified equivalent audit record is created without changing holdings.

### Requirement: Cash and constraints are explicit
Recommendations SHALL state cash policy, limits, turnover/cost assumptions and solver feasibility. A minimum-risk result MUST NOT silently substitute full investment or imply a return forecast absent from its record.

#### Scenario: Infeasible policy
- **WHEN** the constraints admit no feasible portfolio
- **THEN** the result reports infeasibility and no trade is represented as executable.
