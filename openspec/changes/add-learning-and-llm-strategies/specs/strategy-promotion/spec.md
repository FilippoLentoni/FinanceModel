# Spec Delta

## Purpose

Defines self-improvement as a controlled loop: candidates are proposed, evaluated within a budget against approved criteria, and promoted only with human approval, without any claim of guaranteed improvement.

## ADDED Requirements

### Requirement: Candidate registration
A candidate SHALL be a registered `model_version` plus `configuration_id` with a stated hypothesis and a proposer (human or agent). Registering a candidate MUST NOT run any job.

#### Scenario: Agent proposes a candidate
- **WHEN** an agent proposes a new swarm prompt template version
- **THEN** a candidate record is created with status `candidate` and no job starts

### Requirement: Versioned, approved promotion criteria
Promotion criteria SHALL be a versioned document approved by the user (metrics, comparison baseline, minimum margins, maximum drawdown and cost limits, minimum seeds, required prospective period length). Criteria MUST NOT be changed by agents, and a candidate MUST be evaluated against the criteria version current at its registration.

#### Scenario: Agent edits criteria
- **WHEN** an agent or tool role attempts to modify promotion criteria
- **THEN** the request fails with `FORBIDDEN`

### Requirement: Budgeted evaluation
Each evaluation cycle SHALL have a cost budget and a run-count limit. Evaluation runs MUST pass the execution controls, and a cycle MUST stop when its budget is exhausted, reporting the candidates left unevaluated.

#### Scenario: Budget exhausted mid-cycle
- **WHEN** the cycle's remaining budget is below the next run's estimate
- **THEN** no further runs start and the cycle report lists the unevaluated candidates

### Requirement: Promotion decision
A candidate SHALL be promoted only if it meets every criterion on validation and walk-forward results, then on one holdout evaluation, and, where the criteria require, on the prospective paper period. A human approver MUST record the promotion. Promotion changes only registry status and MUST NOT publish plans or change risk preferences.

#### Scenario: Candidate fails a criterion
- **WHEN** a candidate beats the baseline on return but exceeds the drawdown limit
- **THEN** it is not promoted and the report names the failed criterion

#### Scenario: Promotion approved
- **WHEN** a candidate meets every criterion and a human approves
- **THEN** its registry status becomes `promoted`, and no plan version, publication or risk preference changes

### Requirement: No guaranteed-improvement claims
Promotion reports SHALL state the evidence and its limitations (seed dispersion, holdout reuse, leakage risk, synthetic or real data) and MUST NOT claim that a promoted candidate will perform better in future.

#### Scenario: Report wording check
- **WHEN** a promotion report is generated
- **THEN** it contains the limitations section and no forward-looking performance guarantee
