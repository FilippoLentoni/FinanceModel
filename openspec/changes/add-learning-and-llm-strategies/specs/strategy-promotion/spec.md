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
Promotion criteria SHALL be a versioned document approved by the user. Criteria MUST NOT be changed by agents, and a candidate MUST be evaluated against the criteria version current at its registration. Criteria v1 (approved by the user 2026-10-07) SHALL contain exactly the comparison rules of the requirement "Deterministic promotion check (criteria v1)" plus the recorded user approval.

#### Scenario: Agent edits criteria
- **WHEN** an agent or tool role attempts to modify promotion criteria
- **THEN** the request fails with `FORBIDDEN`

#### Scenario: Criteria v1 in force
- **WHEN** a candidate is registered while criteria v1 is current
- **THEN** its candidate record references criteria version 1, and later criteria versions do not apply to it

### Requirement: Budgeted evaluation
Each evaluation cycle SHALL have a cost budget and a run-count limit. Evaluation runs MUST pass the execution controls, and a cycle MUST stop when its budget is exhausted, reporting the candidates left unevaluated.

#### Scenario: Budget exhausted mid-cycle
- **WHEN** the cycle's remaining budget is below the next run's estimate
- **THEN** no further runs start and the cycle report lists the unevaluated candidates

### Requirement: Deterministic promotion check (criteria v1)
The promotion check SHALL be a deterministic function of stored common-evaluator results on the test period: the dataset's untouched holdout range, evaluated once for the frozen candidate with purpose `holdout_evaluation` and for the incumbent on the identical range. It MUST pass only if the candidate's net-of-costs cumulative return is strictly greater than the incumbent's AND the candidate's maximum drawdown is not worse (not larger in magnitude) than the incumbent's.

#### Scenario: Candidate passes both rules
- **WHEN** on the holdout the candidate's net-of-costs return is 6.1% versus the incumbent's 5.4%, and its maximum drawdown is -8.0% versus the incumbent's -9.5%
- **THEN** the check result is `pass` with both rules recorded as passed

#### Scenario: Higher return but worse drawdown
- **WHEN** the candidate's net-of-costs return beats the incumbent's but its maximum drawdown is -12.0% versus the incumbent's -9.5%
- **THEN** the check result is `fail`, the report names the drawdown rule, and the candidate is not promoted

#### Scenario: Equal return
- **WHEN** the candidate's net-of-costs return equals the incumbent's and its drawdown is better
- **THEN** the check result is `fail`, because the return rule requires strictly beating the incumbent

#### Scenario: Gross return beats, net return does not
- **WHEN** the candidate beats the incumbent before transaction costs but not after them
- **THEN** the check result is `fail` on the return rule

#### Scenario: Re-running the check
- **WHEN** the check is re-run on the same stored run results and criteria version
- **THEN** it produces an identical result record and checksum

### Requirement: Comparable evaluation inputs
The check SHALL compare the candidate with the incumbent only when both results come from the same dataset, the same holdout range, the same simulation configuration and cost model, and the same evaluator version. The incumbent SHALL be the currently `promoted` model version for the same instrument and universe, or the `buy_and_hold` control when none is promoted.

#### Scenario: Different evaluator version
- **WHEN** the incumbent's holdout result was computed with an older evaluator version than the candidate's
- **THEN** the check returns `not_comparable`, the incumbent is re-evaluated on the same range before any decision, and no promotion occurs meanwhile

#### Scenario: No promoted incumbent
- **WHEN** no model version is `promoted` for the instrument
- **THEN** the `buy_and_hold` control on the same dataset and holdout range is the incumbent

#### Scenario: Ineligible run
- **WHEN** the candidate's holdout run is flagged `model_drift` or used an out-of-configuration evaluation
- **THEN** the check returns `ineligible` and no promotion occurs

### Requirement: Promotion requires recorded user approval
A candidate SHALL be promoted only when the check result is `pass` and the user's approval is recorded with approver identity, time and the check result reference. Promotion changes only registry status and MUST NOT publish plans or change risk preferences.

#### Scenario: Approval without a passing check
- **WHEN** an approver tries to approve a candidate whose check result is `fail`, `not_comparable` or `ineligible`
- **THEN** the approval is refused and the registry status stays `candidate`

#### Scenario: Passing check without approval
- **WHEN** the check result is `pass` but no user approval is recorded
- **THEN** the registry status stays `candidate`

#### Scenario: Promotion approved
- **WHEN** a candidate's check result is `pass` and the user approves
- **THEN** its registry status becomes `promoted` with the approval record, and no plan version, publication or risk preference changes

### Requirement: No guaranteed-improvement claims
Promotion reports SHALL state the evidence and its limitations (seed dispersion, holdout reuse, leakage risk, synthetic or real data) and MUST NOT claim that a promoted candidate will perform better in future.

#### Scenario: Report wording check
- **WHEN** a promotion report is generated
- **THEN** it contains the limitations section and no forward-looking performance guarantee
