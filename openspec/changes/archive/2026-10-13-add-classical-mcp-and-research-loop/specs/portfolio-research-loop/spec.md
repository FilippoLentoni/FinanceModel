# Spec Delta

## Purpose

Provide reproducible portfolio research loop for auditable beta portfolio decisions, explanations and controlled research.

## ADDED Requirements

### Requirement: Stored feedback and literature
Reviews SHALL bind user feedback, prior analyses, sourced literature metadata, hypotheses and proposed experiment configurations to immutable records. Unsupported model or feature changes MUST remain proposals.

#### Scenario: Missing feature hypothesis
- **WHEN** a review proposes a feature unsupported by the runner
- **THEN** the proposal records required implementation and does not silently substitute a different experiment.

### Requirement: Budgeted weekly sandbox
The beta weekly loop SHALL schedule one review each Monday at 09:00 America/New_York and launch at most one bounded sandbox job per week within USD 0.50, monthly and project limits. Duplicate events and active jobs MUST prevent duplicate paid work.

#### Scenario: Budget exhausted
- **WHEN** the weekly event cannot cover the estimated job
- **THEN** a persisted skipped review names the budget reason and no job starts.

### Requirement: Explicit strategy activation
Research SHALL benchmark candidates and retain seeds, data timing, evaluation windows, costs and findings. It MUST propose changes without automatically replacing the active strategy or publishing trades.

#### Scenario: Candidate improvement
- **WHEN** a sandbox candidate beats its comparison baseline
- **THEN** the review stores results and a proposed activation while active PPO and classical settings remain unchanged.
