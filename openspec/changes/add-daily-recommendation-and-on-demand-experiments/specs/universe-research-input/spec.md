# Spec Delta

## Purpose

Lets the existing on-demand experiment kinds run on the approved `equity-etf-daily` research universe, so the user can compare strategies before choosing one. Their reports must flag hindsight and survivorship bias.

## ADDED Requirements

### Requirement: Universe snapshots accepted by existing experiment kinds
The existing `backtest` and `benchmark` kinds SHALL accept approved `equity-etf-daily` snapshots. They MUST use `adj_close` returns and the snapshot's declared cash assumption.

#### Scenario: Universe benchmark
- **WHEN** the user's agent submits a benchmark on an approved universe snapshot
- **THEN** every strategy is evaluated over the five instruments plus cash on identical folds and holdout

### Requirement: Reports flag hindsight and survivorship bias
Every report and result summary for an `equity-etf-daily` run SHALL include a "Hindsight and survivorship bias" section reproducing the snapshot's `bias_disclosures`, and SHALL state the cash assumption. A run whose snapshot lacks the disclosures MUST fail with `VALIDATION_FAILED`.

#### Scenario: Report content
- **WHEN** a universe benchmark report is generated
- **THEN** it contains both disclosures and the cash assumption

### Requirement: Experiments are never scheduled
Universe experiments SHALL start only from user-originated submissions (the tool `submitter` role with an on-behalf-of user, or the operator). FinanceModel MUST define no schedule that submits jobs, and the platform trigger role MUST receive `FORBIDDEN` for any kind other than `daily_recommendation`.

#### Scenario: Trigger role submits a benchmark
- **WHEN** the platform trigger principal submits `benchmark`
- **THEN** the call fails with `FORBIDDEN`
