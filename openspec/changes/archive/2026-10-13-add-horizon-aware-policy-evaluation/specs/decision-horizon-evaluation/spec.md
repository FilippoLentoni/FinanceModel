## Purpose
Evaluate recorded portfolio decisions over their objective horizons using reproducible sequential strategy paths and consistent controls.

## ADDED Requirements

### Requirement: Preserved evaluation protocol
New paper decisions SHALL preserve objective, evaluation windows, benchmark settings and cost assumptions at issuance. Legacy decisions SHALL identify retrospectively assigned protocols without claiming predeclaration.

#### Scenario: New and legacy decisions
- **WHEN** a new or legacy decision is evaluated
- **THEN** the response identifies its preserved or retrospective protocol and original portfolio revision, snapshot and strategy identity

#### Scenario: Backdated request after outcomes are known
- **WHEN** a new recommendation is issued after the first forward session has completed
- **THEN** its replay is labeled retrospective and ineligible for prospective skill evidence even when its protocol is saved

### Requirement: Frozen sequential replay
Evaluation SHALL replay the original strategy and subsequent rebalancing using only completed observations visible at each decision, starting from the original book. Missing or incompatible evidence SHALL yield unavailable results without substituting today's policy.

#### Scenario: Actor pin changed after issuance
- **WHEN** the currently selected actor differs from the issued artifact
- **THEN** evaluation uses the issued artifact and original inputs only

#### Scenario: Future data changed
- **WHEN** observations after a replay decision change
- **THEN** that earlier decision output is unchanged

### Requirement: Aligned controls and objective diagnostics
Evaluation SHALL compare sequential strategy, unchanged holdings, equal weight and fixed classical controls on identical saved sessions and execution cost settings, reporting return, drawdown, volatility, costs, turnover and objective-relevant diagnostics.

#### Scenario: Sequential replay differs from initial allocation hold
- **WHEN** a policy recommends a later rebalance
- **THEN** replay executes that rebalance and labels the frozen-allocation benchmark separately

### Requirement: Horizon maturity and cautious interpretation
Each window SHALL be partial or mature based on completed forward sessions. Reports SHALL avoid optimality or model-error claims from one path, disclose accounting and historical-data limits, and recommend independent evaluation when underperformance persists.

#### Scenario: Daily loss before primary horizon
- **WHEN** the observed window has one negative session before the primary horizon
- **THEN** interpretation is preliminary and does not declare the policy defective

#### Scenario: Corporate action lacks accounting
- **WHEN** forward observations contain an unaccounted split or dividend
- **THEN** raw-price replay is unavailable with an explicit reason
