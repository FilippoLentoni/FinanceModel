# hourly-crypto-paper-experiments Specification

## Purpose
Compare eligible hourly crypto strategies through reproducible synthetic portfolio evolution with explicit costs, timing and model eligibility.

## Requirements

### Requirement: Synthetic portfolio evolution
The system SHALL maintain independent USD 100 synthetic portfolios for eligible optimization strategies and cash and buy-and-hold controls; decisions based on completed bars SHALL fill only on a subsequently observed bar with explicit fees, slippage, fractional quantities and no leverage.

#### Scenario: An hourly run retries or arrives out of order
- **WHEN** An hourly run retries or arrives out of order
- **THEN** no fill or book revision is duplicated or backdated

### Requirement: Model eligibility and evidence
The system SHALL expose optimization recommendations and PPO/LLM eligibility using crypto and hourly compatibility, frozen point-in-time evidence and reproducible policy identity. An incompatible equity PPO or unavailable LLM policy SHALL be reported ineligible, never substituted and labeled as that model.

#### Scenario: Only a daily equity PPO artifact exists
- **WHEN** Only a daily equity PPO artifact exists
- **THEN** crypto PPO is unavailable and the comparison states why

### Requirement: Honest experiment metrics
The system SHALL report net return, maximum drawdown, turnover, costs and observation count on comparable windows, distinguish forward simulation from retrospective research, and make no profit guarantee or automatic promotion.

#### Scenario: Only the initial portfolio observation exists
- **WHEN** Only the initial portfolio observation exists
- **THEN** risk metrics needing returns are unavailable rather than fabricated

### Requirement: Supplemental evidence availability
The system SHALL accept supplemental data only with immutable delivery references, availability timestamps, extraction version and declared research rights; training without historical coverage SHALL remain blocked.

#### Scenario: News was retrieved after a simulated decision
- **WHEN** News was retrieved after a simulated decision
- **THEN** it is excluded from that decision
