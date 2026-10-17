## Purpose
Enable reproducible matched news research and isolated small synthetic portfolios with honest evidence and decision-time availability.

## ADDED Requirements

### Requirement: Matched news performance evidence
The experiment SHALL compare price-only and enriched policies using the same frozen prices, initial holdings, constraints and costs. It SHALL report realized simulated return, drawdown, turnover and costs separately from forecast estimates.

#### Scenario: Eligible frozen corpus
- **WHEN** all decision times have eligible frozen price and licensed news coverage
- **THEN** both arms execute after decisions and emit comparable net metrics and source checksums

### Requirement: Coverage blockers remain explicit
The experiment SHALL reject retrospective use of later snapshots or news. Missing historical coverage SHALL produce ineligible performance with reasons and a prospective shadow protocol, without invented historical results.

#### Scenario: Today’s news for past dates
- **WHEN** retrieval or availability follows a historical decision
- **THEN** the enriched historical performance is unavailable rather than using future news

### Requirement: Separate synthetic capital profile
A $100 VOO/GOOGL paper profile SHALL have caller and portfolio identity, synthetic cash and no live trading. Initialization SHALL preserve all other saved portfolios and reject reuse with conflicting identity or state.

#### Scenario: Existing larger paper portfolio
- **WHEN** the caller initializes the small study
- **THEN** a separate portfolio is created and the existing portfolio remains unchanged

### Requirement: Real news reaches served optimizer and PPO recommendations
Completion SHALL require purchased eligible news stored centrally, consumed by actual optimizer and compatible trained PPO computations, and returned in hosted LangGraph recommendations with source and observation checksums. Adapters, readiness or ineligible outputs SHALL NOT satisfy this requirement.

#### Scenario: Compatible checkpoint unavailable
- **WHEN** rights or historical coverage prevents training a compatible news-aware actor
- **THEN** the exit check fails and the completion task stays open; legacy checkpoints cannot substitute
