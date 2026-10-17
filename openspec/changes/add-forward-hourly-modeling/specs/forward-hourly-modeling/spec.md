## ADDED Requirements

### Requirement: Honest approach identity and exact actual model inputs
The system SHALL register every implemented approach and meaningful evaluated
variant with its own isolated paper book, actual framework/model/configuration,
supported universe, cadence and input capabilities. It SHALL capture exact
prepared solver/actor/provider inputs at the invocation boundary with source
references and immutable checksums. Unsupported dependencies SHALL remain visible.

#### Scenario: Frozen RL input with no news
- **WHEN** a price-only frozen actor executes
- **THEN** its named ordered normalized observation and actual price/holdings values SHALL be retained
- **AND** news SHALL be labeled not consumed

#### Scenario: News-aware actor unavailable
- **WHEN** native source, formation time, policy artifact or universe is incompatible
- **THEN** captured actual preconditions and an unavailable outcome SHALL be retained without invented features

### Requirement: Durable hourly comparison and independent research collection
The system SHALL permanently account for every experiment/approach/hour with
captured inputs, output or explicit status, and a portfolio observation. It SHALL
use same-capital/same-policy books within declared fair cohorts. Thirty-minute
research SHALL retrieve actual public financial/S&P evidence, invoke headless
Codex review, and SHALL NOT give the primary book extra trading opportunities.

#### Scenario: Repeated or crashed hourly slot
- **WHEN** a slot retries or an invocation crashes after input capture
- **THEN** permanent keys and lease claims SHALL prevent duplicate inference/fills and preserve original inputs

#### Scenario: Missing historical source
- **WHEN** a missed historical slot is recovered
- **THEN** it SHALL be explicitly skipped without fetching replacement data as historical evidence

### Requirement: Forward execution uses actual eligible quote evidence
Forward simulated fills SHALL use a fresh actual quote captured at the declared
next hourly regular-session opportunity, after the actual decision. The common
evaluator SHALL apply shared constraints, sell-before-buy, cash safety, liquidity
and costs. It SHALL NOT manufacture daily bars or fill a historical open whose
price was acquired later.

#### Scenario: Closed session or stale quote
- **WHEN** no eligible fresh quote is actually captured
- **THEN** the policy SHALL remain pending and no forward fill SHALL be invented

### Requirement: Immutable forecasts and correctly aligned realizations
A genuine value forecast SHALL be frozen with USD units, baseline, target time
and checksum. A separate realization SHALL use the first valid actual valuation
at or after the target, adjusting external flows and retaining price evidence.
A durable pending index SHALL preserve horizons beyond report windows.

#### Scenario: Later price revision
- **WHEN** a forecast is already matched and later observations arrive
- **THEN** the original prediction and first eligible match SHALL remain immutable
