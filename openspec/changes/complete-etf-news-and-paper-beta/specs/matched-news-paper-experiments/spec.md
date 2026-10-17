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

### Requirement: Historical search and prospective accumulation
The system SHALL test bounded historical publication-date search, retain the complete raw result, and exclude unverifiable or out-of-window publication dates from the eligible derived subset. Historical retrieval SHALL NOT backdate availability or establish exhaustive archive coverage. Every successful daily pull SHALL be persisted centrally for subsequent model datasets; retained raw and curated current versions SHALL have no automatic age expiry.

#### Scenario: Historical provider ignores date filters
- **WHEN** results fall outside the requested publication interval
- **THEN** the raw response is retained and only verified in-window rows enter the separately immutable eligible subset without another payment

### Requirement: Two-week forward checkpoint
The operator SHALL collect ETF/macro and GOOGL research once per trading day at 09:35 New York through October 30, preserving the ledger, $10 cumulative cap and per-call cap. The new authorized expiry SHALL retain the original expiry. The October 30 16:15 New York checkpoint SHALL freeze fourteen calendar days using overlapping bounded reads and report actual session coverage and unresolved training inputs.

#### Scenario: Small forward cohort is ready
- **WHEN** actual licensed news, approved real price arrays, sufficient price warmup and fresh whole-run budget evidence pass all checks
- **THEN** matched price-only and news-enriched PPO candidates can train with fixed seeds and costs, with held-out evaluation required before any serving change

#### Scenario: First checkpoint remains ineligible
- **WHEN** rights review, approved price inputs, coverage or budget evidence are missing
- **THEN** the checkpoint persists the frozen corpus and explicit blockers, starts no learner, and leaves strict hosted PPO completion open

#### Scenario: Additional data accumulates
- **WHEN** a later versioned experiment uses a larger retained corpus
- **THEN** the protocol preserves actual availability and nonoverlapping selection and test windows, and makes no automatic improvement or promotion claim

### Requirement: User-selected investment universe
Daily collection SHALL cover GOOGL, AMZN, NVDA, MSFT, AAPL, META, TSLA and VOO separately, retaining immutable provenance and the cumulative $10 cap. The S&P 500 index SHALL be represented by VOO. Existing response bounds SHALL be respected through batching until compatible broader contracts and models are released.

#### Scenario: A symbol has no eligible articles
- **WHEN** a symbol search returns no eligible news or retrieval fails
- **THEN** daily coverage distinguishes no news, rejected results and retrieval failure; no missing input is fabricated

#### Scenario: Training the selected universe
- **WHEN** a PPO experiment targets the eight assets
- **THEN** aligned prices and time-valid news cover the declared universe and dates, and compatible state/artifact evidence is required before claiming completion
