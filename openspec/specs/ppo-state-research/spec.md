# ppo-state-research Specification

## Purpose

Implement backward-compatible state feature groups, controlled released ablations, point-in-time news records and eligibility, matched research experiments, reproducible result/report artifacts and beta pipeline gates.

## Requirements

### Requirement: Reproducible enriched state

The system SHALL support separately declared cross-asset risk, market-regime and short return-sequence features using only completed point-in-time inputs; configuration identity, feature order and actor shape SHALL be preserved across training, evaluation and serving.

#### Scenario: Frozen actor is replayed

- **WHEN** an enriched actor is evaluated or served
- **THEN** its observation and output SHALL match its recorded feature definition; legacy actors SHALL retain their old shapes

### Requirement: Matched state experiments

The system SHALL compare baseline, risk-only, regime-only, sequence-only and combined PPO states with identical seeds, reward, action, training budget, chronological splits, execution assumptions and controls. Selection SHALL use validation only; inspected test evidence SHALL be labeled development and ineligible for automatic promotion.

#### Scenario: A state comparison completes

- **WHEN** all requested seeds and variants complete
- **THEN** the report SHALL retain per-seed results, aggregate dispersion, controls, training diagnostics and costs without declaring a universal winner

### Requirement: News provenance and eligibility

News features SHALL retain immutable content identity, extractor version, source timestamps and actual observation/retrieval times. Historical decisions SHALL NOT see later collected or revised content. Missing coverage SHALL be explicit, and a news-enabled training request SHALL fail when eligible evidence is absent.

#### Scenario: Retrospective news is unavailable

- **WHEN** an article was collected after a historical decision
- **THEN** it SHALL be excluded and SHALL NOT be treated as neutral or contemporaneously available news

### Requirement: Bounded news collection

The system SHALL offer bounded collection of untrusted public news metadata into immutable research evidence and report coverage separately from explanatory context. News text SHALL NOT execute instructions or alter experiment authorization.

#### Scenario: Provider fails or returns unsafe metadata

- **WHEN** news collection receives a failure or invalid source URL
- **THEN** it SHALL retain an explicit unavailable/partial outcome without fabricating evidence

### Requirement: Research release and measured reporting

Feature changes SHALL pass point-in-time, observation parity and legacy compatibility checks before beta release. Study compute SHALL stay within the USD 5 incremental cap and USD 100 total project ceiling; results SHALL NOT update holdings or served policy automatically.

#### Scenario: Study finishes

- **WHEN** measured experiment artifacts are retrieved
- **THEN** tool-visible evidence and the paper SHALL cite the actual protocol, runs, limitations and results

