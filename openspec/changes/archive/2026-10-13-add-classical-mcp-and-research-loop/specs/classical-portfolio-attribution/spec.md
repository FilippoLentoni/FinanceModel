# Spec Delta

## Purpose

Provide reproducible classical portfolio attribution for auditable beta portfolio decisions, explanations and controlled research.

## ADDED Requirements

### Requirement: Today versus keeping
The explanation SHALL compare the selected solution with unchanged holdings and an instrument keep counterfactual. It MUST report objective units, feasibility and exact bounded Shapley contributions under a declared game and baseline.

#### Scenario: Why sell Google
- **WHEN** a user asks why the issued optimizer plan sells GOOGL
- **THEN** evidence shows the optimized and keep outcomes, objective effects and Shapley efficiency check.

### Requirement: Plan over plan
The workflow SHALL compare immutable issued plans across completed decision dates under an explicit compatible horizon. It MUST attribute changed expected returns, risk inputs, holdings and configuration with exact grouped Shapley over at most four groups.

#### Scenario: Two daily plans
- **WHEN** two compatible plans recommend different GOOGL actions
- **THEN** the original dates are retained and per-instrument grouped effects sum to the total change within tolerance.

### Requirement: Unsupported comparisons fail explicitly
Missing provenance, incompatible universes or unsupported horizon/model alignment SHALL return a precondition failure. Infeasible hybrid re-solves MUST be retained and MUST NOT yield fabricated Shapley values.

#### Scenario: Infeasible hybrid
- **WHEN** a required coalition has no feasible solution
- **THEN** the attribution is marked unavailable with the failing coalition and alternative single-switch evidence when available.

### Requirement: Observed performance reconciliation
Performance evidence SHALL anchor the immutable issued plan and approved observed data, reconcile plan and observed paths and classify green/red against stated thresholds. Missing fills, fees or forecast uncertainty MUST remain not_available.

#### Scenario: Paper observations
- **WHEN** only saved paper holdings and market closes exist
- **THEN** the report labels the observed path paper and does not infer executions or calibrated forecast error.

### Requirement: Dated market context
Market-event research SHALL store bounded dated source metadata and citations with availability status. External text MUST be untrusted and MUST NOT be reported as proof that an event caused a portfolio gap.

#### Scenario: News provider unavailable
- **WHEN** the public news source fails or has no matching dated records
- **THEN** the result states unavailable or no evidence and invents no event.
