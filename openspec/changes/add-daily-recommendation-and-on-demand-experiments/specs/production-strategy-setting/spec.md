# Spec Delta

## Purpose

Defines how the user reads, sets and clears the per-environment production strategy. Each choice is validated against the strategy registry and stored in a single-writer SSM key that the daily recommendation job and the platform trigger read.

## ADDED Requirements

### Requirement: Setting is an explicit user action
The job API SHALL provide `get_production_strategy`, `set_production_strategy` and `clear_production_strategy`. Set and clear MUST require an on-behalf-of user, `confirmed_by_user: true` and an `idempotency_key`, and are accepted only from the tool `plan-writer` role or the operator role.

#### Scenario: Missing confirmation
- **WHEN** a set request lacks `confirmed_by_user`
- **THEN** it fails with `PRECONDITION_FAILED` `confirmation_required`, and the key is unchanged

#### Scenario: Nothing selected
- **WHEN** `get_production_strategy` is called and the key is absent
- **THEN** the response is `none`

### Requirement: Validated against the strategy registry
Set SHALL accept a strategy only if it meets all of these:
- it is registered with that `model_version`;
- it is deployed in the environment and not retired;
- it supports `equity-etf-daily`;
- it has at least one succeeded research run on that dataset in the environment;
- if it belongs to a learning family, it is `promoted`.

Each failure MUST return `VALIDATION_FAILED` naming the rule.

#### Scenario: Never evaluated
- **WHEN** the user selects `scenario_cvar` with no succeeded universe run in the environment
- **THEN** the call fails with details `no_evaluation_evidence`

#### Scenario: Valid selection
- **WHEN** the user selects `min_variance` after a succeeded universe benchmark
- **THEN** the key holds `strategy_id`, `model_version`, `configuration_id`, `set_by` and `set_at`, and the response cites the evidence `run_id`

### Requirement: Single writer and audit
Only FinanceModel's selection role SHALL write or delete `/finplan/<env>/financemodel/config/production-strategy`. Every set or clear MUST append an audit event with the old value, the new value, the user and the channel.

#### Scenario: Tool role writes SSM directly
- **WHEN** a tool, agent or platform role is simulated against `ssm:PutParameter` on the key
- **THEN** the result is deny
