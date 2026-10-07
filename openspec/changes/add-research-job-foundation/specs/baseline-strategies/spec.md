# Spec Delta

## Purpose

Defines the control strategies and classical portfolio optimizers that form the reference benchmark against which every later strategy family is compared.

## ADDED Requirements

### Requirement: Control strategies
FinanceModel SHALL provide three controls: `cash` (100% cash), `buy_and_hold` (initial target held without rebalancing) and `equal_weight` (equal weights over the eligible universe at each rebalance). Every benchmark MUST include all three controls run on the same dataset and simulation configuration.

#### Scenario: Benchmark without controls
- **WHEN** a benchmark request omits a control
- **THEN** the control runs are added automatically and listed in the report

#### Scenario: Cash control
- **WHEN** the `cash` control runs
- **THEN** it incurs no trades and its return equals the configured cash rate

### Requirement: Minimum-variance optimizer
FinanceModel SHALL provide a minimum-variance optimizer that minimizes estimated portfolio variance subject to the configured constraint set, using a covariance estimator named in configuration and fitted only on data available at each decision time.

#### Scenario: Covariance window respects availability
- **WHEN** the optimizer decides at time t
- **THEN** its covariance estimate uses only observations available at or before t

### Requirement: Mean-variance optimizer
FinanceModel SHALL provide a mean-variance optimizer that maximizes expected return minus a configured risk-aversion times variance, with the expected-return estimator named in configuration.

#### Scenario: Risk aversion changes result
- **WHEN** risk aversion increases with all else equal
- **THEN** the resulting portfolio's estimated variance does not increase

### Requirement: Scenario-CVaR optimizer
FinanceModel SHALL provide a scenario-based CVaR optimizer that minimizes, or constrains, conditional value-at-risk at a configured confidence level over scenarios generated only from point-in-time data, with scenario generation recorded by seed and method.

#### Scenario: Reproducible scenarios
- **WHEN** the same configuration and seed are used twice
- **THEN** the same scenarios and the same optimal weights are produced

### Requirement: Optimizer outcome reporting
Each optimizer decision SHALL report a solver status. Infeasible or unbounded problems MUST be reported as `solution_status` `infeasible` or `unbounded` with `completion_status` `succeeded`, and the simulator MUST apply the configured fallback (hold current weights) for that decision.

#### Scenario: Infeasible constraints
- **WHEN** constraints require minimum cash of 50% and a minimum invested weight of 60%
- **THEN** the run completes with `solution_status` `infeasible`, no staging output is written, and the error is not a job failure

#### Scenario: Partial infeasibility in a backtest
- **WHEN** 2 of 40 rebalance decisions are infeasible
- **THEN** the run reports those decisions, applies the fallback, and reports an overall `solution_status` of `feasible`

### Requirement: CPU-only execution
Controls and classical optimizers SHALL run as SageMaker Processing Jobs on CPU instance types and MUST NOT request GPU instances.

#### Scenario: GPU requested for a baseline
- **WHEN** a baseline submission requests a GPU instance type
- **THEN** it is rejected with `VALIDATION_FAILED`
