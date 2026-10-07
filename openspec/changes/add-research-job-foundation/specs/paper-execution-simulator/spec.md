# Spec Delta

## Purpose

Defines the single evaluator and paper execution simulator that turns any strategy's target allocations into simulated trades, costs and portfolio outcomes under identical point-in-time inputs, constraints, fees, execution timing and liquidity assumptions.

## ADDED Requirements

### Requirement: One evaluator for all strategy families
Every strategy family (controls, classical optimizers and any later family) SHALL be evaluated by the same evaluator version, which is recorded in each result. A benchmark comparison MUST be refused if its runs used different evaluator versions, datasets, constraint sets, fee models, execution timing or liquidity models.

#### Scenario: Mismatched evaluator settings
- **WHEN** a benchmark comparison includes one run with a 5 bps fee model and another with a 10 bps fee model
- **THEN** the comparison is refused with `VALIDATION_FAILED` listing the differing settings

#### Scenario: Same settings
- **WHEN** equal-weight and minimum-variance runs share dataset, evaluator version and simulation configuration
- **THEN** they appear side by side in one benchmark report

### Requirement: Strategy interface
A strategy SHALL receive only the point-in-time feature view for each decision time plus current simulated holdings, and SHALL return target weights per instrument plus cash. The simulator, not the strategy, MUST produce trades.

#### Scenario: Strategy returns orders
- **WHEN** a strategy returns order quantities instead of target weights
- **THEN** the run fails with `VALIDATION_FAILED`

### Requirement: Portfolio constraints enforced by the simulator
The simulator SHALL validate every target against the configured constraint set (at least: long-only flag, per-instrument weight bounds, minimum cash, maximum turnover per rebalance, weights summing to 1 with cash). A violating target MUST be either projected to the nearest feasible target or rejected, as configured, and every projection or rejection MUST be recorded.

#### Scenario: Weight bound exceeded
- **WHEN** a strategy proposes 60% in one instrument with a 40% cap and the policy is `project`
- **THEN** the simulator executes the projected target, and the result records the violation and the projection distance

#### Scenario: Reject policy
- **WHEN** the policy is `reject` and a target violates the turnover limit
- **THEN** the simulator keeps current holdings for that rebalance and records the rejection

### Requirement: Execution timing
The simulator SHALL execute trades for a decision made at time t no earlier than the next executable price after t, as configured (for example next session open or next close). Same-bar execution at the price used to make the decision MUST NOT be allowed.

#### Scenario: Next-open execution
- **WHEN** a decision uses the close of day d and timing is `next_open`
- **THEN** fills use the open of the next trading session after d

### Requirement: Fees and transaction costs
The simulator SHALL apply the configured fee model (proportional bps and fixed per-trade components) and spread cost to every simulated fill, and SHALL report total costs paid separately from gross returns.

#### Scenario: Costs reported
- **WHEN** a run completes
- **THEN** the result contains gross return, net return and total fees, spread and slippage costs as separate fields

### Requirement: Liquidity and slippage model
The simulator SHALL cap each fill at a configured fraction of the instrument's available volume at the execution time and SHALL apply a configured slippage function. Unfilled quantity MUST carry over or be cancelled per configuration and be recorded.

#### Scenario: Participation cap hit
- **WHEN** a target trade exceeds the participation cap
- **THEN** only the capped quantity fills and the remainder is handled per configuration and reported

### Requirement: Paper only, never live
The simulator SHALL operate only on simulated holdings and MUST NOT hold credentials for, call, or produce instructions for any live broker, exchange or wallet. Simulated fills are research artifacts, not platform executions.

#### Scenario: Live adapter configured
- **WHEN** a simulation configuration names a live execution venue
- **THEN** validation fails with `OPERATION_NOT_PERMITTED`

### Requirement: Deterministic simulation
Given the same dataset, strategy outputs, simulation configuration and evaluator version, the simulator SHALL produce identical trades, holdings and metrics.

#### Scenario: Replay
- **WHEN** a stored run's strategy outputs are replayed through the same evaluator version
- **THEN** the replayed trades and metrics match the stored result exactly

### Requirement: Accounting reconciliation
At every simulated step, holdings value plus cash SHALL equal the previous value plus price changes minus costs, within a fixed numerical tolerance. A reconciliation failure MUST fail the run.

#### Scenario: Reconciliation break
- **WHEN** a step's accounting differs from the reconciliation identity beyond tolerance
- **THEN** the run ends `failed` with code `INTERNAL` and the failing step is reported
