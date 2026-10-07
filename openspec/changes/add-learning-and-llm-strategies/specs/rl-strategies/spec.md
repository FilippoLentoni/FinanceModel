# Spec Delta

## Purpose

Defines how FinanceModel trains and evaluates reinforcement-learning allocation strategies (initially PPO and SAC) inside the common paper execution simulator, with explicit state, action and reward definitions and seed variability.

## ADDED Requirements

### Requirement: Versioned state, action and reward specification
Each RL strategy SHALL declare a versioned environment specification: state features (point-in-time only, including current holdings and cash), the action space and its mapping to target weights, the reward formula with every shaping term and coefficient, the episode definition and the decision frequency. The specification MUST be part of the `configuration_id`.

#### Scenario: Reward coefficient change
- **WHEN** the turnover penalty coefficient changes
- **THEN** a new `configuration_id` results and the earlier trained policy is not reused for it

#### Scenario: Undeclared feature
- **WHEN** the environment reads a feature not listed in the state specification
- **THEN** training fails with `VALIDATION_FAILED`

### Requirement: Simulator-backed environment
The RL environment SHALL step through the common paper execution simulator and dataset from the research foundation, with the same constraints, fees, execution timing and liquidity model used to evaluate all other strategies. Rewards MUST be computed from simulator outputs.

#### Scenario: Evaluator parity
- **WHEN** a trained PPO policy is evaluated
- **THEN** its portfolio metrics come from the same evaluator version and simulation configuration as the controls in the same benchmark

### Requirement: Continuous allocation algorithms
PPO and SAC SHALL act in a continuous action space mapped to feasible target weights by a declared transform (for example a softmax over instruments plus cash), followed by the simulator's constraint policy.

#### Scenario: Action mapping
- **WHEN** SAC outputs a raw action vector
- **THEN** the declared transform produces weights that sum to 1 with cash, and any further constraint projection is recorded by the simulator

### Requirement: DQN only with a discrete action design
A DQN strategy SHALL be accepted only with an explicit finite action set, where each action maps to a defined target-weight change. DQN MUST NOT be configured for continuous allocation.

#### Scenario: DQN without discrete actions
- **WHEN** a DQN configuration declares a continuous action space
- **THEN** submission fails with `VALIDATION_FAILED`

### Requirement: Multiple seeds
Each RL configuration SHALL be trained with at least the configured number of seeds (minimum 3), each recorded. Reports MUST show the distribution of results across seeds, and selection MUST use validation folds only.

#### Scenario: Seed results reported
- **WHEN** PPO is trained with 5 seeds
- **THEN** the report lists all 5 seeds' validation and test metrics, their mean and dispersion, and which seed was selected on validation data

#### Scenario: Selection on test data attempted
- **WHEN** a selection rule references test or holdout metrics
- **THEN** the run fails with `OPERATION_NOT_PERMITTED`

### Requirement: Shaped reward reported separately
Shaped training rewards SHALL be reported in the training-reward section and MUST NOT be presented as portfolio performance.

#### Scenario: Reward vs. return
- **WHEN** a report covers an RL strategy
- **THEN** episode rewards appear only in the training-reward section, and returns, drawdowns and costs appear only in the portfolio section

### Requirement: Training as SageMaker Training Jobs
RL training SHALL run as SageMaker Training Jobs submitted through the job interface, with time limits, lease, budget check and approval from the execution controls. Trained policies MUST be registered with a `model_version` that records the artifact checksum, seed and `configuration_id`.

#### Scenario: Trained policy registered
- **WHEN** a training job succeeds
- **THEN** the policy artifact checksum is recorded and a `model_version` is minted for it

#### Scenario: Training in CodeBuild
- **WHEN** a build-stage test attempts to run an RL learner for more than the fixed fixture step count
- **THEN** the build fails

### Requirement: Retraining on reward or environment changes
A change to the state, action or reward specification, the simulator configuration or the training range SHALL require new training runs before the policy can be a promotion candidate. Evaluating an existing policy under a different specification MUST be flagged as out-of-configuration and excluded from promotion.

#### Scenario: Evaluating a policy under a new fee model
- **WHEN** a benchmark uses a fee model different from the policy's training configuration
- **THEN** the report flags the policy as evaluated out of its training configuration and the policy is not eligible for promotion on that result
