# Spec Delta

## Purpose

Defines the Qwen3.6-27B agent-swarm allocation strategy: fixed agent roles, a declared arbitration policy, deterministic validation of every proposal, complete logging of prompts and messages, and honest reporting of the pretraining-leakage limitation.

## ADDED Requirements

### Requirement: Fixed agent roles
A swarm configuration SHALL declare a fixed, ordered set of roles, each with a versioned prompt template, input view, structured output schema and turn limit. Roles MUST NOT be created, removed or rewritten during a run.

#### Scenario: Role set recorded
- **WHEN** a swarm run starts
- **THEN** the role list, prompt template versions and output schemas are part of the `configuration_id` and the run record

#### Scenario: Role added mid-run
- **WHEN** swarm code attempts to add a role during a run
- **THEN** the run fails with `OPERATION_NOT_PERMITTED`

### Requirement: Declared arbitration policy
The final target weights for each decision SHALL come from a declared arbitration policy that combines role outputs (for example a designated arbiter role choosing among proposals, or a deterministic rule over proposals). The policy MUST be in configuration and recorded with each decision.

#### Scenario: Arbitration trace
- **WHEN** a decision is made
- **THEN** the log shows each role's proposal, the arbitration input and the selected or combined target weights

### Requirement: Structured outputs validated by deterministic code
Every role output SHALL be parsed against its schema, and final targets MUST pass the simulator's constraint validation. Unparseable or invalid outputs MUST trigger the configured fallback (hold current weights) and be counted, never repaired by free-form guessing.

#### Scenario: Malformed JSON
- **WHEN** a role returns text that does not parse against its schema after the configured retry count
- **THEN** the decision uses the fallback, the failure is logged, and the report's invalid-output rate includes it

### Requirement: Point-in-time inputs only
Prompts SHALL be built only from the point-in-time dataset view at the decision time and prior swarm messages from the same run. Swarm jobs MUST have no web or retrieval access beyond that view.

#### Scenario: Future data in prompt
- **WHEN** a prompt builder requests an observation available after the decision time
- **THEN** prompt construction fails and the run ends `failed` with `VALIDATION_FAILED`

### Requirement: Complete prompt and message logging
Each run SHALL log every prompt, every model response, role, decision time, sampling parameters, seed, model ID and revision, and token counts to research storage, linked to `run_id`. Logs MUST be immutable after the run ends.

#### Scenario: Audit of one decision
- **WHEN** a reviewer opens a decision in a finished run
- **THEN** they can read the exact prompts, responses and arbitration that produced it

### Requirement: Reproducibility settings
Swarm runs SHALL use configured, recorded sampling settings (temperature 0 by default) and seeds. Reports MUST state that bit-identical regeneration on GPU is not guaranteed, and repeated runs MUST be reported as separate seeds.

#### Scenario: Repeat run differs
- **WHEN** the same swarm configuration is run twice and decisions differ
- **THEN** both runs are reported as separate seeds with their dispersion

### Requirement: Pretraining-leakage limitation reported
Every swarm report on a historical period SHALL state that the model's pretraining data may contain information from after the decision dates, even with timestamp-filtered inputs. It MUST record the model's documented training-data cutoff, or state that it is unknown, and label results before that cutoff as potentially contaminated.

#### Scenario: Backtest before cutoff
- **WHEN** a swarm backtest covers dates before the model's training-data cutoff
- **THEN** the report marks those results `leakage_risk: true` and shows the limitation statement

#### Scenario: Cutoff unknown
- **WHEN** no documented cutoff is configured
- **THEN** all historical results are marked `leakage_risk: true` and the report says the cutoff is unknown

### Requirement: Prospective evaluation for LLM claims
Only results from the prospective paper period after the swarm configuration is frozen SHALL be presented as free of pretraining leakage.

#### Scenario: Promotion evidence
- **WHEN** a swarm configuration is proposed for promotion
- **THEN** its evidence package separates prospective results from historical results labeled `leakage_risk`
