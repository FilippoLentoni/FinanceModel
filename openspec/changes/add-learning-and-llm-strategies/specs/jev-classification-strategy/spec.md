# Spec Delta

## Purpose

Defines the TypeSafe Jev buy/hold/sell strategy (phase 2, enabled behind per-run approval): one `choice` question per instrument and decision date sent to the external TypeSafe API, text-encoded point-in-time features, recorded response model identity, calibration on FinanceModel's own splits, a deterministic mapping from probabilities to position sizes, rate-limit handling, secret handling by name only, TypeSafe cost tracked outside the AWS budget, leakage flags and a mocked API in CI.

## ADDED Requirements

### Requirement: Approval-gated phase 2 enablement
Jev job types SHALL be deployable in phase 2 behind an enable flag in configuration. While the flag is off in an environment, Jev submissions MUST return `DEPENDENCY_UNAVAILABLE`. While it is on, every Jev run MUST wait in `awaiting_approval` for a human approver regardless of the auto-approve threshold, and the approval request MUST show both the AWS job estimate and the TypeSafe estimate.

#### Scenario: Jev disabled in an environment
- **WHEN** a caller submits a Jev job in an environment where the Jev enable flag is off
- **THEN** the call fails with `DEPENDENCY_UNAVAILABLE`, `retryable` false, and no run is created

#### Scenario: Jev run below the auto-approve threshold
- **WHEN** a Jev run's AWS estimate is below the auto-approve threshold
- **THEN** the run still waits in `awaiting_approval` and shows the AWS and TypeSafe estimates

### Requirement: Single choice question per decision
For each instrument and decision date the strategy SHALL send exactly one `choice` question whose options are exactly `buy`, `hold` and `sell`, using a versioned question template that is part of the `configuration_id`. The decision MUST be derived from that one probability vector, so `buy` and `sell` can never both be selected. A response that lacks a probability for any of the three options MUST be treated as invalid.

#### Scenario: One question per decision
- **WHEN** a decision date has one instrument
- **THEN** exactly one `choice` question with options `buy`, `hold`, `sell` is sent for that instrument and date, and exactly one decision is recorded

#### Scenario: Missing option probability
- **WHEN** a response omits the probability for `sell`
- **THEN** the decision is recorded as invalid, resolves to `hold`, and the invalid-response counter increases

### Requirement: Deterministic decision and tie-break
The decision SHALL be the option with the highest calibrated probability, and an exact tie MUST resolve deterministically to `hold` and be counted.

#### Scenario: Tied probabilities
- **WHEN** calibrated `buy` and `sell` probabilities are equal and highest
- **THEN** the decision is `hold` and the tie counter increases

### Requirement: Text-encoded point-in-time features
The request `state` SHALL encode only features available at the decision time, rendered as text and bucketed descriptors rather than raw numeric values, with bucket edges computed from data available at decision time and recorded in the `configuration_id`. A request whose state plus questions exceed the configured token budget (at most the documented 32k) MUST NOT be sent.

#### Scenario: Feature uses future data
- **WHEN** a bucket edge or feature references an observation available after the decision time
- **THEN** preparation fails with `VALIDATION_FAILED` naming the feature

#### Scenario: Oversized state
- **WHEN** the rendered state and question exceed the configured token budget
- **THEN** the request is not sent and the run fails with `VALIDATION_FAILED`

#### Scenario: No raw retrieved series in the request
- **WHEN** the state is rendered from a dataset prepared from approved real ETF snapshots (retrieved by the platform's yfinance ingestion)
- **THEN** the state contains only bucketed descriptors, never the raw retrieved price series, and the request and response cache stays in FinanceModel research storage, not in the repository

### Requirement: Response model identity recorded
Each Jev request SHALL use the configured `model` value (an alias such as `jev-latest` until an exact version ID is accepted by the API, then the exact ID), and each response's returned `model` field MUST be stored with the decision. A run whose responses report more than one distinct `model` value MUST be flagged `model_drift` and is ineligible for promotion.

#### Scenario: Model recorded per response
- **WHEN** a Jev run completes
- **THEN** every stored decision has the returned `model` value, and the run summary lists the distinct values

#### Scenario: Alias moves during a run
- **WHEN** responses within one run report two different `model` values
- **THEN** the run is flagged `model_drift` and the promotion check rejects it

### Requirement: Explicit label definition
A Jev configuration SHALL define the evaluation label from point-in-time prices: the forward horizon in trading days, the return measure, and the thresholds that assign `buy`, `sell` or `hold`. Labels MUST be used only for calibration, policy selection and evaluation, and the label definition MUST be part of the `configuration_id`.

#### Scenario: Label with 5-day horizon
- **WHEN** the horizon is 5 trading days and thresholds are configured
- **THEN** each example's label is computed only from prices up to 5 days after its decision time, and the embargo between ranges is at least 5 days

### Requirement: Separate development, calibration, policy and test ranges
Chronologically ordered, non-overlapping ranges SHALL be used for question and state template development, probability calibration, sizing-policy selection (threshold and maximum weight), and testing, with an embargo of at least the label horizon between them. The policy MUST NOT be selected on calibration or test data, and the holdout MUST be read only by `holdout_evaluation` runs.

#### Scenario: Threshold chosen on test data
- **WHEN** a configuration selects the probability threshold using the test range
- **THEN** preparation fails with `VALIDATION_FAILED`

### Requirement: Calibration on FinanceModel splits
The strategy SHALL calibrate Jev's per-option probabilities on the calibration range with a configured method (default multinomial temperature scaling) and report calibration quality on the test range with a reliability table, Brier score and expected calibration error, for both raw and calibrated probabilities. Vendor calibration claims MUST NOT be reported as results.

#### Scenario: Calibration report
- **WHEN** a Jev evaluation finishes
- **THEN** the accuracy section contains the reliability table, Brier score and expected calibration error for the test range, for raw and calibrated probabilities

### Requirement: Deterministic probability-to-size mapping
A declared, deterministic sizing policy SHALL convert calibrated probabilities into target weights: for `buy`, target weight = `w_max × clip((p_buy − p_threshold) / (1 − p_threshold), 0, 1)`; for `sell`, target weight 0 (long-only); for `hold`, the current weight. `w_max` and `p_threshold` MUST be part of the `configuration_id`. Targets MUST then pass the simulator's constraint policy, and trades MUST come only from the simulator.

#### Scenario: Buy signal sizing
- **WHEN** an instrument gets `buy` with calibrated probability 0.7, `p_threshold` 0.5 and `w_max` 0.8
- **THEN** the target weight is 0.32, the simulator applies constraints, and the resulting trade is recorded with its cost

#### Scenario: Replay gives identical weights
- **WHEN** a run is replayed from its cached responses with the same configuration
- **THEN** every target weight and trade is identical and no API call is made

### Requirement: TypeSafe API client with rate limits and retries
The Jev client SHALL call `POST /v1/systemone` over HTTPS with bearer authentication, keep its request rate below a configured limit, wait for retry-after on 429, retry 529 and network errors with exponential backoff and jitter up to a configured count, and cache responses by request hash in research storage. A 401 MUST fail the run with `DEPENDENCY_UNAVAILABLE` without retry, and a 422 MUST fail the run with `INTERNAL` without retry.

#### Scenario: Rate limited
- **WHEN** the API returns 429 with a retry-after value
- **THEN** the client waits at least that long, retries, and records the retry

#### Scenario: Overloaded beyond retry budget
- **WHEN** the API keeps returning 529 after the configured retry count
- **THEN** the run ends `failed` with `DEPENDENCY_UNAVAILABLE`, `retryable` true, and the lease is released

#### Scenario: Invalid key
- **WHEN** the API returns 401
- **THEN** the run ends `failed` with `DEPENDENCY_UNAVAILABLE` and no retry is made

### Requirement: API key by secret reference only
The Jev API key SHALL be read at run time from AWS Secrets Manager under the name `finplan/shared/financemodel/jev-api-key` (us-east-2), or through an SSM pointer to that name. Repository files MUST contain only the name or pointer, never the value. The job role MUST be allowed to read only that secret, and the key MUST NOT appear in logs, results, job arguments or reports.

#### Scenario: Secret value in repository
- **WHEN** a commit contains a value that matches the Jev key pattern
- **THEN** the leak scan fails the build

#### Scenario: Key kept out of logs
- **WHEN** a Jev run logs its requests
- **THEN** the logged headers show the authorization value redacted

### Requirement: TypeSafe cost tracked outside the AWS budget
Each Jev run SHALL record input and output token counts and a TypeSafe cost computed from the configured price, labeled `external_billing: typesafe`, separately from its AWS cost. The TypeSafe cost MUST NOT be counted against the USD 50 AWS budget or any AWS budget category; the AWS job that calls Jev counts against `cpu_research`. A configured per-run TypeSafe token cap MUST stop the run when reached.

#### Scenario: Cost sections
- **WHEN** a Jev run ends
- **THEN** the cost section shows the AWS cost under `cpu_research` and the TypeSafe cost under `external_billing: typesafe`, and only the AWS cost reduces the AWS category balance

#### Scenario: Token cap reached
- **WHEN** cumulative input tokens reach the per-run cap
- **THEN** no further requests are sent and the run ends `failed` with `BUDGET_EXCEEDED`

### Requirement: Mocked Jev API in CI
Build, beta and gamma pipeline tests SHALL use a mocked TypeSafe API with recorded fixture responses and injected 429, 529, 401 and 422 errors. No pipeline stage MUST call the real TypeSafe API or read the secret value.

#### Scenario: Pipeline run
- **WHEN** the pipeline runs Jev tests in any stage
- **THEN** all requests go to the mock and no outbound request to the TypeSafe API host is made

### Requirement: Availability cutoff and leakage flag
Each Jev run SHALL record the availability cutoff of its inputs and the model's training-data cutoff, or "unknown". Historical results before the cutoff, and all historical results while it is unknown, MUST be labeled `leakage_risk: true`. Only prospective paper results after the configuration freeze MAY be presented as leakage-free.

#### Scenario: Unknown training cutoff
- **WHEN** a historical Jev backtest runs while the training-data cutoff is unknown
- **THEN** every period of the result is labeled `leakage_risk: true` and the report states the pretraining-leakage limitation

### Requirement: Accuracy is not profitability
Jev reports SHALL show classification metrics (per-class precision, recall, confusion matrix, calibration) in the accuracy section and portfolio results in the portfolio section, and MUST NOT present accuracy as evidence of profitability.

#### Scenario: High accuracy, negative return
- **WHEN** Jev has high accuracy but negative net return after costs
- **THEN** both are reported in their own sections and the promotion criteria are evaluated on portfolio metrics

### Requirement: No fabricated model details
Specifications, configuration and documentation SHALL NOT state Jev's architecture, parameter count, training data or capabilities beyond what the identification record in `docs/jev-identification.md` contains with its evidence (vendor documentation and observed API responses).

#### Scenario: Unverified capability claim
- **WHEN** documentation describes a Jev capability not present in the identification record
- **THEN** review rejects the change
