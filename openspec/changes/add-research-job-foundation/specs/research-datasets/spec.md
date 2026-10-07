# Spec Delta

## Purpose

Defines how FinanceModel derives versioned, point-in-time-correct evaluation datasets from approved snapshots and how it splits them chronologically into training, validation, walk-forward, holdout and prospective paper periods.

## ADDED Requirements

### Requirement: Deterministic, content-addressed datasets
A prepared dataset SHALL be identified by its input `input_snapshot_id` list and the preparation `configuration_id`, and SHALL carry a SHA-256 manifest of its files. Preparing the same inputs with the same configuration and container digest MUST produce byte-identical files and the same manifest checksum.

#### Scenario: Repeat preparation
- **WHEN** a dataset is prepared twice from the same snapshots, configuration and container digest
- **THEN** both runs produce the same manifest checksum and the second run reuses the first dataset instead of writing a new one

#### Scenario: Configuration change
- **WHEN** the feature lookback in the preparation configuration changes
- **THEN** a different `configuration_id` and a new dataset manifest are produced

### Requirement: Dataset lineage record
Every dataset SHALL record its input snapshots with their checksums, the preparation `configuration_id`, the container digest, the contract package version, the domain and `domain_schema_version`, and creation time. Results MUST reference datasets only through this record.

#### Scenario: Lineage lookup
- **WHEN** a reviewer inspects a benchmark result
- **THEN** they can trace every number to the dataset record, and from it to the exact snapshots and checksums

### Requirement: Point-in-time availability
Each observation in a dataset SHALL carry the time at which it became available, and features for a decision at time t MUST use only observations available at or before t. Observations marked `intraday_partial` MUST NOT be treated as completed daily observations.

#### Scenario: Late-arriving observation
- **WHEN** an observation for day d was retrieved after the decision time on day d
- **THEN** it is excluded from features for that decision and used from the next decision onward

#### Scenario: Look-ahead check fails
- **WHEN** a feature pipeline references an observation whose availability time is after the decision time
- **THEN** the preparation job fails with `VALIDATION_FAILED` naming the feature and timestamp

### Requirement: Chronological train, validation and test splits
Datasets SHALL be split by time into non-overlapping training, validation and test ranges, in that order, with an embargo between adjacent ranges at least as long as the longest label or feature horizon. Random shuffling across time MUST NOT be used for evaluation splits.

#### Scenario: Overlapping split rejected
- **WHEN** a split configuration places validation dates before the end of the training range
- **THEN** preparation fails with `VALIDATION_FAILED`

#### Scenario: Embargo enforced
- **WHEN** the label horizon is 20 trading days
- **THEN** at least 20 trading days separate the end of training from the start of validation

### Requirement: Walk-forward evaluation folds
The dataset preparation SHALL generate walk-forward folds from configuration (window type expanding or rolling, train length, test length, step, embargo). Each fold MUST train or calibrate only on data before its test window.

#### Scenario: Fold boundaries
- **WHEN** a walk-forward configuration with a 3-year expanding window and 6-month steps is applied
- **THEN** each fold's test window starts after its training window ends plus the embargo, and the fold list is recorded in the dataset record

### Requirement: Untouched holdout with logged access
The latest evaluation range SHALL be reserved as a holdout that model selection, tuning and calibration never read. Every holdout read MUST be logged with `run_id`, candidate `model_version` and purpose, and a holdout evaluation MUST require a frozen candidate.

#### Scenario: Tuning job reads holdout
- **WHEN** a job with purpose `tuning` requests holdout data
- **THEN** access is refused with `OPERATION_NOT_PERMITTED`

#### Scenario: Repeated holdout use reported
- **WHEN** the same candidate family has been evaluated on the holdout more than once
- **THEN** the benchmark report lists every holdout access and flags the holdout as reused

### Requirement: Prospective paper period
FinanceModel SHALL support a prospective paper period that starts after a candidate is frozen and uses only snapshots ingested after that freeze. Prospective results MUST be reported separately from historical backtests.

#### Scenario: Snapshot older than freeze
- **WHEN** a prospective evaluation is given a snapshot ingested before the candidate freeze time
- **THEN** that snapshot is rejected for the prospective period

### Requirement: Synthetic fixture datasets for phase 1
In phase 1, datasets SHALL be prepared only from synthetic snapshots built from the shared contract fixtures (flagged `synthetic: true`). Results computed on synthetic data MUST be labeled synthetic in every report and result.

#### Scenario: Synthetic label propagates
- **WHEN** a benchmark runs on a dataset prepared from synthetic fixtures
- **THEN** the result and report carry `synthetic: true`

### Requirement: Initial instrument definition
The initial instrument SHALL be S&P 500 exposure through a tracking ETF daily series (SPY by default), with the ETF ticker in the preparation `configuration_id`. The ETF series, the S&P 500 index level and the constituent universe MUST be treated as distinct datasets and never substituted for one another.

#### Scenario: Index level offered for the ETF series
- **WHEN** a preparation configuration for the ETF series receives an input snapshot whose instrument is the S&P 500 index level or the constituent universe
- **THEN** preparation fails with `VALIDATION_FAILED` naming the instrument mismatch

### Requirement: Completed daily observations only
In phases 1 and 2 datasets SHALL use only completed daily observations, and intraday observations MUST NOT be used as daily bars.

#### Scenario: Intraday observation supplied
- **WHEN** an input snapshot contains an intraday observation for the ETF
- **THEN** the observation is not used as a daily bar and preparation fails with `VALIDATION_FAILED` if no completed daily observation exists for that date

#### Scenario: Observation on a non-session date
- **WHEN** an input snapshot contains a daily observation dated on a day that is not an XNYS trading session in the pinned exchange calendar recorded in the snapshot provenance
- **THEN** preparation fails with `VALIDATION_FAILED` naming the date

### Requirement: Mock provider until approved real snapshots exist
Until the platform has approved real ETF snapshots in an environment, datasets SHALL come from synthetic ETF-shaped fixtures served by a mock provider, and requests for real ETF data MUST fail with `DEPENDENCY_UNAVAILABLE`. CI and build-stage tests MUST use only the mock provider and synthetic fixtures (provider decision 2026-10-07: the real provider is the platform's yfinance adapter, which FinanceModel never calls).

#### Scenario: No approved real snapshot yet
- **WHEN** a job requests real ETF data in an environment where no approved real ETF snapshot exists
- **THEN** it fails with `DEPENDENCY_UNAVAILABLE` and fixture-backed preparation remains available

#### Scenario: CI uses the mock provider
- **WHEN** the build stage runs dataset preparation tests
- **THEN** every input comes from synthetic fixtures through the mock provider and no network request to any market-data source is made

### Requirement: Market data only through approved platform snapshots
FinanceModel SHALL obtain real market data only by reading approved platform input snapshots. The platform's ingestion Lambda or container retrieves the S&P 500 tracking ETF (SPY) daily series with its yfinance provider adapter; FinanceModel MUST NOT call yfinance, Yahoo Finance or any other market-data provider directly, and MUST NOT declare yfinance (or any market-data provider client) as a dependency of its jobs, images or libraries.

#### Scenario: Provider dependency planted
- **WHEN** a change adds `yfinance` (or another market-data provider client) to a FinanceModel dependency file or container image
- **THEN** the build-stage dependency check fails and names the forbidden package

#### Scenario: Code attempts a direct provider call
- **WHEN** FinanceModel source code imports a market-data provider client or references a provider endpoint instead of resolving an `input_snapshot_id`
- **THEN** the build-stage static check fails, and at run time jobs read market data only through the approved-snapshot path

#### Scenario: Provider lineage carried into the dataset
- **WHEN** a dataset is prepared from an approved real ETF snapshot
- **THEN** the dataset lineage record copies, from the snapshot provenance, the provider adapter name (`yfinance`), the pinned library version, the retrieval timestamp and the exchange-calendar library and version (XNYS), and preparation fails with `VALIDATION_FAILED` if the snapshot lacks any of them

#### Scenario: Snapshot flagged as empty or partial
- **WHEN** an approved snapshot carries a platform quality flag for an empty or partial provider response on some dates
- **THEN** the flag is recorded in the dataset lineage and the affected dates are excluded or the preparation fails, as set in the preparation configuration, and never silently filled

### Requirement: No retrieved market data in the public repository
Retrieved market data (raw or curated provider series, including yfinance output) SHALL NOT be committed to the FinanceModel repository, which is public; test fixtures MUST stay synthetic. Derived research outputs (datasets, run results, staged bundles and reports) SHALL be kept in FinanceModel research storage or the platform staging area, not in the repository, because Yahoo Finance data is provided for personal and research use.

#### Scenario: Real series committed
- **WHEN** a change adds a file containing a retrieved provider price series to the repository
- **THEN** the build-stage fixture check fails because fixtures must carry the `synthetic: true` marker

#### Scenario: Report derived from real data
- **WHEN** a benchmark report is produced from a dataset prepared from real ETF snapshots
- **THEN** the report is stored in research storage (and, for production candidates, staged to the platform) with aggregate metrics only, it does not embed the raw retrieved price series, and it is not committed to the repository
