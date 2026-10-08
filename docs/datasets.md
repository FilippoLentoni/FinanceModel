# Research datasets

Task group 3 of `add-research-job-foundation`. Spec: `research-datasets` (DS-01 to DS-14). Design:
D6. Code: `src/finplan_model/datasets/`.

> **Defaults on this page are configuration, not market facts.** Phase 1 prepares datasets only from
> **synthetic** snapshots served by the mock provider. Real SPY snapshots come only from the
> platform's yfinance ingestion, read as approved platform snapshots. FinanceModel never calls a
> market-data provider.

## Flow

```
input_snapshot_ids ──► SnapshotReader (approved only, every SHA-256 verified)
                   ──► data-source, instrument, provenance and calendar checks
                   ──► observations: completed daily only, sessions only, revisions, quality policy
                   ──► available_at per observation ──► point-in-time features + look-ahead check
                   ──► splits with embargo, walk-forward folds (or the prospective window)
                   ──► canonical files + SHA-256 manifest (research_dataset reference) + lineage record
```

`prepare_dataset(input_snapshot_ids, config, reader=, store=, catalog=, ctx=, environment_data_source=, period=, candidate=)`
returns a `PreparedDataset`:

- `record` is the lineage record.
- `reused` is true when the catalog already held the dataset.
- `ref` is the `research_dataset` trusted reference.

`PreparedDataset.load(store)` (or `Dataset.load(record, store)`) gives the verified `Dataset`.

## Identity and determinism (DS-01)

- **Dataset key**: SHA-256 of the sorted `input_snapshot_id` list, the preparation
  `configuration_id`, the container image digest (`ctx.image_digest`, `local` offline) and the
  preparation code version (`dataset-prep-v1`). For a prospective dataset, the frozen candidate is
  included too.
- **`dataset_id`**: `rds_` followed by the first 40 hex digits of the key.
- **Files**: canonical JSON (RFC 8785, contract canonicalizer), content-addressed in research
  storage:
  - `calendar`, `observations`, `features` and `splits`;
  - `holdout`, a separate file that only the holdout accessor reads.
- **Manifest**: lists every file with its checksum. It is stored as the `research_dataset` artifact,
  with `artifact_id` = `dataset_id` and `checksum` = manifest checksum, as the contract
  `artifact-ref` describes.
- Equal inputs, configuration and image digest give byte-identical files and the same manifest
  checksum. Only the lineage record's `created_at` can differ, and the record is written once.
  - The **dataset catalog** (`InMemoryDatasetCatalog`, `LocalDatasetCatalog`, `S3DatasetCatalog`
    with a conditional create) maps the key to the first record. A repeat preparation **reuses**
    it and writes nothing.
  - Records carry a self-checksum that is verified on read.
- Changing the feature lookback (or any setting) changes the `configuration_id`, the dataset key
  and the manifest.

## Lineage record (DS-02, DS-12)

The record holds the following, and results reference datasets only through it:

- `input_snapshots`, each with its `manifest_checksum`, dataset, status, coverage, quality flags
  and provenance:
  - provider adapter, provider library and pinned version, and retrieval timestamp;
  - calendar exchange, version, library and library version, and session count.
- `configuration_id` and the full configuration.
- `image_digest`, `contract_version`, `domain` and `domain_schema_version`.
- `instrument`, `calendar`, `coverage` and the `availability` rule.
- `quality`: policy, flags by snapshot, excluded sessions and their reasons, missing sessions.
- `revisions`, `intraday_dropped`, `splits`, `folds` and `prospective`.
- `synthetic`, `real_data`, `created_at`, `created_by_run_id` and `correlation_id`.

A **real** (non-synthetic) snapshot must carry all of the following, or preparation fails with
`VALIDATION_FAILED` (`snapshot_provenance_incomplete`, naming the missing fields):

- `lineage.provider`, a real adapter such as `yfinance`;
- `lineage.provider_library` and `lineage.library_version`;
- `lineage.retrieved_at`;
- `lineage.calendar_version` in the form `<exchange>-<library>-<version>-<start>-<end>`, for example
  `xnys-exchange_calendars-4.13.2-...`;
- the manifest calendar block `{exchange, version, coverage, sessions}`.

> **Contract gap (FM-A5):** the platform snapshot manifest of contracts 0.2.2 and 1.0.0 (no schema change) names the calendar
> (`calendar.exchange`, `calendar.version`) but does not list its sessions. FinanceModel needs the
> session list for trading-day arithmetic and missing-session detection, and has no calendar
> dependency of its own. So a real snapshot without `calendar.sessions` is refused rather than
> guessed. The synthetic fixtures carry the list. Task 3.10 (beta, real SPY snapshots) depends on
> the platform adding it.

## Point-in-time availability (DS-03)

Every dataset observation carries `available_at`. The rule is set by `availability.mode`:

| Mode | `available_at` | Use |
|---|---|---|
| `retrieved_at` (default) | max(session close + lag, first retrieval time of the observation) | Daily ingestion snapshots |
| `session_close` | session close + lag | One historical backfill snapshot (recorded in lineage as the assumption used) |

Settings:

- `availability.assumed_close_utc` defaults to `21:00`. This is conservative: never earlier than
  the XNYS close.
- `availability.publication_lag_minutes` defaults to `30`.
- `decision_time_utc` defaults to `22:00`, the simulator's default decision time.

Features for a decision at time `t` use only observations with `available_at <= t`:

- `alignment: asof` (default) joins as of `t`. An observation for day `d` retrieved after the
  decision time on `d` is excluded for that decision and used from the next one.
- `alignment: session` with `offset` references a fixed session relative to the decision.

After computing, every referenced observation is checked. A reference to an observation available
after the decision time fails with `VALIDATION_FAILED` naming:

- the `feature`;
- the observation's `observation_session_date` and `observation_available_at`;
- the `decision_time`.

Feature kinds are `trailing_return`, `trailing_volatility` and `close`.

## Instrument, observations and quality (DS-09, DS-10, DS-12)

- **Instrument**: an ETF daily series only (`dataset_kind` `etf-daily`, `asset_class` `etf`; SPY by
  default). The tickers are part of the `configuration_id`. A snapshot whose dataset or instrument
  differs fails with `VALIDATION_FAILED` ("instrument mismatch", naming both), for example the
  index level `finance/index-level/SPX` or the constituent universe. The configuration itself
  refuses other dataset kinds and intraday granularity.
- **Intraday observations**: `intraday_partial` observations are never daily bars. A date with only
  an intraday observation fails (`intraday_only`, naming the date).
- **Non-session dates**: a completed observation on a date that is not a session of the snapshot's
  calendar fails (`non_session_date`, naming the date).
- **Quality flags**: flags in `quality.flags` (default `empty_response`, `partial_response`,
  `missing_sessions`, `stale_source`), and calendar sessions without a completed observation, are
  handled by `quality.policy`:
  - `exclude` (default): the dates are removed from the dataset;
  - `fail`: preparation fails (`quality_flagged_dates`).

  The affected dates come from `quality_details[flag]`, or else from the snapshot's coverage. They
  are recorded and never filled.
- **Revisions**: by default (`revisions` `first_seen`), an observation that differs between
  snapshots keeps the first-retrieved values (point in time) and is counted. `fail` refuses it.

## Data sources and the mock provider (DS-08, DS-11)

| `data_source` | Accepted snapshots | Otherwise |
|---|---|---|
| `fixture` (phase 1) | Synthetic, provider `fixture` or `mock` | `VALIDATION_FAILED` |
| `platform_snapshots` (real) | Approved, non-synthetic | `DEPENDENCY_UNAVAILABLE` (`no_approved_real_snapshot`) when the environment serves fixtures only (`config/<env>.json` `instrument.data_source`), or the snapshot is missing or not approved. Fixture preparation stays available |

`MockSnapshotProvider` publishes synthetic ETF-shaped snapshots to the in-process
`FixturePlatformClient`. They are built from the shared contract fixtures
`observation/valid/completed-daily.json` and `snapshot-payload/valid/etf-daily.json`, read from
the pinned package and never copied:

- `publish_backfill(ticker, start, end)` publishes one snapshot;
- `publish_daily(ticker, start, end, late=, skip=)` publishes one snapshot per session.

The synthetic calendar is weekdays minus rule-based holidays (`fixture-synthetic-v1-<start>-<end>`).
Everything is flagged `synthetic: true`. CI makes no network request.

## Splits, folds, holdout, prospective period (DS-04 to DS-07)

- **`splits`**: `train`, `validation` and `holdout` date ranges, plus `embargo_sessions`.
  - The ranges must be non-overlapping and in that order. Validation must start after the end of
    training, and the holdout after the end of validation.
  - `embargo_sessions` must be at least the longest label or feature horizon. Feature lookback
    windows look backward and do not count.
  - At least that many trading sessions of the snapshot calendar must separate adjacent ranges.
  - There is no shuffle option.
- **`walk_forward`**: `window` (`expanding` or `rolling`), `train_length`, `test_length` and `step`
  (`{sessions: n}`, `{months: n}` or `{years: n}`), `embargo_sessions` and `min_folds`.
  - Folds span the start of training to the end of validation, so they never touch the holdout.
  - Each test window starts after its training end plus the embargo. Consecutive test windows are
    disjoint, and incomplete trailing folds are dropped and counted.
  - The folds are recorded in the lineage record. `Dataset.training_market(fold)` stops at the
    fold's training end.
- **Holdout**:
  - `HoldoutAccessor(dataset, log).read(ctx, candidate)` is the only reader.
  - Purposes other than `holdout_evaluation` get `OPERATION_NOT_PERMITTED`: `tuning`, `research`
    (model selection) and `production_candidate`.
  - The candidate must be frozen (`FrozenCandidate` with a `model_version` and a freeze time not
    after now), else `PRECONDITION_FAILED` (`candidate_not_frozen`).
  - Every attempt is logged with run, candidate, family, purpose, outcome and time.
  - `holdout_access_report` lists accesses and flags `reused` when a family was evaluated on the
    holdout in more than one run.
- **Prospective paper period**: `period="prospective_paper"` with the frozen candidate.
  - Snapshots retrieved or committed at or before the freeze are rejected (`VALIDATION_FAILED`,
    naming the snapshot).
  - The window starts after the freeze date and is reported as `prospective_paper`.

## Build-stage guards (DS-12, DS-13) and outbound data (DS-14)

- **`provider-guard` gate**: no market-data provider client (`yfinance` and the others in
  `MARKET_DATA_PACKAGES`) in `pyproject.toml` (all groups), `uv.lock`, `requirements*` or
  `Dockerfile*`. No provider import in non-test source, and no provider endpoint reference.
- **`fixture-check` gate**:
  - Data files under `tests/`, `fixtures/` or `data/` carry `synthetic: true` (JSON) or a first
    line `# synthetic: true` (CSV/TSV).
  - Binary data files are refused there.
  - Price-like JSON or CSV anywhere must carry the marker too.
- **Results and reports**: results and reports derived from real data go only to research storage
  or platform staging, never into a source repository (`OPERATION_NOT_PERMITTED`). Reports carry
  aggregate metrics only (see [benchmarks.md](benchmarks.md)).
- **`check_outbound_payload` / `guarded_send`**:
  - Only research purposes (`research`, `tuning`, `holdout_evaluation`) may send.
  - Rejected: runs of more than 3 consecutive numbers, numbers equal to raw price or volume
    observations (also when rounded), tables, attachments and payloads over 16 KiB.
  - Rejections are logged with the run ID and reason codes, never the values. The sender is not
    called.

## Example preparation configuration

<!-- example: preparation-config -->
```json
{
  "instrument": {"tickers": ["SPY"], "dataset_kind": "etf-daily", "calendar": "XNYS", "currency": "USD", "granularity": "daily"},
  "data_source": "fixture",
  "availability": {"mode": "retrieved_at", "assumed_close_utc": "21:00", "publication_lag_minutes": 30},
  "decision_time_utc": "22:00",
  "features": {"lookback_sessions": 60, "label_horizon_sessions": 20, "feature_horizon_sessions": 0},
  "splits": {
    "train": {"start": "2018-01-01", "end": "2022-06-30"},
    "validation": {"start": "2022-08-01", "end": "2023-03-31"},
    "holdout": {"start": "2023-05-01", "end": "2023-12-31"},
    "embargo_sessions": 20
  },
  "walk_forward": {"window": "expanding", "train_length": {"years": 3}, "test_length": {"months": 6}, "step": {"months": 6}, "embargo_sessions": 20},
  "quality": {"policy": "exclude"},
  "revisions": "first_seen"
}
```
