# Benchmark runs and benchmark reports

Task group 5 of `add-research-job-foundation`. Spec: `benchmark-reporting` (REP-01 to REP-07). Code:
`src/finplan_model/reporting/`.

## Benchmark run

`run_benchmark(request, dataset, simulation, ctx=, holdout_accessor=, candidate=, model_versions=, store=)`:

- `request` is `{"strategies": [{"name", "params"?, "label"?, "seeds"?}], "periods": [...]}`.
  - Periods are a subset of `walk_forward`, `validation` and `holdout`. The default is
    `walk_forward` and `validation`.
  - Missing controls are added automatically.
  - `seeds` applies only to strategies with randomness (scenario-CVaR), which run once per seed.
- Every strategy is evaluated through the common evaluator on the same dataset and simulation
  configuration:
  - each walk-forward fold's test window gives a `walk_forward_fold` row;
  - the validation range gives a `validation` row;
  - for a prospective dataset, its window gives a `prospective_paper` row.
- `holdout` reads the holdout only through `HoldoutAccessor`. That requires purpose
  `holdout_evaluation` and a frozen candidate, and every read is logged.
  - Each holdout evaluation stores a holdout metrics record (REP-07).
- All results pass the comparison guard (SIM-01).
- With `store`, every result document (`run_artifact`) and holdout record
  (`holdout_metrics_record`) goes to research storage and is referenced by trusted reference.
- `BenchmarkRun.to_dict()` is everything a report is generated from.

## Report content

`build_report(run_or_stored_doc, cost_records=, config=, holdout_access=, accuracy=, training_rewards=, narrative=)`
returns a `BenchmarkReport` with `content` and `report_checksum`. The narrative is kept separately.

| Part | Content |
|---|---|
| `sections.portfolio_performance` | Per strategy (and seed) and period: `net_cumulative_return`, `gross_cumulative_return`, `annualized_volatility`, `sharpe_ratio`, `max_drawdown`, `turnover`, `total_transaction_costs`, `total_fees`, `total_spread_cost`, `total_slippage_cost`. `risk_free` states the Sharpe risk-free source (`configured_cash_rate`). Rows are grouped by period type in separately labeled columns: `walk_forward_fold` (each fold plus an `aggregate` row), `validation`, `holdout`, `prospective_paper` |
| `sections.model_accuracy` | Only for strategies that make predictions; `not_applicable` for the others. Its metrics never reuse portfolio metric names |
| `sections.training_rewards` | Only for learned strategies; `not_applicable` otherwise |
| `sections.compute_cost` | Per run: `instance_type`, `instance_count`, `billed_runtime_seconds`, `estimated_cost` (`label` `estimated`, from the pre-flight check), `actual_cost` (`status` `pending` until billing data exists, then `available` with `label` `actual`) |
| `variability` | For strategies with randomness: `n_seeds` and, per period, `mean`, `min`, `max` and `range` of every portfolio metric. Every seed's row is also reported |
| `holdout` | Bounds, the attached holdout access log (required whenever holdout rows are reported), `reused`, and the holdout metrics records |
| Labels | `synthetic` on the report and on every row; `period_type` on every row; `period_labels` |

## Rules

- **No blended score.** A report configuration asking for one is `VALIDATION_FAILED`
  (`blended_score_refused`), for example `blended_score`, `combined_score`, `composite`, `score`
  or `score_weights`. The four sections are fixed and validated.
- **Aggregate of walk-forward folds** (stitched out-of-sample):
  - cumulative returns compound the fold returns;
  - volatility and Sharpe use the concatenated session returns;
  - maximum drawdown uses the chained value path;
  - turnover and costs are summed over folds.
- **Deterministic generation.** `report_checksum` is the SHA-256 of the canonical JSON of
  `content`, and no wall-clock value is included. Regenerating from the same stored results
  (`report_from_stored(run_ref, store, ...)`) gives identical content and checksum.
- **Narrative stays separate.** `store_report` stores the numeric report (`benchmark_report`) and,
  separately, any narrative (`report_narrative`).
- **Aggregate metrics only.** A report embedding a series, meaning more than 8 numbers in a list or
  a field such as `nav`, `prices`, `bars`, `observations` or `fills`, is refused
  (`report_embeds_series`).
- **Research storage only.** Results and reports derived from real data are refused in a store
  rooted inside a source repository (`OPERATION_NOT_PERMITTED`). They go only to research storage
  or platform staging (DS-13).
- **Job results.** `to_run_result_payload(report, strategy)` gives the contract
  `finance/v1/run-result-payload` sections (`performance`, `accuracy`, `compute_cost`) for a job
  result.

## Holdout metrics record (REP-07)

`build_holdout_record` and `validate_holdout_record` produce `holdout-metrics-record-v1`.

The record holds these values and identifiers:

- `net_cumulative_return` and `max_drawdown`;
- `dataset_id`, `dataset_manifest_checksum` and the `holdout` bounds;
- `simulation_configuration` with its `simulation_configuration_id`;
- `cost_model` with its `cost_model_id`;
- `evaluator_version`, `image_digest` and `result_checksum`;
- `run_id`, `model_version`, `strategy`, `seed` and `synthetic`.

The values come from the evaluation result, so they equal the report's holdout column; the report
generator checks that. A record missing any comparability field is `VALIDATION_FAILED` naming it.
