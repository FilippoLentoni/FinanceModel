# Design

## Context

See proposal.md (Why). The following come from `add-research-job-foundation` and are reused unchanged:
- the job API, `dry_run` and purposes;
- the CPU image and the registry;
- category pre-flight checks and the auto-approve threshold;
- the concurrency lease (`ml.m5.xlarge` processing quota 1, shared by environments);
- run-output staging and benchmark reporting.

The platform change `add-research-universe-and-daily-loop` provides contracts 1.1.0, approved universe snapshots and the trigger. Models run only in the deployed stack (decision 16).

## Goals / Non-Goals

**Goals:**
- Zero compute until a strategy is selected.
- The daily job uses the same evaluator and strategy code the user saw in experiments.

**Non-Goals:** new experiment or report APIs, automatic selection, and bias correction.

## Decisions

### M1. Caller-scoped job kinds
The handler maps principals to kinds, mirrored in the API resource policy:
- the platform trigger may submit only `daily_recommendation`;
- the tool `submitter` role (with an on-behalf-of user) and the operator may submit only research kinds.

Purpose-only gating was rejected because the purpose does not identify the caller.

### M2. Strategy resolved at submit, frozen on the run
The submit handler reads the SSM key, validates it, and stores the triple on the run record. The container reads the triple from the run record, so a selection change mid-run has no effect on the running job.

### M3. Selection through a dedicated Lambda role
Only the selection Lambda's role holds `ssm:PutParameter`/`DeleteParameter` on the key. Evidence is a succeeded `research` or `holdout_evaluation` run on `equity-etf-daily`, found through the run store. Clear deletes the parameter, so the platform sees it as absent.

### M4. Cost
- The configured price is about USD 0.23 per hour for `ml.m5.xlarge` processing in us-east-2, so the 0.5 h cap gives at most about USD 0.12 per job. The build check allows USD 0.15.
- The daily job must sit under the auto-approve threshold, because the user's gate for recommendations is publication, not compute.

### M5. Strategy comparison in the job result (beta finding)
- Beta showed that `get_experiment_result` for `run_benchmark` exposed only the configured strategy's metrics; the controls, the weights and the metric units lived only in the research-workspace artifact, which callers cannot read (by design).
- The `run_backtest` and `run_benchmark` results now carry `payload.benchmark` (`finplan_model.jobs.comparison`): one row per evaluated strategy (configured strategy first, then each control) with `total_return`, `cagr`, `ann_volatility`, `sharpe`, `max_drawdown` (positive fraction), `turnover`, `transaction_cost` and `transaction_cost_fraction`, plus `final_weights`/`average_weights` per instrument with the cash weights, the evaluation window, `base_currency`, `initial_capital`, the risk-free assumption and a `units` legend. Weight maps are capped at 40 instruments (`weights_truncated_to`).
- Placement: contract 1.1.0 `finance/v1/run-result-payload.json` is an open object (no `additionalProperties: false`), while its `performance` section admits numbers only, so the block is a new payload key; no contract gap and no contract change. `payload.performance` is unchanged.
- Units (labels fixed, math unchanged): turnover is cumulative traded notional (buys + sells) divided by the initial capital, so the beta value 19.6 means 19.6x the USD 100,000 starting capital was traded; the reported 392 is USD of fees (1 bp) plus half spread (1 bp) on that notional (0.39 % of the starting capital).

## Risks / Trade-offs

- [The daily job and a user benchmark contend for the single quota slot] → the existing lease queues one. The platform's 45 minute wait covers one queued 30 minute job.
- [Results flattered by hindsight-selected tickers] → mandatory report sections, disclosures in staged manifests, and publication only by the user.
- [Deployed tests cost money] → at most one benchmark plus one daily job per beta or gamma suite (about USD 0.25).

## Migration Plan

1. Pin contracts 1.1.0.
2. Deploy to beta with no strategy and verify zero runs.
3. Run the deployed tests, then promote to gamma and to prod.
4. Leave prod unselected until the user chooses a strategy.

Rollback: redeploy the previous release. Clearing the key stops daily runs.
