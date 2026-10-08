# Proposal

## Why

The user decided (decisions 17–22, 2026-10-08) that a scheduled daily job runs only the user-selected production strategy, and that experiments over the new research universe run only on request. On-demand experiments, benchmarks, reports, the job interface and cost controls already exist in `add-research-job-foundation` and are not re-specified. Three things are missing:
- a `daily_recommendation` job kind;
- the per-environment production-strategy setting;
- acceptance of the multi-instrument `equity-etf-daily` dataset, with bias flags, by the existing on-demand experiment kinds.

## What Changes

- **`daily_recommendation` job kind.**
  - It is accepted only from the platform trigger role, with purpose `production_candidate`, on an approved `equity-etf-daily` snapshot.
  - It runs exactly the strategy in `/finplan/<env>/financemodel/config/production-strategy`. A request cannot override it, and with no strategy set it fails with `PRECONDITION_FAILED` and starts nothing.
  - It stages one bundle through the existing run-output staging, carrying the snapshot's `bias_disclosures`.
  - Cost: one `ml.m5.xlarge` with a 30 minute hard cap (at most about USD 0.12 per job), budget category `cpu_research` with the existing pre-flight check. Its estimate must fall under the auto-approve threshold.
- **Production-strategy setting.**
  - Get, set and clear operations on the job API. Set and clear require an on-behalf-of user with explicit confirmation.
  - Set validates against the strategy registry: the strategy is registered, deployed in the environment, not retired, supports `equity-etf-daily`, has at least one succeeded research run on that dataset in the environment, and is `promoted` if it belongs to a learning family.
  - FinanceModel's selection role is the only SSM writer, and every change is audited.
- **Universe input for the existing on-demand experiments.** The existing `backtest` and `benchmark` kinds accept approved `equity-etf-daily` snapshots. They use `adj_close` returns and the snapshot's cash assumption. Existing benchmark reports gain a mandatory "Hindsight and survivorship bias" section copied from the snapshot. These runs stay user-initiated only, with no schedule.
- **Out of scope:** new experiment or report APIs, automatic strategy selection, GPU, RL or Jev in the daily kind, and live trading.

## Capabilities

### New Capabilities

- `daily-recommendation-job`: the `daily_recommendation` kind, its caller and strategy restrictions, its no-strategy refusal, staging with disclosures, and its cost cap.
- `production-strategy-setting`: get/set/clear of the production strategy, registry validation, user confirmation, single-writer key and audit.
- `universe-research-input`: acceptance of `equity-etf-daily` by existing experiment kinds, the return and cash basis, mandatory bias sections in reports, and the no-schedule rule.

### Modified Capabilities

None. `openspec/specs/` is empty. This change adds to `add-research-job-foundation` without altering its requirements.

## Impact

- **Code (future):**
  - a `run_daily_recommendation` entry point in the CPU image;
  - job-API handler branches;
  - a selection Lambda with its own role;
  - universe dataset preparation and a report section;
  - a pin of contracts 1.1.0 (FinancialPlanning `add-research-universe-and-daily-loop`).
- **Consumers:** the platform trigger (submit and status), and the FinanceLambdasTool `production_strategy` tool (`add-approval-and-strategy-tools`).
- **AWS:** no always-on resources. The prod daily job costs about USD 2.50 per month, and only after the user selects a strategy.
