# Selected strategy serving in beta

Offline experiments and serving share a frozen configuration, not a training process. The beta
`prepare_policy` CPU job reads an existing succeeded `model_selection_experiment`, freezes its
validation-selected strategy and exports actors without training. PPO/SAC require the complete
selected seed ensemble with checksum-verified source checkpoints. Classical strategies freeze the
evaluated parameters. Exported bundles are immutable `policy_inference` artifacts.

Submit `prepare_policy` with a finance configuration containing `strategy: prepare_policy`, the
source universe, `policy_source_run_id` and `policy_strategy_id`. The supported selectors are PPO,
SAC, cash, buy-and-hold, equal weight, min-variance, mean-variance and scenario-CVaR. Run a dry run
first and use the existing research budget gate. Unsupported or incomplete sources fail explicitly.

After export succeeds, the beta operator calls `PUT /v1/advisory-policy` with
`export_run_id` and `confirmed_by_user: true`. This updates only
`/finplan/beta/financemodel/config/advisory-policy`; the production-strategy selection and promotion
checks remain separate. Activation is audited. Clearing/replacing this pointer is the serving
rollback; no recommendation updates it.

The release publishes `/finplan/beta/financemodel/api/strategy-function-ref`. The MCP adapter invokes
that same-environment Lambda directly, avoiding the job API's shorter request path. Its timeout is
270 seconds. Its role reads only the selected exported artifacts, the relevant SSM references and
approved snapshot APIs. It cannot train, write model selection or create jobs.

`recommend_portfolio` accepts `{}` to use the saved beta paper portfolio referenced by
`/finplan/beta/financialplanning/config/research-plan-ref` and the latest approved research-universe
snapshot. An optional `portfolio_id` selects another saved paper book. The platform operator must
initialize the book first; missing state returns a typed error without creating holdings.
The invocation contract is `tools/recommend-portfolio-invocation-request`; the original explicit
request contract remains valid for existing callers.

For a reproducible experiment, supply an approved `input_snapshot_id`, an `as_of` completed session
and `holdings` containing weights by instrument, cash weight, positive portfolio value and high
watermark. Snapshot/date overrides must be supplied together; `portfolio_id` and `holdings` cannot
be combined. Explicit holdings must sum to one and match the frozen universe. Data is filtered by availability
at the decision time, and the decision must follow training and validation selection. The tool
returns target weights, buy/sell/hold weight deltas, indicative notionals and fractional share changes,
current and target quantities, reference close prices and dates, turnover, constraint
outcome, source experiment/configuration/export IDs and snapshot/artifact checksums. Rejected
trades hold the existing portfolio, as in the common offline evaluator.

This release is beta advisory/paper only. PPO has not met production promotion criteria. No return
forecast is published by the allocation artifact; a backtest return is not a forecast. No trade is
executed. Saved quantities and cash are marked to unadjusted completed closes; actor features retain
the adjusted-price basis used in training. Drawdown uses the greater of the stored high watermark
and every marked portfolio value since the saved state date. Missing aligned history is rejected.
Responses expose the saved portfolio ID/revision, book date, valuation date, cash and value.
Recommendations leave this state unchanged. Fills, splits, dividends and cash flows require an
explicit paper-book update; the service does not perform automatic brokerage accounting. Explicit
mode uses caller-supplied holdings and high watermark. The current historical universe carries
hindsight-selection bias. Data dates are displayed; a recommendation uses the latest approved
completed data available, which can lag the current calendar date.

The four-pipeline lifecycle and future bounded research controller are described in the
FinanceAgent repository's `docs/strategy-lifecycle.md`. Supported artifact changes do not require
releasing Agent/Tools again; new algorithm implementations or feature transforms do require a
Model release and serving-parity checks.
