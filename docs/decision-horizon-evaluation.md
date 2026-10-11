# Objective-aware evaluation of a recorded decision

`evaluate_portfolio_decision` retains its existing paper-accounting response and adds
`horizon_evaluation`. A daily loss is an observed outcome, not a verdict that PPO is
wrong. The new section compares the exact issued strategy and subsequent rebalances
with unchanged holdings, daily equal weight and fixed min-variance, mean-variance and
CVaR controls. Every path starts from the original shares and cash and uses identical
next-open, proportional-paper-fee assumptions. No current serving-policy pointer,
training job, broker action or strategy activation is used.

New recommendations preserve `provenance.evaluation_contract` before forward
observations exist. It fixes the objective, primary horizon, 1/5/21-session diagnostic
windows, traditional-control configurations, execution assumptions and evidence rule.
For PPO the primary horizon uses its recorded training-episode length (64 sessions in
the beta strategy); an episode does not guarantee performance or specify an investor's
ultimate holding period. A legacy full-window policy uses a clearly identified fixed
63-session review window. Traditional recommendations use their declared objective
horizon. Legacy decisions are explicitly `retrospective_legacy_protocol` and must not
be presented as predeclared experiments. Newly created backdated requests are
`retrospective_historical_request` when Platform issuance occurs after the first forward
session close. Their outcomes are ineligible for prospective skill evidence, even
though the evaluation protocol is preserved. The immutable Platform issuance time is
used, so repeated requests retain proposal idempotency.

Each window is `partial` until the requested number of completed forward sessions
exists, then `mature`. Maturity is descriptive: even a mature window is one realized
market path. Return, PnL, drawdown, volatility, turnover and costs are reported together.
Traditional objective diagnostics are realized-path analogues of the solve objective,
not calibrated expected risk. PPO reward diagnostics are daily components with the
evaluation-window drawdown reset; they are not a critic estimate. Exported historical
actors do not preserve learner gamma, so discounted reward is explicitly unavailable
unless gamma is actually recorded in the frozen bundle. It is never inferred from a
current configuration file.

The agent should explain a negative daily outcome through instrument contributions,
costs and execution evidence, and say when the strategy's primary horizon is still
immature. Persistent, independent out-of-sample underperformance motivates a modeling
experiment; this evaluator never declares a proven model defect or statistically
established optimality. The protocol requests at least three nonoverlapping mature
windows for a review. Aggregation belongs to the research-review layer; overlapping
1/5/21/64-session windows are not independent observations.

The response includes `full_policy_replay`, objective and protocol identity, source
portfolio revision, input snapshot, actor and implementation references, window metrics
and explicit limitations. Full sequential NAV, decision outputs, fills and accounting
reconciliation are persisted in the immutable analysis document's `internal.horizon_replay`
section; the bounded MCP response returns its checksum under `replay_evidence` and the
normal `analysis_ref`. Replay is bounded to the largest declared window, at most 252
sessions, to contain Lambda work and artifact size.

Replay fails closed when frozen implementation/actor identity or initial point-in-time
history changes, when aligned raw open/close bars are missing, or when unaccounted
splits/dividends occur. Saved market data is reused without online retrieval. Historical
provider revisions after issuance cannot be fully excluded without verifying each daily
snapshot vintage; this limit is disclosed. Counterfactual next-open fills are distinct
from the existing acceptance ledger's reference-price paper fills and from any training
simulation assumptions. News is contextual evidence, not proof of causation.

## Offline verification

Run from FinanceModel:

```sh
.venv/bin/pytest tests/unit/classical/test_horizon_evaluation.py tests/unit/classical/test_portfolio_decisions.py tests/unit/sim
openspec validate add-horizon-aware-policy-evaluation --strict
```

Tests cover preserved/legacy protocols, partial/mature horizons, original-book valuation,
original actor despite a changed current pin, exact original target reproduction,
point-in-time future-data independence, policy rebalancing versus first-allocation hold,
cost-consistent controls, unavailable corporate-action evidence, immutable storage and
bounded public responses. Beta deployment and live MCP acceptance are verified separately
by the release coordinator; they are not implied by offline tests.

Offline verification on 2026-10-10: 11 focused horizon tests passed; the combined
classical, RL and simulator suite passed 245 tests in 58.44 seconds using the final
contracts 1.6.0 wheel. Strict OpenSpec validation and source diff checks passed.
Beta verification on 2026-10-11 passed through both direct MCP and hosted AgentCore
using an isolated paper book. The hosted PPO result exactly matched the direct six-path
horizon evidence; immutable retrieval preserved it. Traditional min-variance replay also
passed. Backdated requests were explicitly retrospective/ineligible for prospective skill
evidence, incomplete 64-session horizons remained partial, and latest-snapshot requests
correctly returned no forward observations. No holdings were accepted or changed.

All Model beta pipeline actions passed for source
`3b95f8dffd549a4df297fcc24811f12aad9c07a4`, execution
`38b333be-9986-4825-b047-4c66eb5f05e9`, release
`rel_01M4M56RP6FZNARQ26EVNK1WEJ`. Raw acceptance evidence is retained locally at
`.worktmp/finplan-horizon-recursive/live-acceptance/` in the shared workspace; the
analyses and activity receipts are persisted in the beta services.
