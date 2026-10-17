# Bounded recursive research in beta

`run_recursive_improvement` persists an immutable cycle and every review/iteration in the
existing S3 classical-analysis store. It reviews accepted/rejected decisions, stored user
feedback, observed performance, horizon evaluations, dated primary-research metadata and
previous experiment results. Result/evidence reviews refresh those inputs. Approval or
preflight without new feedback preserves the latest reviewed experiment proposal and its
immutable review identifier; it cannot silently revert to the cycle's original profile.
External prose is data;
it cannot execute code, change the budget, or authorize a strategy.

The states are `evidence_review`, `awaiting_experiment_approval`, `experiment_running`,
`proposal_ready`, and `stopped`. The cycle freezes its portfolio, initial plan/universe and
maximum iterations (default three; hard maximum three). Conditional claims serialize a
launch. Repeating its idempotency key returns the same job; recovery after a crash reuses
the frozen experiment review and the experiment API's idempotency key.

The existing weekly beta schedule advances at most one eligible PPO/classical cycle after
its first explicitly confirmed job. A preview alone never authorizes scheduled compute.
Confirmed cycles allow released CPU experiments within the same
$0.50/week, $2/month and $50 project limits; one compute lease is shared with other jobs.
A week already reserved waits until a later invocation. GPU and external-vendor cycles
require explicit review and cannot displace an eligible CPU cycle. They never launch from
the weekly controller. No cycle changes holdings or activates a strategy.

## Candidate generation and evidence

Released PPO profiles are real model-selection jobs: state features (trailing log returns,
market summaries, current weights and drawdown); turnover-aware reward and limited
rebalancing; or a longer episode/discount horizon. Each compares three PPO seeds as an
ensemble against traditional optimizers and cash, buy-and-hold and equal-weight controls.
Fresh cost/turnover/horizon feedback and the previous experiment's metrics select the next
profile. These are bounded ablations, not automatically invented neural architectures.
Traditional cycles retain the released covariance/lookback/risk presets.
All displayed and submitted candidate configurations rebalance daily. Weekly scheduling
controls how often research runs, not the investment decision frequency. Turnover feedback
changes PPO rewards or classical covariance/risk hypotheses while retaining daily decisions.

The controller stops for failed jobs, no improvement on the declared return/drawdown gate,
reused holdout evidence, the iteration bound, or budget restrictions. A passing gate yields
a proposal requiring human activation and fresh forward validation. Reusing an old test
period may support research but never becomes new promotion evidence.

Horizon review counts only mature, prospective, nonoverlapping windows with the same
configuration and evaluation-protocol checksum. Retrospective/legacy decisions and daily
losses do not establish model failure. Three independent underperforming windows trigger
a hypothesis for review, not causal proof or automatic retraining/promotion.

## Calls

Preview: `run_recursive_improvement({"query":"PPO turnover and missing state features",
"dry_run":true,"max_iterations":3})`. Resume the returned `cycle_id` with new `feedback`.
An explicitly authorized launch uses the same cycle with `dry_run:false`,
`confirmed_by_user:true` and a stable `idempotency_key`. The response includes lineage,
the frozen proposal, evidence references, cost estimate and any run identifier.

Qwen/Jev queries return a matching typed `submit_experiment` dry-run request rather than
an unrelated classical benchmark. A hosted confirmed launch resumes the recursive cycle,
which owns the run lineage; a standalone direct `submit_experiment` remains an independent
advanced experiment. GPU/vendor approval is a further control-plane step.
These requests freeze a limited 21-decision daily pilot over 22 aligned saved market sessions,
retaining earlier history for features. They are not full-year or untouched-holdout results.
Frozen previews from the older monthly protocol fail closed and require a fresh review;
approval never silently changes their configuration or dates. This correction is deployed
in beta release `rel_01M4M6Q83GFWVT2VSDA0J7NHNP`, source
`d6edabee2a8abc18d0876bd84cde45d404db6d86`. All beta pipeline gates and live daily
PPO/Qwen/Jev dry-run requests passed. Paid inference remains separately approval-gated.

## Deliberate remaining work

Automatic source-code rewriting, arbitrary new features/algorithms, unbounded literature
research, automatic production activation and proof of investment profitability are not
implemented. Those require reviewed versioned changes and prospective evaluation. The
controller supplies concrete evidence/proposals and can evaluate released candidates; it
does not claim a generally autonomous software scientist. See [LLM benchmarks](llm-benchmarks.md)
for exact identities, readiness and paid-validation status.
