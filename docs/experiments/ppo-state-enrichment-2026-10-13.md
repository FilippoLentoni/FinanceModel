# PPO state enrichment: measured beta study — October 13, 2026

Five matched state profiles, five seeds each. The July 1–October 7, 2026 evaluation period has 69 sessions and was already inspected. These are development results; promotion remains blocked. No serving policy was activated.

| State | Dimensions | Net return | Sharpe | Max drawdown | Turnover | Trading costs (USD) | Seed return SD (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 37 | 6.2897% | 1.5779 | 5.4806% | 2.8730 | 57.4604 | 0.6432 |
| Risk | 48 | 5.9172% | 1.3838 | 5.9478% | 3.9212 | 78.4248 | 1.0329 |
| Regime | 43 | 5.9828% | 1.4185 | 5.9354% | 3.0945 | 61.8905 | 1.2164 |
| Sequence | 62 | 4.5003% | 1.2690 | 5.0668% | 3.1685 | 63.3700 | 1.3279 |
| Combined | 79 | 4.9215% | 1.3651 | 4.6050% | 4.9493 | 98.9870 | 1.5651 |

Return/Sharpe/drawdown are allocation-ensemble results. Seed standard deviation measures initialization variability on one market path, not market uncertainty.

## Common controls

| Strategy | Net return | Sharpe | Max drawdown | Turnover |
|---|---:|---:|---:|---:|
| cash | 0.0000% | undefined | 0.0000% | 0.0000 |
| buy_and_hold | 5.9377% | 1.501963 | 5.2318% | 0.9998 |
| equal_weight | 6.4473% | 1.612134 | 5.2599% | 1.8137 |
| mean_variance | 5.3142% | 0.877567 | 13.6568% | 15.7319 |
| min_variance | 6.7194% | 1.897044 | 4.0581% | 3.3632 |
| scenario_cvar | 4.5949% | 1.568887 | 2.9769% | 8.6120 |

## Protocol and provenance

Universe: AAPL, GOOGL, NFLX, NVDA, VOO. Training: 250 sessions in 2025; validation: 123 sessions January–June 2026. Equal action transform, reward, episode length, nominal 30,000-step training cap, checkpoint/early-stopping rule, seeds 0–4, splits and evaluator costs. Actual executed steps can differ with early stopping and rollout completion. Initial capital USD 100,000; daily next-open simulation. The aggregate JSON accompanying the paper retains seed results, training diagnostics, evaluator identities and compute estimates.

Risk adds trailing pairwise correlations and portfolio volatility. Regime adds an equal-weight-universe proxy, volatility, dispersion and breadth; it is not a separate market-index feed. Sequence adds five recent log returns per instrument. Combined adds all three. Feature ordering, clipping and point-in-time visibility are versioned.

News collection/extraction and strict as-of joins are implemented, but no frozen historical news corpus exists for this window. News training is explicitly ineligible. The live provider check returned unavailable/zero usable records; absence was not converted into observed neutral sentiment. Headline lexicon scores are uncalibrated hypotheses.

Project ceiling: USD 100. This round is bounded at USD 5 incremental; five preflight job estimates total USD 0.3375 upper bound. Container estimates omit billed startup; actual SageMaker billable usage is retained separately. No GPU/vendor training launched.

## Immutable runs

- recursive_ppo_state_baseline: `run_01M4V0D1K5G70G6WZFSPA266DV`
- recursive_ppo_state_risk: `run_01M4V0V4C0WK18AJZJGY90555J`
- recursive_ppo_state_regime: `run_01M4V1C94J2BDH816FS5ZV6Y52`
- recursive_ppo_state_sequence: `run_01M4V1XDQ6N8QZQ2S6KC10S585`
- recursive_ppo_state_combined: `run_01M4V2AX93R5DJNMZBBXEKKM06`

Snapshot: `snap_01M4EDFP5T1KYF2ABGME115689`.

Release remains beta-only; Gamma/production transitions are restored to their prior state after owned executions are stopped. Legacy actors and decision replay retain their original implementation identities.

## Findings and limits

The baseline has the highest ensemble return and Sharpe among the five PPO states. Risk and regime additions lowered return/Sharpe and increased drawdown. Sequence and combined reduced drawdown while sacrificing return/Sharpe; combined also increased turnover. This is not evidence that additional information is universally harmful: capacity, optimization and short training history can interact with state dimension. No policy was selected for activation from these results.

Minimum variance has higher return and Sharpe than every PPO ensemble on this path. Scenario CVaR has lower drawdown than all invested alternatives shown, at lower return and greater turnover than the baseline PPO. Retain the alternatives rather than reducing performance to a single ranking.

All five jobs completed using the identical image digest. Total SageMaker billable time was 1,741 seconds; at the verified Ohio ml.m5.xlarge training rate of USD 0.23/hour, training charges are approximately USD 0.11123. This includes job billable startup and differs from the container-only estimates; it excludes CodeBuild, storage, APIs and narration and is not an invoice.

Further research should first expand point-in-time market history and prospective evaluation, then vary representation/capacity under a separately frozen protocol. Collect news vintages before testing news features. Do not tune repeatedly on this development path and describe it as fresh test evidence.

## Seed return dispersion

| State | Mean return | SD (pp) | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Baseline | 6.2252% | 0.6432 | 5.5182% | 7.1841% |
| Risk | 5.8657% | 1.0329 | 4.4387% | 7.1642% |
| Regime | 5.9240% | 1.2164 | 3.9305% | 7.2071% |
| Sequence | 4.4542% | 1.3279 | 2.3338% | 5.8965% |
| Combined | 4.8713% | 1.5651 | 3.4713% | 7.0322% |

Evaluator and input identities match exactly across profiles:

```json
{
  "input_snapshot_id": "snap_01M4EDFP5T1KYF2ABGME115689",
  "dataset_checksum": "sha256:abd202bf7dcc78c56bd0a0b6da99080028fe3cee64f0f115a414ed3f42237546",
  "evaluator_version": "1.0.0",
  "evaluation": {
    "cost_model_id": "cfg_77cf03025f5cc901eb15d6242cb171428505bda50cbc02ae55f2163af185093d",
    "evaluator": "finplan_model.evaluate.evaluate (common evaluator; same simulation configuration for every family)",
    "execution_timing": "next_open",
    "long_only": true,
    "max_weight": 1,
    "rebalance_frequency": "daily",
    "simulation_configuration_id": "cfg_e16546f8439c890b881f29a1aff83435e920255731fdb4963d85849c9c4baa35"
  }
}
```

## Release and live verification

All four new OpenSpec changes pass strict validation; all 16 tracked tasks are complete. Full offline Model pre-build gates passed, including unit/contract tests. Model, MCP Lambda targets and Agent passed cloud build and beta integration. Owned releases were stopped before Gamma and original transition settings restored. Final read-only checks confirm the selected advisory policy, production-strategy reference (still absent) and research-plan reference are unchanged.

Post-release direct MCP checks passed legacy frozen-decision explanation, enriched risk-state preview, explicit news-training ineligibility and truthful unavailable news collection. Result retrieval retained the matched profile, five-seed statistics and reused-data warning. Both hosted natural-language checks passed: baseline state preview and news-training refusal. No recommendation was accepted and no holdings were changed by this study.

The generated working paper is at `papers/agentic-financial-planning/main.pdf` in the shared workspace, with LaTeX source, aggregate JSON, figures and an evidence index alongside it. It compiles to 10 pages without LaTeX warnings or unresolved references.
