# Offline model selection (`model_selection`)

This page covers decision 27 (2026-10-08) and the `add-learning-and-llm-strategies` change. Each
experiment runs only when the user asks for it, through the agent or the job API. Nothing schedules
it. The protocol in `config/<env>.json` (`job_types.model_selection.protocol`) is **configuration
pending user review**. Each run freezes a copy of it at submission and records its `protocol_id`.

## Protocol

| Item | Value |
|---|---|
| Data | **2025-2026 only.** The snapshot is cut to `data_start` = 2025-01-01, and nothing earlier is visible to any strategy. |
| Train / calibrate | 2025-01-01 .. 2025-12-31 |
| Validate (tuning, model choice) | 2026-01-01 .. 2026-06-30 |
| Untouched test | 2026-07-01 .. latest session of the snapshot, evaluated **once**, after the selection is frozen |
| Universe | equity-etf-daily research universe: VOO, GOOGL, NFLX, AAPL, NVDA, plus cash (the residual weight) |
| Evaluator | `finplan_model.evaluate.evaluate`, with the same simulation configuration for every family: costs (1 bp fee, 1 bp half-spread, linear-participation slippage), monthly rebalance, next-open execution, long-only, max weight 1 |
| Selection rule | highest **validation** Sharpe ratio, net of costs. Ties go to the first entry in the declared grid and family order. A rule that names the test split or a holdout fails with `OPERATION_NOT_PERMITTED`. |

### Families and hyperparameter grids

| Family | Strategy | Grid, chosen on validation | Fixed |
|---|---|---|---|
| Controls | `cash`, `buy_and_hold`, `equal_weight` | none | none |
| Traditional | `min_variance` | lookback 20, 60, 120 sessions | Ledoit-Wolf covariance, cash at its minimum |
| Traditional | `mean_variance` | lookback 20, 60, 120 x risk aversion 1, 5, 20 | Ledoit-Wolf covariance, historical mean, free cash |
| Traditional | `scenario_cvar` | lookback 60, 120 x alpha 0.90, 0.95 | 500 bootstrap scenarios, seed 0, minimum CVaR |
| RL | `ppo` | reward `risk_penalty` 0.5, 2.0 | 30,000 steps, lr 3e-4, n_steps 256, batch 64, 10 epochs, gamma 0.9, GAE 0.95, clip 0.2, MLP 64x64; validation checkpoint every 2,048 steps, patience 5 |
| RL | `sac` | reward `risk_penalty` 0.5, 2.0 | 8,000 steps, lr 3e-4, buffer 50,000, learning starts at 500, batch 128, tau 0.005, gamma 0.9, MLP 64x64; validation checkpoint every 1,000 steps, patience 4 |

- **Seeds.** Each RL configuration trains with seeds 0, 1, 2, 3 and 4 (at least 3).
- **Torch and determinism.** Training uses CPU torch on one thread. Evaluation is deterministic.
- **RL choice.** For each RL algorithm, the job picks the checkpoint per seed by validation Sharpe
  (the untrained policy is the step-0 checkpoint). It then picks the configuration with the highest
  mean validation Sharpe across seeds, and the best seed within that configuration.
- **Environment.** State, action and reward are defined in [rl-environment.md](rl-environment.md).

## Steps inside the job

1. **Prepare the data.** Restrict the market data to 2025 onward and resolve the three windows to
   sessions.
2. **Controls.** Evaluate the controls on train and validation.
3. **Traditional optimizers.** Evaluate every grid point on validation, keep the best one, and
   evaluate it on train.
4. **RL training.** For each RL algorithm x reward configuration x seed:
   - Train on the training window, with validation checkpoint selection.
   - Store the policy as an `rl_policy` artifact with its SHA-256 checksum, seed and
     `configuration_id`.
   - Evaluate the stored policy on validation with the common evaluator.
   - Evaluate every seed of the chosen configuration on train as well.
5. **Freeze the selection.** Freeze and checksum the selection record (`selection_checksum`). It
   contains the validation ranking, the chosen hyperparameters and the selected candidate, which is
   the best candidate other than the incumbent.
6. **Test, once.** The evaluator refuses the test window until step 5 is done. Evaluate every
   family's chosen model, every seed of the chosen RL configurations and the incumbent on the test
   window.
7. **Promotion check, criteria v1 (decision 15c).** The candidate passes only if its net-of-cost
   test return is strictly higher than the incumbent's and its test maximum drawdown is no worse.
   - The incumbent is the production strategy when one is set and comparable; otherwise it is
     `buy_and_hold`.
   - A pass still needs the user's approval. Nothing is promoted or published by this job.

## Result

| Section | Contents |
|---|---|
| `payload.benchmark` | the **test** comparison, in the standard `finplan.benchmark_comparison/1` shape |
| `payload.model_selection.comparison` | `train`, `validation` and `test` comparisons. Every row adds `family`, `params`, `seed` (RL), `selected` and `incumbent`. |
| `.traditional` | every grid point's validation metrics and the chosen parameters |
| `.rl.<algo>` | the grid (mean validation Sharpe per configuration, with checkpoint curves), the chosen configuration and seed, every seed's train, validation and test metrics, and `seed_statistics` (mean, standard deviation, minimum and maximum per split) |
| `.training_reward` | shaped RL training rewards only (episodes, mean, first and last decile, best step, stop reason). These are never portfolio performance. |
| `.selection` | the frozen selection record and `selection_checksum` |
| `.promotion_check` | the criteria v1 result against the incumbent |
| `.test_access` | test evaluations in this run and earlier succeeded runs on the same dataset (`test_reuse`) |
| `.caveats` | thin RL training data, hindsight and survivorship, the short single test period, validation reuse, and reward versus performance |
| `.compute` | seconds per phase, wall seconds, instance, upper-bound and estimated actual USD |
| `payload.bias_section` | "Hindsight and survivorship bias", reproducing the snapshot's disclosures (mandatory; fails closed) |

The artifacts are the evidence JSON (every evaluation result, the protocol and the selection) and
every trained policy as an SB3 zip.

## Caveats reported with every result

- **Thin data for RL.** About 250 training days (one year, about 12 monthly decisions per episode)
  is thin for reinforcement learning. Expect large seed-to-seed variance and overfitting to 2025, and
  treat RL results as exploratory.
- **Hindsight and survivorship.** The universe was chosen in 2026 knowing that these names did well,
  and it contains no failed or delisted companies. Every family benefits, so compare families with
  each other rather than with the market.
- **Short test period.** The test period is about three months and a single path, so its ranking is
  not statistically significant. Live paper trading afterwards is the forward test.
- **Validation reuse.** Validation picks grid points, checkpoints, reward configurations and seeds,
  so validation scores are optimistic.

## Compute and cost

| | |
|---|---|
| Job | **One CPU SageMaker Training job** (`ml.m5.xlarge`, 4 vCPU, 16 GiB). RL learners never run in Lambda or CodeBuild: the offline test harness refuses learner runs above 512 steps. The account's Training quota for `ml.m5.xlarge` is 15. Its Processing quota is 1 and is left to the other jobs. |
| Cap | 3,000 s `MaxRuntimeInSeconds`. RL training stops `reserve_seconds` (600 s) before the cap, keeps the best checkpoints so far, and records any seeds it skipped. |
| Upper-bound estimate | `price x 3000 s + storage` = USD 0.23/h x 0.833 h + USD 0.01, about **USD 0.20**. That is under the beta/gamma auto-approve threshold of USD 0.25 (decision 24), so it starts without approval there. Prod always asks the approver. The build check keeps `0.28 x 3000/3600` at or below USD 0.25. |
| Expected runtime | About 20 PPO and SAC runs of 20 to 60 s each, plus evaluations and image start-up. That is roughly 15 to 25 min, about **USD 0.06 to 0.10** per experiment, drawn from `cpu_research` (USD 7). |

## Running it

Submit with `job_type: model_selection`, `purpose: research` and
`configuration.payload.strategy: model_selection`. The payload carries the universe (with
`USD_CASH`), monthly rebalance and the constraints. Name the approved research-universe snapshot by
`input_snapshot_id`. A `dry_run: true` submission returns the estimate without recording anything.
