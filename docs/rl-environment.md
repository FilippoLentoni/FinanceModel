# RL environment specification (PPO, SAC)

Task 1.4 of `add-learning-and-llm-strategies`; spec `rl-strategies`. **The values below are
configuration pending user review**, not findings. The specification is versioned
(`finplan-rl-env/1`, `finplan_model.rl.spec.EnvSpec`). The whole document, including the reward
formula, enters every trained policy's `configuration_id`, together with the algorithm
hyperparameters, the simulation configuration id, the training range and the instrument list. So a
changed coefficient means a new configuration and a new training run (RL-01, RL-08).

## Default specification

Used by `model_selection` (`config/<env>.json` `job_types.model_selection.protocol.rl.env`):

```json
{
  "window": 20,
  "return_scale": 50.0,
  "obs_clip": 5.0,
  "action_scale": 5.0,
  "step_sessions": 1,
  "decision_frequency": "daily",
  "reward": {"risk_penalty": 1.0, "drawdown_penalty": 0.5, "turnover_penalty": 0.0, "reward_scale": 100.0}
}
```

The `model_selection` protocol varies `risk_penalty` over `[0.5, 2.0]` (the RL grid) and keeps the
other terms fixed.

## State

The state uses point-in-time data only, observed at the close of the decision session `d`:

- `log_return_window`: the trailing 20 daily log close-to-close returns of each of the `n`
  instruments, oldest first. They are multiplied by `return_scale` and clipped to
  `[-obs_clip, obs_clip]`, giving `20 x n` values.
- `current_weights`: the current weight of each instrument plus the cash weight, marked at the
  decision close, giving `n + 1` values.

For the research universe (VOO, GOOGL, NFLX, AAPL, NVDA plus cash) the observation has
`20 x 5 + 6 = 106` values. A feature that the specification does not declare cannot be read: reading
one fails with `VALIDATION_FAILED`. A policy without `window + 1` visible closes holds (`no_effect`),
as the optimizers do while they lack history.

## Action

The action is a vector `a` in `[-1, 1]^(n+1)`. It maps to target weights by
`w = softmax(action_scale * a)` over the instruments plus cash. The result is long-only, every
weight is at most 1, and the weights sum to 1 with cash (RL-03). The simulator's constraint policy
then applies the same projection and record as for every other strategy. Evaluation is
deterministic (`deterministic=True`).

## Reward

The reward is a training signal only. Reports put it in `training_reward` and never in the
portfolio metrics (RL-06).

```
reward_t = reward_scale * ( ln(V[d_t+1] / V[d_t])
                            - risk_penalty     * sum over sessions s in (d_t, d_t+1] of ln(V[s]/V[s-1])^2
                            - drawdown_penalty * max(0, DD[d_t+1] - DD[d_t])
                            - turnover_penalty * turnover_t )
```

- `V` is the simulated net asset value after fees, spread and slippage.
- `DD[s] = 1 - V[s] / max V` up to `s` within the episode.
- `turnover_t` is the traded notional at the execution session divided by the value at the decision
  close.

## Episode and decisions

- **Training.** One episode is one pass over the training window, starting all in cash. The first
  decision falls at a seeded random offset in `[window, window + step_sessions)` sessions into the
  window. After that there is one decision every `step_sessions` (21) sessions, about one month.
- **Validation checkpoints.** These use the calendar rebalance sessions of the window (first
  session of each month), which is the common evaluator's own rule.

## Accounting (RL-02)

`finplan_model.rl.env` reuses the simulator's rules. Cash interest applies on every session after
the first. Execution happens at the next session's open (`next_open`). Sells come first. Fills are
capped at `participation_cap x volume`, with the remainder cancelled. Every fill pays the
proportional fee, the fixed fee, the half-spread and linear-participation slippage. Buys are scaled
down to the available cash, an instrument without a bar does not trade, and
`apply_constraints` applies.

A unit test drives the same scripted policy through the environment and through
`finplan_model.evaluate.evaluate` and requires the same NAV. The final train, validation and test
numbers of every policy still come from the common evaluator.
