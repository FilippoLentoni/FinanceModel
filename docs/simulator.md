# Paper execution simulator and common evaluator

Task 2.9 of `add-research-job-foundation`. Spec: `paper-execution-simulator` (SIM-01 to SIM-09).
Design: D5 (evaluator version), D7 (simulator conventions). Code: `src/finplan_model/sim/` and
`src/finplan_model/evaluate/`.

> **Every default on this page is configuration pending user review (open question FM-OQ-5),
> not a market fact.** The values live in `simulation_defaults` of `config/<env>.json` and can be
> overridden per run. Every result records the full simulation configuration, its
> `simulation_configuration_id` and its `cost_model_id`, and comparisons across different settings
> are refused.

## One evaluator for every strategy family

`finplan_model.evaluate.evaluate(strategy, market, config, ctx=...)` is the only way results are
produced. Controls, classical optimizers and every later family (RL, LLM swarm, Jev) go through it.
Each result carries an evaluator identity:

| Field | Meaning |
|---|---|
| `evaluator_version` | Semver of the simulator and evaluator (`1.0.0`); bumped on any change that can alter trades or metrics |
| `image_digest` | Digest of the container image that ran the evaluation (from the run context) |
| `dataset_id`, `dataset_checksum` | The prepared dataset the market data came from |
| `simulation_configuration_id` | `cfg_` + SHA-256 of the canonical simulation configuration |
| `cost_model_id` | `cfg_` + SHA-256 of the fee, spread and slippage models alone |

`assert_comparable(results)` refuses a benchmark comparison with `VALIDATION_FAILED` (pointer
`/runs`) and lists the differing settings by name: `evaluator_version`, `image_digest`,
`dataset_id`, `dataset_checksum`, `execution_timing`, `fee_model`, `spread_model`,
`slippage_model`, `liquidity_model`, `constraint_set`, `constraint_policy`, `rebalance_frequency`,
`initial_cash`, `cash_rate_annual` (SIM-01).

`replay(stored_result, market, config)` re-runs the stored strategy outputs and fails with
`INTERNAL` (`replay_mismatch`) unless trades and metrics are identical (SIM-08). Each result has a
`result_checksum` (SHA-256 of the canonical JSON of its numeric evidence).

## Strategy interface

A strategy has a `name` and a method `decide(view, holdings)`.

- `view` is a `PointInTimeView`. It exposes only bars whose session is on or before the decision
  session **and** whose `available_at` is at or before the decision time (`history`, `latest`,
  `price_matrix`, `returns`). A late-arriving bar is invisible until the first decision after it
  arrived. `intraday_partial` observations are never used as daily bars.
- `holdings` is a `HoldingsView` (shares, cash, value and weights at the decision, valued with
  prices visible at the decision time).
- The return value is **target weights per instrument plus cash**: a `TargetWeights`,
  `{"weights": {"SPY": 0.6}, "cash": 0.4}` or the contract allocation shape
  `{"weights": [{"instrument_id": "SPY", "weight": 0.6}], "cash_weight": 0.4}`. An optional
  `solution_status` (`optimal`, `feasible`, `infeasible`, `unbounded`, `no_effect`,
  `not_applicable`) reports the optimizer outcome of that decision. `infeasible` and `unbounded`
  keep current holdings (`fallback_hold_current`); `no_effect` means "no change wanted" and also
  keeps current holdings without placing orders (`hold_no_effect`; buy-and-hold after its initial
  target, an optimizer without enough history). Holds are neutral in the run-level status.
- Order-shaped output (`orders`, `quantity`, `shares`, `side`, `trades`, `notional` ...) fails the
  run with `VALIDATION_FAILED` (SIM-02). The simulator, not the strategy, produces trades.

## Step order

At each session `s` of the evaluation window:

1. **Cash interest**: previous cash accrues `(1 + cash_rate_annual)^(1/periods_per_year) - 1`.
2. **Execution** of the decision made at the previous session (or of carried-over remainders) at
   the executable price of `s`. Orders are sized from the target weights and the portfolio value
   at the execution prices. Sells execute before buys. Buys are scaled down when cash after costs
   would go negative (recorded as `cash_limited`).
3. **Mark to market** at the close of `s` (a missing bar keeps the previous mark).
4. **Reconciliation** (SIM-09):
   `V_s = V_{s-1} + interest + sum(q_prev * (P_exec - P_prev)) + sum(q_new * (P_close - P_exec)) - costs`
   within `reconciliation_tolerance`. A break fails the run with `INTERNAL`
   (`reconciliation_break`), naming the failing `step` and `session_date`.
5. **Decision** after the close of `s`, on rebalance sessions only and never on the last session of
   the window.

## Configuration reference

| Setting | Default | Allowed | Meaning |
|---|---|---|---|
| `venue` | `paper` | `paper` only | Any other venue (a broker, exchange or wallet) is `OPERATION_NOT_PERMITTED`; so is any key that looks like live-trading credentials (`api_key`, `secret`, `broker`, `wallet`, ...) (SIM-07) |
| `execution_timing` | `next_open` | `next_open`, `next_close` | Fills for a decision made after the close of day d use the open (or close) of the next session after d. Same-bar values (`same_bar`, `same_close`, `same_open`) are `VALIDATION_FAILED` (SIM-04) |
| `rebalance_frequency` | `monthly` | `daily`, `weekly`, `monthly`, `quarterly` | Decisions on the first session of each period (known from the calendar alone) |
| `initial_cash` | `100000.0` | > 0 | Starting cash in `base_currency` |
| `base_currency` | `USD` | ISO code | |
| `cash_rate_annual` | `0.0` | -0.5 to 1 | Interest on cash; the `cash` control's return equals this rate over a year |
| `periods_per_year` | `252` | 1 to 366 | Sessions per year for accrual and annualization |
| `fees.proportional_bps` | `1.0` | >= 0 | Proportional fee on each fill's notional |
| `fees.fixed_per_trade` | `0.0` | >= 0 | Fixed fee per fill |
| `spread.half_spread_bps` | `1.0` | >= 0 | Half-spread paid on each fill's notional |
| `slippage.model` | `linear_participation` | `none`, `linear_participation` | Slippage in bps = `coefficient_bps x participation`, where participation = fill quantity / session volume |
| `slippage.coefficient_bps` | `10.0` | >= 0 | |
| `liquidity.participation_cap` | `0.05` | (0, 1] or `null` | Maximum fill as a fraction of the execution session's volume (SIM-06) |
| `liquidity.unfilled` | `cancel` | `cancel`, `carry_over` | Capped remainder is cancelled, or retried at later sessions until filled or superseded by the next decision (`cancelled_superseded`); every case is recorded |
| `constraints.long_only` | `true` | boolean | Negative weights are violations |
| `constraints.min_weight` / `max_weight` | `0.0` / `1.0` | -1 to 1 | Per-instrument bounds |
| `constraints.per_instrument` | `{}` | `{"SPY": {"min": 0, "max": 0.4}}` | Bound overrides |
| `constraints.min_cash` / `max_cash` | `0.0` / `1.0` | 0 to 1 | Cash bounds; `max_cash = 1 - minimum invested weight` |
| `constraints.max_turnover` | `null` | >= 0 or `null` | Cap on one-way turnover per rebalance: sum over instruments of abs(target - current) |
| `constraint_policy` | `project` | `project`, `reject` | See below (SIM-03) |
| `infeasible_fallback` | `hold_current` | `hold_current` | Applied to decisions whose `solution_status` is `infeasible` or `unbounded` |
| `reconciliation_tolerance` | `1e-6` | 0 to 1 | Absolute tolerance, in currency units |
| `weight_tolerance` | `1e-9` | 0 to 1e-3 | Tolerance of constraint checks |

### Constraint policies

- `project`: the target is replaced by its exact Euclidean projection onto
  `{lb <= w <= ub, min_cash <= cash <= max_cash, sum(w) + cash = 1}`. If the turnover cap is still
  exceeded, the target moves along the segment from the current weights toward the projection
  until turnover equals the cap. Example: 60% proposed against a 40% cap, with 30% in a second
  instrument and 10% cash, executes 40% / 40% / 20% cash, and the result records the `max_weight`
  violation and the projection distance (about 0.245).
- `reject`: a violating target is discarded and current holdings are kept for that rebalance; the
  rejection and its violations are recorded.
- An infeasible constraint set (for example `min_cash` 0.5 with `max_cash` 0.4, a minimum
  invested weight of 60%) is not a configuration error: every decision is rejected as
  `constraint_set_infeasible` with `solution_status` `infeasible`, and the run still completes.

### Costs and returns

Every fill pays `fee = notional x proportional_bps / 10^4 + fixed_per_trade`,
`spread = notional x half_spread_bps / 10^4` and
`slippage = notional x coefficient_bps x participation / 10^4`. Costs come out of cash; fills are
recorded at the reference price. Results report gross return (net P&L plus costs paid), net
return, and total fees, spread and slippage costs as separate fields (SIM-05). Worked example
(unit-tested): 10,000 cash, 50% target at 100 with volume 1,000, 10 bps + 1.00 fee, 5 bps
half-spread, 10 bps slippage coefficient: 50 shares, fee 6.00, spread 2.50, slippage 0.25, net
return -0.0875%, gross return 0.

### Metrics

Net and gross cumulative return, annualized volatility (sample standard deviation of session
returns x sqrt(`periods_per_year`)), Sharpe ratio (risk-free source stated as
`sharpe_risk_free_source`, default the configured cash rate; `null` when volatility is zero),
maximum drawdown (positive fraction), turnover (traded notional / initial cash) and total
transaction costs.

## Example configuration

The block below is validated by `tests/unit/sim/test_simulator_docs.py`.

<!-- example: simulation-config -->
```json
{
  "venue": "paper",
  "execution_timing": "next_open",
  "rebalance_frequency": "monthly",
  "initial_cash": 100000.0,
  "base_currency": "USD",
  "cash_rate_annual": 0.0,
  "periods_per_year": 252,
  "fees": {"proportional_bps": 1.0, "fixed_per_trade": 0.0},
  "spread": {"half_spread_bps": 1.0},
  "slippage": {"model": "linear_participation", "coefficient_bps": 10.0},
  "liquidity": {"participation_cap": 0.05, "unfilled": "cancel"},
  "constraints": {"long_only": true, "min_weight": 0.0, "max_weight": 0.4, "min_cash": 0.02, "max_cash": 1.0, "max_turnover": 0.5, "per_instrument": {"SPY": {"max": 0.4}}},
  "constraint_policy": "project",
  "infeasible_fallback": "hold_current",
  "reconciliation_tolerance": 1e-6,
  "weight_tolerance": 1e-9
}
```

## Paper only

The simulator only moves simulated cash and shares. It never holds credentials for, calls or
produces instructions for any live broker, exchange or wallet, and simulated fills are research
artifacts, not platform executions. The build's `live-perm-scan` gate fails on any live-trading
client in dependencies, Dockerfiles or source, and on live-financial IAM permissions in repository
documents and synthesized templates.
