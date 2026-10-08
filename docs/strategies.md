# Baseline strategies

Task group 4 of `add-research-job-foundation`. Spec: `baseline-strategies` (BASE-01 to BASE-06).
Code: `src/finplan_model/strategies/`.

> **Defaults are configuration pending user review, not recommendations.**

Every baseline implements the simulator's strategy interface: `name` plus
`decide(view, holdings)`, which returns target weights and cash. Baselines therefore run only
through the common evaluator (`finplan_model.evaluate.evaluate`). The optimizers optimize under
the simulation's constraint set: `build_strategy(name, params, constraints=sim.constraints)`.

## Strategies

| Name | Family | Decision | `solution_status` |
|---|---|---|---|
| `cash` | control | 100% cash: no trades, return = configured cash rate | `not_applicable` |
| `buy_and_hold` | control | Initial target on the first decision of a run (equal weights over the eligible universe, or `initial_weights`), then holds and never rebalances | `not_applicable`, then `no_effect` |
| `equal_weight` | control | Equal weights over the eligible universe (instruments with a price visible at the decision time) at every rebalance; `cash` sets the cash weight | `not_applicable` |
| `min_variance` | classical optimizer | Minimizes estimated variance `w'Σw` | `optimal` / `feasible` / `infeasible` |
| `mean_variance` | classical optimizer | Maximizes `μ'w + r_c·cash - risk_aversion · w'Σw` | as above |
| `scenario_cvar` | classical optimizer | Minimizes CVaR at `alpha` over scenarios (`min_cvar`), or maximizes the scenario mean return subject to `CVaR <= cvar_limit` (`max_return`) | `optimal` / `infeasible` / `unbounded` |

Every benchmark includes the three controls on the same dataset and simulation configuration.
`with_controls` and `BenchmarkRequest` add any control a request omits, and the report lists it
under `auto_added_controls`.

## Optimizer parameters

| Parameter | Default | Strategies | Meaning |
|---|---|---|---|
| `lookback` | 60 | all optimizers | Return observations visible at the decision time (`view.returns`) |
| `min_history` | min(lookback, 20) | all optimizers | Fewer observations: hold (`no_effect`, `insufficient_history`) |
| `cash_policy` | `min` (min-variance, CVaR); `free` (mean-variance) | all optimizers | `min`: cash fixed at the constraint set's `min_cash`; `free`: cash in `[min_cash, max_cash]` |
| `cash_return_annual` | 0.0 | all optimizers | Return of cash in the objective; set it equal to the simulation's `cash_rate_annual` |
| `periods_per_year` | 252 | all optimizers | Session conversion of `cash_return_annual` |
| `covariance_estimator` | `ledoit_wolf` | min-/mean-variance | `sample`, `ledoit_wolf`, `diagonal`, `ewma` |
| `return_estimator` | `historical_mean` | mean-variance | `historical_mean`, `ewma_mean`, `zero` |
| `ewma_halflife` | 20 | min-/mean-variance | Sessions |
| `risk_aversion` | 5.0 | mean-variance | `>= 0` |
| `alpha` | 0.95 | CVaR | Confidence level |
| `objective` | `min_cvar` | CVaR | or `max_return` with `cvar_limit` (per-session loss) |
| `n_scenarios` | 500 | CVaR | Draws per decision (`bootstrap`, `gaussian`) |
| `scenario_method` | `bootstrap` | CVaR | `historical` (every visible row, no randomness), `bootstrap`, `gaussian` |
| `seed` | 0 | CVaR | Per-decision seed = SHA-256 of `seed` and the decision session; the same seed gives the same scenarios and weights (recorded with `scenario_checksum`) |

Point in time:

- Every estimate reads returns only through the decision's `PointInTimeView`. Bars after the
  decision session, or that arrived after the decision time, are invisible (BASE-02).
- Estimates are per session and not annualized.

## Solver outcomes (BASE-05)

Solvers are open source and deterministic: SciPy SLSQP for the quadratic programs, and HiGHS via
SciPy `linprog` for the CVaR linear program. No commercial solver is used (FM-A4).

| Outcome | Decision `solution_status` |
|---|---|
| Constraint set admits no portfolio (for example minimum cash 50% with a minimum invested weight of 60%) | `infeasible` (before solving) |
| SLSQP success, or its point satisfies the KKT conditions of the box-plus-budget QP | `optimal` |
| SLSQP stopped at a feasible, uncertified point | `feasible` |
| SLSQP point infeasible | `infeasible` |
| HiGHS status 0 / 2 / 3 | `optimal` / `infeasible` / `unbounded` |
| HiGHS status 1 or 4 (iteration limit, numerical trouble) | `infeasible` (no certified solution) |

What happens next:

- For `infeasible` and `unbounded` decisions the simulator keeps current weights
  (`fallback_hold_current`).
- The run completes with `completion_status` `succeeded`, so these are never job failures.
- Run-level status: all infeasible gives `infeasible`; some infeasible gives `feasible` (for
  example 2 of 40 decisions).
- `job_outcome(result)` returns `{completion_status, solution_status, fallback_decisions,
  stage_output_allowed}`. Infeasible and unbounded runs stage nothing.

## CPU only (BASE-06)

Controls and classical optimizers run as CPU SageMaker Processing Jobs: the `financemodel-cpu`
image with job types `prepare_dataset`, `run_backtest`, `run_benchmark` and `report`.

- `validate_baseline_submission` and `require_cpu_instance` reject a GPU or accelerator instance
  type (`ml.p*`, `ml.g*`, `ml.inf*`, `ml.trn*`, `ml.dl*`) with `VALIDATION_FAILED`
  (`/instance_type`).
- With the environment configuration, they also reject any instance type not configured for the
  job type.

## Registry identity

`descriptor(name)` gives the identity the model registry mints `model_version` from:

- `strategy`, `family` and `param_schema_version` (`"1"`);
- `makes_predictions` and `has_randomness`;
- `compute_class` `cpu`.

`strategy.configuration_id` is the `cfg_` id of `{strategy, param_schema_version, params}`.
