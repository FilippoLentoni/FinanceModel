# Beta PPO learning and stability

## Intent

Diagnose generalization failure in the first daily PPO experiment. A stronger single seed is not
the success criterion. Seek consistent improvement over each seed's untrained checkpoint, lower
turnover, and narrower seed dispersion while comparing against the same traditional controls.
Outperformance is a research question, not an implementation guarantee.

## Requirements

- Version state and action changes in each policy configuration. Beta uses 5/20/60-session
  return means and volatilities, current instrument/cash weights, and portfolio drawdown (37 inputs
  for five instruments). Features use only observations available at the decision close.
- Record portfolio drawdown because the reward depends on its high water mark. Daily training
  and evaluation observations and accounting must agree under a state-dependent scripted policy.
- Sample seeded 64-session training episodes with 60 sessions of preceding history inside the
  training split. Artificial boundaries must supply a final observation for value bootstrapping.
- Soften actions with softmax scale 1 and a 0.25 rebalance fraction. Add a 0.001 penalty per unit
  traded NAV alongside actual fees/spread/slippage. These are hypotheses, not optimized findings.
- Preserve the first experiment's 2025 training, 2026 H1 validation and July onward development
  periods for this diagnostic comparison. More diverse training history is a separate experiment.
- Train five PPO seeds for both risk-penalty settings. Choose reward settings by average validation
  Sharpe across every requested seed. Exclude incomplete seed grids; evaluate the equal-weight
  ensemble of target portfolios plus each seed individually. Never average network parameters.
- Report validation curves, step-zero comparisons, PPO explained variance, KL, clip fraction,
  entropy, value/policy losses, training reward and net portfolio performance separately.
- Preserve the full summary in the evidence artifact before response compaction. Keep all validation
  curves in the API; sample only optimizer histories and identify the full diagnostic artifact.
- A test path inspected in previous runs is reused development evidence. Flag reuse and prevent
  the run's promotion check from passing. Independent future evaluation is required for promotion.
- Use existing beta pipeline and CloudFormation stacks. Gamma/prod settings and production
  strategy must not receive this experiment. Keep the Gamma inbound transition disabled during
  the beta deployment, stop its execution after beta, then restore the transition.

## Bounded experiment budget

The user's project ceiling is USD 50. This takeover authorizes at most three sequential CPU runs,
each with a dry-run estimate no greater than USD 0.25 and a 3000-second runtime cap. Initial work
reserves at most USD 2 including CodeBuild, deployment and incremental storage. No GPU, endpoint,
scheduled training, or automatic indefinite search is needed. Check account/project charges and
the research ledger before submission; record estimates separately from invoice charges. AWS
billing data and budget actions are delayed, so the application ledger and finite run/runtime limits
control new experiment commitments. Stop after the diagnostic run if it is inconclusive; subsequent
runs need a specific hypothesis rather than repeated selection on the observed market path.

## Verification

Synthetic tests cover observable-state and evaluator parity, seeded episode boundaries and TD
bootstrapping, ensemble evaluation and reused-test promotion blocking. Required build gates and
beta integration tests must pass before the paid diagnostic experiment.
