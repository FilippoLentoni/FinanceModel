# Daily portfolio training and serving

## Authorized outcome

Retain genuine daily source vintages, train/evaluate portfolio models, export
compatible paper actors, and let LangGraph request dated buy/sell/hold allocations
and explain their actual inputs and outputs. Deployment is advisory/paper only;
no brokerage orders or automatic changes to saved holdings are authorized.

## Data and timing

- PPO needs a sequence of state/action/reward/next-state transitions. A single
  approved historical price snapshot containing many ordered market sessions is
  enough to construct price-only training episodes. One current market row is
  insufficient. Multiple retained snapshots are required to recover first-seen
  daily data, revisions and genuine contemporaneous news availability.
- Read approved snapshots through SnapshotReader and verify every artifact
  checksum. Record the exact universe, adjusted-return basis, session calendar,
  missing bars, execution prices, source retrieval time and revision/bias flags.
  Retrospective session-close price availability is an explicit assumption; it
  does not turn a revised history into a point-in-time vintage.
- The daily investable subset is GOOGL, AMZN, NVDA, MSFT, AAPL, META, TSLA and
  VOO. Snapshot ingestion may include NFLX to keep existing five-asset portfolios
  covered. Models and isolated profiles select their exact frozen subset.
- At a morning decision, prices end at the last completed available session and
  news ends at the actual current decision timestamp. Publication time alone
  cannot backdate a news retrieval. Persist price as_of separately from the news
  cutoff and identify whether the actor actually consumed news features.
- The central news API remains bounded to five symbols per response. Wider
  corpora use four-symbol partitions, each validated against the existing
  contract, with disjoint complete universe coverage and matching cutoffs.
  Conflicting versions of an immutable delivery fail closed. Rights, quality,
  original retrieval/availability and missingness are preserved.
- Headline lexicon features are search metadata, not a calibrated sentiment
  model. GDELT numeric archive tone/count features require their own version and
  matched historical/live feature definition; never masquerade as lexicon news.

## Learning and evaluation

- Every training/validation trajectory must be chronological, nonoverlapping,
  finite and sufficiently long for warmup. Learner array construction rejects
  bars unavailable at their decision cutoff rather than exposing future data.
- Preserve legacy/state-ablation protocols. Add the separately versioned
  recursive_ppo_daily_history profile: training 2013–2024, validation 2025,
  development evaluation 2026. Use five independent seeds, 30,000 requested PPO
  steps per seed, existing validation checkpoint selection, randomized 64-session
  episodes with 60-session warmup, cost-aware reward, observed portfolio/cash
  weights and drawdown, and 5/20/60-session return/volatility summaries.
- Keep the released softmax action scale, partial rebalance and risk/drawdown/
  turnover penalties frozen. More market regimes is the hypothesis under test;
  this experiment does not establish a universally best policy architecture.
- Evaluate equal-weight seed allocations through the common portfolio evaluator
  with identical daily schedule, universe, costs and constraints as cash,
  buy-and-hold, equal weight, minimum variance, mean variance and scenario CVaR.
  Select only on validation. Previously observed 2026 paths remain development
  evidence and cannot support a production promotion claim.
- Save model bytes, actor exports, configuration/source checksums, diagnostics,
  requested and actual steps, early stops, seed dispersion and net metrics.
  An unchanged step-zero checkpoint must be reported honestly.
- A news-aware PPO additionally requires licensed, eligible, point-in-time
  coverage at every training/selection decision and historical/live timing
  parity. Missing news is not a zero-event observation. Current research-only
  deliveries remain training-ineligible pending actual rights evidence.

## Deployment and recurrence

- Export only complete evaluated ensembles. Pin a new eight-symbol actor to its
  explicit isolated saved paper portfolio using portfolio-scoped activation;
  preserve the existing default five-symbol actor and saved holdings.
- Price-only and mandatory-news profiles are explicit metadata. A price-only
  profile cannot claim news influenced allocation. Existing two-symbol research
  profile behavior remains backward compatible.
- Retain source captures and versioned frozen dataset manifests daily. Retraining
  is readiness-driven and budget/lease-gated; it is distinct from daily inference.
  Reuse source data only through new on-policy simulated rollouts when fitting
  PPO, rather than pretending archived actions are fresh on-policy rollouts.
- A recurring research controller may submit a released profile only after fresh
  data/rights/readiness and producer budget checks pass. Existing weekly/monthly
  caps stay enforced. No implicit daily paid training or automatic promotion.
- Deploy code through the existing pipeline, run required offline gates and beta
  integration, stop before Gamma/Prod and restore transition state. Report actual
  checkpoints, live deployed identities, freshness and unresolved blockers.

## Primary research basis

- PPO alternating fresh rollout collection and clipped policy optimization:
  https://arxiv.org/abs/1707.06347
- FinRL historical market environments and separate training/testing/trading:
  https://arxiv.org/abs/2011.09607 and https://arxiv.org/abs/2111.09395
- FinRL-Meta data/environment quality, survivorship and overfitting concerns:
  https://arxiv.org/abs/2304.13174

These support the design choices, not a promise of positive returns.
