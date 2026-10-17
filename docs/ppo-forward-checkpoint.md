# First forward-news PPO checkpoint

The first operator checkpoint is **October 30, 2026 at 16:15 America/New_York**
(`2026-10-30T20:15:00Z`). Its fourteen-calendar-day interval begins October16
at16:15, so the first full market session is MondayOctober19. Ten scheduled
weekday captures are expected through FridayOctober30. This is a candidate
feasibility experiment, not sufficient evidence of financial performance.

The checkpoint does not schedule anything, purchase data, submit AWS jobs,
execute portfolio orders, modify the existing paper book, or promote actors.
Root orchestration owns daily collection and the one-shot invocation.

## Read and freeze

Run from the repository with its virtual environment:

```sh
.venv/bin/python scripts/ppo_forward_checkpoint.py \
  --prices /private/approved-price-bundle.json \
  --endpoint "$FINPLAN_PLATFORM_ENDPOINT" --region us-east-2 \
  --output /private/ppo-checkpoint-20261030
```

The caller supplies authenticated credentials already authorized to read news.
The client invokes the existing bounded latest-news route at start+6days, start+12days and
the end of the fourteen-day interval. Seven calendar-prefix reads overlap to
retain late-day boundary arrivals; a rolling168-hour assumption would miss them. It unions immutable delivery IDs and validates
the resulting contract. Saturated168-record responses or conflicting versions
fail closed. API retrieval failures are not empty no-news observations. Original
publication, retrieval, availability, payment provenance, quality and rights
remain unchanged. `--news PATH` instead replays a previously frozen corpus.

The operator-frozen price JSON supplies:

- `instruments`: canonical ordered symbols, normally `["VOO", "GOOGL"]`.
- `decision_times`: unique increasing actual market-session UTC close timestamps.
- Aligned two-dimensional `close`, `exec_price`, `volume`, `has_bar` arrays.
- `input_snapshot_id`, `snapshot_checksum`, `snapshot_status: "approved"`,
  `synthetic: false`, copied from the actual approved snapshot. The operator
  obtains/verifies these through the existing SnapshotReader; this script does
  not independently download or authenticate the declared price source.
- `simulation`: the existing frozen paper simulation configuration, including
  costs/holdings/constraints. Both PPO branches receive exactly this same object.
- `news_start: "2026-10-16T20:15:00Z"` and
  `checkpoint_at: "2026-10-30T20:15:00Z"`.

Provide at least sixty price sessions preceding the first news decision. Those
warmup sessions do **not** need news. No future price rows may be supplied to an
early readiness run. It will honestly report checkpoint_not_due and inadequate
forward sessions rather than synthesize future observations.

Outputs are `corpus.json`, `manifest.json`, `request.json`, `report.json`.
They include corpus/price checksums, delivery count/read counts, decision-level
missingness and event counts. Missing data remains missing; zero events is not
inferred from an empty or stale search. A training-ineligible current metadata
corpus reports `rights_review_pending`. That is a conservative unresolved rights
review, not a claim that providers impose a blanket training prohibition.

## Explicit candidate execution

Add `--execute --budget /private/fresh-budget-reservation.json` only after a
licensed forward corpus and real price bundle pass readiness. Budget JSON:

```json
{
  "checked_at": "2026-10-30T20:15:00Z",
  "estimated_total_usd_upper_bound": 0.1,
  "estimate_basis": "REPLACE with actual operator quote for the entire matched run including export and overhead",
  "project_spent_usd": 0,
  "project_reserved_usd": 0,
  "weekly_spent_usd": 0,
  "weekly_reserved_usd": 0,
  "exclusive_research_lease": true
}
```

The numbers above illustrate shape, **not a current price quote or authorization**.
Evidence must be at most one hour old; estimate plus existing reservations/spend
must fit the $100 project ceiling and $0.50 weekly research cap. The operator must
hold the existing exclusive research lease; concurrent work must not race a
self-declared budget balance. Ten separate $.0675 jobs would exceed the cap.
This local execution uses one shared900-second deadline for five fixed seeds
paired across price-only/news-enriched PPO branches,256timesteps per learner,
and small32×32networks. Price inputs, decision windows, seeds, hyperparameters,
simulation configuration and costs match. The only feature difference is the
five-dimensional per-symbol news group. Existing `train_policy` performs full
point-in-time training preflight before learner creation, then existing
`export_actor` produces candidate artifacts; no fake/padded legacy policy.

For the minimum ten-session corpus, six sessions form the training span, two
validation and two reserved test. The sixty-session warmup precedes training
and is excluded from its news coverage requirement. The test period stays
unopened in this checkpoint runner. Exported actors require the existing common
portfolio evaluator on this reserved window, additional prospective evaluation,
and an explicit release decision before serving. The result says
`test_status: reserved_unopened_requires_common_evaluator`; it does **not** claim
out-of-sample returns, a better policy, or completed hosted PPO integration.

As the retained corpus grows, rerun with a newly versioned longer protocol and
frozen nonoverlapping selection/test windows. Do not repeatedly tune on the same
holdout. Historical publication-filter search can supplement a separately
labeled retrospective dataset; it cannot backdate actual availability in this
forward cohort. Operational exit criteria remain open until a genuine compatible
actor is evaluated and served through the hosted agent.

If the verified price bundle is not yet available, the one-shot operator can
still freeze central news and persist a useful blocked report:

```sh
.venv/bin/python scripts/ppo_forward_checkpoint.py --capture-only \
  --endpoint "$FINPLAN_PLATFORM_ENDPOINT" --region us-east-2 \
  --news-start 2026-10-16T20:15:00Z --checkpoint-at 2026-10-30T20:15:00Z \
  --symbols VOO,GOOGL --output /private/ppo-checkpoint-20261030
```

This writes `missing_price_bundle` and observed rights statuses, returns exit2,
and never imports learners. It cannot be combined with `--execute`. Exit2 is an
honest readiness outcome, which the operator scheduler should retain as evidence
rather than interpret as a guarantee that retraining launched.
