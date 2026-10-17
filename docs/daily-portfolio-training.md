# Daily portfolio model operations

The requested eight-symbol portfolio uses GOOGL, AMZN, NVDA, MSFT, AAPL, META,
TSLA and VOO. Existing five-symbol books and the VOO/GOOGL USD100 profile retain
their identities. A new isolated USD100 profile can explicitly set
`news_required: false` to disable its mandatory paid-headline path. A price-only
actor must disclose that news was not consumed. The released eight-asset
classical numeric adapter separately requires GDELT machine metadata and records
its actual sensitivity inputs. A news-trained PPO requires its compatible
versioned actor and complete eligible training corpus.

PPO needs many ordered transitions, not many files. An approved price snapshot
with years of daily sessions can supply historical environment trajectories.
The learner generates fresh actions and rewards as it rolls through that
environment. Daily immutable snapshots are still needed for source revisions,
news availability and prospective evaluation. Daily inference does not require
daily retraining.

The new `recursive_ppo_daily_history` release profile changes only the history
split from the released state baseline: 2013–2024 training, 2025 validation,
2026 development evaluation. It requests five seeds of 30,000 PPO steps and
compares classical/control families under the same common evaluator. Prior
dataset-wide test access continues to prevent production promotion. Reports
must include completed steps and validation-selected checkpoints; requested
steps alone are not evidence a fit completed.

The checkpoint CLI can now obtain prices directly through the approved source
reader instead of requiring a manually assembled bundle:

```sh
.venv/bin/python scripts/ppo_forward_checkpoint.py \
  --snapshot-id snap_<approved-id> --endpoint "$FINPLAN_PLATFORM_ENDPOINT" \
  --symbols GOOGL,AMZN,NVDA,MSFT,AAPL,META,TSLA,VOO \
  --output /private/daily-ppo-checkpoint
```

The caller needs the existing authenticated snapshot-read permission. It writes
`prices.json` with verified artifact identities, calendar, adjusted basis,
source retrieval/revisions and the explicit retrospective price-availability
assumption. The fourteen-day news feasibility runner retains its rights,
coverage, deadline and combined research budget gates. An early run reports
its blockers without training. It is separate from the standard paid job API,
whose own cost/approval/lease gates govern full historical price-only training.

For eight news symbols the client freezes two four-symbol partitions. Every
partition satisfies the existing central API contract. The internal versioned
envelope preserves original delivery bytes, rejects conflicting identities and
keeps the complete corpus checksum for actor lineage and replay. It grants no
rights or historical availability.

After a complete evaluated PPO ensemble has been exported, an authorized beta
operator may call the existing advisory activation endpoint with
`export_run_id`, `confirmed_by_user: true` and the isolated `portfolio_id`.
This adds a scoped pin under the existing advisory parameter and preserves the
default artifact. Omitting `portfolio_id` updates the default while retaining
scoped pins. The read-only recommendation path chooses the saved portfolio's
pin and verifies profile/model universe compatibility before inference.
Activation never modifies holdings. Rollback restores the previous parameter
document; artifacts and paper decisions remain immutable.

News-aware inference uses current decision time and latest completed prices;
explicit historical requests retain their historical cutoff. Responses expose
`decision_at`, price `as_of`, and actual consumed `news_evidence` separately.
Future news-aware learners must match the morning timing and feature version;
the existing close-timed feasibility checkpoint does not establish that parity.

Daily collection retains source vintages. Recurring retraining remains a
separate readiness-driven action through the existing budgeted research
controller; no automatic daily paid runs or actor promotion are enabled here.
The requirement and primary literature links are in
[the daily training specification](../openspec/changes/daily-portfolio-training/specs/daily-training/spec.md).
