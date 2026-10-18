# Approval-gated current-book Jev

The `finplan-jev-forward/1` backend is a **single forward paper-policy inference**. The managed queue's existing transport discriminator remains `jev_backtest`; the explicit internal `forward_one_decision` mode and `forward_paper_decision` objective select a separate worker path. No historical evaluator/window is used, no ledger is updated, and no portfolio-value forecast is invented from classification probabilities or weights. Native public descriptor preparation, egress guard, probability validation and sizing remain unchanged.

The original hourly `unavailable` / `not_invoked` run stays immutable. A later hourly opportunity can explicitly reuse the approved worker signal only when the approved snapshot identity, public request, private preprocessing, universe, holdings/cash/revision, constraints and native sizing still match. New hourly news/quote bundle checksums may differ because Jev does not consume those layers. The new captured bundle retains the exact original worker input receipt, public response, returned model, original source cutoff, actual invocation/response time, managed job ID and verified artifact references; preflight IDs are separately labeled. Signals expire after 24 hours. Only the common paper engine may apply them at a later eligible opportunity.

## Operator commands

Run from the FinanceModel repository, using the installed beta environment and FinancialPlanning source on `PYTHONPATH`. The command verifies the actual AWS STS identity against the configured beta EC2 development role. No caller-identity option exists.

```sh
python -m finplan_model.forward.submit_jev --experiment beta-forward-hourly-v1 --slot 2026-10-18T02:00:00Z --output /home/ec2-user/projects/.worktmp/beta-modeling/jev-forward-review.json
```

Default is a real control-plane **dry run**: bounded source/evidence/eligibility reads plus the existing estimate and budget checks, no job, model, approval, artifact write or managed-compute launch. The output shows the exact public request, its checksum, prospective immutable binding reference, CPU estimate and unchanged vendor caps ($0.01, 64 calls, 200,000 input tokens). A missing published worker capability remains visible.

After reviewing the concrete request, the authorized operator may add `--submit`. This seals the input and records `awaiting_approval`; it does **not** approve or launch compute. Submission requires the beta Jev job reference to publish `forward_protocol=finplan-jev-forward/1` and pins that reference's immutable image digest. The dispatcher rejects an image/capability change before compute. Do not submit using an unpublished rollout.

Only a separate authorized human, with that human's supported credentials, may run the existing interactive command from the repository root:

```sh
python scripts/approve_run.py --env beta RUN_ID
```

Never auto-approve, impersonate the approver, or reuse approval for a different request. Each approval covers one current-book managed inference, not recurring hourly paid calls. Snapshot/ownership/book revision are checked before submission and again immediately before creating the vendor client. Requests expire one hour after their original availability cutoff; late approval produces an explicit failed expired attempt without a paid invocation. Beta managed-job API reads are restricted by the platform to isolated synthetic experiment books. Ordinary user portfolios, writes, direct platform DynamoDB access, and higher-environment grants remain forbidden.

## Permanent recovery and artifacts

All new artifacts use existing Model-owned `artifacts/run_artifact/`; no new table, public contract wheel or bucket is introduced. The existing job-control head uses a deterministic run ID derived from the entire verified binding and survives native idempotency TTL. A strongly consistent Model-owned context HEAD is reserved by CAS in the same transaction as the managed run and idempotency record. Reusing a still-valid pending context requires its original run/input binding. A fresh separately approved successor is permitted only after expiry with permanent never-started metadata or a verified no-call receipt, or after a known captured daily signal expires. An indeterminate predecessor remains visible indefinitely; missing artifacts never prove no call. Original job heads, bindings, claims and responses remain permanent.

The worker seals input **before** a permanent conditional invocation claim, then seals the public response in a deterministic journal before policy/result completion. S3 creates use `If-None-Match: *`. A retry with a captured response validates/reuses the original input and response without another vendor call, even if the original invocation window has since expired. A claim without a captured response is explicitly `forward_invocation_indeterminate`: a paid side effect might already have occurred, so independent human resolution is required. No blind paid retry is permitted. Native HTTP retries retain their existing bounds and reservations. Failed outputs retain original input/failure artifacts; the container exits unsuccessfully.

Managed worker eligibility also reads only its own Model context HEAD (beta GetItem restricted to FORWARD_CONTEXT# keys), with no control/budget/ledger writes. Current book, expiry and context are rechecked immediately before every vendor transport attempt. Native probabilities and returned model identity remain inspectable. The portfolio forecast is explicitly unavailable. `managed_cost_event` identifies the original job once at its completion time, separating observed compute cost (possibly unavailable), compute estimate, vendor planning estimate and actual token/call usage. Reused hourly runs have zero new model invocation cost; original events must be counted once across reuse, never shifted to a later reporting day or counted once per signal reuse.

## Remaining live requirements

Offline tests do not prove paid inference. Deployment must publish the bridge-capable worker/control source and beta-only paired API grants. A real managed invocation remains pending independent human approval and actual vendor authentication/service availability. Qwen remains a separate preflight: missing verified pinned checkpoint and GPU approval are not resolved by this Jev bridge. No Qwen result or dependent worker output is fabricated.

Cost events are committed atomically with terminal Model job transitions into private existing job-control outbox rows. Hourly EC2 reads at most100 pending events, imports byte-identical ancillary `managed_model_cost:<job-id>` observations at the actual event slot with actual import availability, and enqueues a private durable `finplan-report-refresh/1` task before acknowledging the Model outbox. Set `FINPLAN_REPORT_REFRESH_DIR` to a persistent EBS directory. Queue tasks contain only experiment/day/event/observation IDs and checksums; a full or missing queue fails visibly without losing the cost event. Only verified archived report inclusion clears a refresh task. No new table, index, public wheel or portfolio ledger is involved.
