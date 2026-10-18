# Public model preflight

The hourly `jev_systemone` and `qwen_swarm` adapters prepare native requests; they
remain **unavailable / not invoked** until the approved managed inference path
can consume that exact current-book request and return a captured result. This is
not evidence that either model was evaluated. News is not consumed.

Jev uses the existing `JevStrategy` and `encode_state`: actual eligible return
history and marked synthetic holdings become the same native trend, volatility
and exposure buckets. The captured public request contains those buckets and
native questions. Exact numeric preparation, holdings revision and transformations
are retained separately as local preparation, explicitly not supplied to Jev.
The existing outbound guard still applies. No vendor transport or secret value
read occurs. Secret metadata existence does not establish vendor authentication.

Qwen uses the existing frozen model/revision and native first market-analyst
messages. Subsequent role messages depend on real preceding model outputs and
are explicitly not prepared. Only bounded central staging metadata is checked;
no weights are downloaded and no GPU job starts. A matching metadata manifest
alone does not verify all weight bytes.

Current configured caps remain unchanged: Jev 64 calls, 200,000 tokens and
$0.01 external vendor cost. The native conditional managed cost estimator reads
the configured, dated instance prices and applies the existing runtime bounds.
Its result is neither a remaining-budget check nor approval. Zero vendor usage
in preflight means no invocation; it does not describe the managed job cost.

A genuinely historical-vintage source can produce a contract-validated native
`submit_experiment` dry-run proposal. Forward-retrieved historical bars cannot be
backdated to make a chronological pilot eligible. The pilot evaluates its own
historical book and cannot substitute for this hourly synthetic book's decision.
No snapshot ID, approval, output, dependent role or historical vintage is invented.

The supported signed service route is `POST /v1/jobs` with `dry_run: true` for
validation/cost review. Submission with `dry_run: false` creates an approval-gated
Jev/GPU request. Approval uses `POST /v1/jobs/RUN_ID/approve` or
`scripts/approve_run.py --env beta RUN_ID`; the service enforces an authorized
independent approver and rejects the submitting identity. Preflight never submits,
approves, changes roles or treats an old synthetic pilot as recurring approval.
An approved current-book capture/result bridge remains required before these
approaches can be included as evaluated portfolio performers.
