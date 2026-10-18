# Beta forward hourly experiments

The EC2 orchestration entrypoints live in `finplan_model.forward`. They use the
FinancialPlanning `ModelingEvidence` backend with normal beta EC2 credentials.
The platform-owned `scripts/collect_modeling_sources.py --kind research` must persist the half-hour source before the model cycle, and `--kind quotes` must persist each hourly valuation quote bundle. FinanceModel never calls raw market/news providers. The installed release must put both repositories' Python package roots on
PYTHONPATH, plus FinanceAgent for the hosted LangGraph adapter. No browser or
brokerage credential is needed. Run only in beta/us-east-2.

```
python -m finplan_model.forward.research cycle --slot <UTC-half-hour-slot> --experiment beta-forward-hourly-v1 --output <EBS-cycle-directory>
python -m finplan_model.forward.hourly --slot <UTC-hour-slot> --experiment beta-forward-hourly-v1 --output <EBS-hour-directory>
python -m finplan_model.forward.hourly --slot <UTC-hour-slot> --experiment beta-forward-hourly-v1-cohort8 --cohort eight_assets --output <EBS-hour8-directory>
```

The root-owned systemd units invoke these commands with nonoverlap locks and
bounded deadlines. `FINPLAN_EXPERIMENT_TRIGGER=systemd_scheduled` distinguishes
actual recurring proof from `manual_validation`. Set the immutable installed
source SHAs through `FINPLAN_MODEL_SOURCE_COMMIT`,
`FINPLAN_PLATFORM_SOURCE_COMMIT`, and `FINPLAN_AGENT_SOURCE_COMMIT`.

A slot identifies its scheduled hour. Actual inference time and the shared
availability cutoff are captured separately, after the public quote collection
and before inference. All approaches have the same hourly opportunity, starting
at minute five, and the same configured initial cash, universe, costs, and
next valid hourly regular-session quote rule. The half-hour research experiment has suffix
`-interim`; its recommendations cannot create extra primary fills. Original
source, prepared solver/actor/provider inputs, outputs, run claims, books and
observations are stored through the platform backend. The local result JSON and
`cycle-receipt.json` are read-back receipts, not the authoritative ledger.

A permanent `(experiment, approach, slot)` record prevents a second invocation or
fill. A crashed invocation with captured inputs is recorded as failed on an
expired lease instead of replaying vendor costs or inventing its missing output.
Terminal retries repair a missing valuation using original captured holdings and
source. Historical missing slots use `--account-missed`, capture explicit
`skipped_missing_point_in_time`, and never use today's news as historical input.
No expensive model is trained by an hourly tick.

The registry includes the three controls, all baseline optimizers and evaluated
finite lookback/risk/CVaR variants, the separate MCP classical implementation and
its weekly evaluated lookbacks, PPO/SAC supported feature variants, all discovered
frozen exported policies, direct Codex, the deployed LangGraph orchestration,
TypeSafe Jev and Qwen fixed-role swarm. Missing policy artifacts, vendor approval,
GPU approval or machine interfaces appear as explicit unavailable entries. A
research controller that proposes candidates is not itself a portfolio policy.
An actual new candidate requires a versioned registry/experiment.

The default frozen PPO supports five instruments. The eight-instrument frozen
price policy and GDELT intraday policy belong to a separate comparison cohort.
An incompatible universe never gets projected into a frozen actor. GDELT
intraday accepts only its frozen 09:00 New York formation with actual platform
availability before that time; no raw article text is manufactured from tone and
count metadata. Its hourly records disclose reused native formation inputs or
explicit source unavailability. The standalone price policy consumes no news.

The hourly adapter selects the latest completed **calendar-day** 09:00 formation:
before 09:00 it reads the prior day's already captured immutable metadata; from
09:00 onward it requires the current day's eligible capture. Missing or late
current-day data remains unavailable and does not silently fall back. This fixes
the New York midnight rollover gap without changing native formation validation,
policy shape, or the native 1.5-hour window-age rule evaluated at formation.
The existing eight-instrument observation remains 98 values. Actual source age
at hourly reuse, source/central availability, native cutoff and adapter version
are captured separately as selection/acquisition provenance, not added policy
features or fresh intraday news. Current synthetic holdings remain current.
Successful hourly inference is labeled `inferred_from_reused_native_inputs`;
a failed/noninvoked actor never receives that success label. Earlier unavailable
hourly records stay immutable. The scheduled morning collector's existence does
not prove a future successful capture; strict availability checks still apply.

Constituent research uses a newly fetched, dated published S&P 500 constituent
table, retaining original HTML, per-row supplied date-added and unknown effective
dates. Official SPDR holdings are preserved as a separately labeled fund proxy
and discrepancy check. Neither source proves historical index membership. All
503 currently published symbols are assigned to bounded public RSS batches;
missing/failed coverage remains explicit. Public feed titles are retained, not
article bodies. Publication, retrieval and platform availability are separate.
This is forward-only evidence. The reviewer sees a deterministic bounded
selection of headlines, derived market descriptors, the dated research universe,
coverage and its own current synthetic holdings. Exact raw provider history remains source/evaluator evidence; the prompt receives the approved bucketed momentum, volatility and exposure descriptors through the existing outbound guard. Its full public prompt, actual
configured Codex model/CLI version, unmodified public final response and normalized
policy are captured. No hidden model reasoning or authentication is stored.

Yahoo one-minute quote responses are preserved with original bytes, checksums,
actual retrieval/availability and freshness. They are used for current portfolio valuation and eligible common point-quote paper execution, never as daily frozen policy inputs or retrospectively fabricated fills.
Closed-market quotes are carried forward visibly. Cash-only books have an exact
cash valuation and require no market quote. Missing quote coverage prevents a
fresh valued risky portfolio from entering a comparable leaderboard.

`paper.apply_forward_quotes` invokes the public common evaluator point-price helper,
which shares the daily simulator's exact numerical execution kernel. It consumes
native completed minute quotes and actual minute volume, never fabricated daily
bars. A prior-hour recommendation applies only at the next valid regular-session
hourly opportunity, with a quote captured this cycle, younger than 120 seconds,
and after the actual decision. Execution time is actual availability/capture,
separate from quote time. Sells execute before cash-limited buys; liquidity caps,
fees, spread, slippage and constraints are common across every book. No eligible
fresh quote means pending. Old completed daily opens are never forward fills.
`paper.apply_next_session` is retained only for clearly synthetic controlled
replay; it refuses real forward-market data. Controlled replay tests prove buy
and sell mechanics and are never mixed with forward performance. The latest
pending hourly policy supersedes older pending policies before an eligible
opportunity, with immutable execution-disposition observations.

A genuine forecast is frozen at decision time with target time, USD baseline,
expected value, assumptions and checksum. An allocation or solver estimated
return diagnostic is not a portfolio-value forecast. The sparse pending-forecast
index survives report windows, and the observation-time index finds the first
fresh valid valuation at/after target. Reconciliation writes a separate immutable
match and links the mutable head; it never rewrites the forecast. Signed error is
flow-adjusted realized value minus expected value. MAE and RMSE use matched
observations only; unsupported calibration is labeled unavailable.

Known limitations remain visible: restricted model-native investable cohorts,
no trained artifacts for unexported supported RL variants, approved compute/vendor
dependencies for Qwen/Jev, native GDELT formation/source availability, and no
completed future day/week or matured forecast until actual observations exist.

Captured provider dividends/splits on held assets block paper execution and a
comparable valuation until an authoritative action posting is implemented. A
missing action feed is not asserted neutral. The current raw daily source retains
dividend/split fields and the current quote adapter preserves actual provider
events/coverage; no payment date, dividend cash or split adjustment is invented.
This explicit corporate-action reconciliation limitation remains a coverage
blocker for affected forward observations.

Hosted numerical serving freezes its actual prepared policy observation before
actor execution in the content-addressed `artifacts/served_model_input/` research
prefix. Beta inference has only a conditional create grant to that prefix; writes
elsewhere, overwrites, deletes, DynamoDB writes, training and configuration writes
remain denied. Higher-environment policy grants are unchanged. Exact payloads,
checksums and references are returned in diagnostics and on actor failures so the
existing platform graph activity trace can index the original evidence. Classical
replays use their existing immutable classical-analysis store instead.

Hourly source bundles freeze every model's original `policy_inference` artifact
reference, canonical model checksum, configuration, algorithm, fixed instruments
and ordered feature specification in `frozen_policy_refs`. They do not duplicate
actor weights into every hourly input/report. Crash recovery reads those original
immutable bytes and verifies both artifact and model identity; it never selects
SSM's newer policy or recomputes an input vector. One inaccessible or mismatched
artifact gives that approach an explicit unavailable precondition while other
books continue. Older archived `frozen_policies` bundles remain readable. Exact
prepared solver arrays and policy observations are still captured separately at
the actual inference boundary and remain downloadable in the input explorer.

The bounded evaluated-checkpoint inventory includes 70 real PPO checkpoints and
10 real SAC checkpoints, linked to succeeded research runs and original evaluated
configurations/seeds. CPU export performs no training. The 2026-10-17 provenance
review exported 75 distinct PPO single/selected-ensemble inference artifacts,
conditionally created 72 and read-verified three already existing. Together with
existing SAC and ensemble artifacts this produced 85 immutable inference exports.
Each distinct export has its own book; identical artifact aliases are deduplicated.
An incompatible fixed-universe export remains visibly unavailable in that cohort,
without projecting its frozen input shape onto another universe. The selected
GDELT ensemble retains its verified original news protocol; this does not invent
independent exported news-seed models or missing ETF-news/daily-GDELT policies.

Native licensed search-metadata reads retain the existing platform API limit of
five symbols per request. The eight-asset cohort performs exactly two bounded
reads at one shared cutoff. `finplan.batched-native-news/1` preserves each original
schema-validated response/checksum, query-symbol coverage and partial failures,
then deduplicates identical immutable delivery IDs for native feature extraction.
It never relabels that aggregate as an original eight-symbol API response.
Conflicting IDs, mismatched cutoffs/symbols and future-available deliveries cannot
enter the merged metadata. Successful empty queries remain distinct from failed
queries. Missing eligible licensed metadata is exposed by the native extractor;
public RSS headlines are not substituted for that model's declared metadata input.

The direct classical Lambda's `get_etf_research_inputs` dispatcher explicitly
applies the existing contract `no_storage_locations` semantic check before JSON
validation or producer invocation. This closes the direct runtime rejection gap
for location-valued `contract_version`/`decision_at` fields while preserving valid
existing requests. The immutable legacy v1 request schema still lacks its check
marker: the three pre-existing producer-schema audit failures are not waived,
rewritten or represented as passing by this source hardening.

Execution recovery uses metadata-only approach queries and a verified permanent
execution frontier. The frontier retains one explicitly pending original policy;
older eligible policies have separate immutable superseded dispositions. These
records never stand in for paper ledger receipts or change the recommendation.
Only the selected original policy artifact is expanded. The backend verifies the
covered interval before accepting a frontier, and query overflow or unresolved
running heads leave the hour explicitly incomplete.

A running invocation retains its original600-second lease and captured inputs.
An immediate restart returns temporary exit75 and does not re-invoke the model.
After expiry, recovery records an explicit interrupted-invocation failure using
the original input; missing historical valuations remain missing. Other completed
approaches are not invoked again. Late recovery after the shared input bundle has
been frozen repairs bookkeeping only and cannot execute a trade retroactively.
The scheduler retries only exit75 with the same permanent slot within a bounded
command deadline; generic failures remain failures.

The shared source checksum is taken from the verified captured source artifact
once per hour, rather than repeatedly canonicalizing the full source for each
approach. Every model capture retains that exact checksum; the storage boundary
checks it against the authoritative shared-input reference before creating the
model artifact or invoking inference.

Existing valuation metadata includes the original run/input/book bindings for
new records. Same-slot retries validate those bindings and avoid reopening model
artifacts when the valuation already exists. Legacy observations verify their
original stored bytes; missing valuations are repaired from captured inputs.
A mismatched observation fails integrity checks rather than being skipped.

The interim `research cycle` command waits in30-second steps for the original
lease for up to900 seconds, retaining the same slot. After expiry it records an
explicit interrupted-cycle failure from the existing input and emits the durable
receipt without calling the reviewer or retrieving sources again. An unresolved
lease at the deadline returns exit75 and no successful receipt.

New missed-hour and source-failure evidence retains the actual model/platform/
agent release SHAs from the EC2 process environment. `processing_code_versions`
identifies the workflow that recorded the failure, while `not_invoked` and
`code_version_scope` explicitly prevent interpreting it as historical model
execution. New interim failure records retain all three identities as well.
Same-slot retries preserve the original input/source bytes and earlier recorded
versions; the current process never backfills provenance into archived evidence.

Corporate-action safety: captured daily dividends/splits and current quote events
use the same execution and valuation eligibility guard. An unresolved held-symbol
event produces `PENDING_ACTION_DATA`, no simulated fill and no rankable valuation.
The mutable paper state's `as_of` is never treated as acquisition proof. Explicit
position acquisition metadata, when available, or immutable book creation bounds
can exclude demonstrably older events; otherwise the guard is conservative.
Actual dividend cash entitlements, split share adjustments and their immutable
ledger receipts remain unimplemented coverage, rather than fabricated returns.

An unresolved captured action is permanently latched on the isolated book using
its original immutable execution-attempt observation and checksum. The latch is
written before cleanup or execution and survives failed cleanup, source-window
rollover, policy supersession and a late action after an existing frontier proof.
New pending valuation evidence retains the original content and observation time;
it is execution/valuation evidence, not additional model-consumed context. Only
real future action accounting may justify resolving the entitlement; the current
implementation cannot clear this safety latch or fabricate adjustments.
