# Exact Qwen and TypeSafe Jev research backends

These beta job types are runnable adapters, not production allocation strategies.
All strategies use the same point-in-time market view, common simulator, costs, fills,
constraints and cash/buy-and-hold/equal-weight controls. Messages, choices, model identities,
configuration, usage and outcomes are sealed in run artifacts. Pretraining cutoffs are
unknown, so all historical LLM performance carries a leakage disclosure and requires
prospective paper validation. No strategy is automatically activated.

The daily-protocol correction is deployed in beta release `rel_01M4M6Q83GFWVT2VSDA0J7NHNP`
from source `d6edabee2a8abc18d0876bd84cde45d404db6d86`. Build/deploy/integration gates and
live daily Qwen/Jev dry-run requests passed. Older monthly previews must be refreshed
before approval and are rejected without launching. Real paid inference remains unvalidated.

## Daily pilot scope

Every family makes decisions daily, as required by user decision 28. Qwen and Jev previews
freeze the latest **22 actual aligned completed market sessions**, giving **21 daily
decisions** and a final session to fill the last decision. The saved snapshot remains whole:
earlier point-in-time observations still supply trailing features, including the 60-return
lookback when available. Missing/late bars, insufficient warmup and windows above 32
decisions are rejected before inference; the submission control also checks the approved
snapshot before recording a paid run. Calendar holidays are taken from the snapshot.

Cash, buy-and-hold and equal-weight controls use that exact daily window, simulator and
cost model. Results label this a **limited daily pilot**, not the full 2026 benchmark or an
untouched PPO/optimizer holdout comparison. Comparing it with old metrics from different
dates or decision frequencies is invalid. A broader aligned benchmark remains a separately
approved experiment with reviewed runtime/call bounds; this pilot cannot justify promotion.

## Qwen swarm

The read-only AWS example identified `Qwen/Qwen3.6-27B`, revision
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, Apache-2.0, about 51.8 GiB of files,
`qwen3_5` / `Qwen3_5ForConditionalGeneration`. The reused deployment pattern is an
offline vLLM SageMaker Training Job on `ml.g6.12xlarge` (four L4 GPUs), tensor parallel four,
200 GiB disk. No always-on endpoint is created.

The verified AWS DLC is
`<AWS-DLC-registry>/vllm@sha256:8d2afde34771171cc25d1fbb4353b1a8a4b9bd08d2308eff7799e1b002213f6d`,
original tag `0.29.0-gpu-py312-cu130-ubuntu24.04-sagemaker-v1.2`.
The operator supplies this digest in
`/finplan/beta/financemodel/config/vllm-image`. The release publishes an immutable small
code channel built with CPython 3.12 validation dependencies; it does not rebuild the DLC.
Referenced code is retained under `releases/qwen-code/<release_id>/` in research storage,
outside the expiring `scratch/` prefix. Temporary weights and run hand-off files retain
their separate scratch lifecycle; an old release's code remains available after that window.

`rl_weight_staging` stages only the exact revision into FinanceModel's own short-lived
S3 scratch prefix, bounded to 60 GiB. It verifies pinned metadata, LFS digests and license,
writes a manifest/checksummed ready marker last, and requires approval before downloading.
`swarm_mode_a` checks that marker and supplies approved input, weights and code as three
S3 channels. With network isolation enabled, SageMaker downloads the channels and uploads
outputs outside the container; inference has no network or AWS credentials. The engine
verifies every local weight file before loading and becomes ready before any decision.

Five fixed roles run sequentially: market analyst, risk analyst, allocator, critic, arbiter.
The allocator proposes a complete long-only simplex; bounded retries and critic revision
are allowed, and the arbiter chooses only among validated proposals. Invalid messages
hold the existing book. The released caps are 32 decisions, 224 role calls and 256 output
tokens per response, deterministic sampling and disabled thinking. The 21-decision daily
pilot needs at most 147 role calls, including retries/revisions. The 32-decision/224-call
hard bounds are unchanged. These caps do not guarantee that model loading plus inference
fits within the runtime limit.

The existing dispatcher controls provisioning, GPU lease, quota retry, cancellation,
runtime stop, terminal result import and lease release. Output imports enforce run identity,
trusted S3 prefix, tar member/size bounds and artifact checksums. Engine teardown runs even
on evaluation failure; SageMaker destroys the run-scoped compute. There is no endpoint to
forget to delete.

## TypeSafe Jev

The verified interface is `POST https://api.typesafe.ai/v1/systemone` with `{state,model,
questions}`. Each named question requests `type:"choice"` and buy/hold/sell criteria.
Responses contain named choices, probabilities, confidence, returned model and token usage.
Read-only authenticated `/v1/models` discovery returned `jev-latest` and `jev-preview`.
The [vendor model documentation](https://docs.typesafe.ai/models), checked on 2026-10-10,
also documents `jev-1.13.0` and says explicit version IDs are accepted even when discovery
omits them. This beta configuration uses `jev-latest` and retains the returned model/version
and drift status in artifacts; it does not claim an immutable vendor model pin. A paid
request with either identity has not been validated in this release.

`jev_backtest` reads `finplan/shared/financemodel/jev-api-key` in memory. Only bucketed text
descriptors of point-in-time trend, volatility and exposure leave AWS. The shared outbound
guard runs before cache lookup and each retry; no raw Yahoo series or market dates are sent.
Named probabilities determine bounded target-weight changes with fixed thresholds; the
simulator applies executable portfolio constraints and costs.

HTTP 429/529 and transient failures have bounded backoff; 401/422 fail without retry.
The client refuses redirects, invalid vectors/confidence, contradictory choices and
missing usage. Maximum 64 calls and 200,000 conservatively reserved input tokens include
failed attempts. Responses are cached within the run. Secret values and provider error
bodies are never logged. Jev always requires approval despite the small AWS CPU estimate.

Calibration is measured separately with next-21-session buy/hold/sell labels: multiclass
Brier, negative log likelihood and classification accuracy. Portfolio profitability is
not inferred from classification accuracy or vendor calibration claims. Current sizing
thresholds are fixed, not tuned on test; validation-only calibration/threshold fitting is
still a separate research task. This 21-session forecast horizon is distinct from the daily
decision frequency. Within the 22-session pilot only the first decision's full forecast
horizon can mature; calibration is correspondingly thin or unavailable and excludes labels
beyond the requested end. Overlapping labels reduce independent evidence.

Primary API references: [vendor documentation index](https://docs.typesafe.ai/llms.txt),
[vendor OpenAPI](https://api.typesafe.ai/openapi.json). AWS lifecycle reference:
[network isolation](https://docs.aws.amazon.com/sagemaker/latest/dg/mkt-algo-model-internet-free.html).

## Cost and remaining validation

Read-only AWS Price List checks on 2026-10-10 found `ml.g6.12xlarge` Training $5.752/hour,
and `ml.m5.xlarge` Training and Processing both $0.23/hour in us-east-2. Operator parameters
must remain fresh; missing/stale prices fail closed. At those prices the 900-second Qwen
runtime plus disclosed 600-second startup allowance and $.01 storage estimates $2.406667.
Staging includes a conservative $.70 scratch-storage planning allowance, giving $.7575.
These are planning estimates, not hard cloud billing caps; startup/storage must be reconciled.

Jev's independent vendor planning rate, checked against the
[vendor model documentation](https://docs.typesafe.ai/models) on 2026-10-10, is
$.042/million input tokens, output free; the
200,000 token bound implies $.0084 vendor credits, separately from the $.0675 AWS CPU/storage
estimate. Vendor pricing/credits must be verified before approval. The external cap is
disclosed; this charge is not counted as AWS spend.

Unit tests verify actual dispatch/approval guards, exact manifest integrity, fixed-role
arbitration, common evaluation, outbound guards, retry/call/token bounds, calibration and
failure lease cleanup with fake compute. **No paid GPU, weight-staging, or Jev inference
benchmark was launched during implementation.** The observed earlier AWS example lasted
about 4,754 seconds; a 900-second run may time out loading or evaluating the exact model.
Full real-GPU readiness/performance validation remains pending specific budget approval.
