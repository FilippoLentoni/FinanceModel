## Context

Existing classical research stores immutable analyses and enforces one CPU experiment per week, USD 0.50 weekly, USD 2 monthly and USD 50 total. Model selection freezes chronological splits and selects on validation; reused tests are explicitly research-only. The account's other-project batch pattern discovered Qwen/Qwen3.6-27B revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 in offline vLLM Training Jobs. Jev's official OpenAPI uses named choice questions with criteria and probability answers; authenticated read-only model discovery returns jev-latest and jev-preview.

## Goals / Non-Goals

Goals: real dispatchable adapters, durable bounded review, honest evaluation, no idle GPU, reproducible inputs and logs. Non-goals: guaranteed performance, arbitrary automatic source execution, automatic activation, live trading, substituting model releases, claiming historical LLM results are leakage-free.

## Decisions

Cycles freeze max_iterations and portfolio identity. Each immutable iteration references its cycle, parent iteration, evidence, exact experiment request and job/result. Conditional claims serialize advances and bind paid launches to idempotency. Weekly scheduling resumes eligible cycles within the same existing weekly/monthly controls; dry runs never start compute. Evaluated research results can suggest the next server-approved PPO or optimizer configuration; unsupported features and algorithms become reviewable code/spec proposals. Reused holdout cannot authorize promotion; fresh forward evidence and explicit activation remain required.

Qwen uses run-scoped network-isolated SageMaker Training, staged verified weights and local vLLM, with a hard runtime and shared GPU concurrency lease. No endpoint is created. A completed job releases compute; retries create distinct attempts. Fixed analyst/risk/allocator/critic/arbiter roles produce sealed message logs. Invalid targets fall back to holding. Jev sends only bucketed point-in-time descriptors under research purpose, never raw series; one choice per instrument prevents contradictory buy/sell labels. Calls are bounded and retries limited, responses cached and drift recorded. Calibration and portfolio profitability are reported separately.

User decision 28 supersedes the earlier monthly setting: every family and comparison control decides daily. LLM previews freeze 22 aligned completed snapshot sessions for exactly 21 decisions while retaining the full earlier history for point-in-time warmup. Submission and runtime guards reject frequency mismatch, missing/late bars, insufficient history and over-32-decision windows before inference. This is a limited same-window pilot, not the full 2026 or untouched PPO holdout benchmark. Jev's next-21-session classification horizon remains separate from its daily decision cadence. Existing monthly previews fail closed and need a fresh review; their approved configuration is never silently rewritten. Costs, approvals and runtime limits remain unchanged.

## Risks / Trade-offs

GPU startup/loading may consume the pilot runtime. Discovery is not a successful finance benchmark. Exact Jev version is recorded from responses, because the available model list exposes aliases only. Historical pretraining leakage remains unresolved; prospective evaluation is required.

## Migration Plan

Publish schemas, deploy beta handlers and controller, then expose the tools/skills. CPU mocks exercise lifecycle and failures in CI. GPU and Jev real inference require separately reviewable estimates and approval. Rollback redeploys the previous immutable image, retains all research evidence and creates no endpoint.

## Open Questions

Whether the current runtime cap is adequate for the exact Qwen checkpoint must be measured in an approved GPU pilot. Its model pretraining cutoff is not verified. Neither uncertainty permits a substituted model or automatic promotion.
