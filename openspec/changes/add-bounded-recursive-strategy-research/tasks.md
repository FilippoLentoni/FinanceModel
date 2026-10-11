## 1. Evidence and recursive controller

- [x] 1.1 Implement immutable bounded cycle/iteration orchestration, asynchronous resume, server-approved PPO/optimizer candidate definitions and weekly scheduling with existing cost gates; test lineage, conflicting resume, dry-run and stopping rules.

## 2. Benchmark adapters and dispatch

- [x] 2.1 Implement exact Qwen checkpoint verification, offline vLLM fixed-role swarm and run-scoped job dispatch lifecycle; test parsing, readiness, teardown and retry conditions without GPU work.
- [x] 2.2 Implement verified Jev client, outbound descriptor guard, bounded caching/retry, deterministic sizing and calibration; test permanent/transient failures, contradictory labels, model drift and common simulator accounting.

## 3. Release and evidence

- [x] 3.1 Publish beta job configurations and documented invocation examples; validate focused OpenSpec and required unit/build checks, retaining explicit real-GPU and vendor validation status.


Beta publication and all pipeline build/integration gates passed on 2026-10-11 for source
`3b95f8dffd549a4df297fcc24811f12aad9c07a4`, execution
`38b333be-9986-4825-b047-4c66eb5f05e9`, release
`rel_01M4M56RP6FZNARQ26EVNK1WEJ`. Qwen release code is retained outside the scratch
lifecycle and read/image permissions were verified. Real paid staging/GPU/vendor inference
is explicitly unvalidated and remains separately approval-gated; no such run is claimed.
Hosted/direct research acceptance is recorded by the cross-repository beta verification.

## 4. Daily protocol correction (after the release above)

- [x] 4.1 Align displayed/submitted PPO and classical research candidates and Qwen/Jev pilots to daily decisions; freeze 22 aligned sessions/21 decisions, retain warmup, bound preflight/runtime before inference, and reject obsolete monthly previews.
- [x] 4.2 Disclose limited-pilot scope separately from full-year/holdout comparisons and Jev forecast horizon; verify focused recursive, benchmark and control tests plus release gates.
- [x] 4.3 Redeploy the daily correction to beta and verify live dry-run requests without paid inference.

The daily correction passed every beta pipeline gate for source
`d6edabee2a8abc18d0876bd84cde45d404db6d86`, execution
`87929817-49cf-484f-974a-5e9fe5a46ed2`, release
`rel_01M4M6Q83GFWVT2VSDA0J7NHNP`. Live PPO, Qwen and Jev requests verified daily
cadence; LLM pilots froze 22 sessions/21 decisions with warmup retained. No paid inference ran.
