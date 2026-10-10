## 1. Evidence and recursive controller

- [x] 1.1 Implement immutable bounded cycle/iteration orchestration, asynchronous resume, server-approved PPO/optimizer candidate definitions and weekly scheduling with existing cost gates; test lineage, conflicting resume, dry-run and stopping rules.

## 2. Benchmark adapters and dispatch

- [x] 2.1 Implement exact Qwen checkpoint verification, offline vLLM fixed-role swarm and run-scoped job dispatch lifecycle; test parsing, readiness, teardown and retry conditions without GPU work.
- [x] 2.2 Implement verified Jev client, outbound descriptor guard, bounded caching/retry, deterministic sizing and calibration; test permanent/transient failures, contradictory labels, model drift and common simulator accounting.

## 3. Release and evidence

- [ ] 3.1 Publish beta job configurations and documented invocation examples; validate focused OpenSpec and required unit/build checks, retaining explicit real-GPU and vendor validation status.

Implementation, local configurations, invocation and paid-validation-status documentation are complete.
Focused unit/infra checks and strict OpenSpec validation pass. Release/build checks and deployed
beta verification are coordinated by the parent rollout agent. No paid staging, GPU or vendor
inference is claimed complete; these remain explicit separately approved validation steps.
