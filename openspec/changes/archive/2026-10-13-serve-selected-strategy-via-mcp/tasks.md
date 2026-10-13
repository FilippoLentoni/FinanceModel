# Tasks

## 1. Implementation and local verification

- [x] 1.1 Freeze evaluated PPO/SAC or baseline configurations from an existing source run and export actors without training; verify missing members, unknown strategies and checksum mismatches fail in unit tests.
- [x] 1.2 Implement the shared recommendation dispatcher with current-state validation, exact actor transforms and frozen constraints; verify PPO/SAC action parity and a classical-strategy recommendation.
- [x] 1.3 Add the dedicated 270-second serving Lambda, publish its same-environment reference and scope its IAM to required reads; verify handler tests, synth and negative permission checks.
- [x] 1.4 Document activation, request inputs and advisory limitations; verify examples against the pinned contracts and retain production-promotion tests.

## 2. Deployed beta verification

- [x] 2.1 Deploy beta through the existing pipeline, export and pin the existing research policy under a dry-run estimate, and verify repeatable inference with no training job created; record latency and incremental cost within USD 2.

## Workflow follow-up

- Review before archiving; existing explanation and gamma/prod tasks retain their own acceptance criteria.
