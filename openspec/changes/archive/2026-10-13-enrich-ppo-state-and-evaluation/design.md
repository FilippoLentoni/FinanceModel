# Design

## Context

See proposal.md. Four independently released repositories already provide beta portfolio advice and bounded research. Preserve the existing ownership boundaries and uncommitted archive work.

## Goals / Non-Goals

Goals: controlled state experiments, honest news eligibility, auditable results and the authorized ceiling. Non-goals: live trading, automatic strategy activation, paid GPU research or retrospective news fabrication.

## Decisions

Use existing recursive_evaluate CPU Training jobs with server-approved state profiles; retain public job and result schemas. Add five profiles with five seeds, fixed risk setting 0.5, 30000 steps, unchanged action/reward and existing splits. Add trailing correlations/portfolio volatility, universe regime summaries, and five recent log returns per instrument. Share construction across all three inference paths. Add timestamped research-news records/collector and a strict point-in-time join; inspect coverage before considering training. Do not backfill historical state with contemporary news. Old actors remain unchanged. Deploy beta only; retain the prior served artifact. The study uses the previously inspected 2026 path as exploratory development. No production promotion.

The user explicitly requested proposal followed by apply in this turn; that authorization overrides the generic propose skill planning-only pause. USD 100 is the total ceiling. Start with at most USD 5 incremental work and check billed/project-ledger headroom before each compute launch.

## Risks / Trade-offs

Inspected market path → development label and no promotion. Thin training history → report seed dispersion without equating seeds to independent market regimes. Missing news vintages → explicit ineligibility. Public repositories → synthetic fixtures only; private live receipts stay outside repos.

## Migration Plan

Validate locally, deploy owner artifacts through beta pipelines, preserve gamma/prod state and transition baselines, verify compatible consumer releases. Retain prior selected policy reference for rollback; no automatic activation.

### Frozen implementation compatibility

Issued decisions compare raw inference, observation and horizon source checksums. Retain the
exact previous inference/observation and horizon sources as versioned modules, and dispatch
by frozen feature definition and recorded implementation identity. Old decisions keep their
original identities; enriched actors receive a new inference identity. This preserves replay
rather than treating an additive source change as an acceptable checksum mismatch.
