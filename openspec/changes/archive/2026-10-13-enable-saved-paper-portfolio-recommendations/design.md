# Design

## Context

See proposal.md for motivation. Existing inference loads a frozen actor ensemble or baseline, validates explicit holdings, and uses approved snapshot observations. Platform owns the paper book; the serving Lambda has no write access or training dependency.

## Goals / Non-Goals

**Goals:** Resolve saved state through platform APIs, preserve frozen actor parity, make valuation and indicative trades auditable, and retain explicit experiment requests.

**Non-Goals:** Broker execution, simulated fills, automatic paper rebalances, retraining, production promotion, or new diagnostic workers.

## Decisions

- Add default-context resolution before existing inference. Read the research-plan reference lazily from environment SSM, then the plan and portfolio state through signed platform GETs. An optional portfolio identifier selects a different saved paper book. This avoids duplicating platform ownership or requiring agent-side state assembly.
- Select the newest approved universe snapshot through the platform interface; derive its newest aligned completed session when no explicit snapshot/date is supplied. Explicit inputs remain supported for reproducible experiments.
- Maintain two price bases. Existing market loading keeps adjusted observations for trained features. Raw snapshot closes value actual saved share quantities and convert target notionals into fractional shares. Treat these quantities as indicative, excluding execution slippage and fees.
- Revalue the unchanged saved book through each completed session since its state date to reconstruct high watermark without a write. Reject missing aligned history instead of understating drawdown. Corporate actions are not automatic ledger events; explicit book updates remain necessary when quantities or cash change.
- Add narrowly scoped SSM and GET permissions for plan/state reads. Retain denies on storage mutation, selection changes and training.

## Risks / Trade-offs

- Stale observations → report the exact completed data date and refresh beta's approved snapshot before acceptance.
- No broker or corporate-action accounting → label the book paper and quantities indicative; require explicit operator state updates for fills, splits, dividends and cash flows.
- Independent pipeline rollouts → publish contracts first, deploy platform before consumers, verify same-environment calls through MCP and the hosted agent.
- Budget and stage spillover → no training/export, cap this round at $2, disable Gamma inbound before owned executions and restore gates only after they are stopped.

## Migration Plan

Publish contracts 1.3.0, update consumer pins, deploy the four beta artifacts, refresh completed market data, and initialize the user-authorized $10,000 equal-weight paper book exactly once via the platform operator API. Verify natural requests, arithmetic and read-only behavior. Existing explicit requests support rollback; no state migration beyond optional paper-state initialization is required.
