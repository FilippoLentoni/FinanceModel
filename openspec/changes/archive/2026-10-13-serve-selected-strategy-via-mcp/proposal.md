# Proposal

## Why

On 2026-10-09 the user clarified that the agent must invoke whichever portfolio strategy is selected after offline experimentation through an MCP Lambda target. The existing research pipeline trains policies but does not provide this strategy-neutral on-demand serving path.

## What Changes

- Freeze a user-selected, evaluated strategy and parameters from a succeeded offline run, and export trained PPO/SAC actors without retraining.
- Serve the pinned strategy through a dedicated Lambda, with a 270-second cap, called directly by the MCP adapter.
- Support the existing control/classical strategies and trained PPO/SAC through one allocation interface, preserving constraints, state transforms and provenance.
- Keep research selection advisory-only in beta until promotion; retain existing production selection and scheduled daily restrictions.

## Capabilities

### New Capabilities

- `selected-strategy-serving`: this repository's part of reproducible, on-demand selected-strategy recommendations.

### Modified Capabilities

None in the archived spec inventory. This complements the existing in-flight daily and explanation changes; it does not replace performance replay with an allocation-hold approximation.

## Impact

Frozen strategy bundle/export job, NumPy inference, baseline dispatcher, selected-strategy Lambda and narrowly scoped IAM. No standing model endpoint or scheduled experiment. Validation starts offline. Any new AWS work in this round stays below USD 2 and within the user's USD 50 project budget; no fresh training is authorized by an inference request.
