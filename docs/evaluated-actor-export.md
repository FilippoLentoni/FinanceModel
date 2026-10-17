# Exporting existing evaluated actors without training

`rl.export_evaluated.export_evaluated_members` supports legacy succeeded beta
research runs that selected one seed, alongside newer runs that recorded a full
diagnostics artifact ID. Each actually evaluated configuration and seed becomes
an independent bundle with the original observation shape, universe, reward
configuration, validation checkpoint and source checksum. It never creates an
ensemble from candidates that were not previously evaluated as an ensemble.

The caller reads the original source job result and spec through supported
storage, verifies them against the succeeded job-control record, and supplies
the existing checksum-verifying artifact store. The exporter verifies the
immutable full diagnostics, policy rows, selection protocol, chronological
splits, validation-grid membership and every original policy reference before
exporting tensors. The existing `export_actor` reads `policy.pth` with
`torch.load(weights_only=True, map_location='cpu')`; SB3's cloudpickled objects
are not deserialized. An observation/action shape check validates the frozen
declaration. This operation does not train, submit compute, alter budgets,
promote a policy or mutate a portfolio.

Persist returned bundles with `ArtifactStore.put_json(bundle,
kind='policy_inference', domain='finance')` and verify a read-back. Assign each
bundle's `evaluated_variant.variant_id` a separate comparison book. The selected
candidate may use its canonical strategy ID; preserve the variant identity in
its model metadata and avoid duplicating it as an extra identically weighted
book. All other meaningful evaluated variants retain their own identities.
Checkpoints whose original `best_step` is zero remain disclosed as such.

News-aware actors require their existing corpus-aware export protocol and are
rejected by this compatibility helper. It must not invent news features or
change a policy's observation shape. Native daily signals remain daily even
when evaluated at hourly comparison opportunities. Allocation artifacts are
not portfolio-value forecasts.

`tests/unit/rl/test_export_evaluated.py` uses controlled tensors, without learner
training or network access. It checks independent member identity, source and
checksum integrity, unchanged constraints, chronology, universe, validation
membership, rejected news features and corrupt payload rejection before tensor
execution. Existing inference parity tests compare exported CPU actor outputs
with native deterministic SB3 PPO/SAC actions.
