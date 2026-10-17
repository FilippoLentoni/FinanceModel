"""Freeze per-approach model identities without dropping hourly accounting.

A registry is immutable experiment configuration. Newly discovered artifacts do
not replace a canonical PPO/SAC policy in an existing book. Operators create a
new experiment for intended changes; individual blocked slots remain visible.
"""
from __future__ import annotations

import copy

from finplan_model.core.artifacts import canonical_json_bytes, sha256_checksum

IDENTITY_FIELDS = (
    "family", "adapter", "version", "native_cadence", "news_layer",
    "strategy_name", "parameters", "input_capabilities", "policy_checksum",
)


def _identity(approach, bundle=None):
    identity = {key: copy.deepcopy(approach.get(key)) for key in IDENTITY_FIELDS}
    if bundle is not None:
        identity["policy_checksum"] = sha256_checksum(canonical_json_bytes(bundle))
        identity["version"] = bundle["configuration_id"]
    return identity


def _by_id(rows):
    result = {}
    for row in rows:
        aid = row["approach_id"]
        if aid in result:
            raise ValueError("duplicate approach identity in experiment registry")
        result[aid] = row
    return result


def freeze_experiment_models(existing_registry, prospective_approaches, current_policies):
    """Return frozen rows, usable policies and per-approach precondition blockers.

    Call after discovery and before registry creation / inference. Persist
    ``blockers`` and ``inventory_changes`` in the immutable shared source, and
    record a not-invoked unavailable run for each blocker. Do not overwrite these
    model identities when enriching actual per-run code/trigger metadata later.

    Same-slot recovery must first adopt its original source/artifacts. This
    function never fetches an artifact or regenerates historical input content.
    """
    prospective = _by_id(prospective_approaches)
    current_policies = current_policies or {}
    if existing_registry is None:
        approaches = copy.deepcopy(prospective_approaches)
        for row in approaches:
            bundle = current_policies.get(row["approach_id"])
            if bundle is not None:
                identity = _identity(row, bundle)
                row.update(version=identity["version"], policy_checksum=identity["policy_checksum"])
        return {"approaches": approaches,
                "policies": {aid: copy.deepcopy(bundle) for aid, bundle in current_policies.items() if aid in prospective},
                "blockers": {}, "inventory_changes": {"new_approach_ids": [], "missing_approach_ids": [],
                    "unregistered_policy_ids": sorted(set(current_policies) - set(prospective))}}

    frozen = _by_id(existing_registry["approaches"])
    blockers, usable = {}, {}
    for aid, original in frozen.items():
        proposed = prospective.get(aid)
        expected = _identity(original)
        bundle = current_policies.get(aid)
        if proposed is None:
            blockers[aid] = {"reason": "registered_approach_missing_from_inventory", "invocation_state": "not_invoked",
                             "expected_identity": expected, "observed_identity": None}
            continue
        observed = _identity(proposed, bundle)
        if expected.get("policy_checksum") and bundle is None:
            # A previously bound model remains bound even if today's inventory
            # no longer returns it. Never accept a label alone as original bytes.
            observed["policy_checksum"] = None
            reason = "frozen_policy_artifact_unavailable"
        else:
            reason = "frozen_model_identity_changed"
        if canonical_json_bytes(expected) != canonical_json_bytes(observed):
            blockers[aid] = {"reason": reason, "invocation_state": "not_invoked",
                "expected_identity": expected, "observed_identity": observed,
                "changed_fields": [field for field in IDENTITY_FIELDS if canonical_json_bytes(expected[field]) != canonical_json_bytes(observed[field])],
                "resolution": "Restore the original frozen model or create a new versioned experiment; do not relabel an existing book"}
            continue
        if bundle is not None:
            usable[aid] = copy.deepcopy(bundle)
    return {"approaches": copy.deepcopy(existing_registry["approaches"]), "policies": usable, "blockers": blockers,
            "inventory_changes": {"new_approach_ids": sorted(set(prospective) - set(frozen)),
                "missing_approach_ids": sorted(set(frozen) - set(prospective)),
                "unregistered_policy_ids": sorted(set(current_policies) - set(frozen))}}
