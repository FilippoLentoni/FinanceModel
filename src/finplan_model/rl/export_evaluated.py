"""Export existing evaluated actor members without training or deserializing code.

Legacy research runs selected a single seed rather than the newer ensemble. Each
actually evaluated configuration/seed remains a distinct candidate; this module
never manufactures an ensemble or changes a frozen observation shape.
"""
from __future__ import annotations
import copy

from finplan_model.core.artifacts import canonical_json_bytes, sha256_checksum
from finplan_model.core.errors import FinplanError
from finplan_model.rl.inference import actor_action, export_actor
from finplan_model.rl.spec import EnvSpec


def _require(condition, reason):
    if not condition:
        raise FinplanError.precondition(reason, reason=reason)


def export_evaluated_members(result, spec, artifacts, *, algorithm="sac"):
    """Return independent actor bundles from a verified succeeded research run.

    ``artifacts`` is the existing checksum-verifying ArtifactStore. Its immutable
    full diagnostics must agree with every selected policy row before any tensor
    export runs. The caller persists returned bundles through the same store as
    ``policy_inference`` artifacts. No provider retrieval, compute submission,
    training, production promotion or portfolio mutation occurs here.
    """
    import numpy as np
    _require(algorithm in ("ppo", "sac"), "unsupported_actor_algorithm")
    _require(result.get("completion_status") == "succeeded" and spec.get("purpose") == "research"
             and spec.get("environment") == "beta" and spec.get("synthetic") is not True,
             "evaluated_real_beta_research_required")
    _require(result.get("run_id") == spec.get("run_id") and result.get("input_snapshot_id") == spec.get("input_snapshot_id"),
             "evaluated_run_identity_mismatch")
    summary = result.get("payload", {}).get("model_selection", {})
    policies = [r for r in summary.get("policies", []) if r.get("algorithm") == algorithm]
    _require(bool(policies), "evaluated_actor_members_missing")
    refs = {r["artifact_id"]: r for r in result.get("artifacts", [])}
    _require(len(refs) == len(result.get("artifacts", [])), "duplicate_evaluated_artifact_reference")
    full = refs.get(summary.get("full_diagnostics_artifact_id"))
    if full is None and not summary.get("full_diagnostics_artifact_id"):
        # Early v1 runs linked one full diagnostics artifact without the later ID field.
        candidates = [r for r in refs.values() if r.get("kind") == "run_artifact"]
        full = candidates[0] if len(candidates) == 1 else None
    _require(full and full.get("kind") == "run_artifact", "evaluated_diagnostics_missing")
    diagnostics = artifacts.get_json(full)
    original_rows = [r for r in diagnostics.get("policies", []) if r.get("algorithm") == algorithm]
    _require(canonical_json_bytes(original_rows) == canonical_json_bytes(policies), "evaluated_policy_provenance_mismatch")
    protocol = spec["selection_protocol"]
    _require(canonical_json_bytes(diagnostics.get("protocol")) == canonical_json_bytes(protocol)
             and canonical_json_bytes(diagnostics.get("selection")) == canonical_json_bytes(summary.get("selection")),
             "evaluated_selection_provenance_mismatch")
    train_end = protocol["splits"]["train"]["end"]
    validation_end = protocol["splits"]["validation"]["end"]
    _require(train_end < protocol["splits"]["validation"]["start"] <= validation_end
             < protocol["splits"]["test"]["start"], "evaluated_split_chronology_invalid")
    grid = summary.get("rl", {}).get(algorithm, {}).get("grid", [])
    env = {k: v for k, v in summary["rl"][algorithm]["environment"].items() if k != "reward_formula"}
    selected = summary.get("selection", {}).get("chosen_hyperparameters", {}).get(algorithm, {})
    instruments = sorted(spec["universe"])
    _require(len(instruments) == len(set(instruments)), "evaluated_universe_invalid")
    identities = set()
    validated = []
    # Validate every candidate before exporting the first candidate.
    for row in policies:
        identity = (row["configuration_id"], row["seed"])
        _require(identity not in identities, "duplicate_evaluated_member")
        identities.add(identity)
        ref = refs.get(row["artifact_id"])
        _require(ref and ref.get("kind") == "rl_policy" and ref.get("owner") == "financemodel"
                 and ref["checksum"] == row["checksum"], "evaluated_policy_checksum_mismatch")
        matches = [g for g in grid if g.get("configuration_id") == row["configuration_id"] and g.get("reward") == row["reward"]]
        _require(len(matches) == 1 and row["seed"] in [c["seed"] for c in matches[0].get("checkpoints", [])]
                 and row["seed"] in protocol["rl"]["seeds"], "evaluated_member_not_in_validation_grid")
        member_env = copy.deepcopy(env)
        member_env["reward"] = {**member_env.get("reward", {}), **row["reward"]}
        parsed = EnvSpec.from_dict(member_env)
        _require(not any("news" in f or "gdelt" in f for f in parsed.features),
                 "news_actor_requires_original_corpus_export_protocol")
        checkpoint = next(c for c in matches[0]["checkpoints"] if c["seed"] == row["seed"])
        validated.append((row, ref, member_env, parsed, checkpoint))
    exported = []
    for row, ref, member_env, parsed, checkpoint in validated:
        # ArtifactStore verifies full original SHA before weights_only tensor load.
        data = artifacts.get(ref)
        actor = export_actor(data, algorithm)
        # Verify tensor dimensions against the original declared observation and action spaces.
        closes = np.ones((parsed.window + 1, len(instruments)), dtype=float)
        from finplan_model.rl.spec import build_observation
        observation = build_observation(parsed, closes, np.zeros(len(instruments)), 1.0, drawdown=0.)
        action = actor_action(actor, observation)
        _require(action.shape == (len(instruments) + 1,) and np.isfinite(action).all(), "evaluated_actor_shape_mismatch")
        chosen = (selected.get("configuration_id") == row["configuration_id"] and selected.get("seed") == row["seed"])
        variant_id = algorithm + "_" + row["configuration_id"].removeprefix("cfg_")[:12] + "_s" + str(row["seed"])
        bundle = {"format": "finplan-strategy-bundle/1", "mode": "advisory_paper", "strategy_id": algorithm,
                  "source_run_id": result["run_id"], "configuration_id": row["configuration_id"],
                  "instruments": instruments, "environment": member_env,
                  "training_end": train_end, "available_after": validation_end,
                  "constraints": copy.deepcopy(spec["simulation"]["constraints"]),
                  "constraint_policy": spec["simulation"].get("constraint_policy", "project"),
                  "members": [{"seed": row["seed"], "source_checksum": ref["checksum"], "actor": actor}],
                  "evaluated_variant": {"variant_id": variant_id, "aggregation": "single_evaluated_seed",
                      "selected_in_original_run": chosen, "reward": copy.deepcopy(row["reward"]),
                      "source_policy_artifact": copy.deepcopy(ref), "full_diagnostics_artifact": copy.deepcopy(full),
                      "source_dataset_checksum": result["dataset_checksum"], "source_snapshot_id": result["input_snapshot_id"],
                      "source_spec_checksum": sha256_checksum(canonical_json_bytes(spec)),
                      "evaluated_checkpoint": copy.deepcopy(checkpoint),
                      "original_hyperparameters": copy.deepcopy(protocol["rl"].get(algorithm, {})),
                      "policy_export_version": "finplan-evaluated-members/1", "training_performed": False}}
        exported.append(bundle)
    return exported
