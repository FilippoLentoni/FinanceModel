"""Choose the frozen price-only PPO export trained on an exact instrument universe.

One rule for both callers so they always pick the same bytes: the forward runner's
direct ``ppo`` book (``forward.hourly.load_current_inputs``, cohort ``eight_assets``) and
the served ``generate_portfolio_recommendation_ppo`` tool (``rl.advisory.recommendation``).
"""
from __future__ import annotations

import json

from finplan_model.core.artifacts import ArtifactRef, sha256_checksum
from finplan_model.core.errors import FinplanError

POLICY_PREFIX = "artifacts/policy_inference/"
#: Bounded inventory: exports larger than this are not frozen PPO inference bundles.
MAX_EXPORT_BYTES = 2_000_000
MAX_EXPORTS = 100


def price_only(bundle):
    return not any("news" in f or "gdelt" in f for f in bundle.get("environment", {}).get("features", []))


def universe_candidates(bundles, instruments):
    """Price-only PPO exports whose fixed universe is exactly ``instruments`` (as a set)."""
    wanted = set(instruments)
    return [b for b in bundles if b.get("strategy_id") == "ppo" and len(b.get("instruments", [])) == len(wanted)
            and set(b.get("instruments", [])) == wanted and price_only(b)]


def select_primary(candidates):
    """Prefer an originally selected ensemble over newly exported individual seeds."""
    if not candidates:
        return None

    def priority(p):
        evaluated = p.get("evaluated_variant")
        selected = evaluated is None or evaluated.get("selected_in_original_run") is True
        selected_ensemble = selected and len(p.get("members", [])) > 1
        return (selected_ensemble, selected, p.get("source_run_id", ""), p.get("configuration_id", ""))
    return max(candidates, key=priority)


def list_exports(s3, bucket):
    """Every bounded frozen export as ``(artifact_ref, bundle)``; content checksum verified."""
    listing = s3.list_objects_v2(Bucket=bucket, Prefix=POLICY_PREFIX, MaxKeys=MAX_EXPORTS)
    if listing.get("IsTruncated"):
        raise FinplanError.precondition("frozen policy inventory exceeds its bound", reason="policy_inventory_bound")
    out = []
    for row in listing.get("Contents", []):
        if row["Size"] > MAX_EXPORT_BYTES:
            continue
        stored = s3.get_object(Bucket=bucket, Key=row["Key"])
        data = stored["Body"].read()
        checksum = sha256_checksum(data)
        if stored.get("Metadata", {}).get("checksum") != checksum:
            raise FinplanError.precondition("frozen export metadata/content checksum mismatch", reason="policy_checksum_mismatch")
        try:
            bundle = json.loads(data)
        except ValueError:
            continue
        if not isinstance(bundle, dict):
            continue
        ref = ArtifactRef.from_dict({"artifact_id": row["Key"].rsplit("/", 1)[-1], "kind": "policy_inference", "checksum": checksum,
                                     "content_type": "application/json", "size_bytes": len(data)}).to_dict()
        out.append((ref, bundle))
    return out


def select_for_universe(exports, instruments):
    """``(ref, bundle)`` of the policy for exactly ``instruments`` or a typed precondition error."""
    by_identity = {}
    for ref, bundle in exports:
        by_identity.setdefault(id(bundle), ref)
    chosen = select_primary(universe_candidates([b for _, b in exports], instruments))
    if chosen is None:
        raise FinplanError.precondition(
            "no frozen PPO policy is trained on exactly the requested instruments",
            reason="no_policy_for_universe", requested_instruments=sorted(set(instruments)))
    return by_identity[id(chosen)], chosen
