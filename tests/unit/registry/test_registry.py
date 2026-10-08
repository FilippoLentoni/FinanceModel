"""Model registry (task 9.2; REG-01, REG-02) with the in-memory store and the S3 store (moto)."""

from __future__ import annotations

import threading

import boto3
import pytest
from moto import mock_aws

from finplan_model.core.errors import ErrorCode, FinplanError
from finplan_model.registry import InMemoryRegistryStore, ModelIdentity, ModelRegistry, S3RegistryStore, model_version_resolver, seed_baselines

D1 = "sha256:" + "1" * 64
D2 = "sha256:" + "2" * 64


@pytest.fixture
def registry(ctx):
    return ModelRegistry(InMemoryRegistryStore(), clock=ctx.clock, ids=ctx.ids)


def test_reg01_same_identity_returns_the_existing_model_version(registry):
    first, created = registry.register(ModelIdentity("min_variance", D1, "1"), actor="finplan-beta-financemodel-job-api-handler-role")
    writes = list(registry.store.writes)
    again, created_again = registry.register(ModelIdentity("min_variance", D1, "1"), actor="someone-else")
    assert created and not created_again
    assert again["model_version"] == first["model_version"] and first["model_version"].startswith("mv_")
    assert registry.store.writes == writes  # no new record


def test_reg01_new_container_digest_mints_a_new_model_version(registry):
    a, _ = registry.register(ModelIdentity("min_variance", D1, "1"), actor="ci")
    b, created = registry.register(ModelIdentity("min_variance", D2, "1"), actor="ci")
    c, _ = registry.register(ModelIdentity("min_variance", D1, "2"), actor="ci")
    d, _ = registry.register(ModelIdentity("min_variance", D1, "1", artifact_checksum="sha256:" + "3" * 64), actor="ci")
    assert created and len({a["model_version"], b["model_version"], c["model_version"], d["model_version"]}) == 4


def test_reg02_records_are_immutable_and_status_moves_by_events(registry, ctx):
    rec, _ = registry.register(ModelIdentity("mean_variance", D1, "1"), actor="ci")
    mv = rec["model_version"]
    with pytest.raises(FinplanError) as exc:
        registry.update(mv, {"image_digest": D2}, actor="mallory")
    assert exc.value.code == ErrorCode.IMMUTABLE_RECORD
    assert registry.get(mv)["image_digest"] == D1
    ctx.clock.advance(minutes=5)
    registry.update(mv, {"status": "candidate"}, actor="reviewer")
    registry.transition(mv, "promoted", actor="approver", reason="holdout passed")
    events = registry.events(mv)
    assert [e["to"] for e in events] == ["registered", "candidate", "promoted"]
    assert all(e["actor"] and e["at"] for e in events) and events[1]["actor"] == "reviewer"
    assert registry.get(mv)["status"] == "promoted"
    with pytest.raises(FinplanError) as exc:
        registry.transition(mv, "candidate", actor="x")
    assert exc.value.code == ErrorCode.PRECONDITION_FAILED
    registry.transition(mv, "retired", actor="x")
    with pytest.raises(FinplanError):
        registry.transition(mv, "promoted", actor="x")  # retired is final


def test_reg02_stored_records_cannot_be_rewritten(registry):
    rec, _ = registry.register(ModelIdentity("cash", D1, "1"), actor="ci")
    assert registry.store.put_new(f"versions/{rec['model_version']}.json", {**rec, "image_digest": D2}) is False
    assert registry.get(rec["model_version"])["image_digest"] == D1


def test_concurrent_registrations_converge(ctx):
    registry = ModelRegistry(InMemoryRegistryStore(), clock=ctx.clock)
    out: list[str] = []
    ident = ModelIdentity("equal_weight", D1, "1")
    threads = [threading.Thread(target=lambda: out.append(registry.register(ident, actor="ci")[0]["model_version"])) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(out)) == 1


def test_s3_registry_store_is_write_once(ctx):
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-2")
        s3.create_bucket(Bucket="example-registry", CreateBucketConfiguration={"LocationConstraint": "us-east-2"})
        registry = ModelRegistry(S3RegistryStore(s3, "example-registry"), clock=ctx.clock, ids=ctx.ids)
        rec, created = registry.register(ModelIdentity("scenario_cvar", D1, "1"), actor="ci")
        again, created_again = registry.register(ModelIdentity("scenario_cvar", D1, "1"), actor="ci")
        assert created and not created_again and again["model_version"] == rec["model_version"]
        assert registry.store.put_new(f"versions/{rec['model_version']}.json", {"tampered": True}) is False
        keys = sorted(o["Key"] for o in s3.list_objects_v2(Bucket="example-registry")["Contents"])
        assert [k.split("/")[0] for k in keys] == ["events", "identity", "versions"]


def test_identity_validation():
    for bad in (dict(strategy="Min Var", image_digest=D1, param_schema_version="1"), dict(strategy="x", image_digest="latest", param_schema_version="1"), dict(strategy="x", image_digest=D1, param_schema_version="")):
        with pytest.raises(FinplanError):
            ModelIdentity(**bad)
    a = ModelIdentity("x", D1, "1")
    assert a.key == ModelIdentity(param_schema_version="1", image_digest=D1, strategy="x").key


def test_resolver_and_seed(registry):
    resolve = model_version_resolver(registry, actor="finplan-beta-financemodel-job-api-handler-role")
    mv = resolve("min_variance", D1)
    assert mv and resolve("min_variance", D1) == mv
    assert resolve("unknown_strategy", D1) is None and resolve("min_variance", None) is None
    seeded = seed_baselines(registry, D1, actor="financemodel-release")
    assert seeded["min_variance"]["model_version"] == mv and seeded["min_variance"]["created"] is False
    assert {k for k, v in seeded.items() if v["created"]} == {"buy_and_hold", "cash", "equal_weight", "mean_variance", "scenario_cvar"}
