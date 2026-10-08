"""The job container stages production candidates (task 9.1 wiring; RST-01, RST-02, RST-03 end to
end through the control plane's run spec) and the deployed control plane wires the task-group-9
hooks (REG-04, RST-05)."""

from __future__ import annotations

import json

import boto3
from moto import mock_aws

from finplan_model.core.artifacts import InMemoryArtifactStore
from finplan_model.jobs.entrypoint import run_job
from finplan_model.registry import InMemoryRegistryStore, ModelRegistry, model_version_resolver
from finplan_model.staging import MANIFEST_NAME, InMemoryStagingStore
from tests.unit.control.support import CANDIDATE, Harness, request
from tests.unit.jobs.support import platform_with_snapshot

PLAN = "pl_01KDVDNAZ83BAMMYCEGWF33DPM"


def _run(purpose: str, *, staging: dict | None = None, store=None, registry=None, strategy: str = "min_variance"):
    from finplan_model.control.auth import Principal

    h = Harness(auto_approve=10.0)
    if registry is not None:
        h.deps.model_version_resolver = model_version_resolver(registry, actor="test")
    body = request(purpose=purpose)
    body["configuration"]["payload"]["strategy"] = strategy
    code, resp = h.service.submit_job(Principal.from_arn(CANDIDATE), body, correlation_id="corr-staging-test-01")
    assert code == 202, resp
    h.service.dispatch()
    (req,) = h.created_jobs()
    rid = resp["run_id"]
    spec = h.run_io.get_spec(rid, None)
    if staging is not None:
        spec["staging"] = staging
    checksum = h.run_io.put_spec(rid, spec)
    env = req["Environment"]
    return rid, run_job(req["AppSpecification"]["ContainerArguments"][0], rid, run_io=h.run_io, platform=platform_with_snapshot(), artifacts=InMemoryArtifactStore(), environment="beta", spec_checksum=checksum, correlation_id=env["FINPLAN_CORRELATION_ID"], image_digest=env["FINPLAN_IMAGE_DIGEST"], staging_store=store, registry=registry)


def test_production_candidate_is_staged_manifest_last_before_the_result(ctx):
    store = InMemoryStagingStore()
    registry = ModelRegistry(InMemoryRegistryStore(), clock=ctx.clock, ids=ctx.ids)
    rid, (code, doc) = _run("production_candidate", staging={"plan_id": PLAN}, store=store, registry=registry)
    assert code == 0, doc
    assert doc["solution_status"] in ("optimal", "feasible", "no_effect"), doc["solution_status"]
    assert doc["staging"]["status"] == "staged" and store.order[-1] == f"{rid}/{MANIFEST_NAME}"
    manifest = json.loads(store.bundle(rid)[MANIFEST_NAME])
    assert manifest["model_version"] == doc["model_version"] and manifest["plan_id"] == PLAN
    assert registry.verify_lineage(rid, doc["model_version"])["matches"] is True
    _rid, (code, doc) = _run("production_candidate", staging={"plan_id": PLAN}, store=InMemoryStagingStore(), registry=registry, strategy="equal_weight")
    assert doc["staging"] == {"status": "not_staged", "reason": "solution_status_not_applicable"}  # a control proposes no plan


def test_research_run_and_missing_target_stage_nothing(ctx):
    store = InMemoryStagingStore()
    _rid, (code, doc) = _run("research", store=store)
    assert code == 0 and "staging" not in doc and store.objects == {}
    _rid, (code, doc) = _run("production_candidate", store=store)  # contract gap: no plan target in the submission
    assert code == 0 and doc["staging"] == {"status": "not_staged", "reason": "no_plan_target"} and store.objects == {}
    _rid, (code, doc) = _run("production_candidate", staging={"plan_id": PLAN}, store=None)
    assert doc["staging"]["reason"] == "staging_unavailable"


def test_invalid_bundle_fails_the_run_and_stages_nothing():
    store = InMemoryStagingStore()
    _rid, (code, doc) = _run("production_candidate", staging={"plan_id": "not-a-plan-id"}, store=store)
    assert code == 1 and doc["completion_status"] == "failed" and store.objects == {}
    assert doc["error"]["code"] in ("INVALID_IDENTIFIER", "VALIDATION_FAILED")


def test_deployed_control_plane_wires_registry_and_outcome_hooks():
    from finplan_model.control.handlers import _wire_registry_and_staging
    from finplan_model.core.platform import FixturePlatformClient

    h = Harness()
    with mock_aws():
        ssm = boto3.client("ssm", region_name="us-east-2")
        s3 = boto3.client("s3", region_name="us-east-2")
        _wire_registry_and_staging(h.deps, h.cfg, ssm, s3, None)
        assert h.deps.model_version_resolver is None and h.deps.result_hooks == []  # nothing published yet
        ssm.put_parameter(Name="/finplan/beta/financemodel/config/registry-storage-ref", Value="example-registry", Type="String")
        _wire_registry_and_staging(h.deps, h.cfg, ssm, s3, FixturePlatformClient())
        assert h.deps.model_version_resolver is not None and len(h.deps.result_hooks) == 2
