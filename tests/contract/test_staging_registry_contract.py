"""Contract tests: staged bundles validate against the pinned staged-output manifest schema (RST-02),
the registry lineage route answers the platform's check (REG-03), and job results carry the
``model_version`` the registry minted (REG-04)."""

from __future__ import annotations

import json

import pytest
from finplan_contracts.validate import validate

from finplan_model.core.clock import FrozenClock
from finplan_model.registry import InMemoryRegistryStore, ModelIdentity, ModelRegistry, lineage_result_hook, model_version_resolver
from finplan_model.registry.api import RegistryApi
from finplan_model.staging import MANIFEST_NAME, InMemoryStagingStore, StagingTarget, stage_run_output

RUN = "run_01KDVDNAZ83BAMMYCEGWF33DPM"
OTHER_RUN = "run_01KDVDNBYGX5V5HY2JSK5XWKHC"
DIGEST = "sha256:" + "4" * 64


def _result(mv: str) -> dict:
    return {
        "run_id": RUN,
        "completion_status": "succeeded",
        "solution_status": "feasible",
        "artifacts": [],
        "artifacts_complete": True,
        "model_version": mv,
        "configuration_id": "cfg_" + "a" * 64,
        "input_snapshot_id": "snap_01KDVDNAZ83BAMMYCEGWF33DPM",
        "evaluator_version": "1.0.0",
        "domain": "finance",
        "domain_schema_version": "1.0",
        "payload": {"performance": {"net_return": 0.01}, "accuracy": {}, "compute_cost": {}, "proposed_allocation": {"weights": [{"instrument_id": "SPY", "weight": 1.0}]}},
        "synthetic": True,
    }


@pytest.fixture
def registry(ctx):
    return ModelRegistry(InMemoryRegistryStore(), clock=ctx.clock, ids=ctx.ids)


def test_rst02_staged_manifest_validates_against_the_contract(registry):
    mv = registry.register(ModelIdentity("equal_weight", DIGEST, "1"), actor="ci")[0]["model_version"]
    store = InMemoryStagingStore()
    stage_run_output(store, _result(mv), purpose="production_candidate", target=StagingTarget("pl_01KDVDNAZ83BAMMYCEGWF33DPM"), clock=FrozenClock("2026-10-08T12:00:00Z"))
    manifest = json.loads(store.bundle(RUN)[MANIFEST_NAME])
    res = validate(manifest, "staged-output-manifest")
    assert res.valid, [i.message for i in res.issues]
    assert manifest["model_version"] == mv and manifest["parent_plan_version_id"] is None
    plan = json.loads(store.bundle(RUN)["plan-content.json"])
    assert validate(plan, "plan-content").valid


def _get(api: RegistryApi, path: str, query: dict | None = None) -> tuple[int, dict]:
    resp = api.handle({"httpMethod": "GET", "path": path, "queryStringParameters": query, "headers": {"x-correlation-id": "cor_registrytest01"}})
    return resp["statusCode"], json.loads(resp["body"])


def test_reg03_platform_lineage_check(registry):
    mv = registry.register(ModelIdentity("min_variance", DIGEST, "1"), actor="ci")[0]["model_version"]
    registry.record_run(RUN, mv, actor="finplan-beta-financemodel-job-execution-role")
    api = RegistryApi(registry)
    status, body = _get(api, f"/v1/registry/lineage/{RUN}", {"model_version": mv})
    assert status == 200 and body["matches"] is True and body["model_version"] == mv and body["image_digest"] == DIGEST
    assert "bucket" not in json.dumps(body) and "s3://" not in json.dumps(body)
    status, body = _get(api, f"/v1/registry/lineage/{OTHER_RUN}", {"model_version": mv})
    assert status == 404 and body["code"] == "NOT_FOUND" and body["details"]["record_type"] == "run_lineage"
    assert validate(body, "error").valid
    other_mv = registry.register(ModelIdentity("cash", DIGEST, "1"), actor="ci")[0]["model_version"]
    status, body = _get(api, f"/v1/registry/lineage/{RUN}", {"model_version": other_mv})
    assert status == 404  # run_Y did not use mv_X
    status, body = _get(api, f"/v1/registry/lineage/{RUN}", {"model_version": "mv_01KDVDNBYGX5V5HY2JSK5XWKHC"})
    assert status == 404 and body["details"]["record_type"] == "model_version"
    status, record = _get(api, f"/v1/registry/model-versions/{mv}")
    assert status == 200 and record["status"] == "registered" and record["strategy"] == "min_variance"
    status, body = _get(api, f"/v1/registry/lineage/{RUN}")
    assert status == 400 and body["code"] == "VALIDATION_FAILED"


def test_reg04_results_carry_the_registered_model_version(registry):
    """Submission -> run spec -> job container -> result: every result names the minted model_version."""
    from finplan_model.control.auth import Principal
    from finplan_model.core.artifacts import InMemoryArtifactStore
    from finplan_model.jobs.entrypoint import run_job
    from tests.unit.control.support import IMAGE_URI, Harness, request
    from tests.unit.jobs.support import platform_with_snapshot

    h = Harness(auto_approve=10.0)
    h.deps.model_version_resolver = model_version_resolver(registry, actor="finplan-beta-financemodel-job-api-handler-role")
    h.deps.result_hooks.append(lineage_result_hook(registry, actor="finplan-beta-financemodel-job-api-handler-role"))
    body = request(job_type="run_backtest")
    body["configuration"]["payload"]["strategy"] = "equal_weight"
    code, resp = h.service.submit_job(Principal.from_arn("arn:aws:sts::<account-id>:assumed-role/finplan-beta-financelambdastool-submitter-role/s"), body, correlation_id="corr-reg-test-0001")
    assert code == 202, resp
    h.service.dispatch()
    (req,) = h.created_jobs()
    env = req["Environment"]
    rc, doc = run_job("run_backtest", resp["run_id"], run_io=h.run_io, platform=platform_with_snapshot(), artifacts=InMemoryArtifactStore(), environment="beta", spec_checksum=env["FINPLAN_RUN_SPEC_SHA256"], correlation_id=env["FINPLAN_CORRELATION_ID"], image_digest=env["FINPLAN_IMAGE_DIGEST"])
    assert rc == 0, doc
    expected = registry.find(ModelIdentity("equal_weight", IMAGE_URI.rsplit("@", 1)[1], "1"))
    assert expected is not None and doc["model_version"] == expected["model_version"]
    assert validate(doc, "job-result").valid
    h.service.handle_sagemaker_event(h.event(req["ProcessingJobName"], "Completed"))
    result = h.service.get_job_result(Principal.from_arn("arn:aws:sts::<account-id>:assumed-role/finplan-beta-financelambdastool-submitter-role/s"), resp["run_id"])
    assert result["model_version"] == expected["model_version"]
    assert registry.verify_lineage(resp["run_id"], expected["model_version"])["matches"] is True  # recorded by the hook
