"""Run-output staging (task 9.1; RST-01, RST-02, RST-03, RST-04 code side) and the platform outcome
link (task 9.4; RST-05)."""

from __future__ import annotations

import json

import boto3
import pytest
from moto import mock_aws

from finplan_model.core.clock import FrozenClock
from finplan_model.core.errors import ErrorCode, FinplanError
from finplan_model.core.platform import FixturePlatformClient
from finplan_model.staging import (
    MANIFEST_NAME,
    PLAN_CONTENT_FILE,
    InMemoryStagingStore,
    S3StagingStore,
    StagingTarget,
    link_platform_outcome,
    parse_staging_ref,
    platform_outcome_hook,
    stage_run_output,
    staging_decision,
)

RUN = "run_01KDVDNAZ83BAMMYCEGWF33DPM"
PLAN = "pl_01KDVDNAZ83BAMMYCEGWF33DPM"
PARENT = "pv_01KDVDNAZ83BAMMYCEGWF33DPM"
TARGET = StagingTarget(PLAN, PARENT)


def result(**kw):
    doc = {
        "run_id": RUN,
        "completion_status": "succeeded",
        "solution_status": "optimal",
        "artifacts": [],
        "artifacts_complete": True,
        "model_version": "mv_01KDVDNAZ83BAMMYCEGWF33DPM",
        "configuration_id": "cfg_" + "9" * 64,
        "input_snapshot_id": "snap_01KDVDNAZ83BAMMYCEGWF33DPM",
        "evaluator_version": "1.0.0",
        "domain": "finance",
        "domain_schema_version": "1.0",
        "payload": {"performance": {"net_return": 0.02, "max_drawdown": -0.05}, "accuracy": {}, "compute_cost": {}, "proposed_allocation": {"weights": [{"instrument_id": "SPY", "weight": 0.6}], "cash_weight": 0.4}},
        "synthetic": True,
    }
    doc.update(kw)
    return doc


def stage(store, res, purpose="production_candidate", target=TARGET, **kw):
    return stage_run_output(store, res, purpose=purpose, target=target, clock=FrozenClock("2026-10-08T12:00:00Z"), **kw)


# ----------------------------------------------------------------- RST-01
@pytest.mark.parametrize("purpose", ["research", "tuning", "holdout_evaluation"])
def test_rst01_non_candidate_runs_stage_nothing(purpose):
    store = InMemoryStagingStore()
    assert stage(store, result(), purpose=purpose) == {"status": "not_staged", "reason": "purpose_not_production_candidate"}
    assert store.objects == {}


@pytest.mark.parametrize("solution", ["infeasible", "unbounded", "not_applicable"])
def test_rst01_infeasible_candidate_stages_nothing_and_says_so(solution):
    store = InMemoryStagingStore()
    block = stage(store, result(solution_status=solution))
    assert block == {"status": "not_staged", "reason": f"solution_status_{solution}"} and store.objects == {}
    assert staging_decision("production_candidate", {"completion_status": "timed_out"}) == (False, "completion_status_timed_out")
    assert stage(store, result(), target=None) == {"status": "not_staged", "reason": "no_plan_target"}


# ----------------------------------------------------------------- RST-02 / RST-03
def test_rst02_rst03_bundle_files_first_manifest_last():
    store = InMemoryStagingStore()
    block = stage(store, result())
    assert block["status"] == "staged" and block["platform_validation"] == "pending" and "plan_version_id" not in block
    assert store.order[-1] == f"{RUN}/{MANIFEST_NAME}"
    assert store.order[:-1] == [f"{RUN}/metrics/summary.json", f"{RUN}/{PLAN_CONTENT_FILE}"]
    bundle = store.bundle(RUN)
    manifest = json.loads(bundle[MANIFEST_NAME])
    assert manifest["plan_id"] == PLAN and manifest["parent_plan_version_id"] == PARENT
    for f in manifest["files"]:
        from finplan_model.core.artifacts import sha256_checksum

        assert sha256_checksum(bundle[f["name"]]) == f["checksum"] and len(bundle[f["name"]]) == f["size_bytes"]
    assert json.loads(bundle[PLAN_CONTENT_FILE]) == manifest["payload"]["plan_content"]  # the platform compares both
    assert not [n for n in bundle if n.endswith((".done", "_SUCCESS", ".complete"))]  # no separate marker file


def test_rst02_schema_failure_stages_nothing():
    store = InMemoryStagingStore()
    bad = result(payload={"performance": {}, "accuracy": {}, "compute_cost": {}, "proposed_allocation": {"weights": [{"instrument_id": "SPY", "weight": 1.7}]}})
    with pytest.raises(FinplanError) as exc:
        stage(store, bad)
    assert exc.value.code == ErrorCode.VALIDATION_FAILED and store.objects == {}
    with pytest.raises(FinplanError) as exc:
        stage(store, result(evaluator_version="not a version!"))
    assert exc.value.code == ErrorCode.VALIDATION_FAILED and store.objects == {}


def test_rst03_job_dying_before_the_manifest_leaves_an_incomplete_bundle():
    store = InMemoryStagingStore(fail_before=MANIFEST_NAME)
    with pytest.raises(RuntimeError):
        stage(store, result())
    assert store.bundle(RUN) and not store.complete(RUN)  # data files only: the platform refuses it


# ----------------------------------------------------------------- RST-04: write-only, write-once
class RecordingS3:
    def __init__(self, inner):
        self.inner = inner
        self.calls: list[str] = []

    def put_object(self, **kw):
        self.calls.append("put_object")
        assert kw["IfNoneMatch"] == "*"
        return self.inner.put_object(**kw)

    def __getattr__(self, name):  # any read, list or delete would show up here
        self.calls.append(name)
        return getattr(self.inner, name)


def test_rst04_s3_staging_writes_once_and_never_reads():
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-2")
        s3.create_bucket(Bucket="example-staging", CreateBucketConfiguration={"LocationConstraint": "us-east-2"})
        rec = RecordingS3(s3)
        store = S3StagingStore(rec, "s3://example-staging/staging/")
        assert stage(store, result())["status"] == "staged"
        assert set(rec.calls) == {"put_object"}
        keys = sorted(o["Key"] for o in s3.list_objects_v2(Bucket="example-staging")["Contents"])
        assert keys == [f"staging/{RUN}/manifest.json", f"staging/{RUN}/metrics/summary.json", f"staging/{RUN}/plan-content.json"]
        with pytest.raises(FinplanError) as exc:  # an existing key (this or another run's) is never overwritten
            store.put_new(RUN, MANIFEST_NAME, b"{}", "application/json")
        assert exc.value.code == ErrorCode.OPERATION_NOT_PERMITTED
    assert parse_staging_ref("s3://example-staging/staging") == ("example-staging", "staging/")
    with pytest.raises(FinplanError):
        parse_staging_ref("https://example.invalid/x")


def test_staging_records_lineage_first(ctx):
    from finplan_model.registry import InMemoryRegistryStore, ModelIdentity, ModelRegistry

    registry = ModelRegistry(InMemoryRegistryStore(), clock=ctx.clock, ids=ctx.ids)
    record, _ = registry.register(ModelIdentity("min_variance", "sha256:" + "a" * 64, "1"), actor="test")
    res = result(model_version=record["model_version"])
    stage(InMemoryStagingStore(), res, registry=registry, spec={"run_id": RUN, "model_version": record["model_version"], "purpose": "production_candidate"})
    assert registry.verify_lineage(RUN, record["model_version"])["matches"] is True


# ----------------------------------------------------------------- RST-05: the platform decides commitment
def _staged():
    store = InMemoryStagingStore()
    return {**result(), "staging": stage(store, result())}


def _outcome(**kw):
    doc = {"run_id": RUN, "plan_id": PLAN, "decided_at": "2026-10-08T13:00:00Z", "model_version": "mv_01KDVDNAZ83BAMMYCEGWF33DPM"}
    doc.update(kw)
    return doc


def test_rst05_platform_invalid_outcome_claims_no_plan_version():
    platform = FixturePlatformClient()
    platform.staged_outputs[RUN] = _outcome(outcome="accepted", plan_version_id="pv_01KDVDNBYGX5V5HY2JSK5XWKHC", plan_version_status="invalid")
    linked = link_platform_outcome(_staged(), platform)
    assert linked["staging"]["status"] == "staged"
    assert linked["staging"]["platform_validation"] == "invalid"
    assert "plan_version_id" not in linked["staging"] and "plan_version_id" not in linked
    assert ("get_staged_output", RUN) in platform.calls
    assert not [c for c in platform.calls if "accept" in str(c[0])]  # FinanceModel never triggers acceptance


def test_rst05_other_outcomes():
    platform = FixturePlatformClient()
    staged = _staged()
    assert link_platform_outcome(staged, platform)["staging"]["platform_validation"] == "pending"  # nothing decided yet
    platform.staged_outputs[RUN] = _outcome(outcome="accepted", plan_version_id="pv_01KDVDNBYGX5V5HY2JSK5XWKHC", plan_version_status="validated")
    ok = platform_outcome_hook(platform)({}, staged)
    assert ok["staging"]["plan_version_id"] == "pv_01KDVDNBYGX5V5HY2JSK5XWKHC" and ok["staging"]["status"] == "staged"
    platform.staged_outputs[RUN] = _outcome(outcome="rejected", rejection_kind="structural", error=FinplanError.validation("staged files do not match the manifest", pointer="/files").to_envelope("corr-platform-0001"))
    rej = link_platform_outcome(ok, platform)["staging"]
    assert rej["platform_validation"] == "rejected" and "plan_version_id" not in rej and rej["platform_error_code"] == "VALIDATION_FAILED"
    not_staged = {**result(), "staging": {"status": "not_staged", "reason": "solution_status_infeasible"}}
    assert link_platform_outcome(not_staged, platform) == not_staged
