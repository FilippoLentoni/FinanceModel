"""Task group 8: time limits, leases, bounded queue, quota requeue, cancellation, failure semantics,
pre-flight cost and category budget checks, approval and cost tags (CTL-01 to CTL-10)."""

from __future__ import annotations

import pytest
from finplan_contracts import iam
from finplan_contracts.validate import validate

from finplan_model.control.auth import Principal
from finplan_model.control.policies import job_api_resource_policy
from finplan_model.core.errors import FinplanError

from .support import AGENT, APPROVER, APPROVER_ROLE, PIPELINE, READER, SUBMITTER, Harness, arn, enable_logs, env_config, events_logged, request

APPROVE = Principal.from_arn(APPROVER)


def _err(fn, *a, **kw) -> FinplanError:
    with pytest.raises(FinplanError) as ei:
        fn(*a, **kw)
    return ei.value


# ================================================================ CTL-01 time limits
def test_stopping_condition_equals_approved_runtime():
    h = Harness(auto_approve=1.0)
    h.start(max_runtime_seconds=1200)
    (req,) = h.created_jobs()
    assert req["StoppingCondition"] == {"MaxRuntimeInSeconds": 1200}
    h2 = Harness(auto_approve=1.0)
    h2.start()
    assert h2.created_jobs()[0]["StoppingCondition"]["MaxRuntimeInSeconds"] == 900  # job type default


def test_runtime_above_ceiling_rejected():
    err = _err(Harness().submit, max_runtime_seconds=3600)
    assert err.code == "VALIDATION_FAILED" and err.details["max_runtime_seconds"] == 1800


@pytest.mark.parametrize("status, extra", [("Failed", {"FailureReason": "MaxRuntimeExceeded: job exceeded its stopping condition"}), ("Stopped", {"ExitMessage": "Stopped by MaxRuntimeInSeconds"})])
def test_max_runtime_maps_to_timed_out_with_incomplete_outputs(status, extra):
    h = Harness(auto_approve=1.0)
    rid = h.start()
    name = h.run(rid)["job_name"]
    h.event(name, "InProgress")
    # partial output written by the job before it was stopped
    from .support import job_result

    partial = job_result(h.run(rid))
    h.run_io.put_result(rid, partial)
    h.clock.advance(seconds=900)
    h.event(name, status, **extra)
    run = h.run(rid)
    assert run["state"] == "timed_out"
    result = h.service.get_job_result(Principal.from_arn(READER), rid)
    assert result["completion_status"] == "timed_out" and result["artifacts_complete"] is False
    assert "solution_status" not in result
    assert validate(result, "job-result").valid


def test_stopped_at_the_runtime_limit_without_reason_is_timed_out():
    h = Harness(auto_approve=1.0)
    rid = h.start()
    name = h.run(rid)["job_name"]
    h.event(name, "InProgress")
    h.clock.advance(seconds=899)
    h.event(name, "Stopped")
    assert h.run(rid)["state"] == "timed_out"


# ================================================================ CTL-02 lease
def test_second_job_stays_queued_until_the_first_releases_its_lease():
    h = Harness(auto_approve=1.0)
    _, a = h.submit(idempotency_key="k-a")
    h.clock.advance(seconds=1)  # the queue is FIFO by submission time
    _, b = h.submit(idempotency_key="k-b")
    s = h.service.dispatch()
    assert s["started"] == [a["run_id"]] and s["waiting"] == [b["run_id"]]
    assert h.run(b["run_id"])["state"] == "queued"
    assert len(h.created_jobs()) == 1
    h.event(h.run(a["run_id"])["job_name"], "InProgress")
    h.service.dispatch()
    assert h.run(b["run_id"])["state"] == "queued"
    h.succeed(a["run_id"])
    assert h.service.leases.holders("cpu") == []
    s = h.service.dispatch()
    assert s["started"] == [b["run_id"]]


def test_lease_limit_comes_from_settings():
    h = Harness(auto_approve=1.0, leases={"cpu": 2})
    h.submit(idempotency_key="k-a")
    h.submit(idempotency_key="k-b")
    assert len(h.service.dispatch()["started"]) == 2


def test_stale_lease_is_reclaimed_only_after_terminal_confirmation(caplog):
    enable_logs(caplog)
    h = Harness(auto_approve=1.0)
    rid = h.start()
    name = h.run(rid)["job_name"]
    h.event(name, "InProgress")
    # heartbeats stop and SageMaker cannot be reached: the lease expires but is kept
    h.clock.advance(seconds=1000)
    for _ in range(2):
        h.sagemaker.fail_next("DescribeProcessingJob", "ThrottlingException")
    h.service.dispatch()
    assert [s["holder"] for s in h.service.leases.holders("cpu")] == [rid]
    assert events_logged(caplog, "lease_stale_unconfirmed")
    assert not events_logged(caplog, "lease_reclaimed")
    # the job is confirmed alive: the stale lease is renewed, not reclaimed
    h.clock.advance(seconds=1000)
    h.service._maybe_reclaim("cpu", h.service.leases.holders("cpu")[0])
    assert [s["holder"] for s in h.service.leases.holders("cpu")] == [rid]
    assert events_logged(caplog, "lease_stale_holder_alive")
    # the job ended while nobody listened: confirmed terminal, then reclaimed and logged
    h.sagemaker.set_status(name, "Failed", ExitMessage="exit 1")
    h.clock.advance(seconds=1000)
    assert h.service._maybe_reclaim("cpu", h.service.leases.holders("cpu")[0]) is True
    assert h.service.leases.holders("cpu") == []
    (log,) = events_logged(caplog, "lease_reclaimed")
    assert log["holder"] == rid
    assert h.run(rid)["state"] == "failed"


def test_orphaned_lease_of_a_finished_run_is_reclaimed_by_the_dispatcher(caplog):
    enable_logs(caplog)
    h = Harness(auto_approve=1.0)
    rid = h.start()
    h.succeed(rid)
    h.store.acquire_slot("beta#cpu", 0, rid, "2026-01-05T09:00:01Z", "2026-01-05T09:00:00Z")  # left behind by a crash
    h.clock.advance(seconds=10)
    h.service.dispatch()
    assert h.service.leases.holders("cpu") == []
    assert events_logged(caplog, "lease_reclaimed")


# ================================================================ CTL-03 bounded queue
def test_queue_full_is_rate_limited_and_records_nothing():
    h = Harness(cfg=env_config(mutate=lambda raw: raw["queue"].update(max_depth=2)))
    h.submit(idempotency_key="k1")
    h.submit(idempotency_key="k2")
    err = _err(h.submit, idempotency_key="k3")
    assert err.code == "RATE_LIMITED" and err.retryable is True
    assert len(h.store.runs) == 2 and len(h.store.idempotency) == 2


# ================================================================ CTL-04 quota requeue
def test_quota_exhaustion_requeues_with_backoff_and_wait_reason():
    h = Harness(auto_approve=1.0)
    _, sub = h.submit()
    rid = sub["run_id"]
    h.sagemaker.fail_next("CreateProcessingJob", "ResourceLimitExceeded")
    s = h.service.dispatch()
    assert s["requeued"] == [rid]
    run = h.run(rid)
    assert run["state"] == "queued" and run["wait_reason"] == "account_quota_exhausted"
    assert h.service.leases.holders("cpu") == []
    status = h.service.get_job_status(Principal.from_arn(READER), rid)
    assert status["wait_reason"] == "account_quota_exhausted"
    assert h.service.dispatch()["waiting"] == [rid]  # backoff not elapsed
    h.clock.advance(seconds=61)
    assert h.service.dispatch()["started"] == [rid]
    assert [c["ProcessingJobName"][-2:] for c in h.created_jobs()] == ["a1", "a2"]


def test_quota_wait_beyond_maximum_fails_dependency_unavailable():
    h = Harness(auto_approve=1.0)
    _, sub = h.submit()
    rid = sub["run_id"]
    for _ in range(30):
        if h.run(rid)["state"] != "queued":
            break
        h.sagemaker.fail_next("CreateProcessingJob", "ResourceLimitExceeded")
        h.service.dispatch()
        h.clock.advance(seconds=901)
    run = h.run(rid)
    assert run["state"] == "failed" and run["error"]["code"] == "DEPENDENCY_UNAVAILABLE"
    assert run["error"]["details"]["reason"] == "quota_wait_exceeded"
    assert h.service.leases.holders("cpu") == []


# ================================================================ CTL-05 cancellation
def test_cancel_before_start_is_immediate():
    h = Harness(auto_approve=1.0)
    _, sub = h.submit()
    _, resp = h.service.cancel_job(Principal.from_arn(SUBMITTER), sub["run_id"], {"idempotency_key": "c"})
    assert resp["state"] == "cancelled" and resp["completion_status"] == "cancelled"
    assert h.sagemaker.calls == []
    assert h.service.dispatch()["started"] == []


def test_cancel_running_job_goes_stopping_then_cancelled_and_releases_lease():
    h = Harness(auto_approve=1.0)
    rid = h.start()
    name = h.run(rid)["job_name"]
    h.event(name, "InProgress")
    _, resp = h.service.cancel_job(Principal.from_arn(SUBMITTER), rid, {"idempotency_key": "c"})
    assert resp["state"] == "stopping"
    assert ("StopProcessingJob", {"ProcessingJobName": name}) in h.sagemaker.calls
    assert [s["holder"] for s in h.service.leases.holders("cpu")] == [rid]
    h.sagemaker.set_status(name, "Stopped")
    h.event(name, "Stopped")
    run = h.run(rid)
    assert run["state"] == "cancelled" and h.service.leases.holders("cpu") == []
    result = h.service.get_job_result(Principal.from_arn(READER), rid)
    assert result["artifacts"] == [] and result["artifacts_complete"] is False


def test_cancel_finished_job_reports_terminal_state_unchanged():
    h = Harness(auto_approve=1.0)
    rid = h.start()
    h.succeed(rid)
    before = h.run(rid)
    _, resp = h.service.cancel_job(Principal.from_arn(SUBMITTER), rid, {"idempotency_key": "c"})
    assert resp["state"] == "succeeded"
    assert h.run(rid) == before


def test_cancel_while_the_job_is_being_created_stops_it():
    h = Harness(auto_approve=1.0)
    _, sub = h.submit()
    rid = sub["run_id"]
    orig = h.sagemaker.create_processing_job

    def create_and_cancel(**kw):
        out = orig(**kw)
        h.service.cancel_job(Principal.from_arn(SUBMITTER), rid, {"idempotency_key": "race"})
        return out

    h.sagemaker.create_processing_job = create_and_cancel
    h.service.dispatch()
    run = h.run(rid)
    assert run["state"] == "cancelled" and run["job_started"] is True
    assert any(op == "StopProcessingJob" for op, _ in h.sagemaker.calls)


# ================================================================ CTL-06 failure semantics
def test_container_crash_is_failed_internal_with_correlation_id():
    h = Harness(auto_approve=1.0)
    code, body, headers = h.call("POST", "/v1/jobs", body=request(), headers={"x-correlation-id": "corr-crash-test-0001"})
    assert code == 202
    rid = body["run_id"]
    h.service.dispatch()
    name = h.run(rid)["job_name"]
    assert h.created_jobs()[0]["Environment"]["FINPLAN_CORRELATION_ID"] == "corr-crash-test-0001"
    h.event(name, "InProgress")
    h.sagemaker.set_status(name, "Failed", ExitMessage="non-zero exit")
    h.event(name, "Failed", ExitMessage="non-zero exit")
    _, res, _ = h.call("GET", f"/v1/jobs/{rid}/result", principal=READER)
    assert res["completion_status"] == "failed" and "solution_status" not in res
    assert res["error"]["code"] == "INTERNAL" and res["error"]["correlation_id"] == "corr-crash-test-0001"
    assert h.service.leases.holders("cpu") == []
    h.service.dispatch()
    assert len(h.created_jobs()) == 1  # container failures are never retried


def test_job_reported_integrity_failure_keeps_its_error_code():
    from finplan_model.core.outcome import failed_job_result
    from finplan_model.core.context import RunContext

    h = Harness(auto_approve=1.0)
    rid = h.start()
    run = h.run(rid)
    ctx = RunContext.for_tests(run_id=rid, correlation_id=run["correlation_id"])
    doc = failed_job_result(ctx, FinplanError.precondition("checksum mismatch", reason="snapshot_checksum_mismatch"))
    h.run_io.put_result(rid, doc)
    h.event(run["job_name"], "Failed")
    run = h.run(rid)
    assert run["state"] == "failed" and run["error"]["code"] == "PRECONDITION_FAILED"
    assert run["error"]["correlation_id"] == run["correlation_id"]


def test_transient_start_failure_requeues_and_records_the_retry():
    h = Harness(auto_approve=1.0)
    _, sub = h.submit()
    rid = sub["run_id"]
    h.sagemaker.fail_next("CreateProcessingJob", "ThrottlingException")
    assert h.service.dispatch()["requeued"] == [rid]
    run = h.run(rid)
    assert run["state"] == "queued" and run["start_retries"] == 1
    assert run["transitions"][-1]["reason"] == "start_retry"
    assert any(e.get("reason") == "start_retry" for e in h.store.list_events(rid))


def test_start_retries_are_bounded():
    h = Harness(auto_approve=1.0)
    _, sub = h.submit()
    rid = sub["run_id"]
    for _ in range(5):
        if h.run(rid)["state"] != "queued":
            break
        h.sagemaker.fail_next("CreateProcessingJob", "ThrottlingException")
        h.service.dispatch()
        h.clock.advance(seconds=1000)
    run = h.run(rid)
    assert run["state"] == "failed" and run["error"]["details"]["reason"] == "start_retries_exhausted"
    assert len(h.created_jobs()) == 3  # first try + start_retry_limit (2)


def test_unpinned_image_never_starts():
    h = Harness(auto_approve=1.0, job_definitions={"run_backtest": {"image_uri": "registry.invalid/financemodel-cpu:latest"}})
    _, sub = h.submit()
    h.service.dispatch()
    run = h.run(sub["run_id"])
    assert run["state"] == "failed" and run["error"]["details"]["reason"] == "image_not_digest_pinned"
    assert h.created_jobs() == []


# ================================================================ CTL-07 pre-flight estimate
def test_budget_exceeded_shows_estimate_and_remaining():
    h = Harness(allocation={"cpu_research": 0.05, "gpu": 25})
    err = _err(h.submit)
    assert err.code == "BUDGET_EXCEEDED" and err.retryable is False
    assert err.details["budget_category"] == "cpu_research"
    assert err.details["estimated_usd_upper_bound"] == pytest.approx(0.0725)
    assert err.details["remaining_allocation_usd"] == pytest.approx(0.05)
    assert h.store.runs == {}


def test_missing_price_is_precondition_failed():
    h = Harness(prices={"retrieved_at": "2026-01-01T00:00:00Z", "usd_per_hour": {}})
    err = _err(h.submit)
    assert err.code == "PRECONDITION_FAILED" and err.details["reason"] == "instance_price_missing"
    assert h.store.runs == {}
    assert _err(Harness(prices=None).submit).details["reason"] == "instance_price_missing"


def test_stale_price_is_precondition_failed():
    h = Harness(prices={"retrieved_at": "2025-10-01T00:00:00Z", "usd_per_hour": {"ml.m5.xlarge": 0.25}})
    assert _err(h.submit).details["reason"] == "instance_price_stale"


def test_estimate_is_upper_bound_of_price_runtime_count_plus_storage():
    h = Harness()
    _, resp = h.submit(dry_run=True, max_runtime_seconds=1800)
    assert resp["cost_estimate"]["estimated_usd_upper_bound"] == pytest.approx(0.25 * 0.5 + 0.01)
    assert resp["cost_estimate"]["price_retrieved_at"] == "2026-01-01T00:00:00Z"


# ================================================================ CTL-10 categories
def test_cpu_category_exhausted_while_gpu_has_funds():
    h = Harness(allocation={"platform_infra": 8, "cpu_research": 0.1, "bedrock_explanations": 5, "gpu": 25, "reserve": 5})
    h.submit(idempotency_key="k1")  # 0.0725 reserved
    err = _err(h.submit, idempotency_key="k2")
    assert err.code == "BUDGET_EXCEEDED" and err.details["budget_category"] == "cpu_research"
    assert err.details["remaining_allocation_usd"] == pytest.approx(0.0275)


def test_runs_cancelled_before_start_do_not_count():
    h = Harness(allocation={"cpu_research": 0.1, "gpu": 25})
    _, a = h.submit(idempotency_key="k1")
    h.service.cancel_job(Principal.from_arn(SUBMITTER), a["run_id"], {"idempotency_key": "c"})
    code, _ = h.submit(idempotency_key="k2")
    assert code == 202


def test_budget_deny_action_refuses_every_paid_submission():
    h = Harness(state='{"state": "enforced", "enforced": true, "since": "2026-01-05T00:00:00Z"}')
    err = _err(h.submit)
    assert err.code == "BUDGET_EXCEEDED" and err.retryable is False and err.details["budget_state"] == "enforced"
    assert _err(h.submit, dry_run=True).code == "BUDGET_EXCEEDED"


def test_budget_deny_action_also_blocks_queued_runs_at_start():
    h = Harness(auto_approve=1.0)
    _, sub = h.submit()
    h.settings.state = {"state": "enforced", "enforced": True}
    h.service.dispatch()
    run = h.run(sub["run_id"])
    assert run["state"] == "failed" and run["error"]["code"] == "BUDGET_EXCEEDED"
    assert h.created_jobs() == []


def test_job_type_without_budget_category_is_precondition_failed():
    h = Harness(cfg=env_config(mutate=lambda raw: raw["job_types"]["run_backtest"].update(budget_category=None)))
    err = _err(h.submit)
    assert err.code == "PRECONDITION_FAILED" and err.details["reason"] == "budget_category_missing"
    assert h.store.runs == {}


# ================================================================ CTL-08 approval
def test_tool_agent_and_pipeline_roles_cannot_approve():
    h = Harness()
    _, sub = h.submit()
    rid = sub["run_id"]
    for who in (SUBMITTER, READER, AGENT, PIPELINE):
        err = _err(h.service.approve_run, Principal.from_arn(who), rid, {})
        assert err.code == "FORBIDDEN"
        assert h.run(rid)["state"] == "awaiting_approval"


def test_submitter_cannot_approve_its_own_run():
    h = Harness()
    _, sub = h.submit(principal=APPROVER)
    err = _err(h.service.approve_run, APPROVE, sub["run_id"], {})
    assert err.code == "FORBIDDEN" and err.details["reason"] == "submitter_cannot_approve"


def test_approver_records_identity_and_timestamp():
    h = Harness()
    _, sub = h.submit()
    h.clock.advance(minutes=5)
    status = h.service.approve_run(APPROVE, sub["run_id"], {})
    assert status["state"] == "queued"
    assert status["approval"] == {"approved_by": APPROVER_ROLE, "approved_at": "2026-01-05T09:05:00Z", "approved_estimate_usd": pytest.approx(0.0725)}
    assert "arn:" not in str(status)
    assert h.service.dispatch()["started"] == [sub["run_id"]]


def test_approval_expires_into_cancelled():
    h = Harness()
    _, sub = h.submit()
    h.clock.advance(hours=73)
    assert h.service.dispatch()["expired"] == [sub["run_id"]]
    run = h.run(sub["run_id"])
    assert run["state"] == "cancelled" and run["cancel_reason"] == "approval_expired"
    status = h.service.get_job_status(Principal.from_arn(READER), sub["run_id"])
    assert status["cancel_reason"] == "approval_expired"


def test_gpu_jobs_always_require_approval():
    def add_gpu(raw):
        raw["job_types"]["gpu_fixture"] = {"entry_point": "gpu_fixture", "compute_class": "gpu", "default_runtime_seconds": 600, "max_runtime_seconds": 900, "instance_types": ["ml.g5.xlarge"], "default_instance_type": "ml.g5.xlarge", "max_instance_count": 1, "budget_category": "gpu", "deployed": True}
        raw["lease"]["max_holders"]["gpu"] = 1

    prices = {"retrieved_at": "2026-01-01T00:00:00Z", "usd_per_hour": {"ml.m5.xlarge": 0.25, "ml.g5.xlarge": 1.5}}
    h = Harness(cfg=env_config(mutate=add_gpu), auto_approve=100.0, prices=prices)
    _, resp = h.submit(job_type="gpu_fixture")
    assert resp["state"] == "awaiting_approval" and resp["cost_estimate"]["budget_category"] == "gpu"
    rid = resp["run_id"]
    h.service.dispatch()
    assert h.run(rid)["state"] == "awaiting_approval" and h.created_jobs() == []
    status = h.service.approve_run(APPROVE, rid, {})
    assert validate(status, "tools/get-job-status-response").valid


def test_iam_policy_simulation_denies_approve_to_tool_agent_and_pipeline_roles():
    api = "arn:aws:execute-api:us-east-2:<account-id>:api0000001/beta"
    policy = job_api_resource_policy(
        "beta",
        invoker_role_patterns=["finplan-beta-financelambdastool-*", "finplan-beta-financeagent-*", "finplan-beta-financemodel-pipeline-role", "finplan-beta-financialplanning-*"],
        api_resource="arn:aws:execute-api:us-east-2:<account-id>:api0000001/*",
        partition="aws",
        account="<account-id>",
    )
    approve = f"{api}/POST/v1/jobs/run_01KDVDNAZ83BAMMYCEGWF33DPM/approve"
    status = f"{api}/GET/v1/jobs/run_01KDVDNAZ83BAMMYCEGWF33DPM"

    def decide(role: str, resource: str) -> str:
        return iam.evaluate(iam.Request("execute-api:Invoke", resource, {"aws:PrincipalArn": f"arn:aws:iam::<account-id>:role/{role}"}), [policy]).decision

    for role in ("finplan-beta-financelambdastool-submitter-role", "finplan-beta-financelambdastool-reader-role", "finplan-beta-financeagent-runtime-role", "finplan-beta-financemodel-pipeline-role"):
        assert decide(role, approve) == iam.EXPLICIT_DENY, role
    assert decide(APPROVER_ROLE, approve) == iam.ALLOWED
    assert decide("finplan-beta-financelambdastool-reader-role", status) == iam.ALLOWED
    assert decide("finplan-beta-someone-else-role", status) == iam.IMPLICIT_DENY


# ================================================================ CTL-09 tags
def test_cost_allocation_tags_on_every_job():
    h = Harness(auto_approve=1.0)
    rid = h.start()
    (req,) = h.created_jobs()
    tags = {t["Key"]: t["Value"] for t in req["Tags"]}
    assert tags == {"project": "finplan", "owner-repo": "financemodel", "environment": "beta", "logical-role": "research-job", "run-id": rid}
    assert validate(tags, "cost-allocation-tags").valid
    run = h.run(rid)
    assert run["estimated_cost_usd"] == pytest.approx(0.0725) and run["actual_cost_usd"] is None


def test_processing_request_is_cpu_digest_pinned_and_carries_no_storage_location():
    from finplan_model.control.validation import find_storage_location

    h = Harness(auto_approve=1.0)
    h.start()
    (req,) = h.created_jobs()
    assert req["ProcessingResources"]["ClusterConfig"] == {"InstanceCount": 1, "InstanceType": "ml.m5.xlarge", "VolumeSizeInGB": 10}
    assert "@sha256:" in req["AppSpecification"]["ImageUri"]
    assert req["AppSpecification"]["ContainerArguments"] == ["run_backtest"]
    assert find_storage_location(req["Environment"]) is None
    assert set(req["Environment"]) == {"FINPLAN_ENVIRONMENT", "FINPLAN_RUN_ID", "FINPLAN_JOB_TYPE", "FINPLAN_CORRELATION_ID", "FINPLAN_RUN_SPEC_SHA256", "FINPLAN_IMAGE_DIGEST"}


def test_run_spec_is_written_with_the_checksum_passed_to_the_job():
    h = Harness(auto_approve=1.0)
    rid = h.start()
    (req,) = h.created_jobs()
    spec = h.run_io.get_spec(rid, req["Environment"]["FINPLAN_RUN_SPEC_SHA256"])
    assert spec["run_id"] == rid and spec["simulation"]["fees"]["proportional_bps"] == 1.0
    assert spec["universe"] == ["AGG", "SPY"]
    with pytest.raises(FinplanError) as ei:
        h.run_io.get_spec(rid, "sha256:" + "0" * 64)
    assert ei.value.details["reason"] == "run_spec_checksum_mismatch"


def test_other_principal_kinds_named_by_role():
    assert Principal.from_arn(arn("x-role")).display_name == "x-role"
    with pytest.raises(FinplanError) as ei:
        Principal.from_arn(None)
    assert ei.value.code == "UNAUTHORIZED"


def test_gateway_refusals_are_contract_envelopes():
    import json as _json

    from finplan_model.control.policies import gateway_responses
    from finplan_model.core.errors import contract_version

    for rtype, spec in gateway_responses(contract_version()).items():
        doc = _json.loads(spec["template"].replace("$context.requestId", "0f2b6c1e-aaaa-bbbb-cccc-synthetic01"))
        assert validate(doc, "error").valid, rtype
        assert doc["code"] in ("UNAUTHORIZED", "FORBIDDEN", "RATE_LIMITED")
