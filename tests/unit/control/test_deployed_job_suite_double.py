"""The deployed job-interface suite (``tests/integration/job_suite.py``) against the in-process API
double: proves offline that every request the beta and gamma stages send is valid and that the
lifecycle ends with nothing running (platform lesson L5: deployed suites run real checks)."""

from __future__ import annotations

import pytest

from tests.integration.job_suite import IntegrationError, adapt, run_contract_checks, run_job_lifecycle, run_model_selection_dry_run

from .support import SID, Harness, arn


def _stage_call(h: Harness):
    # the FinanceModel pipeline stage role is the caller in the deployed suites
    stage = arn("finplan-beta-financemodel-pipeline-stage-role")
    return adapt(lambda method, path, body=None: h.call(method, path, principal=stage, body=body))


def test_contract_checks_pass_against_the_double():
    h = Harness()
    steps = run_contract_checks(_stage_call(h))
    assert len(steps) == 3 and h.store.runs == {}


def test_lifecycle_waits_for_approval_and_cancels_with_no_sagemaker_job():
    h = Harness()  # auto-approve threshold 0: every paid run waits for a human
    res = run_job_lifecycle(_stage_call(h), snapshot_id=SID, run_key="rk0001", sleep=lambda s: None)
    assert res.states[0] == "awaiting_approval" and res.states[-1] == "cancelled"
    assert h.created_jobs() == [] and h.sagemaker.calls == []
    assert h.run(res.run_id)["state"] == "cancelled"
    assert len(h.store.runs) == 1  # the dry run and the replay recorded nothing


def test_lifecycle_of_an_auto_approved_run_stops_the_job():
    h = Harness(auto_approve=1.0)
    calls = {"n": 0}

    def sleep(_: float) -> None:  # SageMaker confirms the stop on the next poll
        calls["n"] += 1
        run = h.run(res_holder["run_id"])
        if run.get("job_name"):
            h.sagemaker.set_status(run["job_name"], "Stopped")
            h.event(run["job_name"], "Stopped")

    res_holder: dict[str, str] = {}
    call = _stage_call(h)

    def recording_call(method, path, body=None):
        out = call(method, path, body)
        if method == "POST" and path == "/v1/jobs" and out[0] == 202:
            res_holder["run_id"] = out[1]["run_id"]
            h.service.dispatch()  # the dispatcher starts the queued run before the cancel arrives
        return out

    res = run_job_lifecycle(recording_call, snapshot_id=SID, run_key="rk0002", sleep=sleep)
    assert res.states[-1] == "cancelled"
    assert [op for op, _ in h.sagemaker.calls if op == "StopProcessingJob"]


def test_lifecycle_fails_loudly_on_a_wrong_answer():
    h = Harness()
    call = _stage_call(h)

    def broken(method, path, body=None):
        status, doc = call(method, path, body)
        if method == "POST" and path == "/v1/jobs" and body and not body.get("dry_run") and body["configuration"]["payload"]["strategy"] == "cash":
            return 202, {"run_id": "run_x"}  # an API that ignores IDEMPOTENCY_KEY_REUSED
        return status, doc

    with pytest.raises(IntegrationError, match="IDEMPOTENCY_KEY_REUSED|HTTP 202"):
        run_job_lifecycle(broken, snapshot_id=SID, run_key="rk0003", sleep=lambda s: None)
    # the suite's cleanup still cancelled the run it submitted
    assert [r["state"] for r in h.store.runs.values()] == ["cancelled"]


def test_model_selection_dry_run_against_the_double():
    h = Harness()
    steps = run_model_selection_dry_run(_stage_call(h), snapshot_id=SID, run_key="rk0003")
    assert len(steps) == 2 and "cpu_research" in steps[0]
    assert h.store.runs == {} and h.sagemaker.calls == []
