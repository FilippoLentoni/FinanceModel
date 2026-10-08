"""Task 7.3: lifecycle state machine with append-only events and conditional writes, and the
SageMaker state-change handler (JOB-05); result retrieval semantics (JOB-06 unit side)."""

from __future__ import annotations

import pytest
from finplan_contracts.validate import validate

from finplan_model.control.auth import Principal
from finplan_model.control.states import TRANSITIONS, check_transition
from finplan_model.control.store import ConditionFailed
from finplan_model.control.validation import find_storage_location
from finplan_model.core.errors import FinplanError

from .support import READER, SUBMITTER, Harness, enable_logs, events_logged


def test_full_lifecycle_with_transition_timestamps():
    h = Harness(auto_approve=1.0)
    rid = h.start()
    run = h.run(rid)
    h.clock.advance(seconds=60)
    h.event(run["job_name"], "InProgress")
    h.clock.advance(seconds=120)
    status = h.service.get_job_status(Principal.from_arn(READER), rid)
    assert status["state"] == "running" and "completion_status" not in status
    assert [t["state"] for t in status["transitions"]] == ["queued", "starting", "running"]
    assert all(t["at"].endswith("Z") for t in status["transitions"])
    assert status["elapsed_seconds"] == 120
    assert find_storage_location(status) is None and "example-research-bucket" not in str(status)
    assert validate(status, "tools/get-job-status-response").valid
    h.clock.advance(seconds=60)
    h.succeed(rid)
    done = h.service.get_job_status(Principal.from_arn(READER), rid)
    assert done["state"] == "succeeded" and done["completion_status"] == "succeeded"
    assert done["elapsed_seconds"] == 180
    result = h.service.get_job_result(Principal.from_arn(READER), rid)
    assert result["completion_status"] == "succeeded" and result["solution_status"] == "optimal"
    assert result["evaluator_version"] and result["dataset_checksum"].startswith("sha256:")
    assert result["artifacts"][0]["checksum"].startswith("sha256:")
    assert validate(result, "tools/get-experiment-result-response").valid
    events = h.store.list_events(rid)
    assert [e["state"] for e in events] == ["queued", "starting", "running", "succeeded"]
    assert [e["seq"] for e in events] == [0, 1, 2, 3]


def test_late_event_after_cancelled_is_ignored_and_logged(caplog):
    enable_logs(caplog)
    h = Harness(auto_approve=1.0)
    rid = h.start()
    name = h.run(rid)["job_name"]
    h.event(name, "InProgress")
    h.service.cancel_job(Principal.from_arn(SUBMITTER), rid, {"idempotency_key": "c1"})
    h.event(name, "Stopped")
    assert h.run(rid)["state"] == "cancelled"
    before = h.run(rid)
    h.event(name, "Completed")
    h.event(name, "InProgress")
    assert h.run(rid) == before
    assert len(events_logged(caplog, "late_event_ignored")) == 2


def test_duplicate_events_change_nothing():
    h = Harness(auto_approve=1.0)
    rid = h.start()
    name = h.run(rid)["job_name"]
    h.event(name, "InProgress")
    rev = h.run(rid)["revision"]
    h.event(name, "InProgress")
    assert h.run(rid)["state"] == "running"
    assert len(h.store.list_events(rid)) == 3
    assert h.run(rid)["revision"] == rev


def test_events_for_other_environments_and_unknown_jobs_are_ignored():
    h = Harness()
    assert h.event("fm-gamma-run-01kdvdnaz83bammycegwf33dpm-a1", "Completed") == {"ignored": "not_this_environment"}
    assert h.event("some-other-job", "Completed") == {"ignored": "not_this_environment"}
    assert h.event("fm-beta-run-01kdvdnaz83bammycegwf33dpm-a1", "Completed") == {"ignored": "unknown_run"}


def test_terminal_states_have_no_transitions():
    for s in ("succeeded", "failed", "cancelled", "timed_out"):
        assert TRANSITIONS[s] == frozenset()
        with pytest.raises(FinplanError):
            check_transition(s, "running")


def test_conditional_write_rejects_stale_revision():
    h = Harness()
    _, sub = h.submit()
    run = h.run(sub["run_id"])
    stale = dict(run)
    h.service._update(run, wait_reason="x")
    with pytest.raises(ConditionFailed):
        h.store.update_run({**stale, "revision": stale["revision"] + 1}, stale["revision"], [])


def test_events_are_append_only():
    h = Harness()
    _, sub = h.submit()
    run = h.run(sub["run_id"])
    with pytest.raises(ConditionFailed):
        h.store.update_run({**run, "revision": 1}, 0, [{"run_id": run["run_id"], "seq": 0, "state": "queued", "at": run["submitted_at"]}])


def test_result_of_non_terminal_run_is_precondition_failed_with_state():
    h = Harness()
    _, sub = h.submit()
    with pytest.raises(FinplanError) as ei:
        h.service.get_job_result(Principal.from_arn(READER), sub["run_id"])
    assert ei.value.code == "PRECONDITION_FAILED" and ei.value.details["state"] == "awaiting_approval"


def test_failed_run_result_has_error_and_no_solution_status():
    h = Harness(auto_approve=1.0)
    rid = h.start()
    name = h.run(rid)["job_name"]
    h.sagemaker.set_status(name, "Failed", ExitMessage="Traceback: s3://example-private/key")
    h.event(name, "Failed", ExitMessage="Traceback: s3://example-private/key")
    result = h.service.get_job_result(Principal.from_arn(READER), rid)
    assert result["completion_status"] == "failed" and "solution_status" not in result
    assert result["error"]["code"] == "INTERNAL"
    assert "example-private" not in str(result) and "Traceback" not in str(result)
    assert validate(result, "job-result").valid


def test_unknown_and_malformed_run_ids():
    h = Harness()
    with pytest.raises(FinplanError) as ei:
        h.service.get_job_status(Principal.from_arn(READER), "run_01KDVDNAZ83BAMMYCEGWF33DPM")
    assert ei.value.code == "NOT_FOUND"
    with pytest.raises(FinplanError) as ei:
        h.service.get_job_status(Principal.from_arn(READER), "plan_01KDVDNAZ83BAMMYCEGWF33DPM")
    assert ei.value.code == "INVALID_IDENTIFIER"


def test_infeasible_run_is_succeeded_with_solution_status_infeasible():
    h = Harness(auto_approve=1.0)
    rid = h.start()
    h.event(h.run(rid)["job_name"], "InProgress")
    h.succeed(rid, "infeasible")
    result = h.service.get_job_result(Principal.from_arn(READER), rid)
    assert result["completion_status"] == "succeeded" and result["solution_status"] == "infeasible"


def test_completed_job_without_result_file_fails_internal():
    h = Harness(auto_approve=1.0)
    rid = h.start()
    name = h.run(rid)["job_name"]
    h.sagemaker.set_status(name, "Completed")
    h.event(name, "Completed")
    run = h.run(rid)
    assert run["state"] == "failed" and run["error"]["code"] == "INTERNAL"
    assert run["error"]["details"]["reason"] == "result_missing_or_invalid"
