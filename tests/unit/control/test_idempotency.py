"""Task 7.2: idempotency records for submit_job and cancel_job (JOB-04)."""

from __future__ import annotations

import pytest

from finplan_model.control.auth import Principal
from finplan_model.core.errors import FinplanError

from .support import APPROVER, READER, SUBMITTER, Harness, arn, request


def test_retry_after_timeout_returns_original_run_and_one_job():
    h = Harness(auto_approve=1.0)
    code1, first = h.submit()
    code2, again = h.submit()
    assert (code1, code2) == (202, 202)
    assert again == first
    assert len(h.store.runs) == 1
    h.service.dispatch()
    h.service.dispatch()
    # a third retry after the run started still returns the original response
    assert h.submit()[1] == first
    assert len(h.created_jobs()) == 1


def test_retry_from_a_new_session_of_the_same_role_replays():
    h = Harness()
    _, first = h.submit(principal=arn("finplan-beta-financelambdastool-submitter-role", "session-a"))
    _, again = h.submit(principal=arn("finplan-beta-financelambdastool-submitter-role", "session-b"))
    assert again["run_id"] == first["run_id"]


def test_same_key_with_different_body_is_idempotency_key_reused():
    h = Harness()
    h.submit()
    body = request()
    body["configuration"]["payload"]["strategy"] = "buy_and_hold"
    with pytest.raises(FinplanError) as ei:
        h.service.submit_job(Principal.from_arn(SUBMITTER), body, correlation_id="corr-test-0005")
    assert ei.value.code == "IDEMPOTENCY_KEY_REUSED" and ei.value.retryable is False
    assert len(h.store.runs) == 1


def test_key_scope_includes_the_principal():
    h = Harness()
    _, a = h.submit(principal=SUBMITTER)
    _, b = h.submit(principal=arn("finplan-beta-financelambdastool-plan-writer-role"))
    assert a["run_id"] != b["run_id"]


def test_idempotency_record_retention_is_at_least_seven_days():
    h = Harness()
    h.submit()
    (rec,) = h.store.idempotency.values()
    from finplan_model.core.clock import parse_utc

    assert (parse_utc(rec["retain_until"]) - parse_utc(rec["recorded_at"])).days >= 7
    assert rec["scope"] == {"principal": rec["scope"]["principal"], "environment": "beta", "operation": "submit_job"}
    assert rec["scope"]["principal"].endswith(":role/finplan-beta-financelambdastool-submitter-role")


def test_cancel_job_is_idempotent():
    h = Harness()
    _, sub = h.submit()
    rid = sub["run_id"]
    c1, r1 = h.service.cancel_job(Principal.from_arn(SUBMITTER), rid, {"idempotency_key": "cancel-1"})
    h.clock.advance(seconds=30)
    c2, r2 = h.service.cancel_job(Principal.from_arn(SUBMITTER), rid, {"idempotency_key": "cancel-1"})
    assert (c1, c2) == (200, 200) and r1 == r2 and r1["state"] == "cancelled"
    with pytest.raises(FinplanError) as ei:
        h.service.cancel_job(Principal.from_arn(SUBMITTER), rid, {"idempotency_key": "cancel-1", "reason": "different"})
    assert ei.value.code == "IDEMPOTENCY_KEY_REUSED"


def test_cancel_requires_an_idempotency_key():
    h = Harness()
    _, sub = h.submit()
    with pytest.raises(FinplanError) as ei:
        h.service.cancel_job(Principal.from_arn(SUBMITTER), sub["run_id"], {})
    assert ei.value.code == "VALIDATION_FAILED" and ei.value.details["pointer"] == "/idempotency_key"
    assert h.run(sub["run_id"])["state"] == "awaiting_approval"


def test_reader_and_approver_rights_on_cancel():
    h = Harness()
    _, sub = h.submit()
    with pytest.raises(FinplanError) as ei:
        h.service.cancel_job(Principal.from_arn(READER), sub["run_id"], {"idempotency_key": "k"})
    assert ei.value.code == "FORBIDDEN"
    _, resp = h.service.cancel_job(Principal.from_arn(APPROVER), sub["run_id"], {"idempotency_key": "k"})
    assert resp["state"] == "cancelled"
