"""HTTP layer of the job API (API Gateway proxy events): routing, statuses, envelopes, auth (JOB-01)."""

from __future__ import annotations

from finplan_contracts.validate import validate

from .support import APPROVER, READER, Harness, request


def test_submit_status_result_cancel_and_list_over_http():
    h = Harness(auto_approve=1.0)
    code, sub, headers = h.call("POST", "/v1/jobs", body=request(), headers={"X-Correlation-Id": "corr-http-test-0001"})
    assert code == 202 and headers["X-Correlation-Id"] == "corr-http-test-0001"
    rid = sub["run_id"]
    code, status, _ = h.call("GET", f"/v1/jobs/{rid}", principal=READER)
    assert code == 200 and status["state"] == "queued" and validate(status, "job-status").valid
    code, err, _ = h.call("GET", f"/v1/jobs/{rid}/result", principal=READER)
    assert code == 422 and err["code"] == "PRECONDITION_FAILED" and err["details"]["state"] == "queued"
    assert validate(err, "error").valid
    code, listing, _ = h.call("GET", "/v1/jobs", principal=READER, query={"state": "queued"})
    assert code == 200 and [j["run_id"] for j in listing["jobs"]] == [rid]
    code, cancelled, _ = h.call("POST", f"/v1/jobs/{rid}/cancel", body={"idempotency_key": "cancel-1"})
    assert code == 200 and cancelled["state"] == "cancelled"


def test_dry_run_over_http_is_200():
    h = Harness()
    code, resp, _ = h.call("POST", "/v1/jobs", body=request(dry_run=True))
    assert code == 200 and resp["run_id"] is None


def test_unauthenticated_call_is_401():
    h = Harness()
    code, err, _ = h.call("POST", "/v1/jobs", principal=None, body=request())
    assert code == 401 and err["code"] == "UNAUTHORIZED"
    assert h.store.runs == {}


def test_approve_over_http_by_tool_role_is_403():
    h = Harness()
    _, sub, _ = h.call("POST", "/v1/jobs", body=request())
    code, err, _ = h.call("POST", f"/v1/jobs/{sub['run_id']}/approve", body={})
    assert code == 403 and err["code"] == "FORBIDDEN"
    code, ok, _ = h.call("POST", f"/v1/jobs/{sub['run_id']}/approve", principal=APPROVER, body={})
    assert code == 200 and ok["state"] == "queued"


def test_error_statuses_and_envelopes():
    h = Harness()
    cases = [
        (("POST", "/v1/jobs"), {"body": "{not json"}, 400, "VALIDATION_FAILED"),
        (("POST", "/v1/jobs"), {"body": request(contract_version="7.0.0")}, 400, "UNSUPPORTED_CONTRACT_VERSION"),
        (("GET", "/v1/jobs/run_01KDVDNAZ83BAMMYCEGWF33DPM"), {}, 404, "NOT_FOUND"),
        (("GET", "/v1/jobs/pl_01KDVDNAZ83BAMMYCEGWF33DPM"), {}, 400, "INVALID_IDENTIFIER"),
        (("DELETE", "/v1/jobs"), {}, 404, "NOT_FOUND"),
        (("GET", "/v1/jobs"), {"query": {"bucket": "x"}}, 400, "VALIDATION_FAILED"),
        (("GET", "/v1/jobs"), {"query": {"next_token": "s3-key-like/token"}}, 400, "VALIDATION_FAILED"),
        (("GET", "/v1/jobs"), {"query": {"page_size": "1000"}}, 400, "VALIDATION_FAILED"),
        (("POST", "/v1/jobs"), {"body": request(job_type="rl_train")}, 503, "DEPENDENCY_UNAVAILABLE"),
    ]
    for (method, path), kw, status, code in cases:
        got, body, headers = h.call(method, path, **kw)
        assert (got, body["code"]) == (status, code), (method, path, body)
        assert validate(body, "error").valid
        assert headers["X-Correlation-Id"] == body["correlation_id"]


def test_internal_errors_never_leak_tracebacks():
    h = Harness()

    def boom(*a, **kw):
        raise RuntimeError("s3://example-private/secret traceback")

    h.service.get_job_status = boom
    code, body, _ = h.call("GET", "/v1/jobs/run_01KDVDNAZ83BAMMYCEGWF33DPM")
    assert code == 500 and body["code"] == "INTERNAL"
    assert "example-private" not in str(body) and "traceback" not in str(body)
