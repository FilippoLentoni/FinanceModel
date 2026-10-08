"""Contract tests of the job interface against the pinned finplan-contracts package (JOB-02, JOB-03,
JOB-06, JOB-08; CS-07 completion_status vs solution_status). Shared contract fixtures are read from
the installed package, never copied into this repository."""

from __future__ import annotations

import json

import pytest
from finplan_contracts.schemas import load_store
from finplan_contracts.validate import validate

from finplan_model.control.auth import Principal
from finplan_model.control.validation import find_storage_location
from finplan_model.core.errors import FinplanError, contract_version
from tests.unit.control.support import READER, SUBMITTER, Harness, job_result


def _fixtures(schema: str, kind: str) -> list[tuple[str, dict]]:
    d = load_store().fixtures_dir(schema) / kind
    return [(p.name, json.loads(p.read_text())) for p in sorted(d.glob("*.json"))]


def _as_ours(doc: dict) -> dict:
    """A contract fixture adapted to this deployment: a deployed job type and the pinned version."""
    body = json.loads(json.dumps(doc))
    body.pop("configuration_id", None)
    body.update(job_type="run_backtest", contract_version=contract_version(), compute_class="cpu")
    body["configuration"]["payload"]["universe"] = ["SPY"]
    return body


@pytest.mark.parametrize("name, doc", _fixtures("job-submission", "valid"))
def test_valid_contract_submissions_are_accepted(name, doc):
    if doc.get("compute_class") == "gpu":
        pytest.skip("no GPU job type is deployed in phase 1")
    h = Harness()
    code, resp = h.service.submit_job(Principal.from_arn(SUBMITTER), {**_as_ours(doc), "dry_run": True}, correlation_id="corr-contract-0001")
    assert code == 200 and validate(resp, "tools/submit-experiment-response").valid


@pytest.mark.parametrize("name, doc", _fixtures("job-submission", "invalid"))
def test_invalid_contract_submissions_are_rejected_and_create_nothing(name, doc):
    h = Harness()
    body = json.loads(json.dumps(doc))
    body["contract_version"] = contract_version()
    with pytest.raises(FinplanError) as ei:
        h.service.submit_job(Principal.from_arn(SUBMITTER), body, correlation_id="corr-contract-0002")
    assert ei.value.code in ("VALIDATION_FAILED", "INVALID_IDENTIFIER")
    assert h.store.runs == {} and h.sagemaker.calls == []
    env = ei.value.to_envelope("corr-contract-0002")
    assert validate(env, "error").valid and find_storage_location(env) is None


def _documents() -> list[tuple[str, dict, dict]]:
    """Status and result documents of runs in every terminal and non-terminal state."""
    out = []
    h = Harness(auto_approve=1.0)
    rd = Principal.from_arn(READER)

    def status(rid):
        return h.service.get_job_status(rd, rid)

    _, w = h.submit(idempotency_key="k-wait")
    h.settings.auto_approve = 0.0
    _, a = h.submit(idempotency_key="k-await")
    h.settings.auto_approve = 1.0
    out.append(("awaiting_approval", status(a["run_id"]), {}))
    out.append(("queued", status(w["run_id"]), {}))
    h.service.dispatch()
    out.append(("starting", status(w["run_id"]), {}))
    name = h.run(w["run_id"])["job_name"]
    h.event(name, "InProgress")
    out.append(("running", status(w["run_id"]), {}))
    h.service.cancel_job(Principal.from_arn(SUBMITTER), w["run_id"], {"idempotency_key": "c"})
    out.append(("stopping", status(w["run_id"]), {}))
    h.event(name, "Stopped")
    out.append(("cancelled", status(w["run_id"]), h.service.get_job_result(rd, w["run_id"])))
    for label, finish in (("succeeded", "ok"), ("infeasible", "infeasible"), ("failed", "crash"), ("timed_out", "limit")):
        h2 = Harness(auto_approve=1.0)
        rid = h2.start()
        nm = h2.run(rid)["job_name"]
        h2.event(nm, "InProgress")
        if finish in ("ok", "infeasible"):
            h2.run_io.put_result(rid, job_result(h2.run(rid), "optimal" if finish == "ok" else "infeasible"))
            h2.event(nm, "Completed")
        elif finish == "crash":
            h2.event(nm, "Failed", ExitMessage="exit 137")
        else:
            h2.event(nm, "Failed", FailureReason="MaxRuntimeExceeded")
        out.append((label, h2.service.get_job_status(rd, rid), h2.service.get_job_result(rd, rid)))
    return out


DOCS = _documents()


@pytest.mark.parametrize("label, status, result", DOCS, ids=[d[0] for d in DOCS])
def test_status_and_result_documents_validate(label, status, result):
    assert validate(status, "job-status").valid
    assert validate(status, "tools/get-job-status-response").valid
    assert find_storage_location(status) is None and "arn:" not in json.dumps(status)
    if result:
        assert validate(result, "job-result").valid
        assert validate(result, "tools/get-experiment-result-response").valid
        assert find_storage_location(result) is None and "arn:" not in json.dumps(result)
        assert result["completion_status"] == status["completion_status"]
        if result["completion_status"] == "succeeded":
            assert "solution_status" in result and "error" not in result
            assert set(result["payload"]) >= {"performance", "accuracy", "compute_cost"}
        if result["completion_status"] == "failed":
            assert "solution_status" not in result and result["error"]["code"] == "INTERNAL"
        if result["completion_status"] in ("cancelled", "timed_out"):
            assert result["artifacts_complete"] is False


def test_infeasible_is_a_successful_completion():
    (_, status, result) = next(d for d in DOCS if d[0] == "infeasible")
    assert status["completion_status"] == "succeeded" and result["solution_status"] == "infeasible"
