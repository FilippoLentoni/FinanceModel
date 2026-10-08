"""Task 7.1, 7.4, 7.5: submit_job validation, configuration_id, run_id minting, dry run, purpose
authorization and DEPENDENCY_UNAVAILABLE (JOB-02, JOB-03, JOB-07, JOB-08, JOB-09)."""

from __future__ import annotations

import pytest
from finplan_contracts.canonical import configuration_id
from finplan_contracts.validate import validate

from finplan_model.control.auth import Principal
from finplan_model.core.errors import FinplanError
from finplan_model.core.ids import RUN_ID_RE
from finplan_model.core.platform import FixturePlatformClient, build_synthetic_snapshot

from .support import CANDIDATE, SID, SUBMITTER, Harness, request


def _err(fn, *a, **kw) -> FinplanError:
    with pytest.raises(FinplanError) as ei:
        fn(*a, **kw)
    return ei.value


def _nothing_recorded(h: Harness) -> None:
    assert h.store.runs == {} and h.store.idempotency == {} and h.store.leases == {}
    assert h.sagemaker.calls == []


# ---------------------------------------------------------------- JOB-02
def test_valid_submission_mints_run_id_and_returns_immediately():
    h = Harness()
    code, resp = h.submit()
    assert code == 202
    assert RUN_ID_RE.match(resp["run_id"])
    assert resp["configuration_id"] == configuration_id(request()["configuration"])
    assert resp["state"] == "awaiting_approval"  # auto-approve threshold defaults to 0 (FM-OQ-3)
    assert resp["dry_run"] is False and resp["cost_estimate"]["budget_category"] == "cpu_research"
    assert validate(resp, "tools/submit-experiment-response").valid
    # asynchronous: nothing started, no SageMaker call during submission
    assert h.sagemaker.calls == []
    run = h.run(resp["run_id"])
    assert run["state"] == "awaiting_approval" and run["revision"] == 0


def test_below_threshold_submission_is_queued_and_kicks_dispatcher():
    h = Harness(auto_approve=1.0)
    code, resp = h.submit()
    assert code == 202 and resp["state"] == "queued"
    assert h.kicks == 1


def test_caller_supplied_run_id_is_rejected():
    h = Harness()
    err = _err(h.submit, run_id="run_01KDVDNAZ83BAMMYCEGWF33DPM")
    assert err.code == "VALIDATION_FAILED" and err.details["pointer"] == "/run_id"
    _nothing_recorded(h)


# ---------------------------------------------------------------- JOB-03
@pytest.mark.parametrize(
    "over, pointer",
    [
        ({"output_location": "s3://example-bucket/out/"}, "/output_location"),
        ({"input_location": "s3://example-bucket/in/data.json"}, "/input_location"),
    ],
)
def test_storage_paths_are_rejected_without_echo(over, pointer):
    h = Harness()
    err = _err(h.submit, **over)
    assert err.code == "VALIDATION_FAILED" and err.details["pointer"] == pointer
    env = err.to_envelope("corr-test-0001")
    assert "s3://" not in str(env) and "example-bucket" not in str(env)
    _nothing_recorded(h)


def test_storage_path_inside_configuration_is_rejected():
    h = Harness()
    body = request()
    body["configuration"]["payload"]["universe"] = ["s3://example-bucket/key"]
    err = _err(h.service.submit_job, Principal.from_arn(SUBMITTER), body, correlation_id="corr-test-0002")
    assert err.code == "VALIDATION_FAILED" and err.details["pointer"] == "/configuration/payload/universe/0"
    _nothing_recorded(h)


def test_unsupported_contract_major_lists_served_majors():
    h = Harness()
    err = _err(h.submit, contract_version="0.2.2")  # FinanceModel serves major 1 only (never deployed on 0.x)
    assert err.code == "UNSUPPORTED_CONTRACT_VERSION"
    assert err.details["served_contract_majors"] == [1]
    _nothing_recorded(h)


@pytest.mark.parametrize(
    "over, code, pointer",
    [
        ({"job_type": "fixture_optimizer"}, "VALIDATION_FAILED", "/job_type"),
        ({"domain": "supply_chain"}, "VALIDATION_FAILED", None),
        ({"domain_schema_version": "9.0"}, "VALIDATION_FAILED", None),
        ({"purpose": "trading"}, "VALIDATION_FAILED", None),
        ({"idempotency_key": None}, "VALIDATION_FAILED", None),
        ({"max_runtime_seconds": 1801}, "VALIDATION_FAILED", "/max_runtime_seconds"),
        ({"compute_class": "gpu"}, "VALIDATION_FAILED", "/compute_class"),
        ({"configuration_id": "cfg_" + "0" * 64}, "VALIDATION_FAILED", "/configuration_id"),
    ],
)
def test_invalid_submissions_create_nothing(over, code, pointer):
    h = Harness()
    err = _err(h.submit, **over)
    assert err.code == code
    if pointer is not None:
        assert err.details["pointer"] == pointer
    _nothing_recorded(h)


def test_unknown_strategy_is_rejected():
    h = Harness()
    body = request()
    body["configuration"]["payload"]["strategy"] = "magic_alpha"
    err = _err(h.service.submit_job, Principal.from_arn(SUBMITTER), body, correlation_id="corr-test-0003")
    assert err.code == "VALIDATION_FAILED" and err.details["pointer"] == "/configuration/payload/strategy"
    _nothing_recorded(h)


def test_runtime_at_ceiling_is_accepted_and_recorded():
    h = Harness()
    code, resp = h.submit(max_runtime_seconds=1800)
    assert code == 202
    assert h.run(resp["run_id"])["max_runtime_seconds"] == 1800


# ---------------------------------------------------------------- JOB-08
def test_dry_run_returns_estimate_and_creates_nothing():
    h = Harness()
    code, resp = h.submit(dry_run=True)
    assert code == 200
    assert resp["run_id"] is None and resp["state"] is None and resp["dry_run"] is True
    assert resp["configuration_id"] == configuration_id(request()["configuration"])
    assert resp["cost_estimate"]["estimated_usd_upper_bound"] == pytest.approx(0.0725)
    assert validate(resp, "tools/submit-experiment-response").valid
    _nothing_recorded(h)
    assert h.kicks == 0


def test_dry_run_then_real_submission_with_same_key_is_not_a_reuse():
    h = Harness()
    h.submit(dry_run=True)
    code, resp = h.submit()
    assert code == 202 and resp["run_id"]


# ---------------------------------------------------------------- JOB-07
def test_production_candidate_without_grant_is_forbidden():
    h = Harness()
    err = _err(h.submit, purpose="production_candidate")
    assert err.code == "FORBIDDEN"
    _nothing_recorded(h)


def test_production_candidate_with_grant_is_accepted():
    h = Harness()
    code, resp = h.submit(principal=CANDIDATE, purpose="production_candidate")
    assert code == 202
    assert h.run(resp["run_id"])["purpose"] == "production_candidate"


# ---------------------------------------------------------------- JOB-09
def test_announced_job_type_not_deployed_is_dependency_unavailable():
    h = Harness()
    err = _err(h.submit, job_type="rl_train")
    assert err.code == "DEPENDENCY_UNAVAILABLE" and err.retryable is False
    _nothing_recorded(h)


def test_job_type_without_published_definition_is_dependency_unavailable():
    h = Harness(job_definitions={})
    err = _err(h.submit)
    assert err.code == "DEPENDENCY_UNAVAILABLE" and err.retryable is False
    _nothing_recorded(h)


def test_fixture_backed_jobs_accepted_in_every_environment():
    from .support import env_config

    for env in ("beta", "gamma", "prod"):
        h = Harness(cfg=env_config(env))
        code, resp = h.submit(dry_run=True)
        assert code == 200 and resp["cost_estimate"]["synthetic"] is True


# ---------------------------------------------------------------- WS-03 at submission
def _platform(status: str) -> FixturePlatformClient:
    p = FixturePlatformClient()
    obs = [{"instrument_id": "SPY", "session_date": "2026-01-05", "kind": "completed_daily", "close": 100.0, "synthetic": True}]
    payload = {"dataset_id": "finance/etf-daily/SPY", "calendar": "XNYS", "instruments": [{"instrument_id": "SPY", "asset_class": "etf", "currency": "USD", "synthetic": True}], "observations": obs, "synthetic": True}
    rec, blobs = build_synthetic_snapshot(payload, input_snapshot_id=SID, status=status)
    p.add_snapshot(rec, blobs)
    return p


def test_unapproved_snapshot_is_refused_at_submission():
    h = Harness(platform=_platform("committed"))
    err = _err(h.submit)
    assert err.code == "PRECONDITION_FAILED" and err.details["reason"] == "snapshot_not_approved"
    _nothing_recorded(h)


def test_unknown_snapshot_is_not_found_at_submission():
    h = Harness(platform=FixturePlatformClient())
    err = _err(h.submit)
    assert err.code == "NOT_FOUND"
    _nothing_recorded(h)


def test_approved_synthetic_snapshot_marks_the_run_synthetic():
    h = Harness(platform=_platform("approved"))
    body = request(synthetic=None)
    body["configuration"].pop("synthetic")
    code, resp = h.service.submit_job(Principal.from_arn(SUBMITTER), body, correlation_id="corr-test-0004")
    assert code == 202 and h.run(resp["run_id"])["synthetic"] is True
