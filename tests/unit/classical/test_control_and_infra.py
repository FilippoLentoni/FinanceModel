import pytest

from finplan_model.control.auth import Principal
from finplan_model.core.errors import FinplanError
from tests.unit.control.support import Harness, arn, request


def test_only_bounded_trusted_weekly_benchmark_gets_pre_authorized_queue():
    h = Harness(auto_approve=0)
    principal = Principal.from_arn(
        arn("finplan-beta-financemodel-job-api-handler-research-role")
    )
    body = request(
        job_type="run_benchmark",
        max_runtime_seconds=900,
        idempotency_key="classical-weekly-2026-W01",
    )
    body["configuration"]["payload"]["objective"] = "classical_weekly_review"
    code, result = h.service.submit_job(
        principal, body, correlation_id="corr-classical-weekly-test"
    )
    assert code == 202 and result["state"] == "queued" and h.sagemaker.calls == []
    assert h.service.submit_job(
        principal, body, correlation_id="corr-classical-weekly-retry"
    ) == (code, result)
    assert len(h.store.runs) == 1


def test_weekly_job_replays_across_on_demand_and_scheduler_principals():
    h = Harness(auto_approve=0)
    principals = [
        Principal.from_arn(
            arn(f"finplan-beta-financemodel-job-api-handler-{name}-role")
        )
        for name in ("classical", "research")
    ]
    body = request(
        job_type="run_benchmark",
        max_runtime_seconds=900,
        idempotency_key="classical-weekly-2026-W01",
    )
    body["configuration"]["payload"]["objective"] = "classical_weekly_review"
    issued = h.service.submit_job(
        principals[0], body, correlation_id="corr-classical-first"
    )
    assert (
        h.service.submit_job(principals[1], body, correlation_id="corr-research-retry")
        == issued
    )
    assert len(h.store.runs) == 1 and h.sagemaker.calls == []
    body["configuration"]["payload"]["lookback_days"] = 120
    with pytest.raises(FinplanError) as exc:
        h.service.submit_job(
            principals[1], body, correlation_id="corr-research-changed"
        )
    assert exc.value.code == "IDEMPOTENCY_KEY_REUSED"
    assert len(h.store.runs) == 1


@pytest.mark.parametrize(
    "case",
    ["wrong_objective", "wrong_type", "too_long", "wrong_key", "production", "gpu"],
)
def test_controller_cannot_expand_its_pre_authorized_paid_scope(case):
    h = Harness(auto_approve=0)
    principal = Principal.from_arn(
        arn("finplan-beta-financemodel-job-api-handler-research-role")
    )
    body = request(
        job_type="run_benchmark",
        max_runtime_seconds=900,
        idempotency_key="classical-weekly-2026-W01",
    )
    body["configuration"]["payload"]["objective"] = "classical_weekly_review"
    if case == "wrong_objective":
        body["configuration"]["payload"]["objective"] = "backtest"
    if case == "wrong_type":
        body["job_type"] = "run_backtest"
    if case == "too_long":
        body["max_runtime_seconds"] = 1200
    if case == "wrong_key":
        body["idempotency_key"] = "unbounded-request"
    if case == "production":
        body["purpose"] = "production_candidate"
    if case == "gpu":
        body["compute_class"] = "gpu"
    with pytest.raises(FinplanError):
        h.service.submit_job(principal, body, correlation_id="corr-classical-invalid")
    assert h.store.runs == {} and h.sagemaker.calls == []
