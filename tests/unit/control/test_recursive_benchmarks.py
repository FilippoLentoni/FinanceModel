"""Real control dispatch/approval paths with fake compute; never launches AWS jobs."""

from types import SimpleNamespace

import pytest

from finplan_model.control.auth import Principal
from finplan_model.core.errors import FinplanError
from tests.unit.control.support import APPROVER, Harness, PRICES, request


def configuration(strategy, objective="llm_benchmark"):
    cfg = request()["configuration"]
    cfg["payload"].update(strategy=strategy, objective=objective)
    return cfg


@pytest.mark.parametrize("job_type,strategy", [("jev_backtest", "jev"), ("rl_weight_staging", "qwen_weights")])
def test_vendor_and_weight_download_require_approval_even_with_high_threshold(job_type, strategy):
    h = Harness(auto_approve=100.)
    _, result = h.submit(job_type=job_type, configuration=configuration(strategy))
    assert result["state"] == "awaiting_approval"
    h.service.dispatch()
    assert h.sagemaker.calls == []


def test_exact_swarm_dispatch_is_run_scoped_isolated_and_releases_on_failure():
    prices = {**PRICES, "usd_per_hour": {"ml.m5.xlarge": .25, "ml.g6.12xlarge": 6.}}
    h = Harness(auto_approve=100., prices=prices)
    prepared = []
    def prepare(run, spec, definition):
        prepared.append(spec)
        return {"channels": {"bundle": "s3://<research-bucket>/bundle/", "weights": "s3://<research-bucket>/weights/", "code": "s3://<research-bucket>/code/"}, "bundle_checksum": "sha256:" + "a" * 64, "code_checksum": "sha256:" + "b" * 64}
    h.deps.offline_batch = SimpleNamespace(prepare=prepare)
    _, result = h.submit(job_type="swarm_mode_a", configuration=configuration("qwen_swarm"))
    assert result["state"] == "awaiting_approval"
    h.service.dispatch()
    assert not h.sagemaker.calls
    rid = result["run_id"]
    h.service.approve_run(Principal.from_arn(APPROVER), rid, {})
    assert h.service.dispatch()["started"] == [rid]
    creates = [args for op, args in h.sagemaker.calls if op == "CreateTrainingJob"]
    assert len(creates) == 1
    job = creates[0]
    assert job["EnableNetworkIsolation"] is True
    assert {c["ChannelName"] for c in job["InputDataConfig"]} == {"bundle", "weights", "code"}
    assert job["AlgorithmSpecification"]["ContainerEntrypoint"] == ["python", "/opt/ml/input/data/code/bootstrap.py"]
    assert job["StoppingCondition"]["MaxRuntimeInSeconds"] == 900
    assert job["ResourceConfig"]["VolumeSizeInGB"] == 200
    assert job["Environment"]["HF_HUB_OFFLINE"] == "1" and prepared
    name = h.run(rid)["job_name"]
    h.sagemaker.set_status(name, "Failed", FailureReason="fixture readiness failure")
    h.service.handle_sagemaker_event({"source": "aws.sagemaker", "detail-type": "SageMaker Training Job State Change", "detail": {"TrainingJobName": name, "TrainingJobStatus": "Failed"}})
    assert h.run(rid)["state"] == "failed"
    assert h.service.leases.holders("gpu") == []


def test_missing_weights_fail_before_create_and_release_gpu_lease():
    prices = {**PRICES, "usd_per_hour": {"ml.g6.12xlarge": 6.}}
    h = Harness(auto_approve=100., prices=prices)
    def reject(*args):
        raise FinplanError.precondition("not staged", reason="qwen_weights_not_staged")
    h.deps.offline_batch = SimpleNamespace(prepare=reject)
    _, result = h.submit(job_type="swarm_mode_a", configuration=configuration("qwen_swarm"))
    h.service.approve_run(Principal.from_arn(APPROVER), result["run_id"], {})
    h.service.dispatch()
    assert h.run(result["run_id"])["state"] == "failed"
    assert not h.sagemaker.calls and h.service.leases.holders("gpu") == []
