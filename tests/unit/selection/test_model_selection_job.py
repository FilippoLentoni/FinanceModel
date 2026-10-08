"""The ``model_selection`` job kind end to end, offline: submission (protocol frozen from configuration,
validation, cost estimate under the beta/gamma auto-approve threshold), the generated
``CreateTrainingJob`` request (checked against the SageMaker API model), the container run on a
synthetic research-universe snapshot (RL-07: a Training job; the learners never run in the control
plane), the training-job state change, cancel and timeout paths, the configuration build check and
the IaC wiring. All data is synthetic."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from finplan_model.control.auth import Principal
from finplan_model.core.artifacts import InMemoryArtifactStore
from finplan_model.core.config import validate_config
from finplan_model.core.platform import FixturePlatformClient, build_synthetic_snapshot
from finplan_model.jobs.entrypoint import run_job
from tests.unit.control.support import Harness, arn, env_config, request
from tests.unit.rl.support import TICKERS, market, tiny_protocol

SID = "snap_01KDVDP88REHGPBXFX6CHX92KS"
DISCLOSURES = [
    {"kind": "hindsight_selection", "text": "Synthetic disclosure: the instruments were selected with hindsight."},
    {"kind": "survivorship", "text": "Synthetic disclosure: failed or delisted companies are absent."},
]


def universe_payload() -> dict:
    mkt = market()
    obs = []
    for iid in mkt.instruments:
        for b in mkt.bars_of(iid):
            obs.append({"instrument_id": iid, "session_date": b.session_date.isoformat(), "kind": "completed_daily", "open": b.open, "high": b.high, "low": b.low, "close": b.close, "adj_close": b.close, "volume": int(b.volume or 0), "synthetic": True})
    payload = {
        "dataset_id": "finance/equity-etf-daily/research-universe",
        "calendar": "XNYS",
        "instruments": [{"instrument_id": i, "asset_class": "equity", "kind": "equity", "currency": "USD", "synthetic": True} for i in TICKERS]
        + [{"instrument_id": "USD_CASH", "asset_class": "cash", "kind": "cash", "return_assumption": "zero_nominal", "currency": "USD", "synthetic": True}],
        "observations": obs,
        "universe": {"instruments": [{"instrument_id": i, "kind": "equity"} for i in TICKERS] + [{"instrument_id": "USD_CASH", "kind": "cash", "return_assumption": "zero_nominal"}], "history_start": "2010-10-01", "return_basis": "adj_close"},
        "bias_disclosures": copy.deepcopy(DISCLOSURES),
        "synthetic": True,
    }
    return payload


def universe_platform() -> FixturePlatformClient:
    p = FixturePlatformClient()
    rec, blobs = build_synthetic_snapshot(universe_payload(), input_snapshot_id=SID)
    p.add_snapshot(rec, blobs)
    return p


def _cfg(protocol=None):
    return env_config(mutate=lambda raw: raw["job_types"]["model_selection"].update(protocol=protocol or tiny_protocol()))


def selection_request(**over):
    cfg = {"domain": "finance", "domain_schema_version": "1.0", "payload": {"strategy": "model_selection", "objective": "backtest", "universe": [*TICKERS, "USD_CASH"], "rebalance_frequency": "daily", "constraints": {"long_only": True, "max_weight": 1.0}}, "synthetic": True}
    body = {"job_type": "model_selection", "configuration": cfg, "input_snapshot_id": SID, "evaluation_window": None, **over}
    return body


def _harness(**kw) -> tuple[Harness, FixturePlatformClient]:
    platform = universe_platform()
    return Harness(cfg=_cfg(kw.pop("protocol", None)), platform=platform, **kw), platform


def _validate_against_api_model(req: dict) -> None:
    import boto3
    from botocore.validate import ParamValidator

    model = boto3.client("sagemaker", region_name="us-east-2").meta.service_model.operation_model("CreateTrainingJob")
    report = ParamValidator().validate(req, model.input_shape)
    assert not report.has_errors(), report.generate_report()


# ----------------------------------------------------------------- submission and the training-job request
def test_submission_freezes_the_protocol_and_starts_one_cpu_training_job():
    h, _ = _harness()
    run_id = h.start(**selection_request())
    run = h.run(run_id)
    assert run["sagemaker_job"] == "training" and run["selection_protocol"]["protocol_version"] == "finplan-model-selection/1"
    assert run["selection_protocol"]["rl"]["seeds"] == [0, 1, 2] and run["test_period_prior_accesses"] == 0
    assert not [op for op, _ in h.sagemaker.calls if op == "CreateProcessingJob"]
    (req,) = [kw for op, kw in h.sagemaker.calls if op == "CreateTrainingJob"]
    _validate_against_api_model(req)
    assert req["AlgorithmSpecification"]["ContainerEntrypoint"] == ["python", "-m", "finplan_model.jobs"]
    assert req["AlgorithmSpecification"]["ContainerArguments"] == ["model_selection"]
    assert "@sha256:" in req["AlgorithmSpecification"]["TrainingImage"]
    assert req["ResourceConfig"] == {"InstanceType": "ml.m5.xlarge", "InstanceCount": 1, "VolumeSizeInGB": 10}
    assert req["StoppingCondition"] == {"MaxRuntimeInSeconds": 3000}
    assert req["OutputDataConfig"]["S3OutputPath"] == "s3://example-research-bucket/scratch/training-output/"
    assert req["ProfilerConfig"] == {"DisableProfiler": True} and req["EnableNetworkIsolation"] is False
    tags = {t["Key"]: t["Value"] for t in req["Tags"]}
    assert tags["environment"] == "beta" and tags["owner-repo"] == "financemodel" and tags["run-id"] == run_id
    assert req["Environment"]["FINPLAN_JOB_TYPE"] == "model_selection" and req["Environment"]["FINPLAN_RUN_SPEC_SHA256"] != "pending"
    spec = h.run_io.get_spec(run_id, None)
    assert spec["selection_protocol"] == run["selection_protocol"] and spec["compute"]["sagemaker_job"] == "training"
    assert "s3://" not in json.dumps(spec)


def test_estimate_stays_under_the_beta_gamma_auto_approve_threshold():
    h, _ = _harness(auto_approve=0.25)
    code, resp = h.service.submit_job(Principal.from_arn(arn("finplan-beta-financelambdastool-submitter-role")), request(**selection_request(dry_run=True)), correlation_id="corr-test-selection-dry")
    assert code == 200 and resp["cost_estimate"]["budget_category"] == "cpu_research"
    # synthetic test price 0.25 USD/h x 3000 s + 0.01 storage
    est = resp["cost_estimate"]["estimated_usd_upper_bound"]
    assert est == pytest.approx(0.218334, abs=1e-6) and est < 0.25
    code, resp = h.submit(**selection_request())
    assert code == 202 and resp["state"] == "queued"  # auto-approved under USD 0.25 (beta/gamma, decision 24)
    h2, _ = _harness()  # prod-like threshold 0: a human approver must approve
    code, resp = h2.submit(**selection_request())
    assert resp["state"] == "awaiting_approval"


@pytest.mark.parametrize(
    "over,code",
    [
        ({"configuration": {"domain": "finance", "domain_schema_version": "1.0", "payload": {"strategy": "ppo", "universe": ["VOO"]}, "synthetic": True}}, "VALIDATION_FAILED"),
        ({"purpose": "production_candidate"}, "VALIDATION_FAILED"),
    ],
)
def test_invalid_model_selection_submissions(over, code):
    from finplan_model.core.errors import FinplanError

    h, _ = _harness()
    with pytest.raises(FinplanError) as exc:
        h.submit(**{**selection_request(), **over})
    assert exc.value.code == code
    assert not h.sagemaker.calls


def test_the_daily_trigger_cannot_submit_model_selection():
    from finplan_model.core.errors import FinplanError

    h, _ = _harness()
    with pytest.raises(FinplanError) as exc:
        h.submit(principal=arn("finplan-beta-financialplanning-daily-trigger-step-role"), **selection_request())
    assert exc.value.code == "FORBIDDEN"


# ----------------------------------------------------------------- the container run and the state change
def test_container_run_writes_a_contract_result_with_every_section():
    h, platform = _harness()
    run_id = h.start(**selection_request())
    ((_, req),) = [(op, kw) for op, kw in h.sagemaker.calls if op == "CreateTrainingJob"]
    env = req["Environment"]
    artifacts = InMemoryArtifactStore()
    code, doc = run_job("model_selection", run_id, run_io=h.run_io, platform=platform, artifacts=artifacts, environment="beta", spec_checksum=env["FINPLAN_RUN_SPEC_SHA256"], correlation_id=env["FINPLAN_CORRELATION_ID"], image_digest=env["FINPLAN_IMAGE_DIGEST"], clock=h.clock)
    assert code == 0, doc
    payload = doc["payload"]
    ms = payload["model_selection"]
    assert payload["benchmark"]["schema"] == "finplan.benchmark_comparison/1" and payload["benchmark"] == ms["comparison"]["test"]
    assert payload["bias_section"]["title"] == "Hindsight and survivorship bias" and payload["bias_section"]["bias_disclosures"] == DISCLOSURES
    assert set(ms["comparison"]) == {"train", "validation", "test"} and set(ms["rl"]) == {"ppo", "sac"}
    assert ms["data"]["universe"] == sorted(TICKERS)
    kinds = [a["kind"] for a in doc["artifacts"]]
    assert kinds[0] == "run_artifact" and kinds.count("rl_policy") == 6
    assert {p["artifact_id"] for p in ms["policies"]} == {a["artifact_id"] for a in doc["artifacts"] if a["kind"] == "rl_policy"}
    comp = ms["compute"]
    assert comp["sagemaker_job"] == "training" and comp["instance_type"] == "ml.m5.xlarge" and comp["max_runtime_seconds"] == 3000
    assert comp["estimated_usd_upper_bound"] is not None and 0 <= comp["estimated_actual_usd"] <= comp["estimated_usd_upper_bound"]
    assert payload["compute_cost"]["instance_seconds"] > 0
    assert len(json.dumps(doc)) < 300_000
    # the state change of the Training job completes the run with that result
    h.sagemaker.set_status(h.run(run_id)["job_name"], "Completed")
    out = h.service.handle_sagemaker_event({"source": "aws.sagemaker", "detail-type": "SageMaker Training Job State Change", "detail": {"TrainingJobName": h.run(run_id)["job_name"], "TrainingJobStatus": "Completed"}})
    assert out["state"] == "succeeded"
    result = h.service.get_job_result(Principal.from_arn(arn("finplan-beta-financelambdastool-submitter-role")), run_id)
    assert result["payload"]["model_selection"]["selection"]["selection_checksum"].startswith("sha256:")


def test_local_container_run_produces_policy_artifacts_without_aws(tmp_path):
    """Task 2.1 (local run of the job entry point on fixtures, fixture step count, no AWS call)."""
    from finplan_model.jobs.runio import LocalRunIO

    proto = tiny_protocol()
    proto["rl"]["algorithms"] = ["ppo"]
    proto["rl"]["ppo"] = {**proto["rl"]["ppo"], "total_timesteps": 16, "eval_every": 16}
    h, _ = _harness(protocol=proto)
    rid = h.start(**selection_request())
    LocalRunIO(tmp_path).put_spec(rid, h.run_io.get_spec(rid, None))
    pf = tmp_path / "payload.json"
    pf.write_text(json.dumps(universe_payload()), encoding="utf-8")
    root = Path(__file__).resolve().parents[3]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root)])}
    out = subprocess.run([sys.executable, "-m", "finplan_model.jobs", "model_selection", "--local", str(tmp_path), "--run-id", rid, "--fixture-snapshot", str(pf)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300, check=False)
    assert out.returncode == 0, out.stderr[-3000:] + out.stdout[-3000:]
    doc = json.loads((tmp_path / "runs" / rid / "job-result.json").read_text())
    from finplan_contracts.validate import validate

    assert validate(doc, "job-result").valid and doc["completion_status"] == "succeeded"
    policies = doc["payload"]["model_selection"]["policies"]
    assert len(policies) == 3 and all(p["checksum"].startswith("sha256:") and p["algorithm"] == "ppo" for p in policies)
    stored = list((tmp_path / "artifacts").rglob("*"))
    assert sum(1 for f in stored if f.is_file()) >= 4  # evidence + 3 policies
    assert '"event": "rl_training_finished"' in out.stdout and '"event": "selection_frozen"' in out.stdout


def test_cancel_stops_the_training_job_and_timeouts_are_reported():
    h, _ = _harness()
    run_id = h.start(**selection_request())
    name = h.run(run_id)["job_name"]
    h.service.cancel_job(Principal.from_arn(arn("finplan-beta-financelambdastool-submitter-role")), run_id, {"reason": "user request", "idempotency_key": "cancel-key-0001"})
    assert ("StopTrainingJob", {"TrainingJobName": name}) in h.sagemaker.calls
    h.sagemaker.set_status(name, "Stopped", SecondaryStatus="Stopped")
    h.service.dispatch()
    assert h.run(run_id)["state"] == "cancelled"
    assert any(op == "DescribeTrainingJob" for op, _ in h.sagemaker.calls)

    h2, _ = _harness()
    rid = h2.start(**selection_request())
    n2 = h2.run(rid)["job_name"]
    h2.sagemaker.set_status(n2, "Stopped", SecondaryStatus="MaxRuntimeExceeded")
    h2.service.handle_sagemaker_event({"detail-type": "SageMaker Training Job State Change", "detail": {"TrainingJobName": n2, "TrainingJobStatus": "Stopped", "SecondaryStatus": "MaxRuntimeExceeded"}})
    assert h2.run(rid)["state"] == "timed_out"


def test_processing_events_for_a_training_run_are_ignored():
    h, _ = _harness()
    rid = h.start(**selection_request())
    name = h.run(rid)["job_name"]
    out = h.service.handle_sagemaker_event({"detail-type": "SageMaker Processing Job State Change", "detail": {"ProcessingJobName": name, "ProcessingJobStatus": "Completed"}})
    assert out == {"ignored": "job_kind_mismatch"} and h.run(rid)["state"] == "starting"


# ----------------------------------------------------------------- configuration build check
def test_build_check_refuses_an_expensive_or_non_training_model_selection():
    raw = copy.deepcopy(dict(env_config().raw))
    assert validate_config("beta", raw) == []
    ms = raw["job_types"]["model_selection"]
    assert ms["sagemaker_job"] == "training" and ms["max_runtime_seconds"] * ms["cost_check_usd_per_hour"] / 3600 <= 0.25
    for change in ({"cost_check_usd_per_hour": 0.5}, {"cost_cap_usd": 0.5}, {"sagemaker_job": "processing"}, {"sagemaker_job": "lambda"}, {"protocol": {**tiny_protocol(), "selection": {"metric": "test_sharpe"}}}):
        bad = copy.deepcopy(raw)
        bad["job_types"]["model_selection"].update(change)
        assert validate_config("beta", bad), change


def test_beta_stability_protocol_is_isolated_from_gamma_and_prod():
    protos = [env_config(e).raw["job_types"]["model_selection"]["protocol"] for e in ("beta", "gamma", "prod")]
    assert protos[1] == protos[2]
    from finplan_model.selection.protocol import DEFAULT_PROTOCOL

    assert protos[1] == DEFAULT_PROTOCOL
    beta = protos[0]
    assert beta["splits"] == DEFAULT_PROTOCOL["splits"]
    assert beta["traditional"] == DEFAULT_PROTOCOL["traditional"]
    assert beta["rl"]["algorithms"] == ["ppo"] and beta["rl"]["policy_selection"] == "ensemble"
    assert beta["rl"]["seeds"] == [0, 1, 2, 3, 4]
    assert beta["rl"]["env"]["episode_sessions"] == 64


# ----------------------------------------------------------------- IaC
def test_iam_and_events_wire_training_jobs():
    from infra.stacks.policies import control_role_policy, job_execution_policy

    for kind in ("api", "dispatcher", "state"):
        st = {s["Sid"]: s for s in control_role_policy("beta", kind, partition="aws", region="us-east-2", account="<account-id>")["Statement"]}
        assert "sagemaker:CreateTrainingJob" in st["StartTaggedJobs"]["Action"]
        assert any("training-job/fm-beta-" in r for r in st["StartTaggedJobs"]["Resource"])
        assert st["StartTaggedJobs"]["Condition"]["StringEquals"]["aws:RequestTag/environment"] == "beta"
        assert {"sagemaker:StopTrainingJob", "sagemaker:DescribeTrainingJob"} <= set(st["ManageOwnJobs"]["Action"])
    job = {s["Sid"]: s for s in job_execution_policy("beta", partition="aws", region="us-east-2", account="<account-id>")["Statement"]}
    assert any(r.endswith("log-group:/aws/sagemaker/TrainingJobs:*") for r in job["ProcessingJobLogs"]["Resource"])
    assert any(r.endswith("/scratch/*") for r in job["WriteRunOutputs"]["Resource"])  # the training output path
