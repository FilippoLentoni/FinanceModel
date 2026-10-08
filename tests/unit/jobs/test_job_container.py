"""Task group 6: the CPU job container entry points, snapshot resolution and integrity inside the job,
and job result documents (6.1, 6.2 WS-03/WS-04, 6.3 JOB-06), end to end with the control plane,
all offline (no AWS, no docker needed)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from finplan_contracts.validate import validate

from finplan_model.control.auth import Principal
from finplan_model.control.validation import find_storage_location
from finplan_model.core.artifacts import InMemoryArtifactStore
from finplan_model.jobs.entrypoint import run_job
from finplan_model.jobs.runio import LocalRunIO
from finplan_model.jobs.strategy_resolver import register_strategy, unregister_strategy
from tests.unit.control.support import READER, Harness, request

from .support import SID, FixtureCash, FixtureStatic, platform_with_snapshot, synthetic_payload

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def fixture_strategies():
    FixtureStatic.calls = 0
    register_strategy("fixture_static", lambda p: FixtureStatic())
    yield
    unregister_strategy("fixture_static")


def _started(job_type: str = "run_backtest", strategy: str = "fixture_static") -> tuple[Harness, str, dict]:
    h = Harness(auto_approve=10.0)
    body = request(job_type=job_type)
    body["configuration"]["payload"]["strategy"] = strategy
    code, resp = h.service.submit_job(Principal.from_arn("arn:aws:sts::<account-id>:assumed-role/finplan-beta-financelambdastool-submitter-role/s"), body, correlation_id="corr-job-test-0001")
    assert code == 202, resp
    h.service.dispatch()
    rid = resp["run_id"]
    (req,) = h.created_jobs()
    return h, rid, req


def _run(h: Harness, rid: str, req: dict, platform=None, artifacts=None, **kw):
    env = req["Environment"]
    return run_job(
        req["AppSpecification"]["ContainerArguments"][0],
        rid,
        run_io=h.run_io,
        platform=platform or platform_with_snapshot(),
        artifacts=artifacts if artifacts is not None else InMemoryArtifactStore(),
        environment=env["FINPLAN_ENVIRONMENT"],
        spec_checksum=kw.pop("spec_checksum", env["FINPLAN_RUN_SPEC_SHA256"]),
        correlation_id=env["FINPLAN_CORRELATION_ID"],
        image_digest=env["FINPLAN_IMAGE_DIGEST"],
        clock=h.clock,
        **kw,
    )


# ------------------------------------------------------------------ 6.1 / 6.3
def test_backtest_job_end_to_end_with_the_control_plane():
    h, rid, req = _started()
    store = InMemoryArtifactStore()
    code, doc = _run(h, rid, req, artifacts=store)
    assert code == 0, doc
    assert doc["completion_status"] == "succeeded" and doc["solution_status"] in ("optimal", "feasible")
    assert validate(doc, "job-result").valid
    assert FixtureStatic.calls > 0
    ref = doc["artifacts"][0]
    assert ref["owner"] == "financemodel" and ref["checksum"].startswith("sha256:")
    assert store.exists(ref)
    assert set(doc["payload"]) >= {"performance", "accuracy", "compute_cost"}
    assert doc["payload"]["compute_cost"]["estimated_usd"] == pytest.approx(0.0725)
    assert doc["evaluator_version"] == "1.0.0" and doc["dataset_checksum"].startswith("sha256:")
    # SageMaker reports Completed: the control plane takes solution_status from the job's file
    name = h.run(rid)["job_name"]
    h.event(name, "InProgress")
    h.event(name, "Completed")
    result = h.service.get_job_result(Principal.from_arn(READER), rid)
    assert result["solution_status"] == doc["solution_status"] and result["artifacts"] == doc["artifacts"]
    assert validate(result, "tools/get-experiment-result-response").valid
    assert find_storage_location(result) is None and "example-research-bucket" not in json.dumps(result)


def test_benchmark_job_runs_strategy_and_controls_in_one_job():
    for c in ("cash", "buy_and_hold", "equal_weight"):
        register_strategy(c, lambda p, c=c: FixtureCash(c) if c == "cash" else FixtureStatic(c))
    try:
        h, rid, req = _started("run_benchmark")
        store = InMemoryArtifactStore()
        code, doc = _run(h, rid, req, artifacts=store)
        assert code == 0, doc
        bench = store.get_json(doc["artifacts"][0])
        assert bench["strategies"] == ["fixture_static", "cash", "buy_and_hold", "equal_weight"]
        assert len({r["identity"]["simulation_configuration_id"] for r in bench["results"]}) == 1
        assert validate(doc, "job-result").valid
    finally:
        for c in ("cash", "buy_and_hold", "equal_weight"):
            unregister_strategy(c)


def test_prepare_dataset_job_stores_a_dataset_reference():
    h, rid, req = _started("prepare_dataset")
    store = InMemoryArtifactStore()
    code, doc = _run(h, rid, req, artifacts=store)
    assert code == 0, doc
    assert doc["solution_status"] == "not_applicable" and doc["artifacts"][0]["kind"] == "research_dataset"
    desc = store.get_json(doc["artifacts"][0])
    assert desc["synthetic"] is True and desc["input_snapshot_id"] == SID
    assert FixtureStatic.calls == 0


# ------------------------------------------------------------------ 6.2 WS-04 / WS-03
def test_corrupted_snapshot_fails_before_strategy_code():
    h, rid, req = _started()
    platform = platform_with_snapshot()
    payload_id = next(r["artifact_id"] for r in platform.snapshots[SID]["artifacts"] if r["kind"] == "snapshot_payload")
    platform.corrupt(payload_id, b'{"observations": []}')
    store = InMemoryArtifactStore()
    code, doc = _run(h, rid, req, platform=platform, artifacts=store)
    assert code == 1
    assert doc["completion_status"] == "failed" and "solution_status" not in doc
    assert doc["error"]["code"] == "PRECONDITION_FAILED" and doc["error"]["details"]["reason"] == "snapshot_checksum_mismatch"
    assert doc["error"]["correlation_id"] == "corr-job-test-0001"
    assert FixtureStatic.calls == 0 and store.objects == {}
    assert validate(doc, "job-result").valid
    # the control plane keeps the job's own error code
    name = h.run(rid)["job_name"]
    h.event(name, "Failed")
    run = h.run(rid)
    assert run["state"] == "failed" and run["error"]["code"] == "PRECONDITION_FAILED"
    assert run["result"]["artifacts_complete"] is False


def test_job_resolves_snapshots_only_through_the_platform_by_id():
    h, rid, req = _started()
    platform = platform_with_snapshot()
    code, _ = _run(h, rid, req, platform=platform)
    assert code == 0
    assert platform.calls[0] == ("get_snapshot", SID)
    assert all(kind in ("get_snapshot", "download") for kind, _ in platform.calls)


def test_unapproved_snapshot_fails_the_job():
    h, rid, req = _started()
    code, doc = _run(h, rid, req, platform=platform_with_snapshot(status="committed"))
    assert code == 1 and doc["error"]["details"]["reason"] == "snapshot_not_approved"
    assert FixtureStatic.calls == 0


def test_tampered_run_spec_is_refused():
    h, rid, req = _started()
    code, doc = _run(h, rid, req, spec_checksum="sha256:" + "0" * 64)
    assert code == 1 and doc["error"]["details"]["reason"] == "run_spec_checksum_mismatch"


def test_job_type_must_match_the_spec():
    h, rid, req = _started()
    code, doc = run_job("prepare_dataset", rid, run_io=h.run_io, platform=platform_with_snapshot(), artifacts=InMemoryArtifactStore(), spec_checksum=req["Environment"]["FINPLAN_RUN_SPEC_SHA256"], clock=h.clock)
    assert code == 1 and doc["error"]["code"] == "VALIDATION_FAILED"


def test_uncovered_universe_fails_with_precondition():
    h = Harness(auto_approve=10.0)
    body = request()
    body["configuration"]["payload"].update(strategy="fixture_static", universe=["SPY", "QQQ"])
    _, resp = h.service.submit_job(Principal.from_arn("arn:aws:sts::<account-id>:assumed-role/finplan-beta-x-role/s"), body, correlation_id="corr-job-test-0002")
    h.service.dispatch()
    (req,) = h.created_jobs()
    code, doc = _run(h, resp["run_id"], req)
    assert code == 1 and doc["error"]["details"]["reason"] == "universe_not_covered"


# ------------------------------------------------------------------ local container run (no docker)
def _local_setup(tmp_path: Path, job_type: str, payload: dict | None = None) -> tuple[str, Path]:
    h, rid, req = _started(job_type)
    spec = h.run_io.get_spec(rid, None)
    LocalRunIO(tmp_path).put_spec(rid, spec)
    pf = tmp_path / "payload.json"
    pf.write_text(json.dumps(payload or synthetic_payload()), encoding="utf-8")
    return rid, pf


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT)])
    return env


def test_local_container_run_of_prepare_dataset(tmp_path):
    rid, pf = _local_setup(tmp_path, "prepare_dataset")
    out = subprocess.run([sys.executable, "-m", "finplan_model.jobs", "prepare_dataset", "--local", str(tmp_path), "--run-id", rid, "--fixture-snapshot", str(pf)], cwd=tmp_path, env=_env(), capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    doc = json.loads((tmp_path / "runs" / rid / "job-result.json").read_text())
    assert validate(doc, "job-result").valid and doc["completion_status"] == "succeeded"
    assert any((tmp_path / "artifacts").rglob("*"))
    assert rid in out.stdout  # JSON log lines carry run_id and correlation_id


def test_local_container_run_of_backtest(tmp_path):
    rid, pf = _local_setup(tmp_path, "run_backtest")
    code = (
        "import sys\n"
        "from finplan_model.jobs.strategy_resolver import register_strategy\n"
        "from tests.unit.jobs.support import FixtureStatic\n"
        "register_strategy('fixture_static', lambda p: FixtureStatic())\n"
        "from finplan_model.jobs.entrypoint import main\n"
        f"sys.exit(main(['run_backtest', '--local', {str(tmp_path)!r}, '--run-id', {rid!r}, '--fixture-snapshot', {str(pf)!r}]))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=_env(), capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr + out.stdout
    doc = json.loads((tmp_path / "runs" / rid / "job-result.json").read_text())
    assert validate(doc, "job-result").valid and doc["solution_status"] in ("optimal", "feasible")


def test_local_run_refuses_non_synthetic_fixtures(tmp_path):
    payload = synthetic_payload()
    payload["synthetic"] = False
    rid, pf = _local_setup(tmp_path, "prepare_dataset", payload)
    out = subprocess.run([sys.executable, "-m", "finplan_model.jobs", "prepare_dataset", "--local", str(tmp_path), "--run-id", rid, "--fixture-snapshot", str(pf)], cwd=tmp_path, env=_env(), capture_output=True, text=True, timeout=120)
    assert out.returncode == 2
    assert not (tmp_path / "runs" / rid / "job-result.json").exists()


def test_failed_local_run_exits_non_zero_with_failed_result(tmp_path):
    rid, pf = _local_setup(tmp_path, "prepare_dataset")
    out = subprocess.run([sys.executable, "-m", "finplan_model.jobs", "prepare_dataset", "--local", str(tmp_path), "--run-id", rid, "--fixture-snapshot", str(pf), "--spec-checksum", "sha256:" + "1" * 64], cwd=tmp_path, env=_env(), capture_output=True, text=True, timeout=120)
    assert out.returncode == 1
    doc = json.loads((tmp_path / "runs" / rid / "job-result.json").read_text())
    assert doc["completion_status"] == "failed" and validate(doc, "job-result").valid


# ------------------------------------------------------------------ with the real baseline strategies (task group 4)
@pytest.mark.parametrize("strategy, extra", [("equal_weight", {}), ("mean_variance", {"risk_aversion": 3.0, "lookback_days": 20}), ("min_variance", {"lookback_days": 20})])
def test_backtest_with_registered_baseline_strategies(strategy, extra):
    pytest.importorskip("finplan_model.strategies.registry")
    h = Harness(auto_approve=10.0)
    body = request()
    body["configuration"]["payload"].update(strategy=strategy, **extra)
    _, resp = h.service.submit_job(Principal.from_arn("arn:aws:sts::<account-id>:assumed-role/finplan-beta-x-role/s"), body, correlation_id="corr-job-test-0003")
    h.service.dispatch()
    (req,) = h.created_jobs()
    spec = h.run_io.get_spec(resp["run_id"], None)
    assert spec["strategy_params"] == {k: v for k, v in {"risk_aversion": extra.get("risk_aversion"), "lookback": extra.get("lookback_days")}.items() if v is not None and (k != "risk_aversion" or strategy == "mean_variance")}
    code, doc = _run(h, resp["run_id"], req)
    assert code == 0, doc
    assert doc["solution_status"] in ("optimal", "feasible", "no_effect", "not_applicable", "infeasible")
    assert validate(doc, "job-result").valid


def test_benchmark_with_real_controls():
    pytest.importorskip("finplan_model.strategies.registry")
    h, rid, req = _started("run_benchmark", strategy="mean_variance")
    store = InMemoryArtifactStore()
    code, doc = _run(h, rid, req, artifacts=store)
    assert code == 0, doc
    assert store.get_json(doc["artifacts"][0])["strategies"] == ["mean_variance", "cash", "buy_and_hold", "equal_weight"]
