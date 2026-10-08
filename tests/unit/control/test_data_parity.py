"""Data parity across stages (user decision 26): beta, gamma and prod accept real phase 2 platform
snapshots (no ``synthetic`` flag, lineage provider ``yfinance``) exactly alike, and prod keeps
accepting still-synthetic snapshots during the transition. Test-created requests stay synthetic.

The "real" snapshots here are built in memory with invented values (``synthetic`` false only to
model the platform's real-data record shape); nothing retrieved is committed (DS-13)."""

from __future__ import annotations

from typing import Any

import pytest

from finplan_model.control.auth import Principal
from finplan_model.core.config import load_config
from finplan_model.core.platform import FixturePlatformClient, build_synthetic_snapshot
from finplan_model.core.artifacts import InMemoryArtifactStore
from finplan_model.jobs.entrypoint import run_job
from tests.integration.job_suite import adapt, run_job_lifecycle
from tests.unit.jobs.support import synthetic_payload

from .support import SID, Harness, arn, env_config, request

ENVS = ("beta", "gamma", "prod")
REAL_LINEAGE = {"provider_library": "yfinance", "library_version": "0.2.66", "calendar_version": "xnys-exchange_calendars-4.13.2-20100101-20271231"}


def _strip(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k != "synthetic"}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def _platform(kind: str, instruments: tuple[str, ...] = ("AGG", "SPY")) -> FixturePlatformClient:
    payload = synthetic_payload(instruments)
    p = FixturePlatformClient()
    if kind == "real":
        payload = _strip(payload)
        sessions = sorted({o["session_date"] for o in payload["observations"]})
        calendar = {"exchange": "XNYS", "version": REAL_LINEAGE["calendar_version"], "synthetic": False, "sessions": sessions}
        rec, blobs = build_synthetic_snapshot(payload, input_snapshot_id=SID, provider="yfinance", lineage_extra=REAL_LINEAGE, calendar=calendar, synthetic=False)
        assert "synthetic" not in rec and rec["lineage"]["provider"] == "yfinance"
    else:
        rec, blobs = build_synthetic_snapshot(payload, input_snapshot_id=SID)
    p.add_snapshot(rec, blobs)
    return p


def test_instrument_configuration_is_identical_and_real_capable_in_every_stage():
    blocks = [{k: v for k, v in load_config(env).instrument.items() if k != "$comment"} for env in ENVS]
    assert blocks[0] == blocks[1] == blocks[2]
    assert blocks[0]["data_source"] == "platform_snapshots"


@pytest.mark.parametrize("kind", ["real", "synthetic"])
@pytest.mark.parametrize("env", ENVS)
def test_deployed_suite_lifecycle_accepts_real_and_synthetic_snapshots(env, kind):
    h = Harness(cfg=env_config(env), platform=_platform(kind, ("SPY",)))
    stage = arn(f"finplan-{env}-financemodel-pipeline-stage-role")
    call = adapt(lambda method, path, body=None: h.call(method, path, principal=stage, body=body))
    res = run_job_lifecycle(call, snapshot_id=SID, run_key=f"{env}-{kind}", sleep=lambda s: None)
    assert res.states[0] == "awaiting_approval" and res.states[-1] == "cancelled"
    assert h.created_jobs() == []
    assert h.run(res.run_id)["synthetic"] is True  # test-created records stay synthetic


@pytest.mark.parametrize("kind", ["real", "synthetic"])
@pytest.mark.parametrize("env", ENVS)
def test_backtest_job_runs_on_real_and_synthetic_snapshots(env, kind):
    platform = _platform(kind)
    h = Harness(cfg=env_config(env), platform=platform, auto_approve=10.0)
    body = request(synthetic=None)
    body["configuration"].pop("synthetic")
    code, resp = h.service.submit_job(Principal.from_arn(arn(f"finplan-{env}-financelambdastool-submitter-role")), body, correlation_id="corr-parity-0001")
    assert code == 202, resp
    assert h.run(resp["run_id"])["synthetic"] is (kind == "synthetic")
    h.service.dispatch()
    (req,) = h.created_jobs()
    e = req["Environment"]
    assert e["FINPLAN_ENVIRONMENT"] == env
    rc, doc = run_job("run_backtest", resp["run_id"], run_io=h.run_io, platform=platform, artifacts=InMemoryArtifactStore(), environment=env, spec_checksum=e["FINPLAN_RUN_SPEC_SHA256"], correlation_id=e["FINPLAN_CORRELATION_ID"], image_digest=e["FINPLAN_IMAGE_DIGEST"], clock=h.clock)
    assert rc == 0 and doc["completion_status"] == "succeeded", doc
    assert doc.get("synthetic") is (True if kind == "synthetic" else None)
