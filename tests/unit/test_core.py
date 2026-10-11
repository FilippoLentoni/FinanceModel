"""Shared core interfaces: errors, clock, IDs, run context, artifacts, configuration."""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from finplan_model.core.artifacts import ArtifactRef, InMemoryArtifactStore, LocalArtifactStore, S3ArtifactStore, canonical_json_bytes
from finplan_model.core.clock import FrozenClock, parse_utc, utc_iso
from finplan_model.core.config import ConfigError, is_gpu_instance_type, load_all, load_config, validate_config
from finplan_model.core.context import RunContext
from finplan_model.core.errors import ErrorCode, FinplanError, as_finplan_error, registered_codes
from finplan_model.core.ids import IdMinter, RUN_ID_RE, configuration_id, require_id
from finplan_model.sim.config import SimulationConfig

ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ errors
def test_error_codes_come_from_the_contract():
    assert set(registered_codes()) == {c.value for c in ErrorCode}


def test_fixed_retryable_cannot_be_overridden():
    assert FinplanError(ErrorCode.RATE_LIMITED, "slow down").retryable is True
    assert FinplanError(ErrorCode.DEPENDENCY_UNAVAILABLE, "x", retryable=False).retryable is False
    with pytest.raises(ValueError):
        FinplanError(ErrorCode.BUDGET_EXCEEDED, "x", retryable=True)
    with pytest.raises(ValueError):
        FinplanError("NOT_A_CODE", "x")


def test_contract_required_details():
    assert FinplanError.validation("bad").details["pointer"] == ""
    with pytest.raises(ValueError):
        FinplanError(ErrorCode.INVALID_IDENTIFIER, "x")
    with pytest.raises(ValueError):
        FinplanError(ErrorCode.UNSUPPORTED_CONTRACT_VERSION, "x")
    env = FinplanError(ErrorCode.UNSUPPORTED_CONTRACT_VERSION, "unsupported", details={"served_contract_majors": [0]}).to_envelope("corr-12345678")
    assert env["details"]["served_contract_majors"] == [0]


def test_unexpected_exception_maps_to_internal_without_traceback():
    err = as_finplan_error(KeyError("s3" + "://" + "private-research-data/key"))
    env = err.to_envelope("corr-12345678")
    assert env["code"] == "INTERNAL"
    assert "s3://" not in json.dumps(env)


def test_leaking_envelope_is_replaced():
    env = FinplanError.validation("see s3://example-bucket/x", pointer="/a").to_envelope("corr-12345678")
    assert env["code"] == "INTERNAL"
    assert env["details"]["original_code"] == "VALIDATION_FAILED"


# ------------------------------------------------------------------ clock / ids / context
def test_frozen_clock_and_timestamps():
    c = FrozenClock("2026-01-05T00:00:00Z")
    assert utc_iso(c.now()) == "2026-01-05T00:00:00Z"
    c.advance(hours=1)
    assert utc_iso(c.now()) == "2026-01-05T01:00:00Z"
    assert parse_utc("2026-01-05T01:00:00Z") == c.now()
    t = FrozenClock("2026-01-05T00:00:00Z", tick=timedelta(seconds=1))
    assert (t.now(), t.now()) == (parse_utc("2026-01-05T00:00:00Z"), parse_utc("2026-01-05T00:00:01Z"))


def test_seeded_ids_are_deterministic_and_contract_shaped():
    a = IdMinter.seeded(FrozenClock(), 5)
    b = IdMinter.seeded(FrozenClock(), 5)
    ra, rb = a.run_id(), b.run_id()
    assert ra == rb and RUN_ID_RE.match(ra)
    assert a.model_version().startswith("mv_")
    assert require_id("run_id", ra) == ra
    with pytest.raises(FinplanError) as ei:
        require_id("run_id", "mv_" + ra[4:])
    assert ei.value.code == "INVALID_IDENTIFIER" and ei.value.details["field"] == "run_id"


def test_configuration_id_is_order_independent():
    assert configuration_id({"a": 1, "b": [1, 2]}) == configuration_id({"b": [1, 2], "a": 1})
    assert configuration_id({"a": 1}).startswith("cfg_")


def test_run_context_logs_with_correlation_id(caplog):
    ctx = RunContext.for_tests(seed=3, environment="beta", purpose="research")
    with caplog.at_level(logging.INFO, logger="finplan_model"):
        rec = ctx.with_run(ctx.ids.run_id()).log("probe", x=1)
    assert rec["correlation_id"] == ctx.correlation_id
    assert ctx.correlation_id in caplog.text
    with pytest.raises(ValueError):
        RunContext.for_tests(purpose="live")


# ------------------------------------------------------------------ artifacts
@pytest.mark.parametrize("store_kind", ["memory", "local"])
def test_artifact_store_roundtrip_and_integrity(store_kind, tmp_path):
    store = InMemoryArtifactStore() if store_kind == "memory" else LocalArtifactStore(tmp_path)
    ref = store.put_json({"b": 2, "a": 1}, kind="run_artifact", synthetic=True)
    again = store.put(canonical_json_bytes({"a": 1, "b": 2}), kind="run_artifact", synthetic=True)
    assert ref == again  # content-addressed, idempotent
    assert store.get_json(ref.to_dict()) == {"a": 1, "b": 2}
    d = ref.to_dict()
    assert d["owner"] == "financemodel" and "bucket" not in json.dumps(d)
    bad = ArtifactRef(**{**ref.__dict__, "checksum": "sha256:" + "0" * 64})
    with pytest.raises(FinplanError) as ei:
        store.get(bad)
    assert ei.value.code == "PRECONDITION_FAILED"


def test_s3_artifact_store_with_moto():
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-2")
        s3.create_bucket(Bucket="example-research", CreateBucketConfiguration={"LocationConstraint": "us-east-2"})
        store = S3ArtifactStore(s3, "example-research", "research/")
        ref = store.put_json({"x": 1}, kind="research_dataset")
        assert store.exists(ref)
        assert store.get_json(ref) == {"x": 1}
        assert store.put_json({"x": 1}, kind="research_dataset") == ref
        with pytest.raises(FinplanError) as ei:
            store.put(b"different", kind="research_dataset", artifact_id=ref.artifact_id)
        assert ei.value.code == "IMMUTABLE_RECORD"
        missing = ArtifactRef(artifact_id="nothing_here", kind="research_dataset", checksum=ref.checksum, content_type="application/json")
        assert not store.exists(missing)
        with pytest.raises(FinplanError) as ei:
            store.get(missing)
        assert ei.value.code == "NOT_FOUND"


# ------------------------------------------------------------------ configuration
def test_config_files_validate_and_expose_ssm_names():
    cfgs = load_all()
    beta = cfgs["beta"]
    assert beta.ssm["job_endpoint"] == "/finplan/beta/financemodel/api/job-endpoint"
    assert beta.ssm["run_staging_ref"] == "/finplan/beta/financialplanning/config/run-staging-ref"
    assert beta.ssm["budget_allocation"] == "/finplan/shared/financialplanning/config/budget-allocation"
    assert beta.job_type("run_benchmark").budget_category == "cpu_research"
    assert {name for name, jt in beta.job_types.items() if jt.compute_class == "gpu"} == {"swarm_mode_a"}
    assert beta.job_type("swarm_mode_a").max_runtime_seconds == 900
    assert beta.served_contract_majors == (1,)
    assert SimulationConfig.from_dict(beta.simulation_defaults) == SimulationConfig()


def test_config_rejects_gpu_instance_for_cpu_job_and_bad_values():
    raw = json.loads((ROOT / "config" / "beta.json").read_text(encoding="utf-8"))
    raw["job_types"]["run_backtest"]["instance_types"].append("ml.g5.xlarge")
    raw["idempotency"]["retention_days"] = 3
    raw["simulation_defaults"]["venue"] = "coinbase"
    problems = validate_config("beta", raw)
    assert any("GPU instance type" in p for p in problems)
    assert any("retention_days" in p for p in problems)
    assert any("simulation_defaults" in p for p in problems)
    assert is_gpu_instance_type("ml.g6.12xlarge") and not is_gpu_instance_type("ml.m5.xlarge")
    with pytest.raises(ConfigError):
        load_config("dev")
