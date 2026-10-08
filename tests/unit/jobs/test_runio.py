"""Run hand-off IO (run spec and job result) on S3 (moto), locally and in memory."""

from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from finplan_model.core.artifacts import canonical_json_bytes, sha256_checksum
from finplan_model.core.errors import FinplanError
from finplan_model.jobs.runio import InMemoryRunIO, LocalRunIO, S3RunIO

RID = "run_01KDVDNAZ83BAMMYCEGWF33DPM"
SPEC = {"spec_version": "run-spec-v1", "run_id": RID}


def _roundtrip(io) -> None:
    ck = io.put_spec(RID, SPEC)
    assert ck == sha256_checksum(canonical_json_bytes(SPEC))
    assert io.get_spec(RID, ck) == SPEC
    with pytest.raises(FinplanError) as ei:
        io.get_spec(RID, "sha256:" + "0" * 64)
    assert ei.value.details["reason"] == "run_spec_checksum_mismatch"
    assert io.get_result(RID) is None
    io.put_result(RID, {"run_id": RID, "completion_status": "failed"})
    assert io.get_result(RID)["completion_status"] == "failed"
    with pytest.raises(FinplanError) as ei:
        io.get_spec("run_01KES9T7J0GRHJS1P0A0T03EHX", None)
    assert ei.value.details["reason"] == "run_spec_missing"


def test_in_memory():
    _roundtrip(InMemoryRunIO())


def test_local(tmp_path):
    _roundtrip(LocalRunIO(tmp_path))
    assert (tmp_path / "runs" / RID / "spec.json").exists()
    with pytest.raises(FinplanError):
        LocalRunIO(tmp_path).put_spec("../escape", SPEC)


def test_s3():
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-2")
        s3.create_bucket(Bucket="example-research-bucket", CreateBucketConfiguration={"LocationConstraint": "us-east-2"})
        _roundtrip(S3RunIO(s3, "example-research-bucket"))
        keys = [o["Key"] for o in s3.list_objects_v2(Bucket="example-research-bucket")["Contents"]]
        assert sorted(keys) == [f"runs/{RID}/job-result.json", f"runs/{RID}/spec.json"]
