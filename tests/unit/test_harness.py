"""Task 1.3 / DEP-02: the test harness blocks SageMaker, real AWS and network calls."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from tests.harness import AwsCallBlocked, NetworkBlocked

ROOT = Path(__file__).resolve().parents[2]

PROCESSING_JOB = {
    "ProcessingJobName": "blocked-by-harness",
    "RoleArn": "arn:aws:iam::<account-id>:role/placeholder",
    "AppSpecification": {"ImageUri": "placeholder"},
    "ProcessingResources": {"ClusterConfig": {"InstanceCount": 1, "InstanceType": "ml.m5.xlarge", "VolumeSizeInGB": 1}},
}


def test_create_processing_job_is_blocked():
    sm = boto3.client("sagemaker", region_name="us-east-2")
    with pytest.raises(AwsCallBlocked, match="CreateProcessingJob"):
        sm.create_processing_job(**PROCESSING_JOB)


def test_sagemaker_is_blocked_even_under_moto():
    with mock_aws():
        sm = boto3.client("sagemaker", region_name="us-east-2")
        with pytest.raises(AwsCallBlocked):
            sm.list_processing_jobs()


def test_real_aws_http_is_blocked():
    with pytest.raises(AwsCallBlocked):
        boto3.client("sts", region_name="us-east-2").get_caller_identity()


def test_network_is_blocked():
    with pytest.raises(NetworkBlocked):
        socket.create_connection(("example.com", 443), timeout=1)
    with pytest.raises(NetworkBlocked):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect(("192.0.2.1", 443))
        finally:
            s.close()


def test_moto_still_works_for_other_services():
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-2")
        s3.create_bucket(Bucket="example-research-bucket", CreateBucketConfiguration={"LocationConstraint": "us-east-2"})
        assert [b["Name"] for b in s3.list_buckets()["Buckets"]] == ["example-research-bucket"]


def test_only_fake_credentials_are_visible():
    assert os.environ["AWS_ACCESS_KEY_ID"] == "testing"
    assert "AWS_PROFILE" not in os.environ


def test_build_fails_when_a_test_calls_sagemaker(tmp_path):
    """A planted build-stage test that calls SageMaker makes pytest (and so the build) fail."""
    planted = tmp_path / "test_planted.py"
    planted.write_text(
        "import boto3\n\n"
        "def test_starts_a_job():\n"
        "    boto3.client('sagemaker', region_name='us-east-2').create_processing_job(ProcessingJobName='x', RoleArn='r', AppSpecification={'ImageUri': 'i'}, ProcessingResources={'ClusterConfig': {'InstanceCount': 1, 'InstanceType': 'ml.m5.xlarge', 'VolumeSizeInGB': 1}})\n",
        encoding="utf-8",
    )
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "src")])}
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "tests.harness", "-c", str(tmp_path / "pytest.ini"), str(planted)], cwd=ROOT, env=env, capture_output=True, text=True, check=False, timeout=120)
    assert proc.returncode != 0
    assert "AwsCallBlocked" in proc.stdout + proc.stderr
