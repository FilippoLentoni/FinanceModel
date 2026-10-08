"""S3 clients sign SigV4 on the regional endpoint (platform lesson L4; mirrors FinancialPlanning
``tests/unit/test_aws_clients.py``).

The platform's first beta run failed its download grants with HTTP 403: a default
``boto3.client("s3")`` presigned SigV2 URLs on the global ``s3.amazonaws.com`` host. FinanceModel
builds every S3 client through :func:`finplan_model.core.aws_clients.s3_client`.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import boto3
import pytest

from finplan_model.core.aws_clients import s3_client, s3_endpoint

ROOT = Path(__file__).resolve().parents[2]
#: Code that runs against AWS: the package (Lambdas, job container), scripts, deployed suites.
GUARDED = ("src", "scripts", "tests/integration", "tests/smoke", "tests/deployed.py", "infra")
_DEFAULT_S3 = re.compile(r"""(boto3|session|\w+)\.client\(\s*["']s3["']""")


def test_presigned_get_is_sigv4_on_the_regional_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_REGION", "us-east-2")
    url = s3_client().generate_presigned_url("get_object", Params={"Bucket": "example-bucket", "Key": "runs/k.json"}, ExpiresIn=60)
    parts = urlsplit(url)
    assert parts.netloc == "example-bucket.s3.us-east-2.amazonaws.com"
    query = parse_qs(parts.query)
    assert query["X-Amz-Algorithm"] == ["AWS4-HMAC-SHA256"]
    assert "/us-east-2/s3/aws4_request" in query["X-Amz-Credential"][0]
    assert "Signature" not in query and "AWSAccessKeyId" not in query


def test_presigned_post_is_sigv4(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_REGION", "us-east-2")
    post = s3_client().generate_presigned_post(Bucket="example-bucket", Key="scratch/x.json", ExpiresIn=60)
    assert post["url"].startswith("https://example-bucket.s3.us-east-2.amazonaws.com")
    assert post["fields"]["x-amz-algorithm"] == "AWS4-HMAC-SHA256"


def test_session_region_wins_and_region_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    session = boto3.session.Session(region_name="us-east-2")
    assert s3_client(session=session).meta.endpoint_url == s3_endpoint("us-east-2") == "https://s3.us-east-2.amazonaws.com"
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    with pytest.raises(RuntimeError):
        s3_client()


def test_no_code_builds_a_default_s3_client() -> None:
    offenders = []
    for base in GUARDED:
        path = ROOT / base
        files = [path] if path.is_file() else sorted(path.rglob("*.py"))
        for p in files:
            if p.name == "aws_clients.py" or "__pycache__" in p.parts:
                continue
            if _DEFAULT_S3.search(p.read_text(encoding="utf-8")):
                offenders.append(str(p.relative_to(ROOT)))
    assert offenders == [], f"use finplan_model.core.aws_clients.s3_client(): {offenders}"
