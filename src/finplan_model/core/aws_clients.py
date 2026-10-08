"""AWS client construction shared by the Lambda entry points, the job container and the scripts
(mirrors FinancialPlanning ``finplan_platform/core/aws_clients.py``, platform lesson L4).

S3 clients must sign with SigV4 against the regional endpoint. A default ``boto3.client("s3")``
presigns with SigV2 on the global ``s3.amazonaws.com`` host, which S3 rejects for KMS-encrypted
objects and in SigV4-only regions; the platform's first beta run failed its download grants with
HTTP 403 that way. FinanceModel does not presign today, but every S3 client goes through
:func:`s3_client` so a later presigned URL (or a KMS-encrypted object) cannot regress. A unit test
refuses any other way of building an S3 client in ``src/``, ``scripts/`` and the deployed suites.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = ["s3_client", "s3_endpoint"]


def s3_endpoint(region: str) -> str:
    """The regional S3 endpoint (``https://s3.<region>.amazonaws.com``)."""
    if not region:
        raise RuntimeError("S3 clients need the bucket's region")
    return f"https://s3.{region}.amazonaws.com"


def s3_client(region: str | None = None, *, session: Any = None) -> Any:
    """An S3 client that signs (and presigns) SigV4 on the regional virtual-hosted endpoint.

    ``region`` defaults to the session's region, then ``AWS_REGION`` / ``AWS_DEFAULT_REGION``;
    ``session`` defaults to a new :class:`boto3.session.Session` (its credential chain: the Lambda,
    SageMaker or CodeBuild role, or the instance role)."""
    import boto3
    from botocore.config import Config

    region = region or (getattr(session, "region_name", None) if session is not None else None) or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        raise RuntimeError("AWS_REGION is not set; S3 clients need the bucket's region")
    session = session or boto3.session.Session(region_name=region)
    return session.client(
        "s3",
        region_name=region,
        endpoint_url=s3_endpoint(region),
        config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
    )
