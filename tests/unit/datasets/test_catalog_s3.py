"""DS-01 in research storage: the S3 dataset catalog creates each record once and verifies it."""

from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from finplan_model.core.artifacts import S3ArtifactStore
from finplan_model.core.errors import FinplanError
from finplan_model.datasets import S3DatasetCatalog, prepare_dataset

from .support import Env, config

BUCKET = "example-research-bucket"


def test_s3_catalog_reuses_the_first_record_and_detects_tampering():
    env = Env()
    sid = env.backfill()
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-2")
        s3.create_bucket(Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "us-east-2"})
        store, catalog = S3ArtifactStore(s3, BUCKET), S3DatasetCatalog(s3, BUCKET, prefix="datasets")
        first = prepare_dataset([sid], config(), reader=env.provider.reader, store=store, catalog=catalog, ctx=env.ctx)
        env.ctx.clock.advance(hours=1)
        second = prepare_dataset([sid], config(), reader=env.provider.reader, store=store, catalog=S3DatasetCatalog(s3, BUCKET, prefix="datasets"), ctx=env.ctx)
        assert first.reused is False and second.reused is True and second.record == first.record
        rec, created = catalog.create(first.record["dataset_key"], {**first.record, "created_at": "2030-01-01T00:00:00Z"})
        assert created is False and rec == first.record  # first writer wins
        key = f"datasets/dataset-catalog/{first.record['dataset_key']}.json"
        body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read().replace(b'"synthetic":true', b'"synthetic":false', 1)
        s3.put_object(Bucket=BUCKET, Key=key, Body=body)
        with pytest.raises(FinplanError) as ei:
            catalog.get(first.record["dataset_key"])
        assert ei.value.details["reason"] == "dataset_record_checksum_mismatch"
        assert second.load(store).market_data().dataset_checksum == first.manifest_checksum
