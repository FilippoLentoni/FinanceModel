"""DS-13 report scenario (task 3.9): results and reports derived from real data are written only to
research storage, carry aggregate metrics only and never land in the source repository."""

from __future__ import annotations

from pathlib import Path

import pytest

from finplan_model.core.artifacts import InMemoryArtifactStore, LocalArtifactStore, S3ArtifactStore
from finplan_model.core.errors import FinplanError
from finplan_model.reporting import BenchmarkReport, assert_aggregate_only, inside_source_repository, store_report, store_research_artifact

ROOT = Path(__file__).resolve().parents[3]


def _report(synthetic: bool) -> BenchmarkReport:
    content = {"report_version": "benchmark-report-v1", "synthetic": synthetic, "sections": {"portfolio_performance": {"periods": {"holdout": [{"strategy": "cash", "metrics": {"net_cumulative_return": 0.0}}]}}}}
    return BenchmarkReport(content, "sha256:" + "0" * 64)


def test_real_data_report_is_refused_inside_the_repository():
    target = ROOT / "build-output" / "reports"
    assert inside_source_repository(target)
    with pytest.raises(FinplanError) as ei:
        store_report(_report(synthetic=False), LocalArtifactStore(target))
    assert ei.value.code == "OPERATION_NOT_PERMITTED"
    assert not target.exists()  # nothing was written
    with pytest.raises(FinplanError):
        store_research_artifact({"nav": []}, LocalArtifactStore(target), kind="run_artifact", synthetic=False)


def test_real_data_report_goes_to_research_storage(tmp_path):
    assert not inside_source_repository(tmp_path)
    refs = store_report(_report(synthetic=False), LocalArtifactStore(tmp_path / "research"))
    assert refs["report_ref"]["kind"] == "benchmark_report" and "synthetic" not in refs["report_ref"]
    store_report(_report(synthetic=False), InMemoryArtifactStore())


def test_real_data_report_to_s3_research_storage():
    import boto3
    from moto import mock_aws

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-2")
        s3.create_bucket(Bucket="example-research-bucket", CreateBucketConfiguration={"LocationConstraint": "us-east-2"})
        refs = store_report(_report(synthetic=False), S3ArtifactStore(s3, "example-research-bucket"))
        assert "bucket" not in str(refs) and "s3://" not in str(refs)


@pytest.mark.parametrize("doc", [{"nav": [1, 2]}, {"rows": [{"prices": {}}]}, {"x": list(range(20))}, {"fills": []}])
def test_reports_never_embed_series(doc):
    with pytest.raises(FinplanError) as ei:
        assert_aggregate_only(doc)
    assert ei.value.details["reason"] == "report_embeds_series"
