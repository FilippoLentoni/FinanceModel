import copy

import boto3
import pytest
from moto import mock_aws

from finplan_model.classical.storage import S3Store, analysis_id, public
from finplan_model.core.artifacts import InMemoryArtifactStore
from finplan_model.core.clock import FrozenClock
from finplan_model.core.context import RunContext
from finplan_model.core.errors import FinplanError
from finplan_model.jobs.handlers import JobInputs, run_benchmark
from finplan_model.jobs.market_loader import load_market


def test_s3_write_once_checksum_and_newest_index(context):
    a = context.service.recommend({})
    doc = context.service.store.get(a["analysis_id"])
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="example-bucket")
        store = S3Store(s3, "example-bucket")
        assert store.put(doc) == doc and store.put(doc) == doc
        assert public(store.get(a["analysis_id"])) == a
        newer = copy.deepcopy(doc)
        newer["analysis_id"] = analysis_id({"new": 1})
        newer["created_at"] = "2026-10-11T00:00:00Z"
        store.put(newer)
        assert store.list(limit=1)[0]["analysis_id"] == newer["analysis_id"]
        assert (
            store.list(kind="recommendation", portfolio_id=context.pid, limit=2)[1][
                "analysis_id"
            ]
            == a["analysis_id"]
        )
        corrupt = copy.deepcopy(doc)
        corrupt["request_fingerprint"] = "changed"
        with pytest.raises(FinplanError, match="conflicts"):
            store.put(corrupt)
        s3.put_object(
            Bucket="example-bucket",
            Key="classical/records/" + a["analysis_id"] + ".json",
            Body=b"{}",
            Metadata={"checksum": "sha256:" + "a" * 64},
        )
        with pytest.raises(FinplanError, match="checksum"):
            store.get(a["analysis_id"])


def test_weekly_sandbox_runs_finite_grid_controls_and_chronological_selection(context):
    market, content = load_market(context.platform, context.sid)
    ctx = RunContext(
        run_id="run_01KDVDP88REHGPBXFX6CHX92KS",
        environment="beta",
        clock=FrozenClock("2026-10-10T00:00:00Z"),
        synthetic=True,
    )
    spec = {
        "run_id": ctx.run_id,
        "configuration_id": "cfg_" + "a" * 64,
        "input_snapshot_id": context.sid,
        "configuration": {"payload": {"objective": "classical_weekly_review"}},
        "strategy": "min_variance",
        "universe": list(market.instruments),
        "simulation": {
            "rebalance_frequency": "weekly",
            "constraints": {"long_only": True, "max_weight": 0.6},
            "fees": {"proportional_bps": 2},
        },
        "evaluation_window": {
            "start": context.dates[60].isoformat(),
            "end": context.dates[-1].isoformat(),
        },
        "purpose": "research",
    }
    result = run_benchmark(
        JobInputs(ctx, spec, market, content, InMemoryArtifactStore())
    )
    evidence = result["payload"]["weekly_research"]
    assert (
        result["completion_status"] == "succeeded" and len(evidence["variants"]) == 12
    )
    assert (
        evidence["selection"]["split"] == "validation"
        and evidence["activation"] == "proposal_only"
    )
    assert (
        evidence["windows"]["validation"]["end"]
        < evidence["windows"]["research_test"]["start"]
    )
    assert evidence["common_warmup"]["required_completed_returns"] == 120
    assert evidence["windows"]["validation"]["start"] == context.dates[120].isoformat()
    assert {v["algorithm"] for v in evidence["variants"]} == {
        "cash",
        "buy_and_hold",
        "equal_weight",
        "min_variance",
        "mean_variance",
        "scenario_cvar",
    }


def test_expanded_lookback_grid_requires_sufficient_common_warmup(context):
    from types import SimpleNamespace

    from finplan_model.classical.benchmark import run_weekly_benchmark

    with pytest.raises(FinplanError, match="common maximum-lookback"):
        run_weekly_benchmark(
            SimpleNamespace(
                market=context.market,
                universe=context.market.instruments,
                window=(None, None),
                spec={"configuration": {"payload": {"lookback_days": 120}}},
            )
        )
