"""Operator parameters the bootstrap may default (task 10.6; design D4): instance prices come from
the AWS Price List API at bootstrap time, never from the repository; a current operator value is kept."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import boto3
import pytest
from moto import mock_aws

from finplan_model.control import costs
from finplan_model.core.config import load_config
from scripts import bootstrap as fm_bootstrap
from scripts.instance_prices import PriceUnavailable, decide, ensure_instance_prices, fetch_prices, instance_types, parameter_name
from tests.unit.infra.test_build_and_bootstrap import SYNTHETIC_PRICE, _Pricing

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)


@pytest.fixture
def ssm():
    with mock_aws():
        yield boto3.client("ssm", region_name="us-east-2")


def test_instance_types_cover_every_deployed_job_type():
    assert instance_types(load_config("beta")) == ["ml.m5.xlarge"]


def test_fetch_uses_the_processing_component_in_the_deployment_region():
    pricing = _Pricing()
    doc = fetch_prices(pricing, "us-east-2", ["ml.m5.xlarge"], now=NOW)
    assert doc["usd_per_hour"] == {"ml.m5.xlarge": SYNTHETIC_PRICE}  # the Hosting line is ignored
    (call,) = pricing.calls
    filters = {f["Field"]: f["Value"] for f in call["Filters"]}
    assert call["ServiceCode"] == "AmazonSageMaker" and filters == {"regionCode": "us-east-2", "instanceName": "ml.m5.xlarge", "component": "Processing"}
    assert doc["retrieved_at"] == "2026-10-08T12:00:00Z"


def test_fetched_document_is_accepted_by_the_pre_flight_estimate():
    cfg = load_config("beta")
    doc = fetch_prices(_Pricing(), cfg.region, instance_types(cfg), now=NOW)
    est = costs.estimate(cfg, cfg.job_type("run_backtest"), instance_type="ml.m5.xlarge", instance_count=1, max_runtime_seconds=1800, prices=doc, now=NOW)
    assert est.usd_per_hour == SYNTHETIC_PRICE and est.estimated_usd_upper_bound > 0


def test_missing_price_writes_nothing(ssm):
    with pytest.raises(PriceUnavailable):
        ensure_instance_prices(ssm, _Pricing(sagemaker_price=None), envs=["beta"], now=NOW, out=lambda s: None)
    assert ssm.describe_parameters()["Parameters"] == []


def test_absent_or_stale_prices_are_written_current_ones_kept(ssm):
    cfg = load_config("beta")
    current = json.dumps({"retrieved_at": "2026-10-01T00:00:00Z", "currency": "USD", "usd_per_hour": {"ml.m5.xlarge": 0.3}})
    ssm.put_parameter(Name=parameter_name("beta"), Value=current, Type="String")
    stale = json.dumps({"retrieved_at": (NOW - timedelta(days=31)).strftime("%Y-%m-%dT%H:%M:%SZ"), "usd_per_hour": {"ml.m5.xlarge": 0.3}})
    ssm.put_parameter(Name=parameter_name("gamma"), Value=stale, Type="String")
    out: list[str] = []
    decisions = {d.env: d for d in ensure_instance_prices(ssm, _Pricing(), now=NOW, out=out.append)}
    assert decisions["beta"].action == "keep" and decisions["gamma"].reason.startswith("older than") and decisions["prod"].reason == "absent"
    assert ssm.get_parameter(Name=parameter_name("beta"))["Parameter"]["Value"] == current  # operator value untouched
    for env in ("gamma", "prod"):
        assert json.loads(ssm.get_parameter(Name=parameter_name(env))["Parameter"]["Value"])["usd_per_hour"] == {"ml.m5.xlarge": SYNTHETIC_PRICE}
    assert decide(cfg, current, now=NOW, force=True).action == "write"
    assert decide(cfg, json.dumps({"retrieved_at": "2026-10-07T00:00:00Z", "usd_per_hour": {}}), now=NOW).reason == "no price for ml.m5.xlarge"
    assert decide(cfg, "not json", now=NOW).reason == "unreadable"


def test_dry_run_writes_nothing(ssm):
    out: list[str] = []
    ensure_instance_prices(ssm, _Pricing(), envs=["beta"], write=False, now=NOW, out=out.append)
    assert ssm.describe_parameters()["Parameters"] == [] and out[0].startswith("[DRY-RUN]")


def test_bootstrap_writes_the_shared_role_names_idempotently(ssm):
    out: list[str] = []
    first = fm_bootstrap.publish_tooling_role_names(ssm, out=out.append)
    second = fm_bootstrap.publish_tooling_role_names(ssm, out=out.append)
    assert first == second and out[0].startswith("[WROTE]") and out[1].startswith("[OK]")
    status = fm_bootstrap.report_operator_parameters(ssm, out=out.append)
    assert status == {"/finplan/beta/financemodel/config/integration-snapshot-id": False, "/finplan/gamma/financemodel/config/integration-snapshot-id": False}
    ssm.put_parameter(Name="/finplan/beta/financemodel/config/integration-snapshot-id", Value="snap_01KDVDNAZ83BAMMYCEGWF33DPM", Type="String")
    assert fm_bootstrap.report_operator_parameters(ssm, out=out.append)["/finplan/beta/financemodel/config/integration-snapshot-id"] is True
