"""Change add-daily-recommendation-and-on-demand-experiments: DRJ-04 (daily job stages six weights
and the snapshot's disclosures), UNV-01 (universe snapshots in backtest/benchmark: adj_close basis,
cash assumption) and UNV-02 (mandatory bias section, fail closed, deterministic report checksum).
All data is synthetic."""

from __future__ import annotations

import copy
import json

import pytest
from finplan_contracts.validate import validate

from finplan_model.control.auth import Principal
from finplan_model.core.artifacts import InMemoryArtifactStore
from finplan_model.core.platform import FixturePlatformClient, build_synthetic_snapshot
from finplan_model.jobs.entrypoint import run_job
from finplan_model.jobs.market_loader import market_from_content
from finplan_model.jobs.universe import BIAS_SECTION_TITLE, adjusted_observation, full_allocation
from finplan_model.reporting import build_report
from finplan_model.staging import MANIFEST_NAME, InMemoryStagingStore
from finplan_model.sim.market import synthetic_market
from tests.unit.control.support import Harness, arn, request
from tests.unit.reporting.support import benchmark

SID = "snap_01KDVDP88REHGPBXFX6CHX92KS"
TICKERS = ("VOO", "GOOGL", "NFLX", "AAPL", "NVDA")
DISCLOSURES = [
    {"kind": "hindsight_selection", "text": "Synthetic disclosure: the instruments were selected with hindsight."},
    {"kind": "survivorship", "text": "Synthetic disclosure: failed or delisted companies are absent."},
]
TRIGGER = arn("finplan-beta-financialplanning-daily-trigger-step-role")
SUBMITTER = arn("finplan-beta-financelambdastool-submitter-role")
PLAN_ID = "pl_01KDVDNAZ83BAMMYCEGWF33DPM"


def universe_payload(disclosures=DISCLOSURES, n_sessions: int = 80) -> dict:
    market = synthetic_market(TICKERS, n_sessions=n_sessions, seed=5)
    obs = []
    for iid in market.instruments:
        for b in market.bars_of(iid):
            obs.append({"instrument_id": iid, "session_date": b.session_date.isoformat(), "kind": "completed_daily", "open": b.open, "high": b.high, "low": b.low, "close": b.close, "adj_close": round(b.close * 0.98, 6), "volume": int(b.volume or 0), "synthetic": True})
    payload = {
        "dataset_id": "finance/equity-etf-daily/research-universe",
        "calendar": "XNYS",
        "instruments": [{"instrument_id": i, "asset_class": "equity", "kind": "equity", "currency": "USD", "synthetic": True} for i in TICKERS]
        + [{"instrument_id": "USD_CASH", "asset_class": "cash", "kind": "cash", "return_assumption": "zero_nominal", "currency": "USD", "synthetic": True}],
        "observations": obs,
        "universe": {"instruments": [{"instrument_id": i, "kind": "equity"} for i in TICKERS] + [{"instrument_id": "USD_CASH", "kind": "cash", "return_assumption": "zero_nominal"}], "history_start": "2010-10-01", "return_basis": "adj_close"},
        "synthetic": True,
    }
    if disclosures:
        payload["bias_disclosures"] = copy.deepcopy(disclosures)
    return payload


def universe_platform(disclosures=DISCLOSURES) -> FixturePlatformClient:
    p = FixturePlatformClient()
    rec, blobs = build_synthetic_snapshot(universe_payload(disclosures), input_snapshot_id=SID)
    p.add_snapshot(rec, blobs)
    return p


def _selected(h: Harness, strategy: str = "equal_weight") -> None:
    run = {"run_id": "run_01KDVDNBYGX5V5HY2JSK5XWKHC", "environment": "beta", "state": "succeeded", "job_type": "run_benchmark", "purpose": "research", "dataset_id": "finance/equity-etf-daily/research-universe", "configuration": {"payload": {"strategy": strategy}}, "submitted_at": "2026-01-04T09:00:00Z", "completed_at": "2026-01-04T09:20:00Z", "revision": 3, "compute_class": "cpu"}
    h.store.create_run(run, [{"run_id": run["run_id"], "seq": 0, "from_state": None, "state": "succeeded", "at": run["submitted_at"]}])
    h.settings.strategy_raw = json.dumps({"strategy_id": strategy, "environment": "beta", "selected_at": "2026-01-04T10:00:00Z", "selected_by": "finplan-beta-financialplanning-operator-pipeline-stage"})


def _submit(h: Harness, principal: str, body: dict) -> tuple[str, dict]:
    code, resp = h.service.submit_job(Principal.from_arn(principal), body, correlation_id="corr-test-universe-01")
    assert code == 202, resp
    h.service.dispatch()
    req = h.created_jobs()[-1]
    return resp["run_id"], req


def _run(h: Harness, rid: str, req: dict, platform, **kw):
    env = req["Environment"]
    return run_job(req["AppSpecification"]["ContainerArguments"][0], rid, run_io=h.run_io, platform=platform, artifacts=kw.pop("artifacts", InMemoryArtifactStore()), environment="beta", spec_checksum=env["FINPLAN_RUN_SPEC_SHA256"], correlation_id=env["FINPLAN_CORRELATION_ID"], image_digest=env["FINPLAN_IMAGE_DIGEST"], clock=h.clock, **kw)


def _universe_request(job_type: str, strategy: str = "equal_weight", **over) -> dict:
    body = request(job_type=job_type, input_snapshot_id=SID, evaluation_window=None, **over)
    body["configuration"]["payload"].update(strategy=strategy, universe=list(TICKERS))
    return body


# ===================================================================== DRJ-04
def test_drj04_daily_job_stages_six_weights_lineage_and_disclosures():
    platform = universe_platform()
    h = Harness(platform=platform)
    h.deps.model_version_resolver = lambda strategy, digest: "mv_01KDVDNAZ83BAMMYCEGWF33DPM"  # the registry hook in deployments
    _selected(h)
    body = _universe_request("daily_recommendation", purpose="production_candidate", plan_id=PLAN_ID, idempotency_key="daily-beta-2026-01-05")
    body["configuration"]["payload"]["objective"] = "daily_recommendation"
    rid, req = _submit(h, TRIGGER, body)
    run = h.store.get_run(rid)
    assert run["dataset_id"] == "finance/equity-etf-daily/research-universe" and run["bias_disclosures"] == DISCLOSURES
    staging = InMemoryStagingStore()
    code, doc = _run(h, rid, req, platform, staging_store=staging)
    assert code == 0, doc
    assert doc["staging"]["status"] == "staged", doc["staging"]
    manifest = json.loads(staging.bundle(rid)[MANIFEST_NAME])
    assert validate(manifest, "staged-output-manifest").valid
    assert manifest["bias_disclosures"] == DISCLOSURES and manifest["plan_id"] == PLAN_ID
    alloc = manifest["payload"]["plan_content"]["allocation"]
    assert [w["instrument_id"] for w in alloc["weights"]] == list(TICKERS)
    assert len(alloc["weights"]) + 1 == 6 and sum(w["weight"] for w in alloc["weights"]) + alloc["cash_weight"] == pytest.approx(1.0, abs=1e-9)
    assert manifest["configuration_id"] == run["production_strategy"]["configuration_id"]
    assert manifest["model_version"] == run["production_strategy"]["model_version"] == "mv_01KDVDNAZ83BAMMYCEGWF33DPM"
    assert doc["payload"]["bias_section"]["title"] == BIAS_SECTION_TITLE
    assert validate(doc, "job-result").valid


def test_drj_daily_submission_refuses_a_non_universe_snapshot():
    from tests.unit.jobs.support import platform_with_snapshot

    h = Harness(platform=platform_with_snapshot())
    _selected(h)
    body = request(job_type="daily_recommendation", purpose="production_candidate", plan_id=PLAN_ID, evaluation_window=None, idempotency_key="daily-beta-2026-01-07")
    body["configuration"]["payload"]["strategy"] = "equal_weight"
    with pytest.raises(Exception) as ei:
        h.service.submit_job(Principal.from_arn(TRIGGER), body, correlation_id="corr-test-universe-02")
    assert ei.value.code == "VALIDATION_FAILED" and ei.value.details["pointer"] == "/input_snapshot_id"


def test_full_allocation_includes_zero_weights_and_sums_to_one():
    a = full_allocation({"VOO": 0.5, "NVDA": 0.3}, 0.2, list(TICKERS))
    assert [w["instrument_id"] for w in a["weights"]] == list(TICKERS)
    assert sum(w["weight"] for w in a["weights"]) + a["cash_weight"] == pytest.approx(1.0)
    assert full_allocation({}, 0.0, ["VOO"])["cash_weight"] == 1.0


# ===================================================================== UNV-01
def test_unv01_adj_close_basis_and_cash_assumption():
    obs = {"instrument_id": "VOO", "session_date": "2026-01-05", "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "adj_close": 50.5}
    adj = adjusted_observation(obs)
    assert adj["close"] == 50.5 and adj["open"] == pytest.approx(50.0) and adj["high"] == pytest.approx(51.0)
    platform = universe_platform()
    from finplan_model.core.platform import SnapshotReader

    content = SnapshotReader(platform).load(SID)
    market = market_from_content(content)
    assert set(market.instruments) == set(TICKERS)
    first = next(o for o in content.payload["observations"] if o["instrument_id"] == "VOO")
    assert market.bars_of("VOO")[0].close == pytest.approx(first["adj_close"])


@pytest.mark.parametrize("job_type", ["run_backtest", "run_benchmark"])
def test_unv01_unv02_existing_kinds_accept_universe_with_bias_section(job_type):
    from finplan_model.jobs.strategy_resolver import register_strategy, unregister_strategy  # noqa: F401

    platform = universe_platform()
    h = Harness(platform=platform, auto_approve=10.0)
    rid, req = _submit(h, SUBMITTER, _universe_request(job_type))
    store = InMemoryArtifactStore()
    code, doc = _run(h, rid, req, platform, artifacts=store)
    assert code == 0, doc
    section = doc["payload"]["bias_section"]
    assert section["title"] == "Hindsight and survivorship bias" and section["bias_disclosures"] == DISCLOSURES
    assert section["cash_assumption"]["return_assumption"] == "zero_nominal" and section["return_basis"] == "adj_close"
    assert validate(doc, "job-result").valid
    art = store.get_json(doc["artifacts"][0])
    assert art["bias_section"] == section
    if job_type == "run_benchmark":
        assert art["strategies"][0] == "equal_weight" and len({r["identity"]["simulation_configuration_id"] for r in art["results"]}) == 1


def test_unv02_universe_snapshot_without_disclosures_fails():
    platform = universe_platform(disclosures=None)
    h = Harness(platform=platform, auto_approve=10.0)
    with pytest.raises(Exception) as ei:
        h.service.submit_job(Principal.from_arn(SUBMITTER), _universe_request("run_benchmark"), correlation_id="corr-test-universe-03")
    assert ei.value.code == "VALIDATION_FAILED" and ei.value.details["reason"] == "bias_disclosures_missing"
    # the job fails closed too (snapshot changed after submission)
    h2 = Harness(platform=universe_platform(), auto_approve=10.0)
    rid, req = _submit(h2, SUBMITTER, _universe_request("run_backtest"))
    code, doc = _run(h2, rid, req, platform)
    assert code == 1 and doc["completion_status"] == "failed" and doc["error"]["code"] == "VALIDATION_FAILED"


def test_unv02_report_bias_section_and_deterministic_checksum():
    bench = benchmark([{"name": "min_variance", "params": {"lookback": 40}}])
    section = {"title": BIAS_SECTION_TITLE, "bias_disclosures": DISCLOSURES, "cash_assumption": {"instrument_id": "USD_CASH", "return_assumption": "zero_nominal", "annual_rate": 0.0, "weight_allowed": True}, "return_basis": "adj_close"}
    a = build_report(bench["run"], cost_records=[bench["cost"]], holdout_access=bench["access"], bias_section=section)
    b = build_report(bench["run"], cost_records=[bench["cost"]], holdout_access=bench["access"], bias_section=copy.deepcopy(section))
    assert a.report_checksum == b.report_checksum
    assert a.content["bias"]["title"] == BIAS_SECTION_TITLE and a.content["bias"]["bias_disclosures"] == DISCLOSURES
    plain = build_report(bench["run"], cost_records=[bench["cost"]], holdout_access=bench["access"])
    assert plain.report_checksum != a.report_checksum and "bias" not in plain.content
    run = bench["run"].to_dict()
    run["dataset"] = {**run["dataset"], "dataset_id": "finance/equity-etf-daily/research-universe"}
    with pytest.raises(Exception) as ei:
        build_report(run, cost_records=[bench["cost"]], holdout_access=bench["access"])
    assert ei.value.code == "VALIDATION_FAILED"
