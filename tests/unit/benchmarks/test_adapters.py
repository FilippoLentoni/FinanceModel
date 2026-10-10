import io
import json
import urllib.error
from datetime import date

import pytest

from finplan_model.benchmarks.jev import JevClient, JevStrategy, calibration_report, probabilities
from finplan_model.benchmarks.qwen import MODEL_ID, REVISION, QwenSwarm, verify_weights
from finplan_model.core.artifacts import canonical_json_bytes, sha256_checksum
from finplan_model.core.context import RunContext
from finplan_model.core.errors import FinplanError
from finplan_model.evaluate import evaluate
from finplan_model.sim.config import SimulationConfig
from finplan_model.sim.market import HoldingsView, synthetic_market


def answer(p=None):
    p = p or {"buy": .7, "hold": .2, "sell": .1}
    return {"type": "choice", "choice": max(p, key=p.get), "confidence": .8, "probabilities": p}


def payload():
    return {"model": "jev-latest", "state": "positive trend, moderate volatility", "questions": {"SPY": {"type": "choice", "criteria": {"buy": "increase exposure", "hold": "keep exposure", "sell": "reduce exposure"}}}}


def response(request, model="jev-1.13.0"):
    return {"model": model, "answers": {i: answer() for i in request["questions"]}, "usage": {"input_tokens": 100, "output_tokens": 20}}


def test_jev_request_shape_caching_egress_and_model_drift():
    seen = []
    client = JevClient("not-a-real-key", transport=lambda r: seen.append(r) or response(r), sleep=lambda _: None)
    ctx = RunContext.for_tests()
    a = client.ask(payload(), ctx=ctx)
    assert client.ask(payload(), ctx=ctx) == a and len(seen) == 1
    assert client.usage["returned_models"] == ["jev-1.13.0"]
    changed = payload(); changed["state"] = "negative trend"
    client.transport = lambda r: response(r, model="another-returned-version")
    client.ask(changed, ctx=ctx)
    assert client.usage["model_drift"]
    bad = payload(); bad["state"] = "raw prices: 100, 101, 102, 103, 104"
    with pytest.raises(FinplanError):
        client.ask(bad, ctx=ctx, raw_values=[100., 101., 102., 103., 104.])
    assert client.calls == 2


@pytest.mark.parametrize("status,calls", [(401, 1), (422, 1), (429, 3), (529, 3)])
def test_jev_faults_are_bounded_without_key_or_error_body(status, calls):
    def fault(_):
        raise urllib.error.HTTPError("https://api.typesafe.ai/v1/systemone", status, "private provider message", {"Retry-After": "1000"}, io.BytesIO(b"private payload"))
    waits = []
    client = JevClient("planted-key-never-logged", transport=fault, sleep=waits.append)
    with pytest.raises(FinplanError) as caught:
        client.ask(payload(), ctx=RunContext.for_tests())
    assert client.calls == calls and max(waits or [0]) <= 2
    assert "planted-key" not in str(caught.value) and "private provider" not in str(caught.value)


def test_jev_probability_validation_rejects_contradictory_choice_and_nan():
    bad = answer(); bad["choice"] = "sell"
    with pytest.raises(FinplanError): probabilities(bad)
    with pytest.raises(FinplanError): probabilities(answer({"buy": float("nan"), "hold": .5, "sell": .5}))


def test_jev_common_simulator_has_costs_and_separate_calibration():
    market = synthetic_market(n_sessions=100, instruments=("SPY", "AGG"))
    client = JevClient("fixture-key", transport=response, sleep=lambda _: None)
    strategy = JevStrategy(client, ctx=RunContext.for_tests())
    cfg = SimulationConfig.from_dict({"rebalance_frequency": "monthly", "fees": {"proportional_bps": 10.}})
    result = evaluate(strategy, market, cfg, start=market.sessions[21], end=market.sessions[-1])
    assert result.metrics["total_fees"] > 0
    assert client.calls <= 6 and strategy.records
    report = calibration_report(strategy.records, market)
    assert report["status"] == "available" and report["samples"] > 0
    assert report["vendor_calibration_claim_verified"] is False


def test_real_five_asset_descriptor_payload_passes_guard_without_raw_series():
    market = synthetic_market(n_sessions=60, instruments=("AAPL", "GOOGL", "NVDA", "NFLX", "VOO"))
    seen = []
    client = JevClient("fixture-key", transport=lambda r: seen.append(r) or response(r), sleep=lambda _: None)
    raw = [v for i in market.instruments for b in market.bars_of(i) for v in (b.open, b.high, b.low, b.close, b.volume)]
    strategy = JevStrategy(client, ctx=RunContext.for_tests(), raw_values=raw)
    strategy.decide(market.view(market.sessions[-1]), HoldingsView({}, 10000., {}, 10000.))
    assert len(seen) == 1 and set(seen[0]["state"]["instruments"]) == set(market.instruments)
    for descriptor in seen[0]["state"]["instruments"].values():
        assert all(isinstance(value, str) for value in descriptor.values())


def test_buy_signals_cannot_fund_themselves_by_selling_a_hold_position():
    market = synthetic_market(n_sessions=60, instruments=("AAPL", "GOOGL", "NVDA", "NFLX", "VOO"))
    def predictions(request):
        result = response(request)
        result["answers"]["AAPL"] = answer({"buy": .1, "hold": .8, "sell": .1})
        return result
    strategy = JevStrategy(JevClient("fixture", transport=predictions, sleep=lambda _: None), ctx=RunContext.for_tests())
    target = strategy.decide(market.view(market.sessions[-1]), HoldingsView({"AAPL": 100.}, 0., {"AAPL": 100.}, 10000.))
    assert target.weights["AAPL"] == 1. and all(target.weights[i] == 0 for i in market.instruments if i != "AAPL")


def test_exact_qwen_manifest_verification_fails_on_identity_or_corruption(tmp_path):
    files = {"config.json": canonical_json_bytes({"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]}), "tokenizer_config.json": b"{}", "LICENSE": b"Apache License", "model.safetensors": b"synthetic unit fixture only"}
    for name, data in files.items(): (tmp_path / name).write_bytes(data)
    manifest = "".join(sha256_checksum(data).split(":")[1] + "  " + name + "\n" for name, data in sorted(files.items())).encode()
    (tmp_path / "MANIFEST.sha256").write_bytes(manifest)
    metadata = {"model_id": MODEL_ID, "revision": REVISION, "license": "Apache-2.0", "manifest_checksum": sha256_checksum(manifest), "file_count": len(files)}
    (tmp_path / "STAGED.json").write_bytes(canonical_json_bytes(metadata))
    assert verify_weights(tmp_path)["revision"] == REVISION
    (tmp_path / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(FinplanError): verify_weights(tmp_path)


def test_swarm_fixed_roles_arbitration_and_invalid_hold():
    market = synthetic_market(n_sessions=40, instruments=("SPY", "AGG"))
    view = market.view(market.sessions[-1])
    holdings = HoldingsView({}, 10000., {}, 10000.)
    def model(messages):
        role = messages[0]["content"]
        if "market_analyst" in role or "risk_analyst" in role: return '{"summary":"fixture hypothesis"}'
        if "allocator" in role: return '{"weights":{"AGG":0.4,"SPY":0.4},"cash":0.2}'
        if "critic" in role: return '{"revise":false,"reason":"consistent"}'
        return '{"choice":"original"}'
    swarm = QwenSwarm(model, run_id="fixture-run")
    allocation = swarm.decide(view, holdings)
    assert allocation.weights == {"AGG": .4, "SPY": .4}
    assert [r["role"] for r in swarm.records] == ["market_analyst", "risk_analyst", "allocator", "critic", "arbiter"]
    assert swarm.evidence["historical_pretraining_leakage_risk"]
    invalid = QwenSwarm(lambda _: "not JSON", run_id="fixture-run")
    result = invalid.decide(view, holdings)
    assert result.solution_status == "no_effect" and result.cash == 1.
