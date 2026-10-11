"""Pilot scope and budget bounds use actual sessions before loading/sending to models."""

from dataclasses import replace
from datetime import timedelta

import pytest

from finplan_model.benchmarks.jobs import jev_backtest, swarm_mode_a
from finplan_model.benchmarks.jev import JevClient
from finplan_model.benchmarks.protocol import daily_scope, pilot_window
from finplan_model.core.artifacts import InMemoryArtifactStore
from finplan_model.core.context import RunContext
from finplan_model.core.errors import FinplanError
from finplan_model.core.platform import FixturePlatformClient, build_synthetic_snapshot
from finplan_model.jobs.handlers import JobInputs
from finplan_model.jobs.market_loader import load_market
from finplan_model.sim.market import MarketData, synthetic_market
from tests.unit.jobs.support import SID, synthetic_payload


def inputs():
    platform = FixturePlatformClient()
    record, blobs = build_synthetic_snapshot(synthetic_payload(n_sessions=100), input_snapshot_id=SID)
    platform.add_snapshot(record, blobs)
    market, snapshot = load_market(platform, SID)
    ctx = RunContext.for_tests()
    window, _ = pilot_window(market, market.instruments, as_of=market.sessions[-1])
    spec = {"run_id": "run_01KDVDP88REHGPBXFX6CHX92KS", "configuration_id": "cfg_" + "a" * 64,
        "input_snapshot_id": SID, "purpose": "research", "evaluation_window": window,
        "universe": list(market.instruments), "simulation": {"rebalance_frequency": "daily", "fees": {"proportional_bps": 2.}}}
    return JobInputs(ctx, spec, market, snapshot, InMemoryArtifactStore())


def backend(messages):
    role = messages[0]["content"]
    if "market_analyst" in role or "risk_analyst" in role:
        return '{"summary":"synthetic fixture"}'
    if "allocator" in role:
        return '{"weights":{"AGG":0.4,"SPY":0.4},"cash":0.2}'
    if "critic" in role:
        return '{"revise":false,"reason":"consistent"}'
    return '{"choice":"original"}'


@pytest.mark.parametrize("family", ["qwen", "jev"])
def test_handlers_preserve_warmup_and_compare_exactly_21_daily_decisions(family):
    inp = inputs()
    if family == "qwen":
        result = swarm_mode_a(inp, backend=backend)
    else:
        def transport(request):
            return {"model": "jev-1.13.0", "answers": {i: {"type": "choice", "choice": "buy", "confidence": .8,
                "probabilities": {"buy": .7, "hold": .2, "sell": .1}} for i in request["questions"]},
                "usage": {"input_tokens": 100, "output_tokens": 20}}
        result = jev_backtest(inp, client=JevClient("synthetic-fixture", transport=transport, sleep=lambda _: None))
    evidence = inp.artifacts.get_json(result["artifacts"][0])
    scope = result["payload"]["llm_benchmark"]["evaluation_scope"]
    assert scope["decision_count"] == 21 and scope["market_sessions"] == 22
    assert scope["history_prices_at_first_decision"] == 79 and len(inp.market.sessions) == 100
    assert scope["full_2026_benchmark"] is False and scope["untouched_holdout_comparison"] is False
    for evaluation in [evidence["evaluation"], *evidence["controls"]]:
        sim = evaluation["simulation"]
        assert sim["sessions"]["count"] == 22
        assert len(sim["decisions"]) == 21
        assert evaluation["evaluation_window"] == inp.spec["evaluation_window"]
        assert evaluation["identity"]["settings"]["rebalance_frequency"] == "daily"
    if family == "qwen":
        assert evidence["decisions"] == 21 and len(evidence["messages"]) == 105
    else:
        assert len(evidence["responses"]) == 21
        assert evidence["decision_horizon_sessions"] == 21  # Forecast horizon differs from daily decisions.
        assert evidence["calibration"]["samples"] == 2


@pytest.mark.parametrize("handler", [swarm_mode_a, jev_backtest])
def test_over_cap_is_rejected_before_any_backend_or_provider_access(handler, monkeypatch):
    inp = inputs()
    inp.spec["evaluation_window"]["start"] = inp.market.sessions[-34].isoformat()
    loaded = []
    monkeypatch.setattr("finplan_model.benchmarks.jobs.VllmBackend", lambda _: loaded.append(True))
    with pytest.raises(FinplanError) as caught:
        handler(inp)
    assert caught.value.details["max_decisions"] == 32
    assert caught.value.details["requested_decisions"] == 33 and loaded == []


def test_actual_calendar_not_calendar_day_arithmetic_and_alignment_fails_closed():
    original = synthetic_market(("SPY", "AGG"), n_sessions=100)
    holiday = original.sessions[-11]
    sessions = [s for s in original.sessions if s != holiday]
    bars = [b for i in original.instruments for b in original.bars_of(i) if b.session_date != holiday]
    market = MarketData(sessions, bars)
    window, scope = pilot_window(market, market.instruments, as_of=sessions[-1])
    assert window["start"] == sessions[-22].isoformat() and scope["decision_count"] == 21
    for change in ("missing", "late"):
        changed = [b for b in bars if not (change == "missing" and b.instrument_id == "SPY" and b.session_date == sessions[-10])]
        if change == "late":
            changed = [replace(b, available_at=b.available_at + timedelta(days=1)) if b.instrument_id == "SPY" and b.session_date == sessions[-10] else b for b in changed]
        with pytest.raises(FinplanError):
            pilot_window(MarketData(sessions, changed), market.instruments, as_of=sessions[-1])


def test_missing_warmup_or_window_never_reaches_inference():
    market = synthetic_market(n_sessions=22)
    with pytest.raises(FinplanError):
        pilot_window(market, market.instruments, as_of=market.sessions[-1])
    with pytest.raises(FinplanError):
        daily_scope(market, market.instruments, None, None, frequency="daily")
