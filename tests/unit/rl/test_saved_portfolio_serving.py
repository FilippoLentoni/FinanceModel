import copy
import json
from datetime import UTC, date, datetime
from types import SimpleNamespace

import numpy as np
import pytest

from finplan_model import serving
from finplan_model.core.artifacts import InMemoryArtifactStore
from finplan_model.core.errors import FinplanError
from finplan_model.rl.inference import allocation_response
from finplan_model.rl.serving_context import DATASET_ID, raw_market, value_holdings
from finplan_model.sim.market import Bar, MarketData

UID = "01JABCDEFGHJKMNPQRSTVWXY01"
SID, PID, PLAN = "snap_" + UID, "pf_" + UID, "pl_" + UID


@pytest.fixture
def context(monkeypatch):
    days = [date(2026, 10, d) for d in (7, 8, 9)]
    prices = np.array([[100., 50.], [150., 60.], [110., 60.]])
    rows = [{"instrument_id": i, "session_date": d.isoformat(), "close": prices[t, k]}
            for t, d in enumerate(days) for k, i in enumerate(("A", "B"))]
    market = MarketData(days, [Bar(r["instrument_id"], date.fromisoformat(r["session_date"]), r["close"] / 10,
                                 datetime.fromisoformat(r["session_date"] + "T21:00:00+00:00")) for r in rows], synthetic=False)
    content = SimpleNamespace(payload={"observations": rows}, snapshot=SimpleNamespace(manifest_checksum="sha256:" + "b" * 64, record={}))
    book = {"portfolio_id": PID, "revision": 1, "synthetic": True, "contract_version": "1.3.0",
            "paper_state": {"positions": [{"instrument_id": "A", "quantity": 10.}, {"instrument_id": "B", "quantity": 20.}],
                            "cash_balance": 100., "high_watermark": 2300., "as_of": "2026-10-07",
                            "base_currency": "USD", "mode": "paper", "source": "paper_initialization"}}
    calls = []
    def get_plan(plan):
        calls.append(("plan", plan))
        return {"plan": {"plan_id": PLAN, "portfolio_id": PID}}
    def get_state(portfolio):
        calls.append(("state", portfolio))
        return copy.deepcopy(book)
    def latest(dataset):
        calls.append(("latest", dataset))
        return {"snapshot": {"input_snapshot_id": SID, "status": "approved"}}
    platform = SimpleNamespace(get_plan=get_plan, get_portfolio_state=get_state, get_latest_snapshot=latest)
    bundle = {"format": "finplan-strategy-bundle/1", "mode": "advisory_paper", "strategy_id": "ppo",
              "source_run_id": "run_" + UID, "configuration_id": "cfg_" + "a" * 64,
              "instruments": ["A", "B"], "available_after": "2026-01-01", "constraints": {},
              "members": [{"seed": 0}], "environment": {"window": 2}}
    artifacts = InMemoryArtifactStore()
    ref = artifacts.put_json(bundle, kind="policy_inference")
    deps = SimpleNamespace(platform=platform, research_plan_parameter=SimpleNamespace(read=lambda: PLAN), artifacts=artifacts,
                           advisory_parameter=SimpleNamespace(read=lambda: json.dumps({"export_run_id": "run_" + UID, "artifact": ref.to_dict()})))
    svc = SimpleNamespace(env="beta", d=deps, now=lambda: datetime(2026, 10, 10, tzinfo=UTC))
    captured = []
    monkeypatch.setattr("finplan_model.rl.advisory.load_market", lambda p, s: (market, content))
    def recommend(frozen, px, holdings):
        captured.append((px.copy(), copy.deepcopy(holdings)))
        return allocation_response(frozen, holdings, [.25, .25], .5)
    monkeypatch.setattr("finplan_model.rl.advisory.recommend", recommend)
    return SimpleNamespace(service=svc, book=book, market=market, content=content, calls=calls, captured=captured, days=days)


def request(ctx, body):
    return serving.handle({"environment": "beta", "request": body}, ctx.service)


def test_default_saved_book_is_marked_with_raw_closes_and_interim_peak(context):
    result = request(context, {})
    rec = result["recommendation"]
    state = rec["portfolio_state"]
    assert context.calls == [("plan", PLAN), ("state", PID), ("latest", DATASET_ID)]
    assert rec["as_of"] == "2026-10-09" and rec["input_snapshot_id"] == SID
    assert state["source"] == "saved_paper" and state["revision"] == 1
    assert state["portfolio_value"] == 2400 and state["high_watermark"] == 2800
    assert state["current_cash"] == 100 and result["synthetic"] is True
    px, holdings = context.captured[-1]
    np.testing.assert_allclose(px, [[10, 5], [15, 6], [11, 6]])
    assert holdings["weights"][0]["weight"] == pytest.approx(1100 / 2400)
    assert holdings["cash_weight"] == pytest.approx(100 / 2400)
    assert holdings["high_watermark"] == 2800
    a, b = rec["decisions"]
    assert a["action"] == b["action"] == "sell"
    assert a["reference_price"] == 110 and a["current_quantity"] == 10
    assert a["target_quantity"] == pytest.approx(600 / 110)
    assert a["delta_quantity"] * 110 == pytest.approx(a["indicative_notional"])
    assert b["delta_quantity"] == -10
    assert sum(p["target_quantity"] * p["reference_price"] for p in rec["decisions"]) + rec["cash_weight"] * state["portfolio_value"] == pytest.approx(2400)


def test_repeated_recommendations_do_not_change_state_or_selection(context):
    before = copy.deepcopy(context.book)
    pointer = context.service.d.advisory_parameter.read()
    assert request(context, {}) == request(context, {})
    assert context.book == before
    assert context.service.d.advisory_parameter.read() == pointer
    assert not hasattr(context.service.d, "sagemaker") and not hasattr(context.service.d, "run_io")


def test_explicit_holdings_keep_experiment_mode(context):
    rec = request(context, {"input_snapshot_id": SID, "as_of": "2026-10-09",
                            "holdings": {"weights": [], "cash_weight": 1., "portfolio_value": 10000., "high_watermark": 10000.}})["recommendation"]
    assert context.calls == []
    assert rec["portfolio_state"]["source"] == "supplied"
    assert rec["decisions"][0]["delta_quantity"] == pytest.approx(2500 / 110)


def test_explicit_saved_portfolio_skips_default_plan(context):
    request(context, {"portfolio_id": PID, "input_snapshot_id": SID, "as_of": "2026-10-09"})
    assert context.calls == [("state", PID)]


@pytest.mark.parametrize("body", [{"as_of": "2026-10-09"}, {"input_snapshot_id": SID},
                                     {"holdings": {}}, {"portfolio_id": PID, "holdings": {}}])
def test_ambiguous_or_partial_inputs_fail_before_reads(context, body):
    with pytest.raises(FinplanError):
        request(context, body)
    assert context.calls == []


@pytest.mark.parametrize("case", ["future", "state_after_decision", "unknown", "duplicate", "nan", "cash", "missing_history", "missing_close", "zero_value"])
def test_invalid_saved_context_cannot_produce_trades(context, case):
    body = {}
    if case == "future": body = {"input_snapshot_id": SID, "as_of": "2026-10-11"}
    book = context.book["paper_state"]
    if case == "state_after_decision": book["as_of"] = "2026-10-10"
    if case == "unknown": book["positions"][0]["instrument_id"] = "C"
    if case == "duplicate": book["positions"].append(copy.deepcopy(book["positions"][0]))
    if case == "nan": book["positions"][0]["quantity"] = float("nan")
    if case == "cash": book["cash_balance"] = -1
    if case == "missing_history": book["as_of"] = "2026-10-06"
    if case == "missing_close": context.content.payload["observations"].pop(2)
    if case == "zero_value": book.update(positions=[], cash_balance=0)
    with pytest.raises(FinplanError):
        request(context, body)
    assert context.captured == []


def test_missing_book_is_not_initialized_by_recommendation(context):
    def missing(_):
        raise FinplanError.precondition("paper state missing", reason="paper_state_missing")
    context.service.d.platform.get_portfolio_state = missing
    with pytest.raises(FinplanError) as error:
        request(context, {})
    assert error.value.details["reason"] == "paper_state_missing"


def test_unapproved_latest_snapshot_is_rejected(context):
    context.service.d.platform.get_latest_snapshot = lambda _: {"snapshot": {"status": "pending", "input_snapshot_id": SID}}
    with pytest.raises(FinplanError) as error:
        request(context, {})
    assert error.value.details["reason"] == "no_approved_snapshot"


def test_peak_does_not_collapse_when_terminal_value_recovers_only_partly(context):
    holdings, _ = value_holdings({}, context.book, raw_market(context.content, context.market), context.days[-1], ["A", "B"])
    assert 1 - holdings["portfolio_value"] / holdings["high_watermark"] == pytest.approx(1 - 2400 / 2800)
