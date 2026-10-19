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
    proposals = []
    def persist(portfolio, proposal):
        proposals.append(copy.deepcopy(proposal))
        return {"decision": {"portfolio_id": portfolio, "decision_id": "pd_" + UID}}
    def get_portfolio(portfolio):
        calls.append(("portfolio",portfolio))
        return {"portfolio_id":portfolio,"synthetic":True,"settings":{}}
    platform = SimpleNamespace(get_portfolio=get_portfolio,get_plan=get_plan, get_portfolio_state=get_state, get_latest_snapshot=latest, create_portfolio_decision=persist)
    bundle = {"format": "finplan-strategy-bundle/1", "mode": "advisory_paper", "strategy_id": "ppo",
              "source_run_id": "run_" + UID, "configuration_id": "cfg_" + "a" * 64,
              "instruments": ["A", "B"], "available_after": "2026-01-01", "constraints": {},
              "members": [{"seed": 0}], "environment": {"window": 2}}
    artifacts = InMemoryArtifactStore()
    ref = artifacts.put_json(bundle, kind="policy_inference")
    deps = SimpleNamespace(capture_served_input=lambda payload:artifacts.put_json(payload,kind="served_model_input").to_dict(), platform=platform, research_plan_parameter=SimpleNamespace(read=lambda: PLAN), artifacts=artifacts,
                           advisory_parameter=SimpleNamespace(read=lambda: json.dumps({"export_run_id": "run_" + UID, "artifact": ref.to_dict()})))
    svc = SimpleNamespace(env="beta", d=deps, now=lambda: datetime(2026, 10, 10, tzinfo=UTC))
    captured = []
    monkeypatch.setattr("finplan_model.rl.advisory.load_market", lambda p, s: (market, content))
    def recommend(frozen, px, holdings, *, input_capture=None):
        if input_capture:input_capture({"boundary":"controlled_fixed_policy_fixture","prices":px.tolist(),"holdings":holdings})
        captured.append((px.copy(), copy.deepcopy(holdings)))
        return allocation_response(frozen, holdings, [.25, .25], .5)
    monkeypatch.setattr("finplan_model.rl.advisory.recommend", recommend)
    return SimpleNamespace(service=svc, book=book, market=market, content=content, calls=calls, captured=captured, days=days, proposals=proposals)


def request(ctx, body):
    return serving.handle({"environment": "beta", "request": {"instruments": ["A", "B"], **body}}, ctx.service)


def test_default_saved_book_is_marked_with_raw_closes_and_interim_peak(context):
    result = request(context, {})
    rec = result["recommendation"]
    state = rec["portfolio_state"]
    assert context.calls == [("plan", PLAN), ("state", PID), ("latest", DATASET_ID), ("portfolio", PID)]
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
    assert len(context.proposals) == 2
    assert context.proposals[0] == context.proposals[1]
    assert context.proposals[0]["portfolio_revision"] == before["revision"]
    assert context.proposals[0]["execution"]["transaction_cost_bps"] == 2.


def test_policy_is_selected_by_requested_universe_not_by_portfolio_routing(context):
    parameter=context.service.d.advisory_parameter
    pointer=json.loads(parameter.read())
    original=json.loads(context.service.d.artifacts.get(pointer['artifact']))
    # A selected multi-seed ensemble on the same universe outranks the pinned single seed,
    # whichever portfolio asks (same rule as the forward runner's direct ppo book).
    ensemble={**original,'source_run_id':'run_01JABCDEFGHJKMNPQRSTVWXY02','members':[{'seed':0},{'seed':1}]}
    other={**original,'source_run_id':'run_01JABCDEFGHJKMNPQRSTVWXY04','instruments':['A','B','C']}
    ref=context.service.d.artifacts.put_json(ensemble,kind='policy_inference').to_dict()
    other_ref=context.service.d.artifacts.put_json(other,kind='policy_inference').to_dict()
    context.service.d.policy_exports=lambda:[(ref,ensemble),(other_ref,other)]
    saved=request(context,{})['recommendation']
    explicit=request(context,{'input_snapshot_id':SID,'as_of':'2026-10-09',
                              'holdings':{'weights':[],'cash_weight':1.,'portfolio_value':10000.,'high_watermark':10000.}})['recommendation']
    for rec in (saved,explicit):
        assert rec['policy_source_run_id']==ensemble['source_run_id']
        assert rec['policy_artifact_checksum']==ref['checksum']
        assert rec['export_run_id']==ensemble['source_run_id']
        assert rec['diagnostics']['policy_selection']['policy_instruments']==['A','B']
        assert [w['instrument_id'] for w in rec['target_weights']]==['A','B']
    assert parameter.read()==json.dumps(pointer)


def test_pinned_export_run_identifies_the_selected_pinned_policy(context):
    rec=request(context,{'instruments':['B','A']})['recommendation']
    pointer=json.loads(context.service.d.advisory_parameter.read())
    assert rec['export_run_id']==pointer['export_run_id'] and rec['policy_artifact_checksum']==pointer['artifact']['checksum']


@pytest.mark.parametrize('instruments',[['A'],['A','B','C'],['A','C']])
def test_no_policy_for_the_requested_universe_is_a_typed_error(context,instruments):
    with pytest.raises(FinplanError) as exc:
        request(context,{'instruments':instruments})
    assert exc.value.code=='PRECONDITION_FAILED' and exc.value.details['reason']=='no_policy_for_universe'
    assert exc.value.details['requested_instruments']==sorted(instruments)
    assert not context.captured and not context.proposals and context.calls==[]


def test_news_or_other_algorithm_exports_are_never_served_as_ppo(context):
    pointer=json.loads(context.service.d.advisory_parameter.read())
    original=json.loads(context.service.d.artifacts.get(pointer['artifact']))
    news={**original,'members':[{'seed':0},{'seed':1}],'environment':{'window':2,'features':['etf_news_metadata_v1']}}
    sac={**original,'strategy_id':'sac','members':[{'seed':0},{'seed':1}]}
    refs=[(context.service.d.artifacts.put_json(b,kind='policy_inference').to_dict(),b) for b in (news,sac)]
    context.service.d.policy_exports=lambda:refs
    rec=request(context,{})['recommendation']
    assert rec['policy_artifact_checksum']==pointer['artifact']['checksum']


def test_holdings_outside_the_requested_instruments_are_rejected(context):
    with pytest.raises(FinplanError) as exc:
        request(context,{'input_snapshot_id':SID,'as_of':'2026-10-09','holdings':{'weights':[{'instrument_id':'C','weight':0.}],
                 'cash_weight':1.,'portfolio_value':10000.,'high_watermark':10000.}})
    assert exc.value.code=='VALIDATION_FAILED' and exc.value.details['outside_instruments']==['C']


def test_requested_instrument_missing_from_snapshot_is_a_validation_error(context):
    pointer=json.loads(context.service.d.advisory_parameter.read())
    wider={**json.loads(context.service.d.artifacts.get(pointer['artifact'])),'instruments':['A','B','C']}
    ref=context.service.d.artifacts.put_json(wider,kind='policy_inference').to_dict()
    context.service.d.policy_exports=lambda:[(ref,wider)]
    with pytest.raises(FinplanError) as exc:
        request(context,{'instruments':['A','B','C']})
    assert exc.value.code=='VALIDATION_FAILED' and exc.value.details['missing_instruments']==['C']
    assert not context.captured


def test_top_level_instruments_must_match_the_request(context):
    body={'instruments':['A','B']}
    serving.handle({'environment':'beta','request':body,'instruments':['A','B']},context.service)
    with pytest.raises(FinplanError) as exc:
        serving.handle({'environment':'beta','request':body,'instruments':['A']},context.service)
    assert exc.value.code=='VALIDATION_FAILED'


def test_missing_instruments_is_a_validation_error(context):
    with pytest.raises(FinplanError) as exc:
        serving.handle({'environment':'beta','request':{}},context.service)
    assert exc.value.code=='VALIDATION_FAILED'


def test_saved_recommendation_fails_closed_if_decision_cannot_be_persisted(context):
    def unavailable(*_):
        raise FinplanError.dependency_unavailable("decision storage unavailable")
    context.service.d.platform.create_portfolio_decision = unavailable
    with pytest.raises(FinplanError):
        request(context, {})
    assert context.book["revision"] == 1


def test_explicit_holdings_keep_experiment_mode(context):
    rec = request(context, {"input_snapshot_id": SID, "as_of": "2026-10-09",
                            "holdings": {"weights": [], "cash_weight": 1., "portfolio_value": 10000., "high_watermark": 10000.}})["recommendation"]
    assert context.calls == []
    assert context.proposals == []
    assert rec["portfolio_state"]["source"] == "supplied"
    assert rec["decisions"][0]["delta_quantity"] == pytest.approx(2500 / 110)


def test_explicit_saved_portfolio_skips_default_plan(context):
    request(context, {"portfolio_id": PID, "input_snapshot_id": SID, "as_of": "2026-10-09"})
    assert context.calls == [("state", PID), ("portfolio", PID)]


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


def test_incompatible_frozen_policy_cannot_escape_small_profile_universe(context):
    from finplan_model.paper_profile import identity
    metadata={'portfolio_id':PID,'synthetic':True,'settings':{'paper_research_profile':{
        'identity':identity('synthetic-caller','small'),'caller_id':'synthetic-caller','profile_id':'small','initial_cash':100.,'instruments':['VOO','GOOGL']}}}
    context.service.d.platform.get_portfolio=lambda pid:metadata
    with pytest.raises(FinplanError) as exc:request(context,{})
    assert exc.value.details['reason'] in ('paper_profile_price_coverage','paper_profile_policy_incompatible')
    assert not context.captured and not context.proposals
