"""No training or network: test frozen replay, timing, cost and maturity semantics."""
import copy
import json
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from finplan_model.core.errors import FinplanError
from finplan_model.decision_analysis import evaluate_decision
from finplan_model.horizon_evaluation import _Control, _FrozenStrategy, evaluation_contract, evaluate_horizons
from finplan_model.jobs.market_loader import load_market
from finplan_model.rl.serving_context import raw_market
from finplan_model.sim.config import SimulationConfig
from finplan_model.sim.engine import Simulator
from finplan_model.sim.market import HoldingsView
from tests.unit.classical.test_portfolio_decisions import issued, issued_ppo


def evaluate(context, proposal, index):
    doc = context.platform.get_portfolio_decision(context.pid, proposal["decision_id"])["decision"]
    market, content = load_market(context.platform, context.sid)
    return evaluate_horizons(context.service, doc, market, raw_market(content, market), content, context.dates[index])


def test_new_decisions_predeclare_protocol_and_legacy_is_explicit(context):
    from finplan_model.core.clock import FrozenClock
    context.platform.clock = FrozenClock(context.market.decision_time(context.dates[110]) + timedelta(seconds=1))
    proposal = issued_ppo(context)
    doc = context.platform.portfolio_decisions[proposal["decision_id"]]
    protocol = doc["provenance"]["evaluation_contract"]
    assert protocol["horizons_sessions"] == [1, 5, 21, 63]
    assert protocol["objective"]["discount_factor"] is None
    assert protocol["objective"]["discount_status"] == "not_preserved_in_export"
    result = evaluate(context, proposal, 112)
    assert result["status"] == "available"
    assert result["protocol_status"] == "predeclared_at_issuance"
    assert result["primary_horizon_mature"] is False
    assert result["windows"][0]["status"] == "mature"
    assert result["windows"][1]["status"] == "partial"
    assert result["evidence_assessment"]["model_error_proven"] is False
    assert result["evidence_assessment"]["status"] == "preliminary"
    assert result["full_policy_replay"] is True
    del doc["provenance"]["evaluation_contract"]
    legacy = evaluate(context, proposal, 112)
    assert legacy["protocol_status"] == "retrospective_legacy_protocol"


def test_backdated_request_is_retrospective_even_with_a_saved_protocol(context):
    proposal = issued_ppo(context)
    result = evaluate(context, proposal, 112)
    assert result["full_policy_replay"]
    assert result["protocol_status"] == "retrospective_historical_request"
    assert result["protocol_timing"]["retrospective"]
    assert result["evidence_assessment"]["eligible_for_prospective_skill_evidence"] is False


def test_training_episode_and_optimizer_objective_horizons_are_distinct():
    ppo = evaluation_contract("ppo", bundle={"environment": {"episode_sessions": 64}, "members": [{}]})
    assert ppo["primary_horizon_sessions"] == 64
    assert ppo["horizons_sessions"] == [1, 5, 21, 64]
    assert ppo["objective"]["kind"] == "expected_cumulative_reward"
    traditional = evaluation_contract("min_variance", optimizer_settings={"horizon_sessions": 5})
    assert traditional["primary_horizon_sessions"] == 5
    assert traditional["objective"]["kind"] == "minimum_variance"


def test_replay_uses_frozen_actor_and_rebalances_with_consistent_costs(context):
    proposal = issued_ppo(context)
    # A changed serving selection must not be consulted for historical evaluation.
    def forbidden():
        raise AssertionError("historical replay read today's policy")
    context.service.d.advisory_parameter = SimpleNamespace(read=forbidden)
    result = evaluate(context, proposal, 113)
    assert result["full_policy_replay"]
    trajectories = result["sequential_evidence"]
    strategy = trajectories["issued_strategy"]
    assert len(strategy["decisions"]) == 3
    # Constant actor still issues daily rebalances as prices/current weights change.
    assert {f["decision_session"] for f in strategy["fills"]} == {context.dates[k].isoformat() for k in (110, 111, 112)}
    assert all(f["session_date"] > f["decision_session"] for f in strategy["fills"])
    assert all(f["fee"] == pytest.approx(f["notional"] * .0002) for f in strategy["fills"])
    assert trajectories["unchanged_holdings"]["fills"] == []
    for path in trajectories.values():
        assert path["reconciliation"]["status"] == "reconciled"
    initial = proposal["recommendation"]["portfolio_state"]["portfolio_value"]
    assert all(path["nav"][0]["value"] == pytest.approx(initial) for path in trajectories.values())
    assert strategy["nav"][0]["step_costs"] == 0
    metrics = result["windows"][-1]["strategies"]["issued_strategy"]
    assert metrics["costs"] == pytest.approx(sum(f["fee"] for f in strategy["fills"]))
    assert metrics["reward_diagnostics"]["discounted_reward"] is None


def test_changed_future_cannot_change_an_earlier_actor_decision(context):
    proposal = issued_ppo(context)
    baseline = evaluate(context, proposal, 114)
    market, content = load_market(context.platform, context.sid)
    raw = raw_market(content, market)
    changed = copy.deepcopy(market)
    # Tamper a future price path; the earlier point-in-time actor observations stay fixed.
    for iid in changed.instruments:
        bar = changed._bars[iid][context.dates[113]]
        from dataclasses import replace
        changed._bars[iid][context.dates[113]] = replace(bar, close=bar.close * 1.5)
    doc = context.platform.portfolio_decisions[proposal["decision_id"]]
    altered = evaluate_horizons(context.service, doc, changed, raw, content, context.dates[114])
    earlier = baseline["sequential_evidence"]["issued_strategy"]["decisions"][:3]
    assert altered["sequential_evidence"]["issued_strategy"]["decisions"][:3] == earlier
    assert altered["sequential_evidence"]["issued_strategy"]["decisions"][3] != baseline["sequential_evidence"]["issued_strategy"]["decisions"][3]


def test_sequential_policy_return_differs_from_holding_its_first_allocation(context):
    proposal = issued_ppo(context)
    doc = context.platform.portfolio_decisions[proposal["decision_id"]]
    market, content = load_market(context.platform, context.sid)
    raw = raw_market(content, market)
    bundle = json.loads(context.service.d.artifacts.get(doc["provenance"]["policy_artifact"]))
    state = proposal["recommendation"]["portfolio_state"]
    start, end = context.dates[110], context.dates[114]
    initial = HoldingsView({r["instrument_id"]: r["quantity"] for r in state["positions"]}, state["current_cash"],
                           {i: raw.bar(i, start).close for i in raw.instruments}, state["portfolio_value"])
    cfg = SimulationConfig.from_dict({**doc["provenance"]["evaluation_contract"]["simulation"], "initial_cash": initial.value})

    class Once:
        name = "first_allocation_hold"
        first = True
        frozen = _FrozenStrategy(doc, market, bundle=bundle)

        def decide(self, view, holdings):
            if self.first:
                self.first = False
                return self.frozen.decide(view, holdings)
            return TargetWeights(dict(holdings.weights), holdings.cash_weight, "no_effect")

    from finplan_model.sim.strategy import TargetWeights
    held = Simulator(cfg, raw).run(Once(), start=start, end=end, initial_book=initial)
    sequential = Simulator(cfg, raw).run(_FrozenStrategy(doc, market, bundle=bundle), start=start, end=end, initial_book=initial)
    assert len(sequential.fills) > len(held.fills)
    assert abs(sequential.nav[-1]["value"] - held.nav[-1]["value"]) > 1e-6


def test_missing_frozen_identity_and_revised_history_fail_closed(context):
    proposal = issued_ppo(context)
    doc = context.platform.portfolio_decisions[proposal["decision_id"]]
    saved_identity = copy.deepcopy(doc["provenance"]["implementation"])
    doc["provenance"]["implementation"] = {"version": "wrong"}
    assert evaluate(context, proposal, 112)["reason"] == "policy_implementation_mismatch"
    doc["provenance"]["implementation"] = saved_identity
    doc["provenance"]["policy_inputs"]["prices"][0][0] *= 1.01
    result = evaluate(context, proposal, 112)
    assert result["reason"] == "issued_policy_history_changed"
    assert result["full_policy_replay"] is False


def test_mature_classical_window_stays_descriptive_and_evidence_is_persisted(context):
    proposal = issued(context, settings={"horizon_sessions": 5})
    context.platform.portfolio_history[context.pid] = [{**copy.deepcopy(context.state), "recorded_at": context.market.decision_time(context.dates[100]).isoformat(), "reason": "legacy_baseline"}]
    result = evaluate_decision(context.service, {"decision_id": proposal["decision_id"], "end_date": context.dates[115].isoformat()})
    horizon = result["horizon_evaluation"]
    assert horizon["full_policy_replay"] is True
    assert horizon["primary_horizon_mature"]
    assert horizon["evidence_assessment"]["status"] == "single_path_review"
    assert horizon["evidence_assessment"]["model_error_proven"] is False
    assert "sequential_evidence" not in horizon
    assert "internal" not in result
    stored = context.service.store.get(result["analysis_id"])
    assert len(stored["internal"]["horizon_replay"]) == 6
    assert len(json.dumps(result).encode()) < 32768
    # Identical evaluation has the same immutable identity, even at a later call time.
    assert evaluate_decision(context.service, {"decision_id": proposal["decision_id"], "end_date": context.dates[115].isoformat()}) == result


def test_corporate_action_or_no_forward_session_never_fabricates_replay(context):
    proposal = issued_ppo(context)
    assert evaluate(context, proposal, 110)["reason"] == "no_forward_observations"
    market, content = load_market(context.platform, context.sid)
    raw = raw_market(content, market)
    row = next(r for r in content.payload["observations"] if r["session_date"] == context.dates[111].isoformat())
    row["dividend"] = 1
    doc = context.platform.portfolio_decisions[proposal["decision_id"]]
    assert evaluate_horizons(context.service, doc, market, raw, content, context.dates[112])["reason"] == "corporate_action_accounting_not_recorded"


def test_full_primary_ppo_window_is_bounded_and_does_not_prove_optimality(context):
    proposal = issued_ppo(context)
    context.platform.portfolio_history[context.pid] = [{**copy.deepcopy(context.state), "recorded_at": context.market.decision_time(context.dates[100]).isoformat(), "reason": "legacy_baseline"}]
    result = evaluate_decision(context.service, {"decision_id": proposal["decision_id"], "end_date": context.dates[175].isoformat()})
    horizon = result["horizon_evaluation"]
    assert horizon["primary_horizon_mature"]
    assert horizon["replay_window"]["sessions"] == 63
    assert horizon["replay_window"]["bounded_to_protocol"]
    assert horizon["evidence_assessment"]["model_error_proven"] is False
    assert len(json.dumps(result).encode()) < 32768


def test_common_simulator_initial_book_does_not_charge_fictitious_initialization(context):
    market = context.market
    start, end = context.dates[110:112]
    prices = {i: market.bar(i, start).close for i in market.instruments}
    shares = {i: 1000.0 / prices[i] for i in market.instruments}
    initial = HoldingsView(shares, 1000., prices, 6000.)
    cfg = SimulationConfig.from_dict({"initial_cash": 6000., "rebalance_frequency": "daily"})
    result = Simulator(cfg, market).run(_Control("unchanged_holdings"), start=start, end=end, initial_book=initial)
    assert result.fills == []
    assert result.nav[0]["value"] == 6000.
    expected = 1000 + sum(shares[i] * market.bar(i, end).close for i in market.instruments)
    assert result.nav[-1]["value"] == pytest.approx(expected)
    with pytest.raises(FinplanError) as error:
        Simulator(cfg, market).run(_Control("unchanged_holdings"), start=start, end=end, initial_book=HoldingsView(shares, 2000., prices, 7000.))
    assert error.value.details["reason"] == "initial_book_value_mismatch"
