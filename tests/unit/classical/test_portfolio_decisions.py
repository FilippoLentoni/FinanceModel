import copy
import json
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from finplan_contracts.validate import validate

from finplan_model.classical_api import handle
from finplan_model.core.artifacts import InMemoryArtifactStore
from finplan_model.core.errors import FinplanError
from finplan_model.decision_analysis import compare_decisions, decision_ref, evaluate_decision, explain_decision
from finplan_model.portfolio_decisions import persist_proposal
from finplan_model.rl.inference import implementation_identity, recommend


def issued(context, index=110, **extra):
    return context.service.recommend({"as_of": context.dates[index].isoformat(), **extra})


def issued_ppo(context, index=110):
    """Exercise the public serving producer with a real tiny frozen actor, without training."""
    from finplan_model.serving import handle as serve

    instruments = list(context.market.instruments)
    dim = 2 * len(instruments) + len(instruments) + 1
    actor = {"format": "finplan-actor/1", "output_transform": "clip", "layers": [
        {"weight": np.zeros((2, dim)).tolist(), "bias": [0., 0.], "activation": "tanh"},
        {"weight": np.zeros((2, 2)).tolist(), "bias": [0., 0.], "activation": "tanh"},
        {"weight": np.zeros((len(instruments) + 1, 2)).tolist(), "bias": [0.] * (len(instruments) + 1), "activation": "linear"}]}
    bundle = {"format": "finplan-strategy-bundle/1", "mode": "advisory_paper", "source_run_id": "run_01KDVDP88REHGPBXFX6CHX92KS", "configuration_id": "cfg_" + "a" * 64,
              "strategy_id": "ppo", "environment": {"window": 2}, "constraints": {}, "instruments": instruments, "members": [{"seed": 0, "actor": actor}], "available_after": context.dates[0].isoformat()}
    if not getattr(context.service.d, "artifacts", None):
        context.service.d.artifacts = InMemoryArtifactStore()
    artifact = context.service.d.artifacts.put_json(bundle, kind="policy_inference").to_dict()
    context.service.d.advisory_parameter = SimpleNamespace(read=lambda: json.dumps({"export_run_id": "run_01KDVDP88REHGPBXFX6CHX92KS", "artifact": artifact}))
    return serve({"environment": "beta", "request": {"portfolio_id": context.pid, "input_snapshot_id": context.sid, "as_of": context.dates[index].isoformat()}}, context.service)


@pytest.mark.parametrize("family", ["ppo", "classical"])
@pytest.mark.parametrize("operation", ["explain_portfolio_decision", "compare_portfolio_decisions", "evaluate_portfolio_decision"])
def test_complete_generic_public_responses_conform_for_both_algorithm_families(context, family, operation):
    producer = issued_ppo if family == "ppo" else issued
    first, second = producer(context, 110), producer(context, 111)
    body = {"portfolio_id": context.pid}
    if operation == "compare_portfolio_decisions":
        body.update(previous_decision_id=first["decision_id"], current_decision_id=second["decision_id"])
    else:
        body["decision_id"] = first["decision_id"]
        if operation == "explain_portfolio_decision":
            body["instrument_id"] = "GOOGL"
        else:
            body["end_date"] = context.dates[150].isoformat()
            context.platform.portfolio_history[context.pid] = [{**copy.deepcopy(context.state), "recorded_at": context.market.decision_time(context.dates[100]).isoformat(), "reason": "legacy_baseline"}]
    result = handle({"environment": "beta", "operation": operation, "request": body}, context.service)
    assert validate(result, "tools/" + operation.replace("_", "-") + "-response").valid
    assert context.service.store.get(result["analysis_id"])["portfolio_id"] == context.pid
    if operation == "explain_portfolio_decision":
        if family == "ppo":
            assert "recommendation" not in result
            assert result["policy_recommendation"] == first["recommendation"]
            assert result["explanation"]["policy_replay"]["status"] == "verified"
        else:
            assert "policy_recommendation" not in result
            assert result["recommendation"] == first["recommendation"]
            assert result["explanation"]["optimizer"]["reproduction"]["status"] == "verified"
    elif operation == "compare_portfolio_decisions":
        assert result["status"] == "available"
        if family == "ppo":
            assert result["policy_replay"]["previous"]["status"] == result["policy_replay"]["current"]["status"] == "verified"
    else:
        assert result["status"] == "available"
        assert result["forecast"]["status"] == "not_available"
        assert result["real_execution"]["status"] == "not_available"


def test_policy_explanation_does_not_reuse_previously_persisted_invalid_response(context):
    first = issued_ppo(context)
    doc = context.platform.get_portfolio_decision(context.pid, first["decision_id"])["decision"]
    legacy = context.service.issue("explanation", {"summary": "Legacy policy explanation with a classical-only field", "recommendation": first["recommendation"]}, {"decision": decision_ref(doc), "instrument_id": "GOOGL"}, portfolio_id=context.pid)
    assert not validate(legacy, "tools/explain-portfolio-decision-response").valid
    corrected = handle({"environment": "beta", "operation": "explain_portfolio_decision", "request": {"portfolio_id": context.pid, "decision_id": first["decision_id"], "instrument_id": "GOOGL"}}, context.service)
    assert corrected["analysis_id"] != legacy["analysis_id"]
    assert corrected["policy_recommendation"] == first["recommendation"]
    assert context.service.store.get(legacy["analysis_id"])["recommendation"] == first["recommendation"]


def test_saved_optimizer_proposal_is_linked_immutable_and_never_applies_holdings(context):
    before = copy.deepcopy(context.state)
    result = issued(context)
    doc = context.platform.get_portfolio_decision(context.pid, result["decision_id"])["decision"]
    assert doc["source_analysis_id"] == result["analysis_id"]
    assert doc["recommendation"] == result["recommendation"]
    assert doc["portfolio_revision"] == before["revision"]
    assert doc["status"] == "proposed"
    assert issued(context) == result
    assert context.state == before


def test_generic_classical_explanation_and_comparison_delegate_math(context):
    a, b = issued(context, 110), issued(context, 111)
    result = handle({"environment": "beta", "operation": "explain_portfolio_decision", "request": {"decision_id": a["decision_id"], "instrument_id": "GOOGL"}}, context.service)
    assert result["explanation"]["optimizer"]["reproduction"]["status"] == "verified"
    assert context.service.store.get(result["analysis_id"])["source_decision_id"] == a["decision_id"]
    compared = compare_decisions(context.service, {"portfolio_id": context.pid, "previous_decision_id": a["decision_id"], "current_decision_id": b["decision_id"]})
    assert compared["attribution"]["status"] == "available"
    assert len(compared["attribution"]["shapley"]["contributions"]) == 4
    assert compared["alignment"]["model_provenance_changed"] is False


def test_different_model_comparison_reports_changes_without_fake_shapley(context):
    a = issued(context)
    b = issued(context, algorithm="mean_variance")
    result = compare_decisions(context.service, {"previous_decision_id": a["decision_id"], "current_decision_id": b["decision_id"]})
    assert len(result["changes"]) == 5
    assert result["alignment"]["previous_algorithm"] != result["alignment"]["current_algorithm"]
    assert result["attribution"]["status"] == "not_available"


def test_changed_horizon_keeps_comparison_but_does_not_claim_aligned_shapley(context):
    a = issued(context)
    b = issued(context, settings={"horizon_sessions": 5})
    result = compare_decisions(context.service, {"previous_decision_id": a["decision_id"], "current_decision_id": b["decision_id"]})
    assert result["alignment"]["model_provenance_changed"] is True
    assert len(result["changes"]) == 5
    assert result["attribution"]["status"] == "not_available"


def test_fresh_request_key_creates_distinct_acceptable_proposal(context):
    first = issued(context, idempotency_key="first-proposal")
    second = issued(context, idempotency_key="new-proposal-after-rejection")
    assert first["decision_id"] != second["decision_id"]
    assert first["recommendation"] == second["recommendation"]


def test_hypothetical_optimization_has_analysis_but_no_acceptable_proposal(context):
    result = handle({"environment": "beta", "operation": "recommend_classical_portfolio", "request": {
        "input_snapshot_id": context.sid, "as_of": context.dates[110].isoformat(),
        "holdings": {"weights": [], "cash_weight": 1., "portfolio_value": 10000., "high_watermark": 10000.},
    }}, context.service)
    assert result["analysis_id"]
    assert result["recommendation"]["portfolio_state"]["source"] == "supplied"
    assert "decision_id" not in result
    assert context.platform.portfolio_decisions == {}


@pytest.mark.parametrize("representation", ["reversed", "partial"])
def test_hypothetical_weights_are_aligned_by_instrument_and_missing_means_zero(context, representation):
    rows = [{"instrument_id": name, "weight": .4 if k < 2 else 0.} for k, name in enumerate(context.market.instruments)]
    holdings = {"weights": rows, "cash_weight": .2, "portfolio_value": 10000., "high_watermark": 10000.}
    canonical = issued(context, holdings=holdings)
    other = copy.deepcopy(holdings)
    other["weights"] = list(reversed(rows)) if representation == "reversed" else rows[:2]
    represented = issued(context, holdings=other)
    assert represented["recommendation"] == canonical["recommendation"]
    assert represented["analysis_id"] == canonical["analysis_id"]


def accept_for_test(context, result, *, recorded_index=111, costs=5.):
    doc = context.platform.portfolio_decisions[result["decision_id"]]
    rec = result["recommendation"]
    net = rec["portfolio_state"]["portfolio_value"] - costs
    prices = doc["execution"]["reference_prices"]
    book = {**copy.deepcopy(context.state["paper_state"]),
            "positions": [{"instrument_id": r["instrument_id"], "quantity": net * r["weight"] / prices[r["instrument_id"]]} for r in rec["target_weights"] if r["weight"] > 1e-12],
            "cash_balance": net * rec["cash_weight"], "as_of": rec["as_of"]}
    at = (context.market.decision_time(context.dates[recorded_index]) - timedelta(hours=1)).isoformat()
    baseline = {**copy.deepcopy(context.state), "recorded_at": context.market.decision_time(context.dates[100]).isoformat(), "reason": "legacy_baseline"}
    revision = {"portfolio_id": context.pid, "revision": 2, "paper_state": book, "recorded_at": at, "reason": "accepted_paper_decision", "decision_id": doc["decision_id"], "input_snapshot_id": context.sid}
    context.platform.portfolio_history[context.pid] = [baseline, revision]
    context.platform.portfolio_states[context.pid] = {
        "portfolio_id": context.pid, "revision": 2, "paper_state": copy.deepcopy(book),
        "synthetic": True, "contract_version": "1.5.0",
    }
    doc.update(status="accepted", resolution={"before_revision": 1, "after_revision": 2, "paper_execution": True, "recorded_at": at, "paper_state": copy.deepcopy(book), "transaction_cost": costs, "simulated_fills": []})
    return revision


@pytest.mark.parametrize("algorithm", ["min_variance", "mean_variance", "cvar"])
def test_saved_cash_only_book_can_allocate_across_approved_universe(context, algorithm):
    context.state["paper_state"].update(positions=[], cash_balance=10000.)
    before = copy.deepcopy(context.state)
    result = handle({"environment": "beta", "operation": "recommend_classical_portfolio", "request": {"portfolio_id": context.pid, "algorithm": algorithm}}, context.service)
    rec = result["recommendation"]
    assert {r["instrument_id"] for r in rec["target_weights"]} == set(context.market.instruments)
    assert sum(r["weight"] for r in rec["target_weights"]) == pytest.approx(1.)
    assert all(r["current_quantity"] == 0. for r in rec["decisions"])
    assert any(r["action"] == "buy" and r["target_quantity"] > 0. for r in rec["decisions"])
    assert result["decision_id"] and context.state == before


def test_asset_exited_by_accepted_optimizer_can_reenter_on_next_saved_revision(context):
    first = issued(context, len(context.dates) - 1, algorithm="mean_variance")
    exited = {r["instrument_id"] for r in first["recommendation"]["target_weights"] if r["weight"] <= 1e-12}
    assert exited, "fixture must exercise an optimizer that sells at least one asset completely"
    accepted = accept_for_test(context, first, recorded_index=len(context.dates) - 1, costs=0.)
    assert exited.isdisjoint(r["instrument_id"] for r in accepted["paper_state"]["positions"])
    before = copy.deepcopy(context.platform.portfolio_states[context.pid])
    next_plan = context.service.recommend({"portfolio_id": context.pid, "algorithm": "min_variance", "settings": {"turnover_penalty": 0.}})
    rec = next_plan["recommendation"]
    assert rec["portfolio_state"]["revision"] == 2
    assert {r["instrument_id"] for r in rec["target_weights"]} == set(context.market.instruments)
    reentries = [r for r in rec["decisions"] if r["instrument_id"] in exited and r["action"] == "buy"]
    assert reentries and all(r["current_quantity"] == 0. and r["target_quantity"] > 0. for r in reentries)
    assert context.platform.portfolio_states[context.pid] == before
    assert context.platform.portfolio_decisions[next_plan["decision_id"]]["portfolio_revision"] == 2


def test_saved_asset_outside_approved_universe_fails_without_issuing_proposal(context):
    context.state["paper_state"]["positions"].append({"instrument_id": "UNAPPROVED", "quantity": 1.})
    with pytest.raises(FinplanError) as error:
        context.service.recommend({"portfolio_id": context.pid})
    assert error.value.details["reason"] == "paper_state_invalid"
    assert not context.platform.portfolio_decisions


def test_classical_universe_bound_still_rejects_more_than_five_approved_assets(context, monkeypatch):
    from types import SimpleNamespace
    from finplan_model.jobs.market_loader import load_market

    _, content = load_market(context.platform, context.sid)
    monkeypatch.setattr("finplan_model.classical.service.load_market", lambda *args: (SimpleNamespace(instruments=(*context.market.instruments, "EXTRA")), content))
    with pytest.raises(FinplanError) as error:
        context.service.recommend({"portfolio_id": context.pid})
    assert error.value.details["reason"] == "classical_universe_bound"
    assert not context.platform.portfolio_decisions


def test_performance_uses_recorded_revision_and_costs_not_unchanged_holdings(context):
    proposal = issued(context)
    rev = accept_for_test(context, proposal)
    result = handle({"environment": "beta", "operation": "evaluate_portfolio_decision", "request": {"decision_id": proposal["decision_id"], "end_date": context.dates[150].isoformat()}}, context.service)
    assert result["status"] == "available"
    assert result["observed_paper"]["revision"] == 2
    assert result["observed_paper"]["recorded_paper_costs"] == 5.
    prices = context.market.view(context.dates[150]).price_matrix()[1][-1]
    expected = sum(row["quantity"] * prices[k] for k, row in enumerate(rev["paper_state"]["positions"])) + rev["paper_state"]["cash_balance"]
    assert result["observed_paper"]["end_value"] == pytest.approx(expected)
    assert result["gap"]["total"] == pytest.approx(result["gap"]["execution_costs"] + result["gap"]["allocation_and_execution_timing"])
    assert result["forecast"]["status"] == "not_available"


def test_future_acceptance_is_not_backdated_to_historical_reference_close(context):
    proposal = issued(context)
    accept_for_test(context, proposal, recorded_index=160)
    result = evaluate_decision(context.service, {"decision_id": proposal["decision_id"], "end_date": context.dates[150].isoformat()})
    assert result["observed_paper"]["revision"] == 1
    assert result["observed_paper"]["recorded_paper_costs"] == 0.
    assert result["paper_execution_evidence"] == []


def test_operator_edit_makes_performance_accounting_unavailable(context):
    proposal = issued(context)
    rev = accept_for_test(context, proposal)
    rev.pop("decision_id")
    rev["reason"] = "operator_update"
    result = evaluate_decision(context.service, {"decision_id": proposal["decision_id"], "end_date": context.dates[150].isoformat()})
    assert result["status"] == "not_available"
    assert result["reason"] == "unaccounted_portfolio_edit_or_external_cash_flow"
    assert "gap" not in result


def test_incomplete_history_never_reports_actual_return(context):
    proposal = issued(context)
    rev = accept_for_test(context, proposal)
    context.platform.portfolio_history[context.pid] = [rev]
    result = evaluate_decision(context.service, {"decision_id": proposal["decision_id"], "end_date": context.dates[150].isoformat()})
    assert result["reason"] == "portfolio_history_incomplete"


@pytest.mark.parametrize("change", ["timestamp", "state", "revision", "status"])
def test_resolution_must_match_committed_history_before_reporting_performance(context, change):
    proposal = issued(context)
    accept_for_test(context, proposal)
    doc = context.platform.portfolio_decisions[proposal["decision_id"]]
    if change == "timestamp":
        doc["resolution"]["recorded_at"] = context.market.decision_time(context.dates[112]).isoformat()
    elif change == "state":
        doc["resolution"]["paper_state"]["cash_balance"] += 50.
    elif change == "revision":
        doc["resolution"]["before_revision"] = 99
    else:
        doc["status"] = "rejected"
    result = evaluate_decision(context.service, {"decision_id": proposal["decision_id"], "end_date": context.dates[150].isoformat()})
    assert result["status"] == "not_available"
    assert result["reason"] == "paper_fill_ledger_incomplete"
    assert "gap" not in result


def test_corporate_actions_without_dated_accounting_do_not_become_actuals(context, monkeypatch):
    from finplan_model.jobs.market_loader import load_market

    proposal = issued(context)
    accept_for_test(context, proposal)
    market, content = load_market(context.platform, context.sid)
    observations = content.payload["observations"]
    row = next(r for r in observations if r["session_date"] == context.dates[120].isoformat())
    row["dividend"] = 1.
    monkeypatch.setattr("finplan_model.decision_analysis.load_market", lambda *args: (market, content))
    result = evaluate_decision(context.service, {"decision_id": proposal["decision_id"], "end_date": context.dates[150].isoformat()})
    assert result["reason"] == "corporate_action_accounting_not_recorded"
    assert result["trend"] == "not_available"


def test_a_later_baseline_cannot_claim_observed_historical_performance(context):
    proposal = issued(context)
    accept_for_test(context, proposal, recorded_index=160)
    context.platform.portfolio_history[context.pid][0]["recorded_at"] = context.market.decision_time(context.dates[160]).isoformat()
    result = evaluate_decision(context.service, {"decision_id": proposal["decision_id"], "end_date": context.dates[150].isoformat()})
    assert result["reason"] == "baseline_not_recorded_by_observation_time"
    assert result["trend"] == "not_available"


def test_ppo_uses_frozen_actor_replay_with_real_member_diagnostics(context):
    instruments = ["AAPL", "GOOGL", "NFLX", "NVDA", "VOO"]
    dim = 2 * len(instruments) + len(instruments) + 1
    actor = {"format": "finplan-actor/1", "output_transform": "clip", "layers": [
        {"weight": np.zeros((2, dim)).tolist(), "bias": [0., 0.], "activation": "tanh"},
        {"weight": np.zeros((2, 2)).tolist(), "bias": [0., 0.], "activation": "tanh"},
        {"weight": np.zeros((6, 2)).tolist(), "bias": [0.] * 6, "activation": "linear"}]}
    bundle = {"format": "finplan-strategy-bundle/1", "mode": "advisory_paper", "source_run_id": "run_01KDVDP88REHGPBXFX6CHX92KS", "configuration_id": "cfg_" + "a" * 64,
              "strategy_id": "ppo", "environment": {"window": 2}, "constraints": {}, "instruments": instruments, "members": [{"seed": 0, "actor": actor}]}
    prices = context.market.view(context.dates[110], instruments).price_matrix(lookback=3)[1]
    holdings = {"weights": [], "cash_weight": 1., "portfolio_value": 10000., "high_watermark": 10000.}
    rec = recommend(bundle, prices, holdings)
    rec.update(as_of=context.dates[110].isoformat(), input_snapshot_id=context.sid, portfolio_state={"positions": [{"instrument_id": name, "quantity": 0., "reference_price": float(prices[-1, k])} for k, name in enumerate(instruments)]})
    context.service.d.artifacts = InMemoryArtifactStore()
    artifact = context.service.d.artifacts.put_json(bundle, kind="policy_inference").to_dict()
    did = persist_proposal(context.service, rec, context.state, {"implementation": implementation_identity(), "policy_artifact": artifact, "policy_inputs": {"prices": prices.tolist(), "holdings": holdings}})
    # Per-instrument arithmetic is itself recorded evidence.
    doc = context.platform.portfolio_decisions[did]
    result = explain_decision(context.service, {"decision_id": did})
    assert result["explanation"]["policy_replay"]["status"] == "verified"
    assert result["explanation"]["policy_replay"]["diagnostics"]["member_target_weights"][0]["seed"] == 0
    assert result["explanation"]["attribution"]["status"] == "not_available"
    original_identity = copy.deepcopy(doc["provenance"]["implementation"])
    doc["provenance"]["implementation"]["numpy_version"] = "different-version"
    with pytest.raises(FinplanError) as mismatch:
        explain_decision(context.service, {"decision_id": did})
    assert mismatch.value.details["reason"] == "policy_implementation_mismatch"
    doc["provenance"]["implementation"] = original_identity
    doc["recommendation"]["target_weights"][0]["weight"] += .1
    with pytest.raises(FinplanError, match="reproduce"):
        explain_decision(context.service, {"decision_id": did})


def test_weekly_review_consumes_resolution_and_observed_gap_without_starting_compute(context):
    from finplan_model.classical.research import review

    proposal = issued(context)
    accept_for_test(context, proposal)
    perf = evaluate_decision(context.service, {"decision_id": proposal["decision_id"], "end_date": context.dates[150].isoformat()})
    result = review(context.service, {})
    lifecycle = next(r for r in result["portfolio_decision_evidence"] if r["decision_id"] == proposal["decision_id"])
    assert lifecycle["status"] == "accepted"
    assert lifecycle["resolution"]["after_revision"] == 2
    evidence = next(r for r in result["evidence"] if r["analysis_id"] == perf["analysis_id"])
    assert evidence["source_decision_id"] == proposal["decision_id"]
    assert evidence["observed_paper"]["recorded_paper_costs"] == 5.
    assert context.service.d.job_api.calls == []
