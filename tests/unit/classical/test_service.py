import copy

import pytest
from finplan_contracts.validate import validate

from finplan_model.classical.storage import reference
from finplan_model.classical_api import handle
from finplan_model.core.errors import FinplanError


def invoke(ctx, op, body):
    return handle(
        {"environment": "beta", "operation": op, "request": body}, ctx.service
    )


@pytest.mark.parametrize("algorithm", ["min_variance", "mean_variance", "cvar"])
def test_issued_plans_are_immutable_feasible_and_arithmetic_reconciles(
    context, algorithm
):
    before = copy.deepcopy(context.state)
    a = invoke(context, "recommend_classical_portfolio", {"algorithm": algorithm})
    assert (
        invoke(context, "recommend_classical_portfolio", {"algorithm": algorithm}) == a
    )
    assert validate(a, "tools/recommend-classical-portfolio-response").valid
    rec = a["recommendation"]
    nav = rec["portfolio_state"]["portfolio_value"]
    assert sum(
        d["target_quantity"] * d["reference_price"] for d in rec["decisions"]
    ) + rec["cash_weight"] * nav == pytest.approx(nav)
    for d in rec["decisions"]:
        assert d["delta_quantity"] * d["reference_price"] == pytest.approx(
            d["indicative_notional"]
        )
    assert context.state == before
    stored = context.service.store.get(a["analysis_id"])
    assert (
        a["analysis_ref"] == reference(stored)
        and "solve_inputs" in stored
        and "solve_inputs" not in a
    )
    got = invoke(context, "get_classical_analysis", {"analysis_id": a["analysis_id"]})
    assert got == a
    listed = invoke(
        context,
        "list_classical_analyses",
        {"portfolio_id": context.pid, "kind": "recommendation", "limit": 1},
    )
    assert (
        listed["analyses"][0]["analysis_id"] == a["analysis_id"]
        and listed["analyses"][0]["algorithm"] == algorithm
    )
    assert "sources" not in listed["analyses"][0]


def test_explanation_reproduces_pinned_inputs_and_detects_changed_output(context):
    a = invoke(context, "recommend_classical_portfolio", {})
    e = invoke(
        context,
        "explain_classical_recommendation",
        {"analysis_id": a["analysis_id"], "instrument_id": "GOOGL"},
    )
    assert validate(e, "tools/explain-classical-recommendation-response").valid
    assert e["explanation"]["reproduction"]["status"] == "verified"
    assert (
        len(e["explanation"]["force_keep"]) == 1
        and e["recommendation"] == a["recommendation"]
    )
    context.service.store.docs[a["analysis_id"]]["recommendation"]["target_weights"][0][
        "weight"
    ] += 0.01
    with pytest.raises(FinplanError, match="reproduce"):
        invoke(
            context,
            "explain_classical_recommendation",
            {"analysis_id": a["analysis_id"]},
        )


def test_plan_over_plan_attribution_reconciles_target_and_trade_changes(context):
    a = invoke(
        context,
        "recommend_classical_portfolio",
        {"input_snapshot_id": context.sid, "as_of": context.dates[150].isoformat()},
    )
    b = invoke(
        context,
        "recommend_classical_portfolio",
        {
            "input_snapshot_id": context.sid,
            "as_of": context.dates[151].isoformat(),
            "settings": {"turnover_penalty": 0.002},
        },
    )
    result = invoke(
        context,
        "compare_classical_plans",
        {
            "previous_analysis_id": a["analysis_id"],
            "current_analysis_id": b["analysis_id"],
        },
    )
    assert validate(result, "tools/compare-classical-plans-response").valid
    assert result["alignment"]["previous_as_of"] != result["alignment"]["current_as_of"]
    assert result["shapley"]["coalition_count"] == 16
    assert result["reproduction"]["status"] == "verified"
    assert result["reproduction"]["weight_tolerance"] == 1e-8
    for d in result["changes"]:
        assert sum(r["value"] for r in d["attribution"]) == pytest.approx(
            d["target_weight_change"], abs=1e-8
        )
        assert sum(r["value"] for r in d["trade_attribution"]) == pytest.approx(
            d["trade_delta_weight_change"], abs=1e-8
        )


@pytest.mark.parametrize("tamper", ["version", "allocation"])
def test_compare_refuses_unreproduced_issued_endpoints(context, tamper):
    a = context.service.recommend({"as_of": context.dates[150].isoformat()})
    b = context.service.recommend({"as_of": context.dates[151].isoformat()})
    record = context.service.store.docs[a["analysis_id"]]
    if tamper == "version":
        record["solve_inputs"]["implementation"]["version"] = "finplan-classical/2"
    else:
        record["recommendation"]["target_weights"][0]["weight"] += 0.01
    with pytest.raises(FinplanError) as exc:
        context.service.compare(
            {
                "previous_analysis_id": a["analysis_id"],
                "current_analysis_id": b["analysis_id"],
            }
        )
    assert exc.value.details["reason"] == (
        "classical_solver_version_mismatch"
        if tamper == "version"
        else "classical_plan_reproduction_mismatch"
    )
    assert context.service.store.list(kind="comparison") == []


def test_incompatible_horizon_fails_before_counterfactual(context):
    a = invoke(context, "recommend_classical_portfolio", {})
    b = invoke(
        context, "recommend_classical_portfolio", {"settings": {"horizon_sessions": 10}}
    )
    with pytest.raises(FinplanError, match="horizon"):
        invoke(
            context,
            "compare_classical_plans",
            {
                "previous_analysis_id": a["analysis_id"],
                "current_analysis_id": b["analysis_id"],
            },
        )


def test_paper_performance_forward_window_reconciles_and_has_no_invented_actuals(
    context,
):
    a = invoke(
        context,
        "recommend_classical_portfolio",
        {"input_snapshot_id": context.sid, "as_of": context.dates[170].isoformat()},
    )
    result = invoke(
        context, "evaluate_classical_performance", {"analysis_id": a["analysis_id"]}
    )
    assert validate(result, "tools/evaluate-classical-performance-response").valid
    assert (
        result["actual_source"] == "saved_paper"
        and result["real_execution"]["status"] == "not_available"
    )
    assert (
        result["forecast"]["status"] == "not_available"
        and result["window"]["partial_horizon"] is True
    )
    gap = result["gap"]
    assert gap["total"] == pytest.approx(
        gap["exposure"] + gap["raw_vs_adjusted_price_and_corporate_action_basis"]
    )
    assert gap["reconciliation_residual"] == pytest.approx(0, abs=1e-6)
    assert sum(
        r["planned_pnl"] for r in result["instrument_contributions"]
    ) == pytest.approx(result["planned_allocation_hold"]["pnl"])
    assert result["trend"] == (
        "green" if result["observed_paper"]["pnl"] >= 0 else "red"
    )
    assert result["issued_plan_trend"] == (
        "green" if result["planned_allocation_hold"]["pnl"] >= 0 else "red"
    )
    assert result["source_analysis_ref"] == a["analysis_ref"]


def test_no_forward_history_and_future_boundary_are_explicit(context):
    a = invoke(context, "recommend_classical_portfolio", {})
    assert (
        invoke(
            context, "evaluate_classical_performance", {"analysis_id": a["analysis_id"]}
        )["trend"]
        == "not_available"
    )
    with pytest.raises(FinplanError):
        invoke(
            context,
            "evaluate_classical_performance",
            {"analysis_id": a["analysis_id"], "end_date": "2027-01-01"},
        )


def test_changed_paper_revision_has_only_unattributed_gap(context):
    a = invoke(
        context,
        "recommend_classical_portfolio",
        {"input_snapshot_id": context.sid, "as_of": context.dates[170].isoformat()},
    )
    context.platform.portfolio_states[context.pid]["revision"] = 2
    context.platform.portfolio_states[context.pid]["paper_state"]["cash_balance"] = 100
    result = invoke(
        context, "evaluate_classical_performance", {"analysis_id": a["analysis_id"]}
    )
    assert result["status"] == "partial" and result["gap"]["exposure"] is None
    assert result["gap"]["unattributed"] == result["gap"]["total"]


def test_explicit_historical_scenario_is_not_misrepresented_as_saved_actuals(context):
    a = invoke(
        context,
        "recommend_classical_portfolio",
        {
            "input_snapshot_id": context.sid,
            "as_of": context.dates[90].isoformat(),
            "holdings": {
                "weights": [
                    {"instrument_id": i, "weight": 0.2}
                    for i in context.market.instruments
                ],
                "cash_weight": 0.0,
                "portfolio_value": 10000.0,
                "high_watermark": 10000.0,
            },
        },
    )
    assert (
        a["recommendation"]["portfolio_state"]["source"] == "supplied"
        and "portfolio_id" not in a
    )
    p = invoke(
        context, "evaluate_classical_performance", {"analysis_id": a["analysis_id"]}
    )
    assert p["observed_paper"]["status"] == "not_available"
    assert p["actual_source"] == "hypothetical" and p["trend"] == "not_available"


def test_cross_environment_unknown_operation_and_unsupported_settings_rejected(context):
    with pytest.raises(FinplanError):
        handle(
            {
                "environment": "prod",
                "operation": "recommend_classical_portfolio",
                "request": {},
            },
            context.service,
        )
    with pytest.raises(FinplanError):
        invoke(context, "arbitrary_tool", {})
    with pytest.raises(FinplanError):
        invoke(
            context, "recommend_classical_portfolio", {"settings": {"solver": "remote"}}
        )


def test_request_idempotency_replays_original_plan_and_refuses_reused_key(context):
    body = {
        "idempotency_key": "daily-plan-stable-key",
        "input_snapshot_id": context.sid,
        "as_of": context.dates[150].isoformat(),
    }
    a = invoke(context, "recommend_classical_portfolio", body)
    context.state["revision"] = 2
    assert invoke(context, "recommend_classical_portfolio", body) == a
    with pytest.raises(FinplanError) as err:
        invoke(
            context,
            "recommend_classical_portfolio",
            {**body, "as_of": context.dates[151].isoformat()},
        )
    assert err.value.code == "IDEMPOTENCY_KEY_REUSED"


def test_sandbox_strategy_uses_same_solver_and_inputs_as_mcp(context):
    from datetime import date

    from finplan_model.classical.strategy import ClassicalStrategy
    from finplan_model.jobs.market_loader import load_market
    from finplan_model.sim.market import HoldingsView

    p = context.service.recommend({})["recommendation"]
    market, _ = load_market(context.platform, context.sid)
    positions = p["portfolio_state"]["positions"]
    observed = HoldingsView(
        {r["instrument_id"]: r["quantity"] for r in positions},
        p["portfolio_state"]["current_cash"],
        {r["instrument_id"]: r["reference_price"] for r in positions},
        p["portfolio_state"]["portfolio_value"],
    )
    result = ClassicalStrategy("min_variance").decide(
        market.view(date.fromisoformat(p["as_of"])), observed
    )
    assert list(result.weights.values()) == pytest.approx(
        [w["weight"] for w in p["target_weights"]], abs=1e-8
    )
