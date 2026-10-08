"""BASE-05 contract tests (CS-07): optimizer outcomes map to ``solution_status`` with
``completion_status`` ``succeeded`` and the hold-current-weights fallback, checked against the
shared ``job-result`` contract fixtures."""

from __future__ import annotations

import copy

import pytest
from finplan_contracts.validate import validate

from finplan_model.datasets.fixtures import contract_fixture
from finplan_model.evaluate import evaluate
from finplan_model.sim.config import SimulationConfig
from finplan_model.sim.market import synthetic_market
from finplan_model.strategies import MinVariance, ScenarioCVaR, job_outcome

MARKET = synthetic_market(("AGG", "SPY"), n_sessions=260, seed=13, daily_vol=0.01)


def _job_result(outcome: dict, base: str = "succeeded-optimal.json") -> dict:
    doc = copy.deepcopy(contract_fixture("job-result", base))
    doc["completion_status"] = outcome["completion_status"]
    doc["solution_status"] = outcome["solution_status"]
    return doc


def _cfg(**constraints):
    return SimulationConfig.from_dict({"rebalance_frequency": "weekly", "constraints": constraints} if constraints else {"rebalance_frequency": "weekly"})


def test_optimal_run():
    cfg = _cfg()
    res = evaluate(MinVariance({"lookback": 40}, constraints=cfg.constraints), MARKET, cfg)
    out = job_outcome(res)
    assert out == {"completion_status": "succeeded", "solution_status": "optimal", "fallback_decisions": [], "stage_output_allowed": True}
    assert validate(_job_result(out), "job-result").valid


@pytest.mark.parametrize("constraints", [{"min_cash": 0.5, "max_cash": 0.4}, {"min_cash": 0.5, "per_instrument": {"SPY": {"min": 0.6, "max": 1.0}}}])
def test_infeasible_constraints_complete_with_infeasible(constraints):
    """Minimum cash 50% and a minimum invested weight of 60%: no job failure, no staging output."""
    cfg = _cfg(**constraints)
    res = evaluate(MinVariance({"lookback": 40}, constraints=cfg.constraints), MARKET, cfg)
    out = job_outcome(res)
    assert out["completion_status"] == "succeeded" and out["solution_status"] == "infeasible"
    assert out["stage_output_allowed"] is False
    assert all(d["action"] in ("fallback_hold_current", "rejected_hold_current", "hold_no_effect") for d in res.simulation.decisions)
    assert res.metrics["n_fills"] == 0  # every decision held the (all-cash) current weights
    doc = _job_result(out)
    assert validate(doc, "job-result").valid
    assert {k: doc[k] for k in ("completion_status", "solution_status")} == {k: contract_fixture("job-result", "succeeded-infeasible.json")[k] for k in ("completion_status", "solution_status")}
    # the contract's own invalid fixture (infeasible reported as a failure) is what we never produce
    assert not validate(contract_fixture_invalid("infeasible-reported-as-failure.json"), "job-result").valid


def contract_fixture_invalid(name: str) -> dict:
    import json

    from finplan_contracts.schemas import load_store

    return json.loads((load_store().fixtures_dir("job-result") / "invalid" / name).read_text(encoding="utf-8"))


def test_partially_infeasible_backtest_reports_feasible():
    """A CVaR limit that only a few high-volatility decisions cannot meet."""
    from finplan_model.sim.market import Bar, MarketData

    base = synthetic_market(("AGG", "SPY"), n_sessions=260, seed=13, daily_vol=0.004)
    shock_start = base.sessions[150]
    bars = []
    for i in base.instruments:
        prev = None
        for b in base.bars_of(i):
            if b.session_date >= shock_start and b.session_date < base.sessions[165]:
                k = base.sessions.index(b.session_date) - 150
                factor = 1.0 + (0.06 if k % 2 == 0 else -0.06)
                c = (prev or b.close) * factor
                b = Bar(i, b.session_date, round(c, 4), b.available_at, round(c, 4), round(c * 1.001, 4), round(c * 0.999, 4), b.volume)
            elif prev is not None and b.session_date >= base.sessions[165]:
                r = b.close / base.bar(i, base.sessions[base.session_index(b.session_date) - 1]).close
                c = prev * r
                b = Bar(i, b.session_date, round(c, 4), b.available_at, round(c, 4), round(c * 1.001, 4), round(c * 0.999, 4), b.volume)
            prev = b.close
            bars.append(b)
    market = MarketData(base.sessions, bars, synthetic=True, dataset_id="synthetic/test/cvar-shock")
    cfg = _cfg()
    strat = ScenarioCVaR({"objective": "max_return", "cvar_limit": 0.02, "lookback": 20, "min_history": 10, "scenario_method": "historical", "cash_policy": "min"}, constraints=cfg.constraints)
    res = evaluate(strat, market, cfg)
    statuses = [d["solution_status"] for d in res.simulation.decisions]
    n_inf = statuses.count("infeasible")
    assert 0 < n_inf < len(statuses) - 5
    out = job_outcome(res)
    assert out["completion_status"] == "succeeded" and out["solution_status"] == "feasible"
    assert len(out["fallback_decisions"]) == n_inf and out["stage_output_allowed"] is True
    assert all(d["action"] == "fallback_hold_current" for d in res.simulation.decisions if d["solution_status"] == "infeasible")
    assert validate(_job_result(out), "job-result").valid
