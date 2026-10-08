"""BASE-01: control strategies and their automatic addition to benchmarks; task 4.1."""

from __future__ import annotations

import pytest

from finplan_model.core.errors import FinplanError
from finplan_model.evaluate import evaluate
from finplan_model.reporting import BenchmarkRequest
from finplan_model.sim.config import SimulationConfig
from finplan_model.sim.market import synthetic_market
from finplan_model.strategies import CONTROLS, BuyAndHold, CashControl, EqualWeight, build_strategy, with_controls

MARKET = synthetic_market(("AGG", "SPY", "VTI"), n_sessions=120, seed=5)


def test_cash_control_has_no_trades_and_earns_the_cash_rate():
    cfg = SimulationConfig.from_dict({"rebalance_frequency": "weekly", "cash_rate_annual": 0.04})
    res = evaluate(CashControl(), MARKET, cfg)
    assert res.metrics["n_fills"] == 0 and res.metrics["turnover"] == 0.0 and res.metrics["total_transaction_costs"] == 0.0
    per_session = 1.04 ** (1 / 252) - 1
    n = len(res.simulation.nav)
    assert res.metrics["net_cumulative_return"] == pytest.approx((1 + per_session) ** (n - 1) - 1, rel=1e-12)
    assert res.solution_status == "not_applicable"


def test_buy_and_hold_holds_its_initial_target_without_rebalancing():
    cfg = SimulationConfig.from_dict({"rebalance_frequency": "weekly", "liquidity": {"participation_cap": None}})
    res = evaluate(BuyAndHold(), MARKET, cfg)
    decisions = res.simulation.decisions
    assert decisions[0]["strategy_output"]["weights"] == pytest.approx({"AGG": 1 / 3, "SPY": 1 / 3, "VTI": 1 / 3})
    assert all(d["action"] == "hold_no_effect" for d in decisions[1:])
    fill_sessions = {f.session_date for f in res.simulation.fills}
    assert len(fill_sessions) == 1  # bought once, never rebalanced
    assert res.solution_status == "not_applicable"


def test_buy_and_hold_initial_weights_parameter():
    cfg = SimulationConfig.from_dict({"rebalance_frequency": "monthly"})
    res = evaluate(BuyAndHold({"initial_weights": {"SPY": 0.6, "AGG": 0.4}}), MARKET, cfg)
    assert res.simulation.decisions[0]["strategy_output"]["weights"] == {"AGG": 0.4, "SPY": 0.6, "VTI": 0.0}
    with pytest.raises(FinplanError):
        BuyAndHold({"initial_weights": {"SPY": 0.6}})


def test_equal_weight_rebalances_to_equal_weights_over_the_eligible_universe():
    cfg = SimulationConfig.from_dict({"rebalance_frequency": "weekly"})
    res = evaluate(EqualWeight(), MARKET, cfg)
    for d in res.simulation.decisions:
        assert d["strategy_output"]["weights"] == pytest.approx({"AGG": 1 / 3, "SPY": 1 / 3, "VTI": 1 / 3})
        assert d["action"] in ("executed", "projected")
    assert len({f.session_date for f in res.simulation.fills}) > 1  # it rebalances


def test_equal_weight_skips_instruments_without_a_visible_price():
    from finplan_model.sim.market import Bar, HoldingsView, MarketData

    base = synthetic_market(("AGG", "SPY"), n_sessions=10, seed=1)
    bars = [b for i in base.instruments for b in base.bars_of(i)] + [Bar("NEW", s, 50.0, b.available_at, 50.0, 50.0, 50.0, 1e6) for s, b in zip(base.sessions[5:], base.bars_of("AGG")[5:])]
    market = MarketData(base.sessions, bars, synthetic=True, dataset_id="synthetic/test/new-listing")
    view = market.view(base.sessions[2])
    out = EqualWeight().decide(view, HoldingsView({}, 1.0, {}, 1.0))
    assert out.weights == {"AGG": 0.5, "NEW": 0.0, "SPY": 0.5}


def test_missing_controls_are_added_automatically_and_listed():
    specs, added = with_controls([{"name": "min_variance"}, {"name": "cash"}])
    assert added == ["buy_and_hold", "equal_weight"]
    assert [s["name"] for s in specs] == ["min_variance", "cash", "buy_and_hold", "equal_weight"]
    req = BenchmarkRequest.from_dict({"strategies": [{"name": "mean_variance"}]})
    assert req.auto_added_controls == CONTROLS
    assert [s.name for s in req.strategies if s.auto_added] == list(CONTROLS)


def test_unknown_strategy_and_parameter_are_validation_failures():
    with pytest.raises(FinplanError) as ei:
        build_strategy("momentum_magic")
    assert ei.value.code == "VALIDATION_FAILED"
    with pytest.raises(FinplanError) as ei:
        build_strategy("equal_weight", {"leverage": 2}, pointer="/strategies/0")
    assert ei.value.details["pointer"] == "/strategies/0/params/leverage"
