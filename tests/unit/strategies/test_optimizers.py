"""BASE-02 (minimum-variance, point-in-time covariance), BASE-03 (mean-variance risk aversion),
BASE-04 (scenario-CVaR reproducibility); tasks 4.2 to 4.4."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from finplan_model.core.errors import FinplanError
from finplan_model.sim.config import ConstraintSet
from finplan_model.sim.market import Bar, HoldingsView, MarketData, synthetic_market
from finplan_model.strategies import COVARIANCE_ESTIMATORS, MeanVariance, MinVariance, ScenarioCVaR, estimate_covariance, generate_scenarios

MARKET = synthetic_market(("AGG", "SPY", "VTI"), n_sessions=200, seed=17)
CASH = HoldingsView({"AGG": 0.0, "SPY": 0.0, "VTI": 0.0}, 100000.0, {}, 100000.0)
T = MARKET.sessions[120]


def _alter_after(market: MarketData, t, factor: float = 3.0, late: dict | None = None) -> MarketData:
    """Same history up to ``t``; every later bar replaced (wildly different prices)."""
    bars = []
    for i in market.instruments:
        for b in market.bars_of(i):
            if b.session_date > t:
                b = Bar(i, b.session_date, b.close * factor, b.available_at, (b.open or b.close) * factor, None, None, b.volume)
            if late and (i, b.session_date) in late:
                b = Bar(i, b.session_date, b.close * 1.7, late[(i, b.session_date)], b.open, b.high, b.low, b.volume)
            bars.append(b)
    return MarketData(market.sessions, bars, synthetic=True, dataset_id=market.dataset_id)


@pytest.mark.parametrize("estimator", COVARIANCE_ESTIMATORS)
def test_min_variance_uses_only_observations_available_at_decision_time(estimator):
    strat = MinVariance({"lookback": 60, "covariance_estimator": estimator})
    a = strat.decide(MARKET.view(T), CASH)
    b = strat.decide(_alter_after(MARKET, T).view(T), CASH)
    assert a.solution_status == "optimal" and a.weights == b.weights
    assert sum(a.weights.values()) + a.cash == pytest.approx(1.0, abs=1e-12)


def test_min_variance_ignores_a_bar_that_arrived_after_the_decision_time():
    late = {("SPY", T): datetime(T.year, T.month, T.day, 23, 30, tzinfo=UTC)}
    strat = MinVariance({"lookback": 60, "covariance_estimator": "sample"})
    with_late = strat.decide(_alter_after(MARKET, T, factor=1.0, late=late).view(T), CASH)
    # the decision equals one made without that session's SPY bar at all
    dropped = MarketData(MARKET.sessions, [b for i in MARKET.instruments for b in MARKET.bars_of(i) if not (i == "SPY" and b.session_date == T)], synthetic=True)
    assert with_late.weights == strat.decide(dropped.view(T), CASH).weights


def test_min_variance_is_the_minimum_over_feasible_portfolios():
    strat = MinVariance({"lookback": 60, "covariance_estimator": "sample"})
    out = strat.decide(MARKET.view(T), CASH)
    _d, r = MARKET.view(T).returns(60)
    cov = estimate_covariance(np.asarray(r), "sample")
    w = np.array([out.weights[i] for i in ("AGG", "SPY", "VTI")])
    rng = np.random.default_rng(0)
    for _ in range(200):
        x = rng.dirichlet(np.ones(3))
        assert w @ cov @ w <= x @ cov @ x + 1e-15


def test_min_variance_respects_bounds_and_min_cash():
    cs = ConstraintSet(max_weight=0.5, min_cash=0.1)
    out = MinVariance({"lookback": 60}, constraints=cs).decide(MARKET.view(T), CASH)
    assert out.cash == pytest.approx(0.1) and max(out.weights.values()) <= 0.5 + 1e-12


def test_insufficient_history_holds():
    out = MinVariance({"lookback": 60, "min_history": 30}).decide(MARKET.view(MARKET.sessions[10]), CASH)
    assert out.solution_status == "no_effect" and out.diagnostics["reason"] == "insufficient_history"


def _variance(out, cov):
    w = np.array([out.weights[i] for i in ("AGG", "SPY", "VTI")])
    return float(w @ cov @ w)


@pytest.mark.parametrize("cash_policy", ["free", "min"])
def test_variance_does_not_increase_as_risk_aversion_rises(cash_policy):
    view = MARKET.view(T)
    _d, r = view.returns(60)
    cov = estimate_covariance(np.asarray(r), "ledoit_wolf")
    variances = []
    for lam in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 200.0, 1000.0):
        out = MeanVariance({"lookback": 60, "risk_aversion": lam, "cash_policy": cash_policy}).decide(view, CASH)
        assert out.solution_status == "optimal"
        variances.append(_variance(out, cov))
    assert all(b <= a + 1e-12 for a, b in zip(variances, variances[1:])), variances
    assert variances[-1] < variances[0]


def test_mean_variance_return_estimator_is_configurable():
    view = MARKET.view(T)
    a = MeanVariance({"lookback": 60, "return_estimator": "historical_mean", "risk_aversion": 1.0}).decide(view, CASH)
    z = MeanVariance({"lookback": 60, "return_estimator": "zero", "risk_aversion": 1.0, "cash_policy": "min"}).decide(view, CASH)
    mv = MinVariance({"lookback": 60}).decide(view, CASH)
    assert z.weights == pytest.approx(mv.weights, abs=1e-6)  # zero expected returns = minimum variance
    assert a.diagnostics["estimated_return"] != z.diagnostics["estimated_return"]
    with pytest.raises(FinplanError):
        MeanVariance({"return_estimator": "oracle"})


@pytest.mark.parametrize("method", ["bootstrap", "gaussian", "historical"])
def test_same_seed_gives_same_scenarios_and_weights(method):
    view = MARKET.view(T)
    _d, r = view.returns(60)
    s1, m1 = generate_scenarios(np.asarray(r), method, 300, 7, T.isoformat())
    s2, m2 = generate_scenarios(np.asarray(r), method, 300, 7, T.isoformat())
    assert np.array_equal(s1, s2) and m1 == m2
    p = {"lookback": 60, "n_scenarios": 300, "scenario_method": method, "seed": 7}
    a, b = ScenarioCVaR(p).decide(view, CASH), ScenarioCVaR(p).decide(view, CASH)
    assert a.solution_status == "optimal" and a.weights == b.weights
    assert a.diagnostics["scenario_checksum"] == b.diagnostics["scenario_checksum"] and a.diagnostics["seed"] == 7


def test_different_seed_gives_different_scenarios():
    view = MARKET.view(T)
    a = ScenarioCVaR({"lookback": 60, "seed": 1}).decide(view, CASH)
    b = ScenarioCVaR({"lookback": 60, "seed": 2}).decide(view, CASH)
    assert a.diagnostics["scenario_checksum"] != b.diagnostics["scenario_checksum"]
    assert a.diagnostics["scenario_method"] == "bootstrap" and a.diagnostics["n_scenarios"] == 500


def test_scenarios_use_only_point_in_time_data():
    p = {"lookback": 60, "seed": 3}
    a = ScenarioCVaR(p).decide(MARKET.view(T), CASH)
    b = ScenarioCVaR(p).decide(_alter_after(MARKET, T).view(T), CASH)
    assert a.weights == b.weights and a.diagnostics["scenario_checksum"] == b.diagnostics["scenario_checksum"]


def test_cvar_limit_mode_and_parameter_validation():
    view = MARKET.view(T)
    out = ScenarioCVaR({"lookback": 60, "objective": "max_return", "cvar_limit": 0.5}).decide(view, CASH)
    assert out.solution_status == "optimal"
    tight = ScenarioCVaR({"lookback": 60, "objective": "max_return", "cvar_limit": 0.0}).decide(view, CASH)
    assert tight.solution_status == "infeasible"
    with pytest.raises(FinplanError):
        ScenarioCVaR({"cvar_limit": 0.1})
    with pytest.raises(FinplanError):
        ScenarioCVaR({"alpha": 1.5})
