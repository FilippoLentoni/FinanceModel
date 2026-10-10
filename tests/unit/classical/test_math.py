import numpy as np
import pytest

from finplan_model.classical.math import (
    cvar,
    exact_shapley,
    explain,
    objective,
    settings,
    solve,
    violations,
)


def inputs(algorithm="min_variance"):
    r = np.array([[0.01, -0.02], [-0.01, 0.01], [0.02, 0.0], [-0.02, 0.015]])
    return {
        "algorithm": algorithm,
        "settings": settings({"max_weight": 0.8, "turnover_penalty": 0}),
        "instruments": ["A", "B"],
        "current_weights": [0.5, 0.5],
        "covariance": np.cov(r.T).tolist(),
        "expected_returns": r.mean(axis=0).tolist(),
        "scenarios": r.tolist(),
    }


def test_exact_shapley_additive_and_interacting_efficiency():
    additive = exact_shapley(
        ["a", "b", "c"],
        lambda m: 5 + sum([2, 3, -1][i] for i in range(3) if m & (1 << i)),
    )
    assert [r["value"] for r in additive["contributions"]] == pytest.approx([2, 3, -1])
    interaction = exact_shapley(
        ["a", "b"],
        lambda m: (1 if m & 1 else 0) + (2 if m & 2 else 0) + (6 if m == 3 else 0),
    )
    assert [r["value"] for r in interaction["contributions"]] == pytest.approx([4, 5])
    assert interaction["reconciliation_residual"] == pytest.approx(0)
    assert interaction["coalition_count"] == 4


@pytest.mark.parametrize("algorithm", ["min_variance", "mean_variance", "cvar"])
def test_convex_solver_and_instrument_keep_evidence(algorithm):
    x = inputs(algorithm)
    result = solve(x)
    assert result["status"] == "optimal" and violations(x, result["weights"]) == []
    assert sum(result["weights"]) + result["cash_weight"] == pytest.approx(1)
    assert result["objective"] >= objective(x, x["current_weights"]) - 1e-8
    evidence = explain(x, result)
    assert sum(
        r["value"] for r in evidence["shapley"]["contributions"]
    ) == pytest.approx(evidence["objective_gain"])
    assert sum(
        r["variance_contribution"] for r in evidence["risk_contributions"]
    ) == pytest.approx(result["metrics"]["variance"])
    assert all(r["objective_loss_from_keep"] >= -1e-8 for r in evidence["force_keep"])
    assert evidence["causal_market_explanation"]["status"] == "not_available"
    assert evidence["shapley"][
        "infeasible_hybrids"
    ]  # full-investment trade subsets violate cash constraint


def test_infeasible_force_keep_and_cash_constraint_are_explicit():
    x = inputs()
    x["current_weights"] = [0.9, 0.1]
    assert solve(x, force_keep=0)["reason"] == "unchanged_position_violates_bounds"
    x["settings"]["max_weight"] = 0.2
    assert solve(x)["status"] == "infeasible"


def test_exact_discrete_cvar_fractional_tail():
    assert cvar([1, 2, 3, 4], 0.625) == pytest.approx((4 + 0.5 * 3) / 1.5)


@pytest.mark.parametrize(
    "invalid",
    [
        {"lookback_days": 19},
        {"risk_aversion": 0},
        {"turnover_penalty": float("nan")},
        {"lookback_days": 20.5},
        {"max_weight": True},
        {"unknown": 1},
    ],
)
def test_settings_refuse_invalid_inputs(invalid):
    from finplan_model.core.errors import FinplanError

    with pytest.raises(FinplanError):
        settings(invalid)
