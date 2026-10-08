"""SIM-03 (constraints: project and reject) and SIM-06 (liquidity cap); tasks 2.2 and 2.5."""

from __future__ import annotations

import math

import pytest

from finplan_model.sim.config import ConstraintSet
from finplan_model.sim.constraints import apply_constraints, project_box_simplex
from finplan_model.sim.engine import Simulator

from .support import Fixed, cfg, make_market

TWO = [
    ("2026-01-05", {"A": (100.0, 100.0, 1e9), "B": (50.0, 50.0, 1e9)}),
    ("2026-01-06", {"A": (100.0, 100.0, 1e9), "B": (50.0, 50.0, 1e9)}),
    ("2026-01-07", {"A": (100.0, 100.0, 1e9), "B": (50.0, 50.0, 1e9)}),
]


def test_sixty_percent_over_forty_percent_cap_is_projected_and_recorded():
    c = cfg(constraints={"max_weight": 0.4}, constraint_policy="project", rebalance_frequency="monthly")
    res = Simulator(c, make_market(TWO)).run(Fixed({"A": 0.6, "B": 0.3}, 0.1))
    (dec,) = res.decisions
    assert dec["action"] == "projected"
    con = dec["constraints"]
    assert {"constraint": "max_weight", "instrument_id": "A", "value": 0.6, "limit": 0.4} in con["violations"]
    # Euclidean projection onto {0 <= w <= 0.4, cash in [0, 1], sum = 1}: A 0.4, B 0.4, cash 0.2.
    assert con["final_weights"] == pytest.approx({"A": 0.4, "B": 0.4})
    assert con["final_cash"] == pytest.approx(0.2)
    assert con["projection_distance"] == pytest.approx(math.sqrt(0.2**2 + 0.1**2 + 0.1**2))
    # The simulator executed the projected target, not the proposal.
    nav = res.nav[-1]
    held_a = sum(f.quantity for f in res.fills if f.instrument_id == "A") * 100.0 / nav["value"]
    assert held_a == pytest.approx(0.4)


def test_turnover_limit_with_reject_policy_keeps_current_holdings():
    c = cfg(constraints={"max_turnover": 0.5}, constraint_policy="reject", rebalance_frequency="monthly")
    res = Simulator(c, make_market(TWO)).run(Fixed({"A": 0.4, "B": 0.4}, 0.2))
    (dec,) = res.decisions
    assert dec["action"] == "rejected_hold_current"
    assert dec["constraints"]["violations"] == [{"constraint": "max_turnover", "value": pytest.approx(0.8), "limit": 0.5}]
    assert res.fills == []
    assert res.summary["n_rejections"] == 1


def test_turnover_limit_with_project_policy_moves_partway():
    out = apply_constraints({"A": 0.4, "B": 0.4}, 0.2, ConstraintSet(max_turnover=0.5), current={"A": 0.0, "B": 0.0}, current_cash=1.0, policy="project", tol=1e-9)
    assert out.action == "projected"
    assert out.weights == pytest.approx({"A": 0.25, "B": 0.25})
    assert out.cash == pytest.approx(0.5)


def test_infeasible_constraint_set_rejects_decision():
    # min cash 50% and minimum invested weight 60% (max cash 40%) cannot both hold.
    cs = ConstraintSet(min_cash=0.5, max_cash=0.4)
    out = apply_constraints({"A": 0.6}, 0.4, cs, current={"A": 0.0}, current_cash=1.0, policy="project", tol=1e-9)
    assert (out.action, out.reason, out.weights) == ("rejected", "constraint_set_infeasible", None)


def test_projection_is_exact_and_feasible():
    y = {"a": 0.9, "b": -0.2, "c": 0.5, "cash": 0.1}
    lb = {"a": 0.0, "b": 0.0, "c": 0.0, "cash": 0.05}
    ub = {"a": 0.5, "b": 0.5, "c": 0.5, "cash": 1.0}
    x = project_box_simplex(y, lb, ub)
    assert x is not None
    assert sum(x.values()) == pytest.approx(1.0, abs=1e-15)
    assert all(lb[k] - 1e-15 <= x[k] <= ub[k] + 1e-15 for k in x)
    assert project_box_simplex({"a": 1.0}, {"a": 0.0}, {"a": 0.5}) is None


def test_long_only_violation_recorded():
    c = cfg(rebalance_frequency="monthly")
    res = Simulator(c, make_market(TWO)).run(Fixed({"A": -0.2, "B": 0.6}, 0.6))
    (dec,) = res.decisions
    assert dec["action"] == "projected"
    assert any(v["constraint"] == "long_only" for v in dec["constraints"]["violations"])
    assert dec["constraints"]["final_weights"]["A"] == pytest.approx(0.0)


# ------------------------------------------------------------------ liquidity (SIM-06)
THIN = [(f"2026-01-{d:02d}", {"X": (100.0, 100.0, 1000.0)}) for d in (5, 6, 7, 8, 9, 12)]


def test_participation_cap_cancel_mode():
    c = cfg(initial_cash=10000.0, rebalance_frequency="monthly", liquidity={"participation_cap": 0.01, "unfilled": "cancel"})
    res = Simulator(c, make_market(THIN)).run(Fixed({"X": 0.5}, 0.5))
    assert [f.quantity for f in res.fills] == [pytest.approx(10.0)]
    (u,) = res.unfilled
    assert (u["reason"], u["handling"], u["session_date"]) == ("participation_cap", "cancelled", "2026-01-06")
    assert u["quantity"] == pytest.approx(40.0)


def test_participation_cap_carry_over_mode():
    c = cfg(initial_cash=10000.0, rebalance_frequency="monthly", liquidity={"participation_cap": 0.01, "unfilled": "carry_over"})
    res = Simulator(c, make_market(THIN)).run(Fixed({"X": 0.5}, 0.5))
    assert [f.session_date for f in res.fills] == ["2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-12"]
    assert sum(f.quantity for f in res.fills) == pytest.approx(50.0)
    assert all(f.participation <= 0.01 + 1e-12 for f in res.fills)
    assert all(u["handling"] == "carried_over" for u in res.unfilled)
    assert [u["quantity"] for u in res.unfilled] == pytest.approx([40.0, 30.0, 20.0, 10.0])


def test_new_decision_supersedes_carried_over_remainder():
    c = cfg(initial_cash=10000.0, rebalance_frequency="daily", liquidity={"participation_cap": 0.01, "unfilled": "carry_over"})
    res = Simulator(c, make_market(THIN)).run(Fixed({"X": 0.5}, 0.5))
    assert any(u["handling"] == "cancelled_superseded" for u in res.unfilled)
    assert res.reconciliation["status"] == "reconciled"


def test_infeasible_constraint_set_in_a_run_completes_as_infeasible():
    c = cfg(constraints={"min_cash": 0.5, "max_cash": 0.4}, rebalance_frequency="daily")
    res = Simulator(c, make_market(TWO)).run(Fixed({"A": 0.3, "B": 0.3}, 0.4))
    assert [d["action"] for d in res.decisions] == ["rejected_hold_current", "rejected_hold_current"]
    assert res.solution_status == "infeasible"
    assert res.fills == []
