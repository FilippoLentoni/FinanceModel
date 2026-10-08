"""SIM-02 (strategy interface, point-in-time view) and SIM-09 (reconciliation); tasks 2.1 and 2.6."""

from __future__ import annotations

from datetime import date

import pytest

from finplan_model.core.errors import FinplanError
from finplan_model.core.outcome import failed_job_result
from finplan_model.sim.engine import Simulator
from finplan_model.sim.strategy import TargetWeights, aggregate_solution_status, coerce_target

from .support import Fixed, Script, cfg, make_market

ROWS = [
    ("2026-01-05", {"A": (100.0, 101.0, 1e9), "B": (50.0, 51.0, 1e9)}),
    ("2026-01-06", {"A": (101.0, 102.0, 1e9), "B": (51.0, 52.0, 1e9)}),
    ("2026-01-07", {"A": (102.0, 103.0, 1e9), "B": (52.0, 53.0, 1e9)}),
    ("2026-01-08", {"A": (103.0, 104.0, 1e9), "B": (53.0, 54.0, 1e9)}),
]


@pytest.mark.parametrize(
    "output",
    [
        {"orders": [{"instrument_id": "A", "quantity": 10}]},
        {"weights": {"A": 0.5}, "cash": 0.5, "quantity": {"A": 10}},
        {"weights": [{"instrument_id": "A", "weight": 0.5, "side": "buy"}], "cash_weight": 0.5},
        [("A", 10)],
    ],
)
def test_order_shaped_output_fails_validation(output):
    with pytest.raises(FinplanError) as ei:
        Simulator(cfg(), make_market(ROWS)).run(Script({r[0]: output for r in ROWS}))
    assert ei.value.code == "VALIDATION_FAILED"


def test_accepted_output_shapes():
    universe = ("A", "B")
    a = coerce_target(TargetWeights({"A": 0.5}, 0.5), universe)
    b = coerce_target({"weights": {"A": 0.5}, "cash": 0.5}, universe)
    c = coerce_target({"weights": [{"instrument_id": "A", "weight": 0.5}], "cash_weight": 0.5}, universe)
    assert a.weights == b.weights == c.weights == {"A": 0.5, "B": 0.0}
    assert a.cash == b.cash == c.cash == 0.5
    with pytest.raises(FinplanError) as ei:
        coerce_target({"weights": {"ZZZ": 1.0}, "cash": 0.0}, universe)
    assert ei.value.code == "VALIDATION_FAILED"
    with pytest.raises(FinplanError):
        coerce_target({"weights": {"A": float("nan")}, "cash": 0.0}, universe)


def test_strategy_sees_only_point_in_time_data_and_holdings():
    late = {("2026-01-06", "A"): "2026-01-07T09:00:00Z"}  # day-2 bar of A arrives after day-2's decision time
    strat = Fixed({"A": 0.5, "B": 0.5}, 0.0)
    Simulator(cfg(), make_market(ROWS, late=late)).run(strat)
    v0, v1, v2 = strat.views
    assert [d for d, _ in v0.history("A")] == [date(2026, 1, 5)]
    # Late-arriving observation excluded at its own decision, used from the next decision onward.
    assert [d for d, _ in v1.history("A")] == [date(2026, 1, 5)]
    assert [d for d, _ in v1.history("B")] == [date(2026, 1, 5), date(2026, 1, 6)]
    assert [d for d, _ in v2.history("A")] == [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]
    assert all(d <= v.decision_session for v in (v0, v1, v2) for i in ("A", "B") for d, _ in v.history(i))


def test_infeasible_decisions_fall_back_and_aggregate():
    out = {"2026-01-05": {"weights": {"A": 0.5, "B": 0.5}, "cash": 0.0}, "2026-01-06": {"weights": {"A": 0.0, "B": 0.0}, "cash": 1.0, "solution_status": "infeasible"}, "2026-01-07": {"weights": {"A": 0.5, "B": 0.5}, "cash": 0.0}}
    res = Simulator(cfg(), make_market(ROWS)).run(Script(out))
    assert [d["action"] for d in res.decisions] == ["executed", "fallback_hold_current", "executed"]
    assert res.solution_status == "feasible"
    assert aggregate_solution_status(["infeasible"] * 3) == "infeasible"
    assert aggregate_solution_status(["optimal"] * 38 + ["infeasible"] * 2) == "feasible"
    assert aggregate_solution_status([]) == "no_effect"


class LeakySimulator(Simulator):
    """Injects an accounting break: the fill's fee is never posted to cash."""

    def _apply_fill(self, book, fill):  # noqa: ANN001
        book.cash -= fill.quantity * fill.price + fill.spread_cost + fill.slippage_cost
        book.shares[fill.instrument_id] += fill.quantity


def test_reconciliation_break_fails_run_with_internal(ctx):
    c = cfg(fees={"proportional_bps": 10.0, "fixed_per_trade": 1.0})
    with pytest.raises(FinplanError) as ei:
        LeakySimulator(c, make_market(ROWS)).run(Fixed({"A": 0.5, "B": 0.5}, 0.0))
    err = ei.value
    assert err.code == "INTERNAL"
    assert err.details["reason"] == "reconciliation_break"
    assert (err.details["step"], err.details["session_date"]) == (1, "2026-01-06")
    assert abs(err.details["difference"]) > c.reconciliation_tolerance
    run_ctx = ctx.with_run(ctx.ids.run_id())
    result = failed_job_result(run_ctx, err)
    assert result["completion_status"] == "failed"
    assert result["error"]["code"] == "INTERNAL"
    assert result["error"]["correlation_id"] == run_ctx.correlation_id
    assert "solution_status" not in result


def test_clean_run_reconciles_every_step(market):
    res = Simulator(cfg(fees={"proportional_bps": 3.0, "fixed_per_trade": 0.5}, spread={"half_spread_bps": 2.0}, slippage={"model": "linear_participation", "coefficient_bps": 25.0}, liquidity={"participation_cap": 0.001, "unfilled": "carry_over"}), market).run(Fixed({"AGG": 0.3, "SPY": 0.6}, 0.1))
    assert res.reconciliation["steps"] == len(market.sessions)
    assert res.reconciliation["max_abs_error"] <= res.reconciliation["tolerance"]
    assert min(row["cash"] for row in res.nav) >= -1e-6
