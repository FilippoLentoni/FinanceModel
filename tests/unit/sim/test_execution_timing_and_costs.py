"""SIM-04 (execution timing), SIM-05 (fees and costs), cash accrual; tasks 2.3 and 2.4."""

from __future__ import annotations

import pytest

from finplan_model.core.errors import FinplanError
from finplan_model.evaluate import evaluate
from finplan_model.sim.engine import Simulator

from .support import Fixed, cfg, make_market

# Same-bar fills would change the result: the decision uses d0's close (100); d1 opens at 110.
TIMING = [
    ("2026-01-05", {"X": (100.0, 100.0, 1e9)}),
    ("2026-01-06", {"X": (110.0, 120.0, 1e9)}),
    ("2026-01-07", {"X": (120.0, 120.0, 1e9)}),
]


def test_next_open_fills_at_next_session_open():
    res = Simulator(cfg(), make_market(TIMING)).run(Fixed({"X": 1.0}, 0.0))
    first = res.fills[0]
    assert (first.decision_session, first.session_date, first.price) == ("2026-01-05", "2026-01-06", 110.0)
    assert first.quantity == pytest.approx(100000.0 / 110.0)
    # Final value differs from what a same-bar fill at the decision close (100) would give (120000).
    assert res.nav[-1]["value"] == pytest.approx(100000.0 / 110.0 * 120.0)
    assert res.nav[-1]["value"] != pytest.approx(120000.0)
    assert all(f.session_date > f.decision_session for f in res.fills)


def test_next_close_fills_at_next_session_close():
    res = Simulator(cfg(execution_timing="next_close"), make_market(TIMING)).run(Fixed({"X": 1.0}, 0.0))
    first = res.fills[0]
    assert (first.session_date, first.price) == ("2026-01-06", 120.0)
    assert res.nav[-1]["value"] == pytest.approx(100000.0)


@pytest.mark.parametrize("timing", ["same_bar", "same_close", "same_open"])
def test_same_bar_execution_is_refused(timing):
    with pytest.raises(FinplanError) as ei:
        cfg(execution_timing=timing)
    assert ei.value.code == "VALIDATION_FAILED"
    assert ei.value.details["pointer"] == "/execution_timing"


def test_last_session_never_decides():
    strat = Fixed({"X": 1.0}, 0.0)
    res = Simulator(cfg(), make_market(TIMING)).run(strat)
    assert [d["decision_session"] for d in res.decisions] == ["2026-01-05", "2026-01-06"]


COSTS = [
    ("2026-01-05", {"X": (100.0, 100.0, 1000.0)}),
    ("2026-01-06", {"X": (100.0, 100.0, 1000.0)}),
]


def test_fee_spread_slippage_hand_computed():
    """10000 cash, 50% target at 100 -> buy 50 shares (participation 0.05).

    fee = 5000 * 10bps + 1.00 = 6.00; spread = 5000 * 5bps = 2.50;
    slippage = 5000 * (10bps * 0.05) = 0.25; total 8.75.
    """
    c = cfg(
        initial_cash=10000.0,
        fees={"proportional_bps": 10.0, "fixed_per_trade": 1.0},
        spread={"half_spread_bps": 5.0},
        slippage={"model": "linear_participation", "coefficient_bps": 10.0},
        liquidity={"participation_cap": 0.1, "unfilled": "cancel"},
    )
    r = evaluate(Fixed({"X": 0.5}, 0.5), make_market(COSTS), c)
    (fill,) = r.simulation.fills
    assert fill.quantity == pytest.approx(50.0)
    assert fill.participation == pytest.approx(0.05)
    assert fill.fee == pytest.approx(6.0)
    assert fill.spread_cost == pytest.approx(2.5)
    assert fill.slippage_cost == pytest.approx(0.25)
    s = r.simulation.summary
    assert s["total_costs"] == pytest.approx(8.75)
    assert s["final_value"] == pytest.approx(9991.25)
    assert s["gross_return"] == pytest.approx(0.0, abs=1e-12)
    assert s["net_return"] == pytest.approx(-8.75 / 10000.0)
    # Separate fields: gross, net and every cost component.
    m = r.metrics
    assert m["gross_cumulative_return"] == pytest.approx(0.0, abs=1e-12)
    assert m["net_cumulative_return"] == pytest.approx(-0.000875)
    assert (m["total_fees"], m["total_spread_cost"], m["total_slippage_cost"]) == pytest.approx((6.0, 2.5, 0.25))
    assert m["total_transaction_costs"] == pytest.approx(8.75)


def test_cash_only_portfolio_earns_the_configured_cash_rate():
    # 84 sessions = 83 accrual periods; with periods_per_year = 83 the run spans exactly one year.
    rows = [(f"2026-{m:02d}-{d:02d}", {"X": (100.0, 100.0, 1e6)}) for m in (1, 2, 3) for d in range(1, 29)]
    c = cfg(cash_rate_annual=0.05, periods_per_year=len(rows) - 1)
    res = Simulator(c, make_market(rows)).run(Fixed({"X": 0.0}, 1.0, name="cash"))
    assert res.fills == []
    assert res.summary["net_return"] == pytest.approx(0.05, rel=1e-12)
