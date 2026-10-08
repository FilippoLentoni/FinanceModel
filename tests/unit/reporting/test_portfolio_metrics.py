"""REP-02: portfolio metrics per fold, holdout and aggregate against hand-computed values; task 5.2.

Fixture: one synthetic instrument whose open equals its close, zero costs, no liquidity cap, daily
rebalancing. Buy-and-hold decides at the first session ``a`` of a window and buys at the next
session's price ``P[a+1]``, so its value is ``C`` on ``a`` and ``a+1`` and ``C * P[k] / P[a+1]``
afterwards. The expected numbers below are computed directly from the prices with ``statistics``.
"""

from __future__ import annotations

import math
import statistics
from datetime import date

import pytest

from finplan_model.datasets import FrozenCandidate, HoldoutAccessor, InMemoryHoldoutAccessLog, holdout_access_report
from finplan_model.reporting import build_report, run_benchmark

from ..datasets.support import Env, utc
from .support import hand_observations, hand_prices

C = 100000.0
ZERO = {
    "rebalance_frequency": "daily",
    "initial_cash": C,
    "fees": {"proportional_bps": 0.0, "fixed_per_trade": 0.0},
    "spread": {"half_spread_bps": 0.0},
    "slippage": {"model": "none", "coefficient_bps": 0.0},
    "liquidity": {"participation_cap": None, "unfilled": "cancel"},
}


def _expected(prices: list[float], a: int, b: int) -> dict[str, float | None]:
    """Hand computation for buy-and-hold over window sessions a..b (indexes into prices)."""
    nav = [C, C] + [C * prices[k] / prices[a + 1] for k in range(a + 2, b + 1)]
    rets = [nav[i] / nav[i - 1] - 1 for i in range(1, len(nav))]
    sd = statistics.stdev(rets)
    peak, mdd = C, 0.0
    for v in [C, *nav]:
        peak = max(peak, v)
        mdd = max(mdd, (peak - v) / peak)
    return {"net": prices[b] / prices[a + 1] - 1, "vol": sd * math.sqrt(252), "sharpe": statistics.mean(rets) / sd * math.sqrt(252), "mdd": mdd, "rets": rets, "path": nav}


@pytest.fixture(scope="module")
def hand():
    env = Env()
    sessions = env.provider.sessions(date(2026, 1, 5), date(2026, 3, 31))
    assert len(sessions) == 62
    sid = env.provider.publish(hand_observations(sessions), retrieved_at=utc(2026, 4, 1, 13))
    iso = [s.isoformat() for s in sessions]
    cfg = {
        "instrument": {"tickers": ["SPY"]},
        "availability": {"mode": "session_close"},
        "features": {"lookback_sessions": 2, "label_horizon_sessions": 2},
        "splits": {"train": {"start": iso[0], "end": iso[29]}, "validation": {"start": iso[32], "end": iso[45]}, "holdout": {"start": iso[48], "end": iso[61]}, "embargo_sessions": 2},
        "walk_forward": {"window": "expanding", "train_length": {"sessions": 10}, "test_length": {"sessions": 8}, "step": {"sessions": 8}, "embargo_sessions": 2},
    }
    ds = env.prepare([sid], cfg).load(env.store)
    env.ctx.clock.set("2026-05-01T00:00:00Z")  # the evaluation runs after the candidate freeze
    ctx = env.ctx.with_run(env.ctx.ids.run_id(), purpose="holdout_evaluation")
    log = InMemoryHoldoutAccessLog()
    run = run_benchmark({"strategies": [{"name": "buy_and_hold"}], "periods": ["walk_forward", "validation", "holdout"]}, ds, ZERO, ctx=ctx, holdout_accessor=HoldoutAccessor(ds, log), candidate=FrozenCandidate(ctx.ids.model_version(), "control", utc(2026, 4, 2)))
    report = build_report(run, cost_records=[{"run_id": ctx.run_id, "instance_type": "ml.m5.xlarge", "instance_count": 1, "estimated_usd": 0.05}], holdout_access=holdout_access_report(log, ds.dataset_id))
    return {"sessions": sessions, "prices": hand_prices(62), "ds": ds, "report": report}


def _row(report, period, pid, strategy="buy_and_hold"):
    rows = report.content["sections"]["portfolio_performance"]["periods"][period]
    return next(r for r in rows if r["strategy"] == strategy and r["period_id"] == pid)


def test_fold_boundaries_of_the_fixture(hand):
    folds = [(f.test.start, f.test.end) for f in hand["ds"].folds]
    s = hand["sessions"]
    assert folds == [(s[12], s[19]), (s[20], s[27]), (s[28], s[35]), (s[36], s[43])]


@pytest.mark.parametrize("fold,a,b", [(0, 12, 19), (1, 20, 27), (2, 28, 35), (3, 36, 43)])
def test_per_fold_metrics_match_hand_computation(hand, fold, a, b):
    exp = _expected(hand["prices"], a, b)
    m = _row(hand["report"], "walk_forward_fold", f"fold-{fold:03d}")["metrics"]
    assert m["net_cumulative_return"] == pytest.approx(exp["net"], abs=1e-12)
    assert m["gross_cumulative_return"] == pytest.approx(exp["net"], abs=1e-12)  # zero costs
    assert m["annualized_volatility"] == pytest.approx(exp["vol"], rel=1e-9)
    assert m["sharpe_ratio"] == pytest.approx(exp["sharpe"], rel=1e-9)
    assert m["max_drawdown"] == pytest.approx(exp["mdd"], abs=1e-12)
    assert m["turnover"] == pytest.approx(1.0, abs=1e-12)
    assert m["total_transaction_costs"] == 0.0


def test_validation_and_holdout_metrics_match_hand_computation(hand):
    for period, a, b in (("validation", 32, 45), ("holdout", 48, 61)):
        exp = _expected(hand["prices"], a, b)
        m = _row(hand["report"], period, period)["metrics"]
        assert m["net_cumulative_return"] == pytest.approx(exp["net"], abs=1e-12)
        assert m["max_drawdown"] == pytest.approx(exp["mdd"], abs=1e-12)
        assert m["annualized_volatility"] == pytest.approx(exp["vol"], rel=1e-9)


def test_aggregate_of_folds_matches_hand_computation(hand):
    parts = [_expected(hand["prices"], a, b) for a, b in ((12, 19), (20, 27), (28, 35), (36, 43))]
    net = math.prod(1 + p["net"] for p in parts) - 1
    rets = [r for p in parts for r in p["rets"]]
    level, peak, mdd = 1.0, 1.0, 0.0
    for p in parts:
        path = [C, *p["path"]]
        for k in range(1, len(path)):
            level *= path[k] / path[k - 1]
            peak = max(peak, level)
            mdd = max(mdd, (peak - level) / peak)
    row = _row(hand["report"], "walk_forward_fold", "aggregate")
    assert row["folds"] == 4 and row["aggregation"] == "stitched_out_of_sample"
    m = row["metrics"]
    assert m["net_cumulative_return"] == pytest.approx(net, abs=1e-12)
    assert m["annualized_volatility"] == pytest.approx(statistics.stdev(rets) * math.sqrt(252), rel=1e-9)
    assert m["sharpe_ratio"] == pytest.approx(statistics.mean(rets) / statistics.stdev(rets) * math.sqrt(252), rel=1e-9)
    assert m["max_drawdown"] == pytest.approx(mdd, abs=1e-12)
    assert m["turnover"] == pytest.approx(4.0, abs=1e-12)


def test_every_row_has_every_listed_metric_and_the_risk_free_source(hand):
    pp = hand["report"].content["sections"]["portfolio_performance"]
    assert pp["risk_free"] == {"source": "configured_cash_rate", "cash_rate_annual": 0.0}
    for rows in pp["periods"].values():
        for r in rows:
            assert set(r["metrics"]) == set(pp["metrics"])


def test_cash_control_rows(hand):
    m = _row(hand["report"], "validation", "validation", "cash")["metrics"]
    assert m["net_cumulative_return"] == 0.0 and m["annualized_volatility"] == 0.0 and m["sharpe_ratio"] is None and m["max_drawdown"] == 0.0
