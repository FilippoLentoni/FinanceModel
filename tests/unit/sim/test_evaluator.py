"""SIM-01 (one evaluator, comparison guard) and SIM-08 (deterministic replay); task 2.7."""

from __future__ import annotations

import copy

import pytest

from finplan_model.core.context import EVALUATOR_VERSION
from finplan_model.core.errors import FinplanError
from finplan_model.evaluate import EvaluatorIdentity, assert_comparable, evaluate, replay
from finplan_model.sim.config import SimulationConfig

from .support import Fixed


class EqualWeight:
    name = "equal_weight"

    def decide(self, view, holdings):  # noqa: ANN001
        n = len(view.instruments)
        return {"weights": {i: 0.99 / n for i in view.instruments}, "cash": 0.01}


class LowVolTilt:
    """Stand-in for an optimizer: weights inversely proportional to trailing volatility."""

    name = "low_vol_tilt"

    def decide(self, view, holdings):  # noqa: ANN001
        _, r = view.returns(lookback=10)
        if len(r) < 3:
            return {"weights": {i: 0.0 for i in view.instruments}, "cash": 1.0}
        inv = 1.0 / (r.std(axis=0, ddof=1) + 1e-12)
        w = 0.95 * inv / inv.sum()
        return {"weights": dict(zip(view.instruments, w.tolist())), "cash": 0.05}


def _cfg(**kw):
    return SimulationConfig.from_dict({"rebalance_frequency": "weekly", **kw})


def test_result_records_evaluator_identity(market, ctx):
    r = evaluate(EqualWeight(), market, _cfg(), ctx=ctx.with_run(ctx.ids.run_id()))
    ident = r.identity.to_dict()
    assert ident["evaluator_version"] == EVALUATOR_VERSION
    assert ident["simulation_configuration_id"].startswith("cfg_")
    assert ident["cost_model_id"].startswith("cfg_")
    assert ident["dataset_id"] == market.dataset_id
    assert r.synthetic is True
    assert r.result_checksum.startswith("sha256:")
    assert r.to_dict()["run_id"].startswith("run_")


def test_mismatched_fee_models_are_refused(market):
    a = evaluate(EqualWeight(), market, _cfg(fees={"proportional_bps": 5.0, "fixed_per_trade": 0.0}))
    b = evaluate(LowVolTilt(), market, _cfg(fees={"proportional_bps": 10.0, "fixed_per_trade": 0.0}))
    with pytest.raises(FinplanError) as ei:
        assert_comparable([a, b])
    assert ei.value.code == "VALIDATION_FAILED"
    assert ei.value.details["differing_settings"] == ["fee_model"]
    assert ei.value.details["pointer"] == "/runs"


def test_other_differences_are_listed(market):
    a = EvaluatorIdentity.of(_cfg(), market)
    b = EvaluatorIdentity.of(_cfg(execution_timing="next_close", liquidity={"participation_cap": 0.1}), market)
    c = EvaluatorIdentity(**{**a.__dict__, "evaluator_version": "9.9.9"})
    with pytest.raises(FinplanError) as ei:
        assert_comparable([a, b, c])
    assert ei.value.details["differing_settings"] == ["evaluator_version", "execution_timing", "liquidity_model"]


def test_same_settings_appear_side_by_side(market):
    a = evaluate(EqualWeight(), market, _cfg())
    b = evaluate(LowVolTilt(), market, _cfg())
    assert_comparable([a, b])  # no exception
    assert a.identity.comparison_fields() == b.identity.comparison_fields()


def test_deterministic_and_replay_reproduces_trades_and_metrics(market):
    cfg = _cfg(fees={"proportional_bps": 2.0, "fixed_per_trade": 0.25}, liquidity={"participation_cap": 0.002, "unfilled": "carry_over"})
    first = evaluate(LowVolTilt(), market, cfg)
    second = evaluate(LowVolTilt(), market, cfg)
    assert first.result_checksum == second.result_checksum
    stored = copy.deepcopy(first.canonical_content())
    again = replay(stored, market, cfg)
    assert again.result_checksum == first.result_checksum
    assert [f.to_dict() for f in again.simulation.fills] == stored["simulation"]["fills"]
    assert again.metrics == stored["metrics"]


def test_replay_detects_a_tampered_stored_result(market):
    cfg = _cfg()
    stored = copy.deepcopy(evaluate(EqualWeight(), market, cfg).canonical_content())
    stored["metrics"]["net_cumulative_return"] += 1e-9
    with pytest.raises(FinplanError) as ei:
        replay(stored, market, cfg)
    assert ei.value.code == "INTERNAL"
    assert ei.value.details["sections"] == ["metrics"]


def test_every_family_goes_through_the_same_evaluator(market):
    results = [evaluate(s, market, _cfg()) for s in (EqualWeight(), LowVolTilt(), Fixed({"AGG": 0.0, "SPY": 0.0}, 1.0, name="cash"))]
    assert len({r.identity.evaluator_version for r in results}) == 1
    assert_comparable(results)
