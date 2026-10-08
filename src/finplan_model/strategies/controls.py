"""Control strategies: ``cash``, ``buy_and_hold`` and ``equal_weight`` (BASE-01).

* ``cash`` - 100% cash at every decision: no trades; its return is the configured cash rate
  compounded per session.
* ``buy_and_hold`` - the initial target (equal weights over the eligible universe, or the
  configured ``initial_weights``) on the first decision of a run, then ``no_effect`` holds: the
  position is never rebalanced. Stateless: the "first decision" is recognized from the holdings
  (no position yet), so one instance can evaluate several folds.
* ``equal_weight`` - equal weights over the eligible universe at every rebalance. The eligible
  universe is every instrument with a price visible at the decision time.

Controls optimize nothing, so their decisions report ``solution_status`` ``not_applicable``.
"""

from __future__ import annotations

from typing import Any

from finplan_model.core.errors import FinplanError
from finplan_model.sim.market import HoldingsView, PointInTimeView
from finplan_model.sim.strategy import TargetWeights

from .base import BaselineStrategy, hold, param_num

__all__ = ["BuyAndHold", "CashControl", "EqualWeight"]


def _eligible(view: PointInTimeView) -> list[str]:
    return [i for i in view.instruments if view.latest(i) is not None]


class CashControl(BaselineStrategy):
    name = "cash"
    family = "control"
    PARAMS = frozenset()

    def decide(self, view: PointInTimeView, holdings: HoldingsView) -> TargetWeights:
        return TargetWeights({i: 0.0 for i in view.instruments}, 1.0, "not_applicable")


class EqualWeight(BaselineStrategy):
    name = "equal_weight"
    family = "control"
    PARAMS = frozenset({"cash"})

    def _parse(self, p: dict[str, Any]) -> dict[str, Any]:
        return {"cash": param_num(p, "cash", 0.0, lo=0.0, hi=1.0)}

    def decide(self, view: PointInTimeView, holdings: HoldingsView) -> TargetWeights:
        elig = _eligible(view)
        cash = self.params["cash"]
        if not elig:
            return TargetWeights({i: 0.0 for i in view.instruments}, 1.0, "not_applicable", {"eligible": 0})
        w = (1.0 - cash) / len(elig)
        return TargetWeights({i: (w if i in elig else 0.0) for i in view.instruments}, cash, "not_applicable", {"eligible": len(elig)})


class BuyAndHold(BaselineStrategy):
    name = "buy_and_hold"
    family = "control"
    PARAMS = frozenset({"initial_weights", "cash"})

    def _parse(self, p: dict[str, Any]) -> dict[str, Any]:
        cash = param_num(p, "cash", 0.0, lo=0.0, hi=1.0)
        iw = p.get("initial_weights")
        if iw is not None:
            if not isinstance(iw, dict) or not iw:
                raise FinplanError.validation("initial_weights must map instrument ids to weights", pointer="/params/initial_weights")
            for k in iw:
                param_num(iw, k, 0.0, lo=0.0, hi=1.0)
            if abs(sum(float(v) for v in iw.values()) + cash - 1.0) > 1e-9:
                raise FinplanError.validation("initial_weights plus cash must sum to 1", pointer="/params/initial_weights")
            iw = {str(k): float(v) for k, v in sorted(iw.items())}
        return {"cash": cash, "initial_weights": iw}

    def decide(self, view: PointInTimeView, holdings: HoldingsView) -> TargetWeights:
        invested = any(abs(q) > 0 for q in holdings.shares.values())
        if invested:
            return hold(view, holdings, "no_effect", reason="buy_and_hold_holds")
        iw = self.params["initial_weights"]
        if iw is not None:
            unknown = sorted(set(iw) - set(view.instruments))
            if unknown:
                raise FinplanError.validation("initial_weights names instruments outside the evaluation universe", pointer="/params/initial_weights", instruments=unknown)
            return TargetWeights({i: iw.get(i, 0.0) for i in view.instruments}, self.params["cash"], "not_applicable", {"initial": True})
        elig = _eligible(view)
        if not elig:
            return TargetWeights({i: 0.0 for i in view.instruments}, 1.0, "not_applicable", {"eligible": 0})
        w = (1.0 - self.params["cash"]) / len(elig)
        return TargetWeights({i: (w if i in elig else 0.0) for i in view.instruments}, self.params["cash"], "not_applicable", {"initial": True, "eligible": len(elig)})
