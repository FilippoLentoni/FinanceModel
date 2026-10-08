"""The strategy interface every strategy family implements (spec paper-execution-simulator, SIM-02).

A strategy receives only the :class:`~finplan_model.sim.market.PointInTimeView` of one decision
plus the current simulated :class:`~finplan_model.sim.market.HoldingsView`, and returns **target
weights** per instrument plus cash. The simulator, never the strategy, turns targets into trades.

Accepted return shapes (normalized by :func:`coerce_target`):

* :class:`TargetWeights`;
* ``{"weights": {"SPY": 0.6}, "cash": 0.4}`` (optionally ``"solution_status"``);
* the contract allocation shape ``{"weights": [{"instrument_id": "SPY", "weight": 0.6}], "cash_weight": 0.4}``.

Anything order-shaped (``orders``, ``quantity``, ``shares``, ``side``, ``trades``, ``notional`` ...)
fails the run with ``VALIDATION_FAILED``. A decision whose ``solution_status`` is ``infeasible``
or ``unbounded`` is recorded and the simulator applies the configured fallback (hold current
weights) for that decision (spec baseline-strategies, "Optimizer outcome reporting").
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol, runtime_checkable

from finplan_model.core.errors import FinplanError

from .market import HoldingsView, PointInTimeView

__all__ = [
    "DECISION_SOLUTION_STATUSES",
    "FALLBACK_STATUSES",
    "NO_EFFECT",
    "ReplayStrategy",
    "Strategy",
    "TargetWeights",
    "aggregate_solution_status",
    "coerce_target",
]

DECISION_SOLUTION_STATUSES = ("optimal", "feasible", "infeasible", "unbounded", "no_effect", "not_applicable")
FALLBACK_STATUSES = ("infeasible", "unbounded")
#: A decision that asks for no change: the simulator keeps holdings and places no orders.
NO_EFFECT = "no_effect"
_ORDER_KEYS = frozenset({"orders", "order", "quantity", "quantities", "qty", "shares", "side", "sides", "trades", "trade", "notional", "notionals", "buy", "sell", "limit_price", "order_type"})


@dataclass(frozen=True)
class TargetWeights:
    weights: Mapping[str, float]
    cash: float
    solution_status: str = "optimal"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"weights": {k: float(v) for k, v in sorted(self.weights.items())}, "cash": float(self.cash), "solution_status": self.solution_status}
        if self.diagnostics:
            d["diagnostics"] = dict(self.diagnostics)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TargetWeights":
        return cls(dict(d["weights"]), float(d["cash"]), str(d.get("solution_status", "optimal")), dict(d.get("diagnostics") or {}))


@runtime_checkable
class Strategy(Protocol):
    #: registry name of the strategy (for example ``equal_weight``)
    name: str

    def decide(self, view: PointInTimeView, holdings: HoldingsView) -> TargetWeights | Mapping[str, Any]: ...


def _order_shaped(node: Any) -> bool:
    if isinstance(node, Mapping):
        return any(isinstance(k, str) and k.lower() in _ORDER_KEYS for k in node) or any(_order_shaped(v) for v in node.values())
    if isinstance(node, (list, tuple)):
        return any(_order_shaped(v) for v in node)
    return False


def _finite(value: Any, pointer: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise FinplanError.validation("strategy output weights must be finite numbers", pointer=pointer)
    return float(value)


def coerce_target(output: Any, universe: tuple[str, ...]) -> TargetWeights:
    """Normalize a strategy output to :class:`TargetWeights`, rejecting order-shaped outputs."""
    if isinstance(output, TargetWeights):
        raw: Any = output.to_dict()
    elif isinstance(output, Mapping):
        raw = output
    else:
        raise FinplanError.validation("strategy must return target weights (a mapping with weights and cash)", pointer="")
    if _order_shaped({k: v for k, v in raw.items() if k != "diagnostics"}):
        raise FinplanError.validation("strategy returned orders or quantities instead of target weights; the simulator produces trades", pointer="")
    if "weights" not in raw:
        raise FinplanError.validation("strategy output lacks weights", pointer="/weights")
    w_raw = raw["weights"]
    weights: dict[str, float] = {}
    if isinstance(w_raw, Mapping):
        for iid, v in w_raw.items():
            weights[str(iid)] = _finite(v, f"/weights/{iid}")
    elif isinstance(w_raw, list):
        for n, item in enumerate(w_raw):
            if not isinstance(item, Mapping) or set(item) - {"instrument_id", "weight"}:
                raise FinplanError.validation("allocation items must be {instrument_id, weight}", pointer=f"/weights/{n}")
            weights[str(item["instrument_id"])] = _finite(item.get("weight"), f"/weights/{n}/weight")
    else:
        raise FinplanError.validation("weights must be a mapping or an allocation list", pointer="/weights")
    unknown = sorted(set(weights) - set(universe))
    if unknown:
        raise FinplanError.validation("strategy output names instruments outside the evaluation universe", pointer="/weights", instruments=unknown[:10])
    cash_key = "cash" if "cash" in raw else "cash_weight" if "cash_weight" in raw else None
    cash = _finite(raw[cash_key], f"/{cash_key}") if cash_key else 1.0 - sum(weights.values())
    status = str(raw.get("solution_status", "optimal"))
    if status not in DECISION_SOLUTION_STATUSES:
        raise FinplanError.validation("unknown solution_status in strategy output", pointer="/solution_status")
    full = {i: weights.get(i, 0.0) for i in universe}
    diag = raw.get("diagnostics") if isinstance(raw.get("diagnostics"), Mapping) else {}
    return TargetWeights(full, cash, status, dict(diag or {}))


def aggregate_solution_status(statuses: list[str]) -> str:
    """Run-level ``solution_status`` from decision-level statuses.

    All infeasible -> ``infeasible``; all unbounded -> ``unbounded``; some decisions fell back ->
    ``feasible`` (spec: 2 of 40 infeasible gives ``feasible``); all ``optimal`` -> ``optimal``;
    no decision -> ``no_effect``; otherwise ``feasible``.

    ``no_effect`` decisions (holds) are neutral when other statuses are present, and so are
    ``not_applicable`` ones (controls) next to optimizer statuses: buy-and-hold (one
    ``not_applicable`` target, then ``no_effect`` holds) reports ``not_applicable``, and an
    optimizer that held while it lacked history and then solved every decision reports ``optimal``.
    """
    if not statuses:
        return "no_effect"
    s = set(statuses)
    if len(s) > 1:
        s.discard(NO_EFFECT)
    if len(s) > 1:
        s.discard("not_applicable")
    if s == {"infeasible"}:
        return "infeasible"
    if s == {"unbounded"}:
        return "unbounded"
    if s <= {"infeasible", "unbounded"}:
        return "infeasible"
    if s == {"optimal"}:
        return "optimal"
    if s == {"not_applicable"}:
        return "not_applicable"
    if s <= {"no_effect"}:
        return "no_effect"
    return "feasible"


class ReplayStrategy:
    """Replays stored strategy outputs by decision session (SIM-08 replay)."""

    def __init__(self, name: str, outputs: Mapping[date | str, Mapping[str, Any] | TargetWeights]) -> None:
        self.name = name
        self._outputs = {(d if isinstance(d, str) else d.isoformat()): o for d, o in outputs.items()}

    def decide(self, view: PointInTimeView, holdings: HoldingsView) -> TargetWeights | Mapping[str, Any]:
        key = view.decision_session.isoformat()
        if key not in self._outputs:
            raise FinplanError.validation("replay has no stored output for this decision", pointer="/decisions", decision_session=key)
        return self._outputs[key]
