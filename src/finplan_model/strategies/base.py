"""Shared pieces of the baseline strategies (spec baseline-strategies).

Every baseline implements the simulator's strategy interface (``name`` + ``decide(view,
holdings)`` returning target weights plus cash) and therefore runs only through the common
evaluator. Parameters are validated up front (``VALIDATION_FAILED`` with a JSON pointer); the
canonical ``configuration`` (strategy name, parameter schema version and parameters) gives the
strategy's ``configuration_id``.

Decision statuses (contracts ``solution_status``):

* controls report ``not_applicable`` (nothing is optimized);
* optimizers report ``optimal`` (solver certified), ``infeasible`` or ``unbounded`` - the simulator
  then applies the configured fallback (hold current weights) for that decision - and the run
  still completes (``completion_status`` ``succeeded``);
* ``no_effect`` asks the simulator to keep current holdings without trading (buy-and-hold after its
  initial target; an optimizer that lacks the history its estimator needs).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from finplan_model.core.errors import FinplanError
from finplan_model.core.ids import configuration_id
from finplan_model.sim.config import ConstraintSet
from finplan_model.sim.constraints import project_box_simplex
from finplan_model.sim.market import HoldingsView, PointInTimeView
from finplan_model.sim.strategy import TargetWeights

__all__ = ["BaselineStrategy", "hold", "param_bool", "param_choice", "param_int", "param_num"]


def _ptr(*parts: Any) -> str:
    return "/params" + "".join("/" + str(p) for p in parts)


def param_num(p: Mapping[str, Any], key: str, default: float, *, lo: float | None = None, hi: float | None = None, lo_open: bool = False) -> float:
    v = p.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise FinplanError.validation(f"strategy parameter {key} must be a finite number", pointer=_ptr(key))
    v = float(v)
    if (lo is not None and (v < lo or (lo_open and v == lo))) or (hi is not None and v > hi):
        raise FinplanError.validation(f"strategy parameter {key} is out of range", pointer=_ptr(key), minimum=lo, maximum=hi)
    return v


def param_int(p: Mapping[str, Any], key: str, default: int, *, lo: int = 0, hi: int | None = None) -> int:
    v = p.get(key, default)
    if isinstance(v, bool) or not isinstance(v, int) or v < lo or (hi is not None and v > hi):
        raise FinplanError.validation(f"strategy parameter {key} must be an integer in range", pointer=_ptr(key), minimum=lo, maximum=hi)
    return v


def param_choice(p: Mapping[str, Any], key: str, choices: tuple[str, ...], default: str) -> str:
    v = p.get(key, default)
    if v not in choices:
        raise FinplanError.validation(f"strategy parameter {key} must be one of {', '.join(choices)}", pointer=_ptr(key))
    return str(v)


def param_bool(p: Mapping[str, Any], key: str, default: bool) -> bool:
    v = p.get(key, default)
    if not isinstance(v, bool):
        raise FinplanError.validation(f"strategy parameter {key} must be a boolean", pointer=_ptr(key))
    return v


def hold(view: PointInTimeView, holdings: HoldingsView, status: str, **diagnostics: Any) -> TargetWeights:
    """Keep current weights (the simulator places no order for ``no_effect``/``infeasible``/``unbounded``)."""
    w = holdings.weights
    weights = {i: float(w.get(i, 0.0)) for i in view.instruments}
    return TargetWeights(weights, float(holdings.cash_weight), status, diagnostics)


class BaselineStrategy:
    """Base class: parameter handling, identity and the constraint set the strategy optimizes under."""

    name: str = ""
    family: str = ""  # control | classical_optimizer
    param_schema_version: str = "1"
    makes_predictions: bool = False
    has_randomness: bool = False
    #: allowed parameter names (anything else is VALIDATION_FAILED)
    PARAMS: frozenset[str] = frozenset()

    def __init__(self, params: Mapping[str, Any] | None = None, *, constraints: ConstraintSet | None = None) -> None:
        p = dict(params or {})
        for k in sorted(p):
            if k not in self.PARAMS:
                raise FinplanError.validation(f"unknown parameter {k} for strategy {self.name}", pointer=_ptr(k))
        self.constraints = constraints or ConstraintSet()
        self.params = self._parse(p)

    def _parse(self, p: dict[str, Any]) -> dict[str, Any]:
        return p

    @property
    def configuration(self) -> dict[str, Any]:
        return {"strategy": self.name, "param_schema_version": self.param_schema_version, "params": dict(sorted(self.params.items()))}

    @property
    def configuration_id(self) -> str:
        return configuration_id(self.configuration)

    def describe(self) -> dict[str, Any]:
        return {"strategy": self.name, "family": self.family, "param_schema_version": self.param_schema_version, "makes_predictions": self.makes_predictions, "has_randomness": self.has_randomness, "compute_class": "cpu", "configuration_id": self.configuration_id}

    # ------------------------------------------------------------------ constraint helpers
    def bounds(self, instruments: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
        lb = np.array([self.constraints.bounds(i)[0] for i in instruments], dtype=float)
        ub = np.array([self.constraints.bounds(i)[1] for i in instruments], dtype=float)
        return lb, ub

    def invested_interval(self, lb: np.ndarray, ub: np.ndarray, cash_policy: str) -> tuple[float, float] | None:
        """Feasible range of the invested total ``sum(w)`` (None when the constraint set is infeasible).

        ``cash_policy`` ``min``: cash is held at ``min_cash`` (invested = 1 - min_cash);
        ``free``: cash may be anywhere in ``[min_cash, max_cash]``.
        """
        cs = self.constraints
        if cs.min_cash > cs.max_cash + 1e-12 or np.any(lb > ub + 1e-12):
            return None
        lo_c, hi_c = (cs.min_cash, cs.min_cash) if cash_policy == "min" else (cs.min_cash, cs.max_cash)
        lo = max(float(lb.sum()), 1.0 - hi_c)
        hi = min(float(ub.sum()), 1.0 - lo_c)
        if lo > hi + 1e-12:
            return None
        return lo, hi

    @staticmethod
    def feasible_start(lb: np.ndarray, ub: np.ndarray, total: float) -> np.ndarray:
        """Equal weights projected onto ``{lb <= w <= ub, sum(w) = total}`` (deterministic start point)."""
        n = len(lb)
        if total <= 0:
            return np.clip(np.zeros(n), lb, ub)
        keys = [str(k) for k in range(n)]
        x = project_box_simplex({k: 1.0 / n for k in keys}, {k: float(lb[i]) / total for i, k in enumerate(keys)}, {k: float(ub[i]) / total for i, k in enumerate(keys)})
        if x is None:
            return np.full(n, total / n)
        return np.array([x[k] * total for k in keys])
