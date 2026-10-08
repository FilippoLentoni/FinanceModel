"""Strategy lookup for job containers and submission validation.

The strategy implementations belong to ``finplan_model.strategies`` (task group 4). This module
only resolves a strategy *name* from an experiment configuration to a
:class:`~finplan_model.sim.strategy.Strategy` instance, in this order:

1. factories registered here with :func:`register_strategy` (tests, later strategy families);
2. the strategies package: ``finplan_model.strategies.build_strategy(name, params, constraints=...)``
   (task group 4; the simulation's constraint set is bound so optimizers optimize under the same
   constraints the simulator enforces), else ``get_strategy`` or a ``BASELINES`` / ``STRATEGIES`` /
   ``REGISTRY`` mapping of ``name -> factory(params)``.

:func:`known_strategies` lists the names submissions may use (``VALIDATION_FAILED`` for any other):
the registered and package names, plus the phase 1 names announced by the baseline-strategies spec
(:data:`PHASE1_STRATEGIES`) so validation does not depend on import order.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Mapping
from typing import Any

from finplan_model.core.errors import FinplanError
from finplan_model.sim.strategy import Strategy

__all__ = ["CONTROLS", "PHASE1_STRATEGIES", "known_strategies", "register_strategy", "resolve_strategy", "strategy_params", "unregister_strategy"]

#: Controls every benchmark includes (spec baseline-strategies, "Control strategies").
CONTROLS = ("cash", "buy_and_hold", "equal_weight")
#: Phase 1 strategy names (controls plus the classical optimizers).
PHASE1_STRATEGIES = (*CONTROLS, "min_variance", "mean_variance", "scenario_cvar")

_FACTORIES: dict[str, Callable[[Mapping[str, Any]], Strategy]] = {}


def register_strategy(name: str, factory: Callable[[Mapping[str, Any]], Strategy]) -> None:
    _FACTORIES[name] = factory


def unregister_strategy(name: str) -> None:
    _FACTORIES.pop(name, None)


def _package() -> Any:
    try:
        return importlib.import_module("finplan_model.strategies")
    except ImportError:  # pragma: no cover - the package always exists
        return None


_REGISTRY_ATTRS = ("BASELINES", "STRATEGIES", "REGISTRY")


def _package_names(pkg: Any) -> set[str]:
    names: set[str] = set()
    for attr in _REGISTRY_ATTRS:
        reg = getattr(pkg, attr, None)
        if isinstance(reg, Mapping):
            names |= {str(k) for k in reg}
    fn = getattr(pkg, "strategy_names", None) or getattr(pkg, "available_strategies", None)
    if callable(fn):
        try:
            names |= {str(n) for n in fn()}
        except Exception:  # noqa: BLE001 - a broken hook must not break validation
            pass
    return names


def known_strategies() -> set[str]:
    pkg = _package()
    return set(PHASE1_STRATEGIES) | set(_FACTORIES) | (_package_names(pkg) if pkg is not None else set())


def strategy_params(name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Strategy parameters from the experiment configuration (finance payload), restricted to the
    parameters the strategy accepts (``PARAMS`` of the registered class, when it declares them):
    ``lookback_days`` -> ``lookback``, ``risk_aversion`` -> ``risk_aversion``."""
    raw: dict[str, Any] = {}
    if "lookback_days" in payload:
        raw["lookback"] = payload["lookback_days"]
    if "risk_aversion" in payload:
        raw["risk_aversion"] = payload["risk_aversion"]
    pkg = _package()
    for attr in _REGISTRY_ATTRS:
        reg = getattr(pkg, attr, None) if pkg is not None else None
        cls = reg.get(name) if isinstance(reg, Mapping) else None
        allowed = getattr(cls, "PARAMS", None)
        if allowed is not None:
            return {k: v for k, v in raw.items() if k in allowed}
    return raw


def resolve_strategy(name: str, params: Mapping[str, Any] | None = None, *, constraints: Any = None) -> Strategy:
    """A strategy instance bound to the simulation's constraint set (when the implementation takes one)."""
    params = dict(params or {})
    if name in _FACTORIES:
        return _FACTORIES[name](params)
    pkg = _package()
    if pkg is not None:
        for attr in ("build_strategy", "get_strategy"):
            fn = getattr(pkg, attr, None)
            if callable(fn):
                try:
                    sig = inspect.signature(fn)
                    takes_constraints = "constraints" in sig.parameters
                except (TypeError, ValueError):  # pragma: no cover
                    takes_constraints = False
                return fn(name, params, constraints=constraints) if takes_constraints else fn(name, params)
        for attr in _REGISTRY_ATTRS:
            reg = getattr(pkg, attr, None)
            if isinstance(reg, Mapping) and name in reg:
                return reg[name](params)
    if name in PHASE1_STRATEGIES:
        raise FinplanError.dependency_unavailable("the strategy implementation is not available in this image", retryable=False, strategy=name)
    raise FinplanError.validation("unknown strategy", pointer="/configuration/payload/strategy")
