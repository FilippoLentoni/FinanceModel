"""Constraint validation, projection and rejection (SIM-03).

The constraint set (:class:`~finplan_model.sim.config.ConstraintSet`) covers the long-only flag,
per-instrument weight bounds, minimum (and maximum) cash, the per-rebalance turnover cap and
weights summing to 1 with cash. A violating target is, per ``constraint_policy``:

* ``project`` - replaced by the nearest feasible target: the exact Euclidean projection of
  ``(weights, cash)`` onto ``{lb <= x <= ub, sum(x) = 1}`` (cash is one more coordinate with bounds
  ``[min_cash, max_cash]``), then, if the turnover cap is still exceeded, moved along the segment
  from the current weights toward the projected target until turnover equals the cap. The
  violations and the projection distance are recorded;
* ``reject`` - discarded: current holdings are kept for that rebalance and the rejection is recorded.

If the constraint set itself is infeasible (bounds cannot sum to 1), or the turnover-limited move
cannot reach a feasible point, the decision is rejected (hold current) and recorded.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .config import ConstraintSet

__all__ = ["ConstraintOutcome", "apply_constraints", "check_violations", "project_box_simplex", "turnover"]

CASH = "__cash__"


@dataclass(frozen=True)
class ConstraintOutcome:
    action: str  # accepted | projected | rejected
    weights: dict[str, float] | None  # final target (None when rejected)
    cash: float | None
    violations: list[dict[str, Any]] = field(default_factory=list)
    projection_distance: float = 0.0
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"action": self.action, "violations": self.violations, "projection_distance": self.projection_distance}
        if self.weights is not None:
            d["final_weights"] = {k: self.weights[k] for k in sorted(self.weights)}
            d["final_cash"] = self.cash
        if self.reason:
            d["reason"] = self.reason
        return d


def turnover(target: Mapping[str, float], current: Mapping[str, float]) -> float:
    """One-way turnover: sum over instruments of |target - current| (cash excluded)."""
    keys = set(target) | set(current)
    return float(sum(abs(target.get(k, 0.0) - current.get(k, 0.0)) for k in sorted(keys)))


def check_violations(weights: Mapping[str, float], cash: float, cs: ConstraintSet, current: Mapping[str, float] | None, tol: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for iid in sorted(weights):
        w = weights[iid]
        lo, hi = cs.bounds(iid)
        if cs.long_only and w < -tol:
            out.append({"constraint": "long_only", "instrument_id": iid, "value": w, "limit": 0.0})
        elif w < lo - tol:
            out.append({"constraint": "min_weight", "instrument_id": iid, "value": w, "limit": lo})
        if w > hi + tol:
            out.append({"constraint": "max_weight", "instrument_id": iid, "value": w, "limit": hi})
    if cash < cs.min_cash - tol:
        out.append({"constraint": "min_cash", "value": cash, "limit": cs.min_cash})
    if cash > cs.max_cash + tol:
        out.append({"constraint": "max_cash", "value": cash, "limit": cs.max_cash})
    total = sum(weights.values()) + cash
    if abs(total - 1.0) > tol:
        out.append({"constraint": "sum_to_one", "value": total, "limit": 1.0})
    if cs.max_turnover is not None and current is not None:
        t = turnover(weights, current)
        if t > cs.max_turnover + tol:
            out.append({"constraint": "max_turnover", "value": t, "limit": cs.max_turnover})
    return out


def project_box_simplex(y: Mapping[str, float], lb: Mapping[str, float], ub: Mapping[str, float]) -> dict[str, float] | None:
    """Exact Euclidean projection of ``y`` onto ``{lb <= x <= ub, sum(x) = 1}``; None when empty.

    ``g(tau) = sum(clip(y - tau, lb, ub))`` is continuous, piecewise linear and non-increasing; the
    solution is ``x = clip(y - tau*, lb, ub)`` with ``g(tau*) = 1``, found exactly between sorted
    breakpoints.
    """
    keys = sorted(y)
    if any(lb[k] > ub[k] for k in keys) or sum(lb[k] for k in keys) > 1.0 + 1e-12 or sum(ub[k] for k in keys) < 1.0 - 1e-12:
        return None

    def g(tau: float) -> float:
        return sum(min(max(y[k] - tau, lb[k]), ub[k]) for k in keys)

    bps = sorted({y[k] - ub[k] for k in keys} | {y[k] - lb[k] for k in keys})
    lo_tau, hi_tau = bps[0], bps[-1]
    tau = lo_tau
    if g(lo_tau) <= 1.0:
        tau = lo_tau
    elif g(hi_tau) >= 1.0:
        tau = hi_tau
    else:
        for a, b in zip(bps, bps[1:]):
            ga, gb = g(a), g(b)
            if ga >= 1.0 >= gb:
                tau = a if ga == gb else a + (ga - 1.0) * (b - a) / (ga - gb)
                break
    x = {k: min(max(y[k] - tau, lb[k]), ub[k]) for k in keys}
    # Remove floating residue on a free coordinate so the sum is 1 to machine precision.
    resid = 1.0 - sum(x.values())
    if resid:
        for k in keys:
            if lb[k] < x[k] + resid < ub[k] or (lb[k] <= x[k] + resid <= ub[k]):
                x[k] += resid
                break
    return x


def _distance(a_w: Mapping[str, float], a_c: float, b_w: Mapping[str, float], b_c: float) -> float:
    keys = set(a_w) | set(b_w)
    return math.sqrt(sum((a_w.get(k, 0.0) - b_w.get(k, 0.0)) ** 2 for k in sorted(keys)) + (a_c - b_c) ** 2)


def apply_constraints(weights: Mapping[str, float], cash: float, cs: ConstraintSet, *, current: Mapping[str, float], current_cash: float, policy: str, tol: float) -> ConstraintOutcome:
    proposed = {k: float(v) for k, v in weights.items()}
    violations = check_violations(proposed, cash, cs, current, tol)
    if not violations:
        return ConstraintOutcome("accepted", proposed, float(cash))
    if policy == "reject":
        return ConstraintOutcome("rejected", None, None, violations, reason="constraint_violation")
    y = {**proposed, CASH: float(cash)}
    lb = {k: cs.bounds(k)[0] for k in proposed} | {CASH: cs.min_cash}
    ub = {k: cs.bounds(k)[1] for k in proposed} | {CASH: cs.max_cash}
    x = project_box_simplex(y, lb, ub)
    if x is None:
        return ConstraintOutcome("rejected", None, None, violations, reason="constraint_set_infeasible")
    new_cash = x.pop(CASH)
    if cs.max_turnover is not None:
        t = turnover(x, current)
        if t > cs.max_turnover + tol:
            alpha = cs.max_turnover / t
            keys = set(x) | set(current)
            x = {k: current.get(k, 0.0) + alpha * (x.get(k, 0.0) - current.get(k, 0.0)) for k in sorted(keys) if k in proposed or k in current}
            new_cash = current_cash + alpha * (new_cash - current_cash)
            remaining = [v for v in check_violations(x, new_cash, cs, current, max(tol, 1e-9)) if v["constraint"] != "max_turnover" or v["value"] > cs.max_turnover + 1e-9]
            if remaining:
                return ConstraintOutcome("rejected", None, None, violations, reason="turnover_projection_infeasible")
    return ConstraintOutcome("projected", x, new_cash, violations, _distance(proposed, cash, x, new_cash))
