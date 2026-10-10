"""Bounded deterministic convex solves and explicitly defined exact Shapley games.

Objective values are per chosen horizon. Mean/variance use stationary independent-session
scaling, not a calibrated forecast. Historical CVaR uses horizon-scaled historical scenarios.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linprog, minimize

from finplan_model.core.errors import FinplanError

DEFAULTS = {
    "lookback_days": 60,
    "risk_aversion": 2.0,
    "cvar_alpha": 0.9,
    "max_weight": 0.6,
    "cash_weight": 0.0,
    "horizon_sessions": 21,
    "turnover_penalty": 0.001,
}
BOUNDS = {
    "lookback_days": (20, 252),
    "risk_aversion": (1e-8, 100),
    "cvar_alpha": (0.8, 0.99),
    "max_weight": (0.2, 1),
    "cash_weight": (0, 0.5),
    "horizon_sessions": (1, 63),
    "turnover_penalty": (0, 1),
}


def settings(raw=None):
    if raw is None:
        raw = {}
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise FinplanError.validation(
            "unsupported classical setting", pointer="/settings"
        )
    out = {**DEFAULTS, **raw}
    for k, v in out.items():
        lo, hi = BOUNDS[k]
        if (
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or not lo <= v <= hi
        ):
            raise FinplanError.validation(
                "classical setting outside its finite bounds", pointer="/settings/" + k
            )
        if k in ("lookback_days", "horizon_sessions") and int(v) != v:
            raise FinplanError.validation(
                "setting must be an integer", pointer="/settings/" + k
            )
    return out


def cvar(losses, alpha):
    """Exact discrete expected shortfall including fractional quantile mass."""
    x = np.sort(np.asarray(losses, dtype=float))[::-1]
    mass = len(x) * (1 - alpha)
    whole = math.floor(mass + 1e-12)
    part = max(0.0, mass - whole)
    return float(
        (x[:whole].sum() + (part * x[whole] if part and whole < len(x) else 0.0)) / mass
    )


def metrics(inputs, weights):
    w = np.asarray(weights, dtype=float)
    cfg = inputs["settings"]
    h = cfg["horizon_sessions"]
    cov = np.asarray(inputs["covariance"])
    mu = np.asarray(inputs["expected_returns"])
    scenarios = np.asarray(inputs["scenarios"])
    return {
        "estimated_return": float(h * mu @ w),
        "variance": float(h * w @ cov @ w),
        "cvar_loss": cvar(-h * scenarios @ w, cfg["cvar_alpha"]),
        "turnover": float(np.abs(w - np.asarray(inputs["current_weights"])).sum()),
    }


def objective(inputs, weights):
    m, cfg = metrics(inputs, weights), inputs["settings"]
    risk = m["cvar_loss"] if inputs["algorithm"] == "cvar" else m["variance"]
    gain = m["estimated_return"] if inputs["algorithm"] == "mean_variance" else 0.0
    scale = cfg["risk_aversion"] if inputs["algorithm"] == "mean_variance" else 1.0
    return float(gain - scale * risk - cfg["turnover_penalty"] * m["turnover"])


def violations(inputs, weights):
    w = np.asarray(weights)
    cfg = inputs["settings"]
    out = []
    if np.any(w < -1e-7):
        out.append("negative_weight")
    if np.any(w > cfg["max_weight"] + 1e-7):
        out.append("max_weight")
    cash = 1 - float(w.sum())
    if cash < -1e-7:
        out.append("negative_cash")
    if abs(cash - cfg["cash_weight"]) > 1e-7:
        out.append("fixed_cash_constraint")
    return out


def solve(inputs, *, force_keep=None):
    n = len(inputs["instruments"])
    cfg = inputs["settings"]
    total = 1 - cfg["cash_weight"]
    bounds = [(0.0, cfg["max_weight"])] * n
    if force_keep is not None:
        v = float(inputs["current_weights"][force_keep])
        if not 0 <= v <= cfg["max_weight"]:
            return {
                "status": "infeasible",
                "reason": "unchanged_position_violates_bounds",
            }
        bounds[force_keep] = (v, v)
    if (
        sum(x[0] for x in bounds) > total + 1e-8
        or sum(x[1] for x in bounds) < total - 1e-8
    ):
        return {"status": "infeasible", "reason": "constraint_set_infeasible"}
    current = np.asarray(inputs["current_weights"])
    h = cfg["horizon_sessions"]
    # Epigraph variables make L1 turnover and historical CVaR exactly convex.
    s = len(inputs["scenarios"]) if inputs["algorithm"] == "cvar" else 0
    nv = 2 * n + (1 + s if s else 0)
    eq = np.zeros((1, nv))
    eq[0, :n] = 1
    aub = np.zeros((2 * n + s, nv))
    bub = np.zeros(2 * n + s)
    for k in range(n):
        aub[k, k], aub[k, n + k], bub[k] = 1.0, -1.0, current[k]
        aub[n + k, k], aub[n + k, n + k], bub[n + k] = -1.0, -1.0, -current[k]
    cost = np.zeros(nv)
    cost[n : 2 * n] = cfg["turnover_penalty"]
    allbounds = bounds + [(0.0, None)] * n
    if s:
        cost[2 * n] = 1
        cost[2 * n + 1 :] = 1 / ((1 - cfg["cvar_alpha"]) * s)
        aub[2 * n :, :n] = -h * np.asarray(inputs["scenarios"])
        aub[2 * n :, 2 * n] = -1
        aub[2 * n + np.arange(s), 2 * n + 1 + np.arange(s)] = -1
        allbounds += [(None, None)] + [(0.0, None)] * s
        res = linprog(
            cost,
            A_ub=aub,
            b_ub=bub,
            A_eq=eq,
            b_eq=[total],
            bounds=allbounds,
            method="highs",
        )
        solver = "scipy_highs"
    else:
        cov = h * np.asarray(inputs["covariance"])
        mu = (
            h * np.asarray(inputs["expected_returns"])
            if inputs["algorithm"] == "mean_variance"
            else np.zeros(n)
        )
        ra = cfg["risk_aversion"] if inputs["algorithm"] == "mean_variance" else 1.0
        # Feasible point through bounded water filling, independent of previous allocation.
        w = np.array([x[0] for x in bounds])
        for _ in range(n + 1):
            room = np.array([b[1] for b in bounds]) - w
            active = room > 1e-12
            if not active.any():
                break
            w += np.minimum(room, max(0.0, total - w.sum()) / active.sum()) * active
        x0 = np.r_[w, np.abs(w - current)]
        scale = max(1e-6, float(np.max(np.diag(cov))))

        def f(x):
            return (ra * x[:n] @ cov @ x[:n] - mu @ x[:n] + cost @ x) / scale

        def jac(x):
            return (np.r_[2 * ra * cov @ x[:n] - mu, np.zeros(n)] + cost) / scale

        res = minimize(
            f,
            x0,
            jac=jac,
            bounds=allbounds,
            constraints=[
                {"type": "eq", "fun": lambda x: eq @ x - total, "jac": lambda x: eq},
                {"type": "ineq", "fun": lambda x: bub - aub @ x, "jac": lambda x: -aub},
            ],
            method="SLSQP",
            options={"ftol": 1e-11, "maxiter": 500},
        )
        solver = "scipy_slsqp"
    if not res.success or res.x is None or violations(inputs, np.asarray(res.x[:n])):
        return {
            "status": "infeasible",
            "reason": "solver_not_certified",
            "solver": solver,
            "solver_status": int(res.status),
        }
    w = np.asarray(res.x[:n])
    w[np.abs(w) < 1e-12] = 0
    return {
        "status": "optimal",
        "weights": w.tolist(),
        "cash_weight": float(cfg["cash_weight"]),
        "objective": objective(inputs, w),
        "metrics": metrics(inputs, w),
        "solver": solver,
    }


def exact_shapley(names, value):
    """Every coalition evaluated once, max 5 dimensions (32 coalition values)."""
    n = len(names)
    if not 1 <= n <= 5:
        raise ValueError("exact Shapley is bounded to 1..5 dimensions")
    vals = {mask: np.asarray(value(mask), dtype=float) for mask in range(1 << n)}
    if any(not np.isfinite(v).all() for v in vals.values()):
        raise ValueError("nonfinite coalition value")
    phi = []
    for i in range(n):
        contribution = np.zeros_like(vals[0])
        for mask in range(1 << n):
            if mask & (1 << i):
                continue
            size = mask.bit_count()
            coeff = (
                math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            )
            contribution += coeff * (vals[mask | (1 << i)] - vals[mask])
        phi.append(contribution)
    residual = vals[(1 << n) - 1] - vals[0] - sum(phi)
    if np.max(np.abs(residual)) > 1e-8:
        raise ValueError("Shapley reconciliation failed")

    def output(x):
        return float(x) if x.ndim == 0 else x.tolist()

    return {
        "method": "exact_shapley",
        "groups": names,
        "baseline": output(vals[0]),
        "final": output(vals[(1 << n) - 1]),
        "contributions": [
            {"group": name, "value": output(v)} for name, v in zip(names, phi)
        ],
        "reconciliation_residual": output(residual),
        "coalition_count": len(vals),
        "tolerance": 1e-8,
    }


def explain(inputs, solution):
    current = np.asarray(inputs["current_weights"])
    target = np.asarray(solution["weights"])
    delta = target - current
    n = len(current)
    bad = []

    def game(mask):
        hybrid = current + delta * np.array([bool(mask & (1 << i)) for i in range(n)])
        flags = violations(inputs, hybrid)
        if flags:
            bad.append({"coalition": mask, "violations": flags})
        return objective(inputs, hybrid)

    shapley = exact_shapley(inputs["instruments"], game)
    shapley.update(
        {
            "game": "objective of original weights with only coalition instrument trades applied; residual funded by cash",
            "baseline": objective(inputs, current),
            "objective_units": "horizon objective utility (decimal return/variance units as declared)",
            "hybrid_constraint_handling": "diagnostic algebraic hybrids; infeasible hybrids are disclosed and are not executable plans",
            "infeasible_hybrids": bad,
        }
    )
    forced = []
    for i, instrument in enumerate(inputs["instruments"]):
        alt = solve(inputs, force_keep=i)
        row = {
            "instrument_id": instrument,
            "status": alt["status"],
            "constraint": "keep original instrument weight; optimize remaining holdings under original constraints",
        }
        if alt["status"] == "optimal":
            row.update(
                {
                    "objective_loss_from_keep": solution["objective"]
                    - alt["objective"],
                    "counterfactual_target_weights": alt["weights"],
                }
            )
        else:
            row["reason"] = alt["reason"]
        forced.append(row)
    cov = np.asarray(inputs["covariance"])
    mrc = inputs["settings"]["horizon_sessions"] * target * (cov @ target)
    return {
        "objective_definition": {
            "min_variance": "-horizon variance - transaction-cost proxy",
            "mean_variance": "horizon historical mean return - risk_aversion*horizon variance - transaction-cost proxy",
            "cvar": "-historical horizon-scaled CVaR loss - transaction-cost proxy",
        }[inputs["algorithm"]],
        "objective_units": "decimal return utility; variance is squared decimal returns; risk aversion supplies their trade-off",
        "hold_objective": objective(inputs, current),
        "hold_constraint_violations": violations(inputs, current),
        "optimized_objective": solution["objective"],
        "objective_gain": solution["objective"] - objective(inputs, current),
        "shapley": shapley,
        "force_keep": forced,
        "risk_contributions": [
            {"instrument_id": name, "variance_contribution": float(mrc[i])}
            for i, name in enumerate(inputs["instruments"])
        ],
        "binding_constraints": ["fixed_cash_weight"]
        + [
            f"{name}:" + ("lower_bound" if target[i] < 1e-7 else "max_weight")
            for i, name in enumerate(inputs["instruments"])
            if target[i] < 1e-7 or target[i] > inputs["settings"]["max_weight"] - 1e-7
        ],
        "causal_market_explanation": {
            "status": "not_available",
            "reason": "mathematical attribution explains model decisions, not causal market/news effects",
        },
    }
