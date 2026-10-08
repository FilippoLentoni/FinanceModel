"""Classical optimizers: minimum-variance, mean-variance and scenario-CVaR (BASE-02 to BASE-05).

All three optimize over the simulation's constraint set (per-instrument bounds, long-only, cash
bounds) using only point-in-time estimates (``view.returns``). Solvers are open source and
deterministic: SciPy SLSQP for the quadratic programs, SciPy HiGHS (``linprog``) for the CVaR
linear program (design FM-A4: no commercial solver).

Cash (``cash_policy``):

* ``min`` (default for minimum-variance and scenario-CVaR): cash is held at the constraint set's
  ``min_cash`` and the rest is invested - otherwise "minimize risk" would trivially pick 100% cash;
* ``free`` (default for mean-variance): cash is a decision variable in ``[min_cash, max_cash]``
  earning ``cash_return_annual`` (configure it equal to the simulation's cash rate).

Solver outcome mapping (``solution_status`` of the decision; :data:`SOLVER_STATUS_MAP`):

* the constraint set admits no portfolio (for example minimum cash 50% with a minimum invested
  weight of 60%) -> ``infeasible`` before any solve;
* SLSQP success -> ``optimal``; SLSQP stopped but its point satisfies every constraint ->
  ``feasible``; otherwise ``infeasible``;
* HiGHS 0 -> ``optimal``, 2 -> ``infeasible``, 3 -> ``unbounded``, 1 or 4 (iteration limit,
  numerical trouble) -> ``infeasible`` (no certified solution).

``infeasible`` and ``unbounded`` decisions return the current weights and the simulator applies
the hold-current-weights fallback; the run completes with ``completion_status`` ``succeeded``.
An optimizer without ``min_history`` return observations holds (``no_effect``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.optimize import linprog, minimize

from finplan_model.core.errors import FinplanError
from finplan_model.sim.market import HoldingsView, PointInTimeView
from finplan_model.sim.strategy import TargetWeights

from .base import BaselineStrategy, hold, param_choice, param_int, param_num
from .estimators import COVARIANCE_ESTIMATORS, MEAN_ESTIMATORS, estimate_covariance, estimate_mean, pit_returns

__all__ = ["MeanVariance", "MinVariance", "SOLVER_STATUS_MAP", "ScenarioCVaR", "generate_scenarios"]

SOLVER_STATUS_MAP = {
    "slsqp": {"success": "optimal", "stopped_feasible": "feasible", "stopped_infeasible": "infeasible"},
    "highs": {0: "optimal", 1: "infeasible", 2: "infeasible", 3: "unbounded", 4: "infeasible"},
}
SCENARIO_METHODS = ("historical", "bootstrap", "gaussian")
_TOL = 1e-8


def _r(x: float) -> float:
    return 0.0 if x == 0 else float(x)


class _Optimizer(BaselineStrategy):
    family = "classical_optimizer"
    default_cash_policy = "min"
    COMMON = frozenset({"lookback", "min_history", "cash_policy", "cash_return_annual", "periods_per_year"})

    def _common(self, p: Mapping[str, Any]) -> dict[str, Any]:
        lookback = param_int(p, "lookback", 60, lo=2, hi=5000)
        return {
            "lookback": lookback,
            "min_history": param_int(p, "min_history", min(lookback, 20), lo=2, hi=lookback),
            "cash_policy": param_choice(p, "cash_policy", ("min", "free"), self.default_cash_policy),
            "cash_return_annual": param_num(p, "cash_return_annual", 0.0, lo=-0.5, hi=1.0),
            "periods_per_year": param_int(p, "periods_per_year", 252, lo=1, hi=366),
        }

    def _cash_rate(self) -> float:
        return (1.0 + self.params["cash_return_annual"]) ** (1.0 / self.params["periods_per_year"]) - 1.0

    def _prepare(self, view: PointInTimeView, holdings: HoldingsView) -> tuple[np.ndarray | None, TargetWeights | None]:
        r = pit_returns(view, self.params["lookback"])
        if r.shape[0] < self.params["min_history"]:
            return None, hold(view, holdings, "no_effect", reason="insufficient_history", observations=int(r.shape[0]), min_history=self.params["min_history"])
        return r, None

    def _infeasible(self, view: PointInTimeView, holdings: HoldingsView, reason: str, **diag: Any) -> TargetWeights:
        return hold(view, holdings, "infeasible", reason=reason, **diag)

    def _finish(self, view: PointInTimeView, x: np.ndarray, cash: float, status: str, **diag: Any) -> TargetWeights:
        lb, ub = self.bounds(view.instruments)
        x = np.clip(x, lb, ub)
        if self.params["cash_policy"] == "min":
            total = 1.0 - self.constraints.min_cash
            if abs(x.sum() - total) > 1e-12:
                x = self.feasible_start(lb, ub, total) if x.sum() <= 0 else self._rescale(x, lb, ub, total)
            cash = 1.0 - float(x.sum())
        else:
            cash = float(np.clip(1.0 - x.sum(), self.constraints.min_cash, self.constraints.max_cash))
            if abs(x.sum() + cash - 1.0) > 1e-12:
                x = self._rescale(x, lb, ub, 1.0 - cash)
                cash = 1.0 - float(x.sum())
        weights = {i: _r(float(round(v, 15))) for i, v in zip(view.instruments, x)}
        return TargetWeights(weights, _r(float(round(cash, 15))), status, diag)

    def _rescale(self, x: np.ndarray, lb: np.ndarray, ub: np.ndarray, total: float) -> np.ndarray:
        from finplan_model.sim.constraints import project_box_simplex

        if total <= 0:
            return np.clip(np.zeros_like(x), lb, ub)
        keys = [str(k) for k in range(len(x))]
        p = project_box_simplex({k: float(x[i]) / total for i, k in enumerate(keys)}, {k: float(lb[i]) / total for i, k in enumerate(keys)}, {k: float(ub[i]) / total for i, k in enumerate(keys)})
        return x if p is None else np.array([p[k] * total for k in keys])

    # ------------------------------------------------------------------ quadratic programs
    def _solve_qp(self, view: PointInTimeView, holdings: HoldingsView, cov: np.ndarray, mu: np.ndarray, risk_aversion: float) -> TargetWeights:
        """min  risk_aversion * w'Σw - (mu'w + r_c * cash)   over the constraint set."""
        n = len(view.instruments)
        lb, ub = self.bounds(view.instruments)
        interval = self.invested_interval(lb, ub, self.params["cash_policy"])
        if interval is None:
            return self._infeasible(view, holdings, "constraint_set_infeasible")
        free = self.params["cash_policy"] == "free"
        rc = self._cash_rate()
        scale = float(np.mean(np.diag(cov))) or 1.0
        if free:
            q = np.zeros((n + 1, n + 1))
            q[:n, :n] = cov
            c = np.append(mu, rc)
            lbx = np.append(lb, self.constraints.min_cash)
            ubx = np.append(ub, self.constraints.max_cash)
            total = 1.0
        else:
            q, c, lbx, ubx = cov, mu, lb, ub
            total = 1.0 - self.constraints.min_cash
        x0 = self.feasible_start(lbx, ubx, total)

        def f(x: np.ndarray) -> float:
            return float(risk_aversion * x @ q @ x - c @ x) / scale

        def g(x: np.ndarray) -> np.ndarray:
            return (2.0 * risk_aversion * (q @ x) - c) / scale

        res = minimize(f, x0, jac=g, method="SLSQP", bounds=list(zip(lbx, ubx)), constraints=[{"type": "eq", "fun": lambda x: float(x.sum() - total), "jac": lambda x: np.ones_like(x)}], options={"ftol": 1e-12, "maxiter": 1000})
        x = np.asarray(res.x, dtype=float)
        x = np.where(np.abs(x - lbx) < 1e-10, lbx, np.where(np.abs(x - ubx) < 1e-10, ubx, x))
        ok = abs(x.sum() - total) <= 1e-7 and bool(np.all(x >= lbx - 1e-7)) and bool(np.all(x <= ubx + 1e-7))
        certified = ok and (bool(res.success) or _kkt_ok(x, g(x), lbx, ubx))
        status = SOLVER_STATUS_MAP["slsqp"]["success" if certified else "stopped_feasible" if ok else "stopped_infeasible"]
        if status == "infeasible":
            return self._infeasible(view, holdings, "solver_failed", solver="slsqp", solver_status=int(res.status))
        w = x[:n]
        cash = float(x[n]) if free else 1.0 - float(w.sum())
        var = float(w @ cov @ w)
        return self._finish(view, w, cash, status, solver="slsqp", solver_status=int(res.status), estimated_variance=var, estimated_return=float(mu @ w + rc * cash))


class MinVariance(_Optimizer):
    name = "min_variance"
    PARAMS = _Optimizer.COMMON | frozenset({"covariance_estimator", "ewma_halflife"})

    def _parse(self, p: dict[str, Any]) -> dict[str, Any]:
        return {**self._common(p), "covariance_estimator": param_choice(p, "covariance_estimator", COVARIANCE_ESTIMATORS, "ledoit_wolf"), "ewma_halflife": param_num(p, "ewma_halflife", 20.0, lo=1.0, hi=10000.0)}

    def decide(self, view: PointInTimeView, holdings: HoldingsView) -> TargetWeights:
        r, early = self._prepare(view, holdings)
        if early is not None:
            return early
        cov = estimate_covariance(r, self.params["covariance_estimator"], halflife=self.params["ewma_halflife"])
        return self._solve_qp(view, holdings, cov, np.zeros(len(view.instruments)), 1.0)


class MeanVariance(_Optimizer):
    name = "mean_variance"
    default_cash_policy = "free"
    PARAMS = _Optimizer.COMMON | frozenset({"risk_aversion", "covariance_estimator", "return_estimator", "ewma_halflife"})

    def _parse(self, p: dict[str, Any]) -> dict[str, Any]:
        return {
            **self._common(p),
            "risk_aversion": param_num(p, "risk_aversion", 5.0, lo=0.0, hi=1e9),
            "covariance_estimator": param_choice(p, "covariance_estimator", COVARIANCE_ESTIMATORS, "ledoit_wolf"),
            "return_estimator": param_choice(p, "return_estimator", MEAN_ESTIMATORS, "historical_mean"),
            "ewma_halflife": param_num(p, "ewma_halflife", 20.0, lo=1.0, hi=10000.0),
        }

    def decide(self, view: PointInTimeView, holdings: HoldingsView) -> TargetWeights:
        r, early = self._prepare(view, holdings)
        if early is not None:
            return early
        cov = estimate_covariance(r, self.params["covariance_estimator"], halflife=self.params["ewma_halflife"])
        mu = estimate_mean(r, self.params["return_estimator"], halflife=self.params["ewma_halflife"])
        return self._solve_qp(view, holdings, cov, mu, self.params["risk_aversion"])


def _kkt_ok(x: np.ndarray, grad: np.ndarray, lb: np.ndarray, ub: np.ndarray, tol: float = 1e-6) -> bool:
    """KKT conditions of ``min f(x) s.t. sum(x) = const, lb <= x <= ub`` for a convex ``f``.

    Optimal iff some multiplier ``nu`` has ``grad_i = nu`` on free coordinates, ``grad_i >= nu`` at a
    lower bound and ``grad_i <= nu`` at an upper bound (a certificate, used when SLSQP stops on a
    line-search message at an optimal vertex).
    """
    at_lb = np.abs(x - lb) <= 1e-9
    at_ub = np.abs(x - ub) <= 1e-9
    free = ~(at_lb | at_ub)
    scale = max(1.0, float(np.max(np.abs(grad))))
    lo = max([float(v) for v in grad[at_ub & ~at_lb]] + ([float(np.min(grad[free]))] if free.any() else []), default=-np.inf)
    hi = min([float(v) for v in grad[at_lb & ~at_ub]] + ([float(np.max(grad[free]))] if free.any() else []), default=np.inf)
    if free.any() and float(np.max(grad[free]) - np.min(grad[free])) > tol * scale:
        return False
    return lo <= hi + tol * scale


def _decision_seed(seed: int, session_iso: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{session_iso}".encode()).digest()[:8], "big")


def generate_scenarios(r: np.ndarray, method: str, n_scenarios: int, seed: int, session_iso: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Scenario matrix (``S x N`` per-session returns) from point-in-time history ``r``.

    ``historical`` uses every visible return row (no randomness); ``bootstrap`` resamples rows and
    ``gaussian`` draws from a normal with the sample mean and covariance, both with a per-decision
    seed derived from the configured ``seed`` and the decision session (same seed, same scenarios).
    """
    ds = _decision_seed(seed, session_iso)
    if method == "historical":
        scen = np.array(r, dtype=float)
    elif method == "bootstrap":
        rng = np.random.default_rng(ds)
        scen = r[rng.integers(0, r.shape[0], size=n_scenarios)]
    elif method == "gaussian":
        rng = np.random.default_rng(ds)
        cov = estimate_covariance(r, "sample")
        scen = rng.multivariate_normal(r.mean(axis=0), cov, size=n_scenarios, method="eigh")
    else:  # pragma: no cover - validated at construction
        raise FinplanError.validation("unknown scenario method", pointer="/params/scenario_method")
    scen = np.ascontiguousarray(scen, dtype=float)
    meta = {"scenario_method": method, "seed": seed, "decision_seed": str(ds), "n_scenarios": int(scen.shape[0]), "scenario_checksum": "sha256:" + hashlib.sha256(scen.tobytes()).hexdigest()}
    return scen, meta


class ScenarioCVaR(_Optimizer):
    name = "scenario_cvar"
    has_randomness = True
    PARAMS = _Optimizer.COMMON | frozenset({"alpha", "objective", "cvar_limit", "n_scenarios", "scenario_method", "seed"})

    def _parse(self, p: dict[str, Any]) -> dict[str, Any]:
        out = {
            **self._common(p),
            "alpha": param_num(p, "alpha", 0.95, lo=0.5, hi=0.999),
            "objective": param_choice(p, "objective", ("min_cvar", "max_return"), "min_cvar"),
            "n_scenarios": param_int(p, "n_scenarios", 500, lo=10, hi=20000),
            "scenario_method": param_choice(p, "scenario_method", SCENARIO_METHODS, "bootstrap"),
            "seed": param_int(p, "seed", 0, lo=0, hi=2**31 - 1),
        }
        if out["objective"] == "max_return":
            out["cvar_limit"] = param_num(p, "cvar_limit", 0.02, lo=0.0, hi=1.0)
        elif "cvar_limit" in p:
            raise FinplanError.validation("cvar_limit applies only to objective max_return", pointer="/params/cvar_limit")
        return out

    def with_seed(self, seed: int) -> "ScenarioCVaR":
        return ScenarioCVaR({**{k: v for k, v in self.params.items() if k in self.PARAMS}, "seed": seed}, constraints=self.constraints)

    def decide(self, view: PointInTimeView, holdings: HoldingsView) -> TargetWeights:
        r, early = self._prepare(view, holdings)
        if early is not None:
            return early
        n = len(view.instruments)
        lb, ub = self.bounds(view.instruments)
        if self.invested_interval(lb, ub, self.params["cash_policy"]) is None:
            return self._infeasible(view, holdings, "constraint_set_infeasible")
        scen, meta = generate_scenarios(r, self.params["scenario_method"], self.params["n_scenarios"], self.params["seed"], view.decision_session.isoformat())
        s = scen.shape[0]
        alpha = self.params["alpha"]
        free = self.params["cash_policy"] == "free"
        rc = self._cash_rate()
        nc = 1 if free else 0
        # variables: x (n), [cash], zeta, u (s)
        nv = n + nc + 1 + s
        k = 1.0 / ((1.0 - alpha) * s)
        cvar_row = np.zeros(nv)
        cvar_row[n + nc] = 1.0
        cvar_row[n + nc + 1 :] = k
        a_ub = np.zeros((s, nv))
        a_ub[:, :n] = -scen
        if free:
            a_ub[:, n] = -rc
        a_ub[:, n + nc] = -1.0
        a_ub[np.arange(s), n + nc + 1 + np.arange(s)] = -1.0
        b_ub = np.zeros(s)
        a_eq = np.zeros((1, nv))
        a_eq[0, : n + nc] = 1.0
        b_eq = np.array([1.0 if free else 1.0 - self.constraints.min_cash])
        bounds = [(float(lb[i]), float(ub[i])) for i in range(n)]
        if free:
            bounds.append((self.constraints.min_cash, self.constraints.max_cash))
        bounds += [(None, None)] + [(0.0, None)] * s
        mean = scen.mean(axis=0)
        if self.params["objective"] == "min_cvar":
            c = cvar_row
        else:
            c = np.zeros(nv)
            c[:n] = -mean
            if free:
                c[n] = -rc
            a_ub = np.vstack([a_ub, cvar_row])
            b_ub = np.append(b_ub, self.params["cvar_limit"])
        res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
        status = SOLVER_STATUS_MAP["highs"].get(int(res.status), "infeasible")
        if status != "optimal":
            return hold(view, holdings, status, reason="solver_" + status, solver="highs", solver_status=int(res.status), **meta)
        x = np.asarray(res.x, dtype=float)
        w = x[:n]
        cash = float(x[n]) if free else 1.0 - float(w.sum())
        cvar = float(cvar_row @ x)
        return self._finish(view, w, cash, status, solver="highs", solver_status=0, estimated_cvar=cvar, alpha=alpha, **meta)
