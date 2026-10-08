"""The gymnasium portfolio environment (spec rl-strategies, "Simulator-backed environment"; RL-02, RL-03).

The environment steps one **decision** at a time. Its accounting is the common simulator's
(:mod:`finplan_model.sim.engine`) for the long-only, cancel-unfilled configurations used here, so a
scripted policy gets the same NAV in the environment as in the common evaluator (RL-02 parity test):

* cash earns ``cash_rate_annual / periods_per_year`` (compounded) on every session after the first;
* a decision at the close of session ``d`` executes at session ``d + 1`` at the executable price
  (``next_open``: the open; ``next_close``: the close); orders are sized from the target weights and
  the value at the execution prices; sells execute first; each fill is capped at
  ``participation_cap x volume`` (remainder cancelled) and pays the proportional fee, the fixed fee,
  the half-spread and the linear-participation slippage; buys are scaled down when cash (after costs)
  would go negative; an instrument without a bar does not trade;
* targets pass through the simulator's constraint policy (:func:`finplan_model.sim.constraints.apply_constraints`);
* values are marked at every session close.

Rewards are computed from these simulated values only (net of every cost); see
:mod:`finplan_model.rl.spec` for the formula. The final evaluation of every trained policy still runs
through :func:`finplan_model.evaluate.evaluate`; the environment serves training and validation
checkpoint selection.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np

from finplan_model.sim import metrics as m
from finplan_model.sim.config import SimulationConfig
from finplan_model.sim.constraints import apply_constraints

from .arrays import PriceArrays, calendar_schedule
from .spec import EnvSpec, allocation_action, build_observation

__all__ = ["Book", "PortfolioEnv", "run_schedule", "schedule_metrics"]


@dataclass
class Book:
    """Simulated holdings (shares per instrument, in the arrays' sorted order) and cash."""

    shares: np.ndarray
    cash: float
    costs: float = 0.0
    traded: float = 0.0
    log: list[float] = field(default_factory=list)  # value at each marked session

    def value(self, prices: np.ndarray) -> float:
        return float(self.cash + float(self.shares @ prices))


class Accounting:
    """The simulator's execution and marking rules over :class:`PriceArrays` (see module docstring)."""

    def __init__(self, arrays: PriceArrays, config: SimulationConfig) -> None:
        self.a = arrays
        self.cfg = config
        self.per_period = (1.0 + config.cash_rate_annual) ** (1.0 / config.periods_per_year) - 1.0
        self.fee = config.fees.proportional_bps / 1e4
        self.fixed = float(config.fees.fixed_per_trade)
        self.spread = config.spread.half_spread_bps / 1e4
        self.slip_bps = config.slippage.coefficient_bps if config.slippage.model == "linear_participation" else 0.0
        self.cap = config.liquidity.participation_cap
        self.n = len(arrays.instruments)

    def weights(self, book: Book, k: int) -> tuple[np.ndarray, float]:
        px = self.a.close[k]
        v = book.value(px)
        if v <= 0:
            return np.zeros(self.n), 1.0
        return book.shares * px / v, book.cash / v

    def target(self, book: Book, k: int, w: np.ndarray, cash: float) -> tuple[np.ndarray, float] | None:
        """The constraint policy applied to a target at decision session ``k`` (None: rejected, hold)."""
        names = self.a.instruments
        cur_w, cur_c = self.weights(book, k)
        out = apply_constraints({i: float(w[j]) for j, i in enumerate(names)}, float(cash), self.cfg.constraints, current={i: float(cur_w[j]) for j, i in enumerate(names)}, current_cash=float(cur_c), policy=self.cfg.constraint_policy, tol=self.cfg.weight_tolerance)
        if out.weights is None:
            return None
        return np.array([out.weights[i] for i in names], dtype=float), float(out.cash or 0.0)

    def _rates(self, qty: float, volume: float) -> tuple[float, float]:
        participation = abs(qty) / volume if volume else 0.0
        return participation, self.slip_bps * participation / 1e4

    def execute(self, book: Book, k: int, target_w: np.ndarray) -> float:
        """Trade to ``target_w`` at session ``k``'s executable prices; returns the traded notional."""
        p = self.a.exec_price[k]
        vol = self.a.volume[k]
        has = self.a.has_bar[k]
        v_exec = book.cash + float(np.nansum(book.shares * p))
        planned: list[tuple[int, float]] = []
        for j in range(self.n):
            if not (p[j] > 0):
                continue
            q = target_w[j] * v_exec / p[j] - book.shares[j]
            if abs(q) * p[j] <= 1e-9:
                continue
            if not has[j]:
                continue  # no bar: unfilled (cancelled)
            if self.cap is not None:
                q = math.copysign(min(abs(q), self.cap * (vol[j] or 0.0)), q)
            planned.append((j, q))
        traded = 0.0
        for j, q in planned:
            if q < 0:
                traded += self._fill(book, j, q, p[j], vol[j])
        buys = [(j, q) for j, q in planned if q > 0]
        if buys:
            variable = 0.0
            for j, q in buys:
                _, slr = self._rates(q, vol[j])
                variable += q * p[j] * (1.0 + self.fee + self.spread + slr)
            fixed = self.fixed * len(buys)
            scale = 1.0
            if variable + fixed > book.cash:
                scale = max(0.0, min(1.0, (book.cash - fixed) / variable)) * (1.0 - 1e-12) if variable > 0 else 0.0
            for j, q in buys:
                if q * scale > 0:
                    traded += self._fill(book, j, q * scale, p[j], vol[j])
        book.traded += traded
        return traded

    def _fill(self, book: Book, j: int, q: float, price: float, volume: float) -> float:
        notional = abs(q) * price
        _, slr = self._rates(q, volume)
        cost = notional * self.fee + self.fixed + notional * self.spread + notional * slr
        book.cash -= q * price + cost
        book.shares[j] += q
        book.costs += cost
        return notional

    def hold_until(self, book: Book, start: int, stop: int, target_w: np.ndarray | None) -> list[float]:
        """Sessions ``start..stop`` (inclusive): interest, execution of ``target_w`` at ``start``, marks."""
        values: list[float] = []
        for k in range(start, stop + 1):
            book.cash += book.cash * self.per_period
            if k == start and target_w is not None:
                self.execute(book, k, target_w)
            v = book.value(self.a.close[k])
            values.append(v)
        book.log.extend(values)
        return values

    def observation(self, spec: EnvSpec, book: Book, k: int) -> np.ndarray:
        w, c = self.weights(book, k)
        closes = self.a.close[k - spec.window : k + 1]
        value = book.value(self.a.close[k])
        peak = max(book.log, default=value)
        drawdown = max(0.0, min(1.0, 1 - value / peak)) if peak > 0 else 0.0
        return build_observation(spec, closes, w, c, drawdown=drawdown)


Predict = Callable[[np.ndarray], Any]


def run_schedule(arrays: PriceArrays, config: SimulationConfig, spec: EnvSpec, i0: int, i1: int, decide: Callable[[Book, int], tuple[np.ndarray, float] | None], schedule: Sequence[int]) -> list[float]:
    """NAV over ``[i0, i1]`` starting all cash, deciding at ``schedule`` (the evaluator's rules).

    ``decide(book, k)`` returns target ``(weights, cash)`` or None (hold, no orders). Returns the
    value at every session ``i0..i1`` (the common evaluator's NAV series for the same decisions).
    """
    acc = Accounting(arrays, config)
    book = Book(np.zeros(acc.n), float(config.initial_cash))
    sched = sorted(k for k in schedule if i0 <= k < i1)
    values = [book.value(arrays.close[i0])]
    book.log.append(values[0])
    pos = i0
    pending: np.ndarray | None = None
    if sched and sched[0] == i0:
        pending = _decide_target(acc, book, i0, decide)
        sched = sched[1:]
    for nxt in [*sched, i1]:
        values += acc.hold_until(book, pos + 1, nxt, pending)
        pos = nxt
        pending = _decide_target(acc, book, nxt, decide) if nxt < i1 else None
    return values


def _decide_target(acc: Accounting, book: Book, k: int, decide: Callable[[Book, int], tuple[np.ndarray, float] | None]) -> np.ndarray | None:
    out = decide(book, k)
    if out is None:
        return None
    tgt = acc.target(book, k, out[0], out[1])
    return None if tgt is None else tgt[0]


def schedule_metrics(values: Sequence[float], config: SimulationConfig) -> dict[str, float | None]:
    vals = list(values)
    return {
        "sharpe": m.sharpe_ratio(vals, config.periods_per_year, config.cash_rate_annual),
        "net_return": m.cumulative_return(vals),
        "max_drawdown": m.max_drawdown(vals),
    }


class PortfolioEnv(gym.Env):
    """One decision per step; observation and action per :mod:`finplan_model.rl.spec`.

    ``mode="train"``: an episode covers ``[i0, i1]`` from a seeded random first decision in
    ``[i0 + window, i0 + window + step_sessions)`` (at least ``window`` sessions of history inside the
    window), then every ``step_sessions`` sessions. ``mode="eval"``: decisions at ``schedule``
    (the calendar rebalance sessions; history before ``i0`` may be observed, rewards start at ``i0``).
    """

    metadata = {"render_modes": []}

    def __init__(self, arrays: PriceArrays, spec: EnvSpec, config: SimulationConfig, *, i0: int, i1: int, mode: str = "train", schedule: Sequence[int] | None = None) -> None:
        super().__init__()
        if mode not in ("train", "eval"):
            raise ValueError(mode)
        self.arrays, self.env_spec, self.cfg = arrays, spec, config
        self.acc = Accounting(arrays, config)
        self.n = self.acc.n
        self.i0, self.i1, self.mode = i0, i1, mode
        first = i0 + spec.window
        if mode == "train" and first + 1 > i1:
            from finplan_model.core.errors import FinplanError

            raise FinplanError.validation("the training window is shorter than the observation window", pointer="/splits/train", window=spec.window)
        if mode == "eval":
            sched = list(schedule if schedule is not None else calendar_schedule(arrays, i0, i1, spec.decision_frequency))
            self._eval_schedule = [k for k in sched if k >= spec.window and i0 <= k < i1]
        size = spec.observation_size(self.n)
        self.observation_space = gym.spaces.Box(low=-max(spec.obs_clip, 1.0), high=max(spec.obs_clip, 1.0), shape=(size,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.n + 1,), dtype=np.float32)
        self._schedule: list[int] = []
        self._t = 0
        self._episode_end = i1
        self.book = Book(np.zeros(self.n), float(config.initial_cash))
        self._peak = float(config.initial_cash)
        self._dd = 0.0
        self.episode_reward = 0.0

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if self.mode == "train":
            first = self.i0 + self.env_spec.window
            length = self.env_spec.episode_sessions
            if length:
                if self.i1 - first < length:
                    from finplan_model.core.errors import FinplanError
                    raise FinplanError.validation("training range cannot fit the requested episode and history", pointer="/rl/env/episode_sessions")
                start = int(self.np_random.integers(first, self.i1 - length + 1))
                self._episode_end = start + length
            else:
                offset = int(self.np_random.integers(0, self.env_spec.step_sessions)) if self.env_spec.step_sessions > 1 else 0
                start = min(first + offset, self.i1 - 1)
                self._episode_end = self.i1
            self._schedule = list(range(start, self._episode_end, self.env_spec.step_sessions))
        else:
            self._schedule = list(self._eval_schedule)
            self._episode_end = self.i1
        self._t = 0
        self.book = Book(np.zeros(self.n), float(self.cfg.initial_cash))
        v0 = self.book.value(self.arrays.close[self._schedule[0]]) if self._schedule else float(self.cfg.initial_cash)
        self.book.log.append(v0)
        self._peak, self._dd, self.episode_reward = v0, 0.0, 0.0
        if not self._schedule:
            return np.zeros(self.observation_space.shape, dtype=np.float32), {"empty": True}
        return self.acc.observation(self.env_spec, self.book, self._schedule[0]), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        d = self._schedule[self._t]
        nxt = self._schedule[self._t + 1] if self._t + 1 < len(self._schedule) else self._episode_end
        current, current_cash = self.acc.weights(self.book, d)
        w, c = allocation_action(self.env_spec, action, current, current_cash)
        tgt = self.acc.target(self.book, d, w, c)
        v_d = self.book.value(self.arrays.close[d])
        traded_before = self.book.traded
        values = self.acc.hold_until(self.book, d + 1, nxt, None if tgt is None else tgt[0])
        v_exec_ref = max(v_d, 1e-12)
        turnover = (self.book.traded - traded_before) / v_exec_ref
        reward, terms = self._reward(v_d, values, turnover)
        self.episode_reward += reward
        self._t += 1
        finished = self._t >= len(self._schedule)
        truncated = finished and self._episode_end < self.i1
        terminated = finished and not truncated
        info: dict[str, Any] = {"value": values[-1], "turnover": turnover, "reward_terms": terms}
        if finished:
            info["episode_reward"] = self.episode_reward
            info["episode_net_return"] = values[-1] / float(self.cfg.initial_cash) - 1.0
            obs = self.acc.observation(self.env_spec, self.book, nxt) if truncated else np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self.acc.observation(self.env_spec, self.book, self._schedule[self._t])
        return obs, float(reward), terminated, truncated, info

    def _reward(self, v_d: float, values: list[float], turnover: float) -> tuple[float, dict[str, float]]:
        r = self.env_spec.reward
        prev, sq = v_d, 0.0
        for v in values:
            if prev > 0 and v > 0:
                sq += math.log(v / prev) ** 2
            prev = v
            self._peak = max(self._peak, v)
        dd_new = 1.0 - values[-1] / self._peak if self._peak > 0 else 0.0
        dd_inc = max(0.0, dd_new - self._dd)
        self._dd = dd_new
        log_ret = math.log(values[-1] / v_d) if v_d > 0 and values[-1] > 0 else -10.0
        raw = log_ret - r.risk_penalty * sq - r.drawdown_penalty * dd_inc - r.turnover_penalty * turnover
        return r.reward_scale * raw, {"log_return": log_ret, "sum_sq_log_return": sq, "drawdown_increase": dd_inc, "turnover": turnover}

    # ------------------------------------------------------------------ evaluation helper
    def evaluate(self, predict: Predict) -> dict[str, Any]:
        """Deterministic pass over the evaluation schedule (``mode='eval'``): NAV from ``i0`` and metrics."""
        spec = self.env_spec

        def decide(book: Book, k: int) -> tuple[np.ndarray, float] | None:
            if k < spec.window:
                return None
            obs = self.acc.observation(spec, book, k)
            current, cash = self.acc.weights(book, k)
            return allocation_action(spec, predict(obs), current, cash)

        values = run_schedule(self.arrays, self.cfg, spec, self.i0, self.i1, decide, self._eval_schedule if self.mode == "eval" else calendar_schedule(self.arrays, self.i0, self.i1, spec.decision_frequency))
        return {"values": values, **schedule_metrics(values, self.cfg)}
