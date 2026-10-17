"""The paper execution simulator (spec paper-execution-simulator; design D7).

One implementation of execution timing, fees, spread, slippage, liquidity, constraints, rebalancing
and accounting, used by every strategy family. It only moves simulated cash and shares; it never
holds credentials for, calls or produces instructions for any broker, exchange or wallet (SIM-07).

Step at each session ``s`` (in order):

1. **Cash interest** on the previous cash at ``cash_rate_annual / periods_per_year`` (compounded).
2. **Execution** at the configured executable price of ``s`` - its open (``next_open``) or its close
   (``next_close``) - of the orders implied by the decision made at the *previous* session (or of
   carried-over remainders). A decision is never filled on the bar it was made on (SIM-04).
   Orders are sized from the target weights and the portfolio value at the execution prices;
   sells execute before buys; each fill is capped at ``participation_cap`` times the session
   volume, the remainder is cancelled or carried over (SIM-06); buys are scaled down if cash
   (after costs) would go negative. Every fill pays the proportional fee plus the fixed fee, the
   half-spread and the slippage ``coefficient_bps * participation`` (SIM-05).
3. **Mark to market** at the close of ``s``.
4. **Reconciliation** (SIM-09): ``V_s = V_{s-1} + interest + sum(q_prev * (P_exec - P_prev)) +
   sum(q_new * (P_close - P_exec)) - costs`` within ``reconciliation_tolerance``; a break raises
   ``INTERNAL`` naming the failing step.
5. **Decision** (rebalance sessions only, never the last session): the strategy sees the
   point-in-time view and holdings, returns target weights, and the constraint policy projects or
   rejects them (SIM-03). Infeasible/unbounded strategy outcomes keep current holdings (fallback);
   a ``no_effect`` decision keeps current holdings without orders (``hold_no_effect``).

All iteration is in sorted instrument order, so equal inputs give identical trades and metrics
(SIM-08). Gross values add back cumulative costs: gross P&L = net P&L + costs paid.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from finplan_model.core.errors import ErrorCode, FinplanError

from .config import SimulationConfig
from .constraints import apply_constraints
from .market import HoldingsView, MarketData
from .strategy import FALLBACK_STATUSES, NO_EFFECT, Strategy, TargetWeights, aggregate_solution_status, coerce_target

__all__ = ["Fill", "SimulationResult", "Simulator", "rebalance_sessions"]


@dataclass
class _Book:
    cash: float
    shares: dict[str, float]


@dataclass(frozen=True)
class Fill:
    session_date: str
    decision_session: str
    instrument_id: str
    quantity: float
    price: float
    notional: float
    participation: float
    fee: float
    spread_cost: float
    slippage_cost: float

    @property
    def cost(self) -> float:
        return self.fee + self.spread_cost + self.slippage_cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_date": self.session_date,
            "decision_session": self.decision_session,
            "instrument_id": self.instrument_id,
            "quantity": self.quantity,
            "price": self.price,
            "notional": self.notional,
            "participation": self.participation,
            "fee": self.fee,
            "spread_cost": self.spread_cost,
            "slippage_cost": self.slippage_cost,
        }


@dataclass
class SimulationResult:
    strategy: str
    config: SimulationConfig
    sessions: list[str]
    nav: list[dict[str, Any]]
    fills: list[Fill]
    decisions: list[dict[str, Any]]
    unfilled: list[dict[str, Any]]
    reconciliation: dict[str, Any]
    synthetic: bool
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def solution_status(self) -> str:
        return aggregate_solution_status([d["solution_status"] for d in self.decisions])

    def strategy_outputs(self) -> dict[str, dict[str, Any]]:
        """Stored strategy outputs by decision session (input to a replay, SIM-08)."""
        return {d["decision_session"]: d["strategy_output"] for d in self.decisions}

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "simulation_configuration_id": self.config.configuration_id,
            "cost_model_id": self.config.cost_model_id,
            "sessions": {"start": self.sessions[0], "end": self.sessions[-1], "count": len(self.sessions)} if self.sessions else None,
            "summary": self.summary,
            "solution_status": self.solution_status,
            "nav": self.nav,
            "fills": [f.to_dict() for f in self.fills],
            "decisions": self.decisions,
            "unfilled": self.unfilled,
            "reconciliation": self.reconciliation,
            "synthetic": self.synthetic,
        }


def _period_key(d: date, frequency: str) -> tuple[int, ...]:
    if frequency == "daily":
        return (d.toordinal(),)
    if frequency == "weekly":
        iso = d.isocalendar()
        return (iso[0], iso[1])
    if frequency == "monthly":
        return (d.year, d.month)
    if frequency == "quarterly":
        return (d.year, (d.month - 1) // 3)
    raise ValueError(frequency)


def rebalance_sessions(sessions: Sequence[date], frequency: str) -> set[date]:
    """First session of each period (knowable from the calendar alone, no look-ahead)."""
    out: set[date] = set()
    last: tuple[int, ...] | None = None
    for s in sessions:
        k = _period_key(s, frequency)
        if k != last:
            out.add(s)
            last = k
    return out


def _r(x: float) -> float:
    """Normalize -0.0 and tiny float noise in recorded values (keeps canonical JSON stable)."""
    return 0.0 if x == 0 else x


class Simulator:
    def __init__(self, config: SimulationConfig, market: MarketData, *, universe: Sequence[str] | None = None) -> None:
        if config.venue != "paper":  # defensive: SimulationConfig.from_dict already refuses
            raise FinplanError.not_permitted("simulation venue must be 'paper'")
        self.cfg = config
        self.market = market
        self.universe: tuple[str, ...] = tuple(sorted(universe or market.instruments))
        missing = [i for i in self.universe if i not in market.instruments]
        if missing:
            raise FinplanError.validation("universe instruments missing from the market data", pointer="/universe", instruments=missing[:10])

    # ------------------------------------------------------------------ helpers
    def _exec_field(self) -> str:
        return "open" if self.cfg.execution_timing == "next_open" else "close"

    def _cost_rates(self, qty: float, volume: float | None) -> tuple[float, float, float, float]:
        participation = abs(qty) / volume if volume else 0.0
        fee_rate = self.cfg.fees.proportional_bps / 1e4
        spread_rate = self.cfg.spread.half_spread_bps / 1e4
        slip_rate = (self.cfg.slippage.coefficient_bps * participation / 1e4) if self.cfg.slippage.model == "linear_participation" else 0.0
        return participation, fee_rate, spread_rate, slip_rate

    def _cap(self, qty: float, volume: float | None) -> float:
        cap = self.cfg.liquidity.participation_cap
        if cap is None:
            return qty
        limit = cap * (volume or 0.0)
        return math.copysign(min(abs(qty), limit), qty)

    # ------------------------------------------------------------------ run
    def run(self, strategy: Strategy, *, start: date | None = None, end: date | None = None,
            initial_book: HoldingsView | None = None) -> SimulationResult:
        cfg = self.cfg
        sessions = [s for s in self.market.sessions if (start is None or s >= start) and (end is None or s <= end)]
        if len(sessions) < 2:
            raise FinplanError.validation("evaluation window needs at least two sessions", pointer="/evaluation_window")
        rebal = rebalance_sessions(sessions, cfg.rebalance_frequency)
        per_period = (1.0 + cfg.cash_rate_annual) ** (1.0 / cfg.periods_per_year) - 1.0
        exec_field = self._exec_field()
        tol = cfg.reconciliation_tolerance

        if initial_book is None:
            book = _Book(float(cfg.initial_cash), {i: 0.0 for i in self.universe})
        else:
            # Decision evaluation starts from the original shares, without charging a
            # fictitious liquidation and purchase merely to initialize the simulation.
            shares_in = dict(initial_book.shares)
            quantities = [shares_in.get(i, 0.0) for i in self.universe]
            if (set(shares_in) - set(self.universe) or not math.isfinite(initial_book.cash)
                    or initial_book.cash < 0 or any(not math.isfinite(q) or q < 0 for q in quantities)):
                raise FinplanError.validation("invalid initial paper book", pointer="/initial_book")
            bars_in = [self.market.bar(i, sessions[0]) for i in self.universe]
            if any(b is None for b in bars_in):
                raise FinplanError.precondition("initial book lacks raw marks", reason="initial_book_marks_missing")
            initial_value = initial_book.cash + sum(q * b.close for q, b in zip(quantities, bars_in))
            if abs(initial_value - cfg.initial_cash) > cfg.reconciliation_tolerance:
                raise FinplanError.precondition("initial paper book and simulation value differ", reason="initial_book_value_mismatch")
            book = _Book(float(initial_book.cash), dict(zip(self.universe, quantities)))
        shares = book.shares
        cash = book.cash
        marks: dict[str, float] = {}
        for i in self.universe:  # initial marks: last close at or before the first session
            b = self.market.bar(i, sessions[0])
            if b is not None:
                marks[i] = b.close
        nav: list[dict[str, Any]] = []
        fills: list[Fill] = []
        decisions: list[dict[str, Any]] = []
        unfilled: list[dict[str, Any]] = []
        pending_target: tuple[str, TargetWeights] | None = None  # (decision_session, target)
        carry: dict[str, tuple[str, float]] = {}  # instrument -> (decision_session, remaining qty)
        cum = {"fees": 0.0, "spread": 0.0, "slippage": 0.0, "traded_notional": 0.0}
        max_err = 0.0
        prev_value = float(cfg.initial_cash)

        for k, s in enumerate(sessions):
            s_iso = s.isoformat()
            # 1. interest on previous cash
            interest = book.cash * per_period if k > 0 else 0.0
            book.cash += interest
            # executable and closing prices of this session (carry the mark when a bar is missing)
            bars = {i: self.market.bar(i, s) for i in self.universe}
            p_exec = {i: (bars[i].price(exec_field) if bars[i] is not None and bars[i].price(exec_field) is not None else marks.get(i)) for i in self.universe}
            p_close = {i: (bars[i].close if bars[i] is not None else marks.get(i)) for i in self.universe}
            shares_prev = dict(shares)
            mtm_pre = sum(shares_prev[i] * ((p_exec[i] or 0.0) - (marks.get(i) or 0.0)) for i in self.universe if shares_prev[i])
            # 2. execution
            orders: dict[str, tuple[str, float]] = {}
            if k > 0 and pending_target is not None:
                dsess, target = pending_target
                for i, (old_dsess, qty) in sorted(carry.items()):
                    unfilled.append({"session_date": s_iso, "decision_session": old_dsess, "instrument_id": i, "quantity": _r(qty), "handling": "cancelled_superseded"})
                carry = {}
                v_exec = book.cash + sum(shares[i] * (p_exec[i] or 0.0) for i in self.universe)
                for i in self.universe:
                    if p_exec[i] is None or p_exec[i] <= 0:
                        continue
                    want = target.weights.get(i, 0.0) * v_exec / p_exec[i]
                    q = want - shares[i]
                    if abs(q) * p_exec[i] > 1e-9:
                        orders[i] = (dsess, q)
                pending_target = None
            elif k > 0 and carry:
                orders, carry = dict(carry), {}

            # sells first (they raise cash), then buys scaled to the cash available after costs
            n_fills_before = len(fills)
            planned = [(i, d, self._plan(unfilled, carry, s_iso, d, i, q, bars[i], p_exec[i])) for i, (d, q) in sorted(orders.items())]
            for i, d, qf in planned:
                if qf < 0:
                    fills.append(self._fill(book, s_iso, d, i, qf, p_exec[i], bars[i].volume))
            buys = [(i, d, qf) for i, d, qf in planned if qf > 0]
            if buys:
                variable = 0.0
                for i, _d, qf in buys:
                    _, fr, sr, slr = self._cost_rates(qf, bars[i].volume)
                    variable += qf * p_exec[i] * (1.0 + fr + sr + slr)
                fixed = cfg.fees.fixed_per_trade * len(buys)
                scale = 1.0
                if variable + fixed > book.cash:
                    scale = max(0.0, min(1.0, (book.cash - fixed) / variable)) * (1.0 - 1e-12) if variable > 0 else 0.0
                for i, d, qf in buys:
                    if scale < 1.0:
                        self._unfilled(unfilled, carry, s_iso, d, i, qf * (1.0 - scale), "cash_limited")
                    if qf * scale > 0:
                        fills.append(self._fill(book, s_iso, d, i, qf * scale, p_exec[i], bars[i].volume))
            step_costs = sum(f.cost for f in fills[n_fills_before:])
            for f in fills[n_fills_before:]:
                cum["fees"] += f.fee
                cum["spread"] += f.spread_cost
                cum["slippage"] += f.slippage_cost
                cum["traded_notional"] += f.notional
            cash = book.cash

            # 3. mark to market
            mtm_post = sum(shares[i] * ((p_close[i] or 0.0) - (p_exec[i] or 0.0)) for i in self.universe if shares[i])
            for i in self.universe:
                if p_close[i] is not None:
                    marks[i] = p_close[i]
            value = cash + sum(shares[i] * (marks.get(i) or 0.0) for i in self.universe)

            # 4. reconciliation
            self._check_reconciliation(k, s_iso, value, prev_value + interest + mtm_pre + mtm_post - step_costs, tol)
            err = abs(value - (prev_value + interest + mtm_pre + mtm_post - step_costs))
            max_err = max(max_err, err)
            total_costs = cum["fees"] + cum["spread"] + cum["slippage"]
            nav.append({"session_date": s_iso, "value": value, "cash": cash, "gross_value": value + total_costs, "costs_cumulative": total_costs, "step_costs": step_costs, "interest": interest})
            prev_value = value

            # 5. decision (after the close of s; executes at the next session)
            if s in rebal and k < len(sessions) - 1:
                view = self.market.view(s, self.universe)
                # Holdings are valued with prices visible at the decision time (a late bar is not).
                pit = {i: p for i in self.universe if (p := view.latest(i)) is not None or (p := marks.get(i)) is not None}
                holdings = HoldingsView({i: shares[i] for i in self.universe}, cash, pit, cash + sum(shares[i] * pit.get(i, 0.0) for i in self.universe))
                raw = strategy.decide(view, holdings)
                target = coerce_target(raw, self.universe)
                cur_w = holdings.weights
                rec: dict[str, Any] = {
                    "decision_session": s_iso,
                    "decision_time": view.decision_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "solution_status": target.solution_status,
                    "strategy_output": target.to_dict(),
                }
                if target.solution_status in FALLBACK_STATUSES:
                    rec.update(action="fallback_hold_current", fallback=cfg.infeasible_fallback)
                elif target.solution_status == NO_EFFECT:
                    # The strategy asks for no change (buy-and-hold after its initial target, an
                    # optimizer without enough history): keep holdings, place no orders.
                    # Carried-over remainders of earlier decisions keep executing.
                    rec.update(action="hold_no_effect")
                else:
                    outcome = apply_constraints(target.weights, target.cash, cfg.constraints, current=cur_w, current_cash=holdings.cash_weight, policy=cfg.constraint_policy, tol=cfg.weight_tolerance)
                    rec["constraints"] = outcome.to_dict()
                    rec["action"] = {"accepted": "executed", "projected": "projected", "rejected": "rejected_hold_current"}[outcome.action]
                    if outcome.reason == "constraint_set_infeasible":
                        rec["solution_status"] = "infeasible"
                    if outcome.weights is not None:
                        pending_target = (s_iso, TargetWeights(outcome.weights, float(outcome.cash or 0.0), target.solution_status))
                decisions.append(rec)

        first, last = nav[0], nav[-1]
        init = float(cfg.initial_cash)
        total_costs = cum["fees"] + cum["spread"] + cum["slippage"]
        summary = {
            "initial_value": init,
            "final_value": last["value"],
            "net_return": last["value"] / init - 1.0,
            "gross_return": (last["value"] + total_costs) / init - 1.0,
            "total_fees": cum["fees"],
            "total_spread_cost": cum["spread"],
            "total_slippage_cost": cum["slippage"],
            "total_costs": total_costs,
            "traded_notional": cum["traded_notional"],
            "turnover": cum["traded_notional"] / init if init else 0.0,
            "n_fills": len(fills),
            "n_decisions": len(decisions),
            "n_projections": sum(1 for d in decisions if d.get("action") == "projected"),
            "n_rejections": sum(1 for d in decisions if d.get("action") == "rejected_hold_current"),
            "n_fallbacks": sum(1 for d in decisions if d.get("action") == "fallback_hold_current"),
            "n_holds": sum(1 for d in decisions if d.get("action") == "hold_no_effect"),
            "first_session": first["session_date"],
            "last_session": last["session_date"],
        }
        return SimulationResult(
            strategy=strategy.name,
            config=cfg,
            sessions=[s.isoformat() for s in sessions],
            nav=nav,
            fills=fills,
            decisions=decisions,
            unfilled=unfilled,
            reconciliation={"steps": len(nav), "max_abs_error": max_err, "tolerance": tol, "status": "reconciled"},
            synthetic=self.market.synthetic,
            summary=summary,
        )

    # ------------------------------------------------------------------ internals
    def _plan(self, log: list[dict[str, Any]], carry: dict[str, tuple[str, float]], s_iso: str, dsess: str, i: str, q: float, bar: Any, price: float | None) -> float:
        """Fillable quantity after the liquidity cap; records the unfilled remainder."""
        if bar is None or price is None:
            self._unfilled(log, carry, s_iso, dsess, i, q, "no_bar")
            return 0.0
        q_fill = self._cap(q, bar.volume)
        rem = q - q_fill
        if abs(rem) > 1e-12:
            self._unfilled(log, carry, s_iso, dsess, i, rem, "participation_cap")
        return q_fill

    def _fill(self, book: "_Book", s_iso: str, dsess: str, i: str, q: float, price: float, volume: float | None) -> Fill:
        notional = abs(q) * price
        part, fr, sr, slr = self._cost_rates(q, volume)
        fill = Fill(s_iso, dsess, i, _r(q), price, notional, part, notional * fr + self.cfg.fees.fixed_per_trade, notional * sr, notional * slr)
        self._apply_fill(book, fill)
        return fill

    def _apply_fill(self, book: "_Book", fill: Fill) -> None:
        """Ledger posting of one fill: cash pays the notional and every cost component."""
        book.cash -= fill.quantity * fill.price + fill.fee + fill.spread_cost + fill.slippage_cost
        book.shares[fill.instrument_id] += fill.quantity

    def _unfilled(self, log: list[dict[str, Any]], carry: dict[str, tuple[str, float]], s_iso: str, dsess: str, i: str, qty: float, reason: str) -> None:
        handling = "carried_over" if self.cfg.liquidity.unfilled == "carry_over" else "cancelled"
        log.append({"session_date": s_iso, "decision_session": dsess, "instrument_id": i, "quantity": _r(qty), "reason": reason, "handling": handling})
        if handling == "carried_over":
            prev = carry.get(i, (dsess, 0.0))[1]
            carry[i] = (dsess, prev + qty)

    def _check_reconciliation(self, step: int, session_iso: str, value: float, expected: float, tol: float) -> None:
        diff = value - expected
        if not math.isfinite(value) or abs(diff) > tol:
            raise FinplanError(
                ErrorCode.INTERNAL,
                "accounting reconciliation failed: holdings value plus cash differs from the reconciliation identity",
                details={"reason": "reconciliation_break", "step": step, "session_date": session_iso, "difference": diff if math.isfinite(diff) else None, "tolerance": tol},
            )

