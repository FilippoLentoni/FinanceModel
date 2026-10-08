"""Simulation configuration of the common paper execution simulator (design D7).

One frozen, JSON-serializable :class:`SimulationConfig` holds every convention that can change a
result: execution timing, fee, spread and slippage models, the liquidity cap, the constraint set
and policy, the cash rate and the reconciliation tolerance. Its canonical form gives the
``simulation_configuration_id`` and the ``cost_model_id`` recorded in every result and compared by
the benchmark comparison guard (SIM-01).

Validation (``from_dict``):

* unknown keys, wrong types and out-of-range values: ``VALIDATION_FAILED`` with a JSON pointer;
* any venue other than ``paper`` (for example a broker, exchange or wallet) and any key that looks
  like live-trading credentials: ``OPERATION_NOT_PERMITTED`` (SIM-07; paper only, never live);
* same-bar execution (``same_close``/``same_bar``): ``VALIDATION_FAILED`` (SIM-04).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from finplan_model.core.errors import FinplanError
from finplan_model.core.ids import configuration_id

__all__ = [
    "ConstraintSet",
    "EXECUTION_TIMINGS",
    "FeeModel",
    "LiquidityModel",
    "REBALANCE_FREQUENCIES",
    "SimulationConfig",
    "SlippageModel",
    "SpreadModel",
]

EXECUTION_TIMINGS = ("next_open", "next_close")
REBALANCE_FREQUENCIES = ("daily", "weekly", "monthly", "quarterly")
CONSTRAINT_POLICIES = ("project", "reject")
UNFILLED_POLICIES = ("cancel", "carry_over")
SLIPPAGE_MODELS = ("none", "linear_participation")
#: The only permitted venue. Simulated fills are research artifacts, not platform executions.
PAPER_VENUES = ("paper",)
_LIVE_KEY_RE = re.compile(r"(?i)(api[_-]?key|secret|password|token|credential|broker|brokerage|exchange_account|wallet|private[_-]?key|account[_-]?id|order[_-]?endpoint)")


def _pointer(*parts: Any) -> str:
    return "".join("/" + str(p).replace("~", "~0").replace("/", "~1") for p in parts)


def _num(d: Mapping[str, Any], key: str, ptr: tuple[Any, ...], *, lo: float | None = None, hi: float | None = None, default: Any = ...) -> float:
    if key not in d:
        if default is ...:
            raise FinplanError.validation(f"missing required setting {key}", pointer=_pointer(*ptr, key))
        return default
    v = d[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise FinplanError.validation(f"setting {key} must be a finite number", pointer=_pointer(*ptr, key))
    v = float(v)
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise FinplanError.validation(f"setting {key} is out of range", pointer=_pointer(*ptr, key), minimum=lo, maximum=hi)
    return v


def _choice(d: Mapping[str, Any], key: str, choices: tuple[str, ...], ptr: tuple[Any, ...], default: str) -> str:
    v = d.get(key, default)
    if v not in choices:
        raise FinplanError.validation(f"setting {key} must be one of {', '.join(choices)}", pointer=_pointer(*ptr, key))
    return str(v)


def _no_unknown(d: Mapping[str, Any], allowed: set[str], ptr: tuple[Any, ...]) -> None:
    for k in d:
        if k == "$comment":
            continue
        if k not in allowed:
            raise FinplanError.validation(f"unknown setting {k}", pointer=_pointer(*ptr, k))


def _scan_live(node: Any, path: tuple[Any, ...] = ()) -> None:
    if isinstance(node, Mapping):
        for k, v in node.items():
            if isinstance(k, str) and _LIVE_KEY_RE.search(k) and path[-1:] != ("per_instrument",):
                raise FinplanError.not_permitted("live trading settings are not permitted; the simulator is paper only", pointer=_pointer(*path, k))
            _scan_live(v, (*path, k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _scan_live(v, (*path, i))


@dataclass(frozen=True)
class FeeModel:
    proportional_bps: float = 1.0
    fixed_per_trade: float = 0.0


@dataclass(frozen=True)
class SpreadModel:
    half_spread_bps: float = 1.0


@dataclass(frozen=True)
class SlippageModel:
    model: str = "linear_participation"
    #: slippage in bps per unit of participation (fill quantity / session volume)
    coefficient_bps: float = 10.0


@dataclass(frozen=True)
class LiquidityModel:
    #: maximum fill as a fraction of the instrument's session volume at execution (None = no cap)
    participation_cap: float | None = 0.05
    unfilled: str = "cancel"


@dataclass(frozen=True)
class ConstraintSet:
    long_only: bool = True
    min_weight: float = 0.0
    max_weight: float = 1.0
    min_cash: float = 0.0
    max_cash: float = 1.0
    #: one-way turnover cap per rebalance, sum over instruments of |target - current| (None = no cap)
    max_turnover: float | None = None
    #: per-instrument bound overrides: {"SPY": {"min": 0.0, "max": 0.4}}
    per_instrument: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def bounds(self, instrument_id: str) -> tuple[float, float]:
        o = self.per_instrument.get(instrument_id, {})
        lo = float(o.get("min", self.min_weight if self.long_only else -self.max_weight))
        hi = float(o.get("max", self.max_weight))
        if self.long_only:
            lo = max(lo, 0.0)
        return lo, hi

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["per_instrument"] = {k: dict(v) for k, v in sorted(self.per_instrument.items())}
        return d


@dataclass(frozen=True)
class SimulationConfig:
    venue: str = "paper"
    execution_timing: str = "next_open"
    rebalance_frequency: str = "monthly"
    initial_cash: float = 100000.0
    base_currency: str = "USD"
    cash_rate_annual: float = 0.0
    periods_per_year: int = 252
    fees: FeeModel = field(default_factory=FeeModel)
    spread: SpreadModel = field(default_factory=SpreadModel)
    slippage: SlippageModel = field(default_factory=SlippageModel)
    liquidity: LiquidityModel = field(default_factory=LiquidityModel)
    constraints: ConstraintSet = field(default_factory=ConstraintSet)
    constraint_policy: str = "project"
    infeasible_fallback: str = "hold_current"
    reconciliation_tolerance: float = 1e-6
    weight_tolerance: float = 1e-9

    # ------------------------------------------------------------------ parsing
    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "SimulationConfig":
        d = dict(d or {})
        _scan_live(d)
        venue = d.get("venue", "paper")
        if venue not in PAPER_VENUES:
            raise FinplanError.not_permitted("simulation venue must be 'paper'; live execution venues are not permitted", pointer="/venue")
        _no_unknown(d, {f for f in cls.__dataclass_fields__}, ())
        timing = d.get("execution_timing", "next_open")
        if timing in ("same_close", "same_bar", "same_open", "decision_close"):
            raise FinplanError.validation("same-bar execution at the decision price is not allowed", pointer="/execution_timing")
        timing = _choice(d, "execution_timing", EXECUTION_TIMINGS, (), "next_open")
        fees_d = dict(d.get("fees") or {})
        _no_unknown(fees_d, {"proportional_bps", "fixed_per_trade"}, ("fees",))
        spread_d = dict(d.get("spread") or {})
        _no_unknown(spread_d, {"half_spread_bps"}, ("spread",))
        slip_d = dict(d.get("slippage") or {})
        _no_unknown(slip_d, {"model", "coefficient_bps"}, ("slippage",))
        liq_d = dict(d.get("liquidity") or {})
        _no_unknown(liq_d, {"participation_cap", "unfilled"}, ("liquidity",))
        con_d = dict(d.get("constraints") or {})
        _no_unknown(con_d, {"long_only", "min_weight", "max_weight", "min_cash", "max_cash", "max_turnover", "per_instrument"}, ("constraints",))
        long_only = con_d.get("long_only", True)
        if not isinstance(long_only, bool):
            raise FinplanError.validation("long_only must be a boolean", pointer="/constraints/long_only")
        per: dict[str, dict[str, float]] = {}
        for iid, o in dict(con_d.get("per_instrument") or {}).items():
            if not re.fullmatch(r"[A-Z0-9][A-Z0-9._-]{0,31}", str(iid)) or not isinstance(o, Mapping):
                raise FinplanError.validation("per_instrument keys must be instrument ids mapping to {min, max}", pointer=_pointer("constraints", "per_instrument", iid))
            _no_unknown(o, {"min", "max"}, ("constraints", "per_instrument", iid))
            per[str(iid)] = {k: _num(o, k, ("constraints", "per_instrument", iid), lo=-1.0, hi=1.0) for k in ("min", "max") if k in o}
        cap = liq_d.get("participation_cap", 0.05)
        max_turnover = con_d.get("max_turnover")
        cfg = cls(
            venue="paper",
            execution_timing=timing,
            rebalance_frequency=_choice(d, "rebalance_frequency", REBALANCE_FREQUENCIES, (), "monthly"),
            initial_cash=_num(d, "initial_cash", (), lo=1e-9, default=100000.0),
            base_currency=str(d.get("base_currency", "USD")),
            cash_rate_annual=_num(d, "cash_rate_annual", (), lo=-0.5, hi=1.0, default=0.0),
            periods_per_year=int(_num(d, "periods_per_year", (), lo=1, hi=366, default=252)),
            fees=FeeModel(_num(fees_d, "proportional_bps", ("fees",), lo=0, hi=10000, default=1.0), _num(fees_d, "fixed_per_trade", ("fees",), lo=0, default=0.0)),
            spread=SpreadModel(_num(spread_d, "half_spread_bps", ("spread",), lo=0, hi=10000, default=1.0)),
            slippage=SlippageModel(_choice(slip_d, "model", SLIPPAGE_MODELS, ("slippage",), "linear_participation"), _num(slip_d, "coefficient_bps", ("slippage",), lo=0, hi=10000, default=10.0)),
            liquidity=LiquidityModel(None if cap is None else _num(liq_d, "participation_cap", ("liquidity",), lo=1e-9, hi=1.0, default=0.05), _choice(liq_d, "unfilled", UNFILLED_POLICIES, ("liquidity",), "cancel")),
            constraints=ConstraintSet(
                long_only=long_only,
                min_weight=_num(con_d, "min_weight", ("constraints",), lo=-1.0, hi=1.0, default=0.0),
                max_weight=_num(con_d, "max_weight", ("constraints",), lo=0.0, hi=1.0, default=1.0),
                min_cash=_num(con_d, "min_cash", ("constraints",), lo=0.0, hi=1.0, default=0.0),
                max_cash=_num(con_d, "max_cash", ("constraints",), lo=0.0, hi=1.0, default=1.0),
                max_turnover=None if max_turnover is None else _num(con_d, "max_turnover", ("constraints",), lo=0.0),
                per_instrument=per,
            ),
            constraint_policy=_choice(d, "constraint_policy", CONSTRAINT_POLICIES, (), "project"),
            infeasible_fallback=_choice(d, "infeasible_fallback", ("hold_current",), (), "hold_current"),
            reconciliation_tolerance=_num(d, "reconciliation_tolerance", (), lo=0.0, hi=1.0, default=1e-6),
            weight_tolerance=_num(d, "weight_tolerance", (), lo=0.0, hi=1e-3, default=1e-9),
        )
        # An infeasible constraint set (for example min_cash 0.5 with max_cash 0.4, i.e. a minimum
        # invested weight of 60%) is NOT a configuration error: every decision is then rejected as
        # constraint_set_infeasible and reported with solution_status infeasible, and the run still
        # completes (spec baseline-strategies, "Optimizer outcome reporting").
        if not re.fullmatch(r"[A-Z]{3}", cfg.base_currency):
            raise FinplanError.validation("base_currency must be an ISO currency code", pointer="/base_currency")
        return cfg

    # ------------------------------------------------------------------ identity
    def to_dict(self) -> dict[str, Any]:
        return {
            "venue": self.venue,
            "execution_timing": self.execution_timing,
            "rebalance_frequency": self.rebalance_frequency,
            "initial_cash": self.initial_cash,
            "base_currency": self.base_currency,
            "cash_rate_annual": self.cash_rate_annual,
            "periods_per_year": self.periods_per_year,
            "fees": asdict(self.fees),
            "spread": asdict(self.spread),
            "slippage": asdict(self.slippage),
            "liquidity": asdict(self.liquidity),
            "constraints": self.constraints.to_dict(),
            "constraint_policy": self.constraint_policy,
            "infeasible_fallback": self.infeasible_fallback,
            "reconciliation_tolerance": self.reconciliation_tolerance,
            "weight_tolerance": self.weight_tolerance,
        }

    @property
    def configuration_id(self) -> str:
        """``cfg_`` id of the full simulation configuration."""
        return configuration_id(self.to_dict())

    @property
    def cost_model_id(self) -> str:
        """``cfg_`` id of the cost model alone (fees, spread, slippage)."""
        return configuration_id({"fees": asdict(self.fees), "spread": asdict(self.spread), "slippage": asdict(self.slippage)})

    def comparability_settings(self) -> dict[str, Any]:
        """The settings the comparison guard compares (SIM-01), by name."""
        return {
            "execution_timing": self.execution_timing,
            "fee_model": asdict(self.fees),
            "spread_model": asdict(self.spread),
            "slippage_model": asdict(self.slippage),
            "liquidity_model": asdict(self.liquidity),
            "constraint_set": self.constraints.to_dict(),
            "constraint_policy": self.constraint_policy,
            "rebalance_frequency": self.rebalance_frequency,
            "initial_cash": self.initial_cash,
            "cash_rate_annual": self.cash_rate_annual,
        }
