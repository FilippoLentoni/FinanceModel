"""Hand-built synthetic markets for simulator tests (``synthetic=True``; values are invented)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, time
from typing import Any

from finplan_model.sim.config import SimulationConfig
from finplan_model.sim.market import Bar, MarketData

ZERO_COSTS: dict[str, Any] = {
    "fees": {"proportional_bps": 0.0, "fixed_per_trade": 0.0},
    "spread": {"half_spread_bps": 0.0},
    "slippage": {"model": "none", "coefficient_bps": 0.0},
    "liquidity": {"participation_cap": None, "unfilled": "cancel"},
}


def make_market(rows: Sequence[tuple[str, Mapping[str, tuple[float, float, float]]]], *, late: Mapping[tuple[str, str], str] | None = None) -> MarketData:
    """``rows`` = [(session, {instrument: (open, close, volume)})]; ``late`` overrides available_at."""
    bars = []
    sessions = []
    for s, per in rows:
        d = date.fromisoformat(s)
        sessions.append(d)
        for iid, (o, c, v) in per.items():
            avail = datetime.combine(d, time(21, 0), tzinfo=UTC)
            if late and (s, iid) in late:
                avail = datetime.fromisoformat(late[(s, iid)].replace("Z", "+00:00"))
            bars.append(Bar(iid, d, c, avail, o, max(o, c), min(o, c), v))
    return MarketData(sessions, bars, synthetic=True, dataset_id="synthetic/test/hand-built")


def cfg(**overrides: Any) -> SimulationConfig:
    base: dict[str, Any] = {"rebalance_frequency": "daily", "initial_cash": 100000.0, **ZERO_COSTS}
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return SimulationConfig.from_dict(base)


class Fixed:
    """A strategy that always returns the same target."""

    def __init__(self, weights: Mapping[str, float], cash: float, name: str = "fixed", status: str = "optimal") -> None:
        self.name = name
        self._out = {"weights": dict(weights), "cash": cash, "solution_status": status}
        self.views: list[Any] = []

    def decide(self, view, holdings):  # noqa: ANN001
        self.views.append(view)
        return dict(self._out)


class Script:
    """A strategy returning a scripted output per decision session (ISO date -> output)."""

    def __init__(self, outputs: Mapping[str, Any], name: str = "scripted") -> None:
        self.name = name
        self.outputs = dict(outputs)

    def decide(self, view, holdings):  # noqa: ANN001
        return self.outputs[view.decision_session.isoformat()]
