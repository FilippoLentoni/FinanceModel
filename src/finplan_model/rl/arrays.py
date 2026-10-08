"""Aligned price arrays and decision schedules for the RL environment (numpy only).

:class:`PriceArrays` holds, for the evaluation universe (sorted instrument order, as the simulator),
the session calendar and per session the close, the executable price (the open for ``next_open``,
the close for ``next_close``) and the volume. A missing bar keeps the last close as its mark and
cannot trade at that session, exactly as the simulator marks and refuses to fill it.

Schedules (indices into the calendar):

* :func:`calendar_schedule` - the rebalance sessions of a window (first session of each period, the
  last session never decides), the simulator's own rule (:func:`finplan_model.sim.engine.rebalance_sessions`);
* :func:`offset_schedule` - every ``step`` sessions from an offset (training episodes).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from finplan_model.core.errors import FinplanError
from finplan_model.sim.engine import rebalance_sessions
from finplan_model.sim.market import MarketData

__all__ = ["PriceArrays", "calendar_schedule", "offset_schedule"]


@dataclass(frozen=True)
class PriceArrays:
    sessions: tuple[date, ...]
    instruments: tuple[str, ...]
    close: np.ndarray  # T x n, marks (last close carried over a missing bar)
    exec_price: np.ndarray  # T x n, executable price (carried mark when the bar is missing)
    volume: np.ndarray  # T x n, 0 when unknown
    has_bar: np.ndarray  # T x n, bool

    @classmethod
    def from_market(cls, market: MarketData, universe: Sequence[str] | None = None, *, execution_timing: str = "next_open") -> PriceArrays:
        instruments = tuple(sorted(universe or market.instruments))
        sessions = market.sessions
        t, n = len(sessions), len(instruments)
        close = np.full((t, n), np.nan)
        exec_p = np.full((t, n), np.nan)
        vol = np.zeros((t, n))
        has = np.zeros((t, n), dtype=bool)
        field = "open" if execution_timing == "next_open" else "close"
        for j, iid in enumerate(instruments):
            mark = np.nan
            for k, s in enumerate(sessions):
                bar = market.bar(iid, s)
                if bar is not None:
                    px = bar.price(field)
                    exec_p[k, j] = px if px is not None else mark
                    mark = float(bar.close)
                    close[k, j] = mark
                    vol[k, j] = float(bar.volume or 0.0)
                    has[k, j] = True
                else:
                    close[k, j] = mark
                    exec_p[k, j] = mark
        if np.isnan(close).any():
            raise FinplanError.validation("every instrument needs a bar on the first session of the RL data", pointer="/universe", session_date=sessions[0].isoformat())
        return cls(tuple(sessions), instruments, close, exec_p, vol, has)

    def index_of(self, day: date, *, side: str = "left") -> int:
        """First session on or after ``day`` (``left``) or last session on or before it (``right``)."""
        idx = int(np.searchsorted(np.array([s.toordinal() for s in self.sessions]), day.toordinal(), side=side))
        return idx if side == "left" else idx - 1

    def window_indices(self, start: date, end: date) -> tuple[int, int]:
        i0, i1 = self.index_of(start, side="left"), self.index_of(end, side="right")
        if i0 >= len(self.sessions) or i1 < 0 or i1 <= i0:
            raise FinplanError.validation("the window has fewer than two sessions in the data", pointer="/splits", start=start.isoformat(), end=end.isoformat())
        return i0, i1


def calendar_schedule(arrays: PriceArrays, i0: int, i1: int, frequency: str) -> list[int]:
    """Decision indices of the window ``[i0, i1]``: its rebalance sessions except the last session."""
    window = arrays.sessions[i0 : i1 + 1]
    rebal = rebalance_sessions(window, frequency)
    return [i0 + k for k, s in enumerate(window) if s in rebal and i0 + k < i1]


def offset_schedule(i0: int, i1: int, offset: int, step: int) -> list[int]:
    return list(range(i0 + offset, i1, max(1, step)))
