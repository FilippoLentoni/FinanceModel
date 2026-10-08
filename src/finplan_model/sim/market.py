"""Market data as the simulator sees it, and the point-in-time view strategies receive.

* :class:`Bar` - one completed daily observation of one instrument with its ``available_at`` time
  (when it became available; from the dataset's point-in-time record).
* :class:`MarketData` - the session calendar plus bars, with one **decision time** per session
  (default: ``decision_time_utc`` on the session date). Only ``completed_daily`` bars are ever used
  as daily bars; ``intraday_partial`` observations are dropped (DS-03, DS-10).
* :class:`PointInTimeView` - what a strategy may see at a decision: bars whose session is on or
  before the decision session **and** whose ``available_at`` is at or before the decision time. A
  bar that arrived late is invisible until the first decision after it arrived (DS-03 "late-arriving
  observation"). Nothing after the decision time is reachable through the view (SIM-02).
* :func:`synthetic_market` - deterministic synthetic ETF-shaped bars for tests and fixtures,
  always ``synthetic=True`` (no retrieved market data in this public repository, DS-13).
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import numpy as np

from finplan_model.core.clock import parse_utc
from finplan_model.core.errors import FinplanError

__all__ = ["Bar", "HoldingsView", "MarketData", "PointInTimeView", "synthetic_market"]

COMPLETED = "completed_daily"
INTRADAY = "intraday_partial"


@dataclass(frozen=True)
class Bar:
    instrument_id: str
    session_date: date
    close: float
    available_at: datetime
    open: float | None = None
    high: float | None = None
    low: float | None = None
    volume: float | None = None
    kind: str = COMPLETED

    def price(self, field: str) -> float | None:
        v = getattr(self, field)
        return None if v is None else float(v)

    @classmethod
    def from_observation(cls, obs: Mapping[str, Any], available_at: datetime | str) -> "Bar":
        """From a contract ``finance/v1/observation`` document plus its availability time."""
        return cls(
            instrument_id=str(obs["instrument_id"]),
            session_date=date.fromisoformat(str(obs["session_date"])),
            close=float(obs["close"]),
            open=None if obs.get("open") is None else float(obs["open"]),
            high=None if obs.get("high") is None else float(obs["high"]),
            low=None if obs.get("low") is None else float(obs["low"]),
            volume=None if obs.get("volume") is None else float(obs["volume"]),
            kind=str(obs.get("kind", COMPLETED)),
            available_at=parse_utc(available_at) if isinstance(available_at, str) else available_at,
        )


class MarketData:
    def __init__(
        self,
        sessions: Sequence[date],
        bars: Iterable[Bar],
        *,
        decision_times: Mapping[date, datetime] | None = None,
        decision_time_utc: time = time(22, 0),
        synthetic: bool = True,
        dataset_id: str | None = None,
        dataset_checksum: str | None = None,
    ) -> None:
        self.sessions: tuple[date, ...] = tuple(sorted(set(sessions)))
        if not self.sessions:
            raise FinplanError.validation("market data has no sessions", pointer="/sessions")
        self.synthetic = synthetic
        self.dataset_id = dataset_id
        self.dataset_checksum = dataset_checksum
        session_set = set(self.sessions)
        self._bars: dict[str, dict[date, Bar]] = {}
        self.dropped_intraday = 0
        for b in bars:
            if b.kind != COMPLETED:
                self.dropped_intraday += 1
                continue
            if b.session_date not in session_set:
                raise FinplanError.validation("bar dated on a day that is not a session of the calendar", pointer="/bars", session_date=b.session_date.isoformat())
            if not (math.isfinite(b.close) and b.close > 0):
                raise FinplanError.validation("bar close must be a positive finite number", pointer="/bars", session_date=b.session_date.isoformat())
            per = self._bars.setdefault(b.instrument_id, {})
            if b.session_date in per:
                raise FinplanError.validation("duplicate completed bar for one instrument and session", pointer="/bars", session_date=b.session_date.isoformat())
            per[b.session_date] = b
        self.instruments: tuple[str, ...] = tuple(sorted(self._bars))
        dt = dict(decision_times or {})
        self._decision_times = {s: dt.get(s) or datetime.combine(s, decision_time_utc, tzinfo=UTC) for s in self.sessions}
        self._sorted_dates = {i: sorted(per) for i, per in self._bars.items()}

    # ------------------------------------------------------------------ access
    def bar(self, instrument_id: str, session: date) -> Bar | None:
        return self._bars.get(instrument_id, {}).get(session)

    def decision_time(self, session: date) -> datetime:
        return self._decision_times[session]

    def session_index(self, session: date) -> int:
        i = bisect.bisect_left(self.sessions, session)
        if i >= len(self.sessions) or self.sessions[i] != session:
            raise KeyError(session)
        return i

    def view(self, session: date, instruments: Sequence[str] | None = None) -> "PointInTimeView":
        return PointInTimeView(self, session, self.decision_time(session), tuple(instruments or self.instruments))

    def bars_of(self, instrument_id: str) -> list[Bar]:
        per = self._bars.get(instrument_id, {})
        return [per[d] for d in self._sorted_dates.get(instrument_id, [])]


class PointInTimeView:
    """Read-only, as-of view for one decision. Only data available at ``decision_time`` is visible."""

    __slots__ = ("_market", "decision_session", "decision_time", "instruments")

    def __init__(self, market: MarketData, decision_session: date, decision_time: datetime, instruments: tuple[str, ...]) -> None:
        self._market = market
        self.decision_session = decision_session
        self.decision_time = decision_time
        self.instruments = instruments

    def _visible(self, instrument_id: str) -> list[Bar]:
        dates = self._market._sorted_dates.get(instrument_id, [])
        end = bisect.bisect_right(dates, self.decision_session)
        per = self._market._bars[instrument_id] if dates else {}
        return [per[d] for d in dates[:end] if per[d].available_at <= self.decision_time]

    def history(self, instrument_id: str, field: str = "close", lookback: int | None = None) -> list[tuple[date, float]]:
        """``(session_date, value)`` pairs available at the decision time, oldest first."""
        rows = [(b.session_date, b.price(field)) for b in self._visible(instrument_id)]
        rows = [(d, v) for d, v in rows if v is not None]
        return rows[-lookback:] if lookback else rows

    def latest(self, instrument_id: str, field: str = "close") -> float | None:
        h = self.history(instrument_id, field, 1)
        return h[-1][1] if h else None

    def price_matrix(self, field: str = "close", lookback: int | None = None) -> tuple[list[date], np.ndarray]:
        """Aligned ``T x N`` matrix over sessions where every instrument has a visible value."""
        hist = {i: dict(self.history(i, field)) for i in self.instruments}
        common = sorted(set.intersection(*(set(h) for h in hist.values()))) if hist else []
        if lookback:
            common = common[-lookback:]
        mat = np.array([[hist[i][d] for i in self.instruments] for d in common], dtype=float).reshape(len(common), len(self.instruments))
        return common, mat

    def returns(self, lookback: int | None = None) -> tuple[list[date], np.ndarray]:
        """Simple close-to-close returns over the last ``lookback`` return periods (point in time)."""
        dates, px = self.price_matrix("close", None if lookback is None else lookback + 1)
        if len(dates) < 2:
            return [], np.zeros((0, len(self.instruments)))
        return dates[1:], px[1:] / px[:-1] - 1.0


@dataclass(frozen=True)
class HoldingsView:
    """Current simulated holdings at a decision (marked at the decision session's close)."""

    shares: Mapping[str, float]
    cash: float
    prices: Mapping[str, float]
    value: float

    @property
    def weights(self) -> dict[str, float]:
        if self.value <= 0:
            return {i: 0.0 for i in self.shares}
        return {i: q * self.prices[i] / self.value for i, q in sorted(self.shares.items()) if i in self.prices}

    @property
    def cash_weight(self) -> float:
        return self.cash / self.value if self.value > 0 else 1.0


def _weekdays(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def synthetic_market(
    instruments: Sequence[str] = ("SPY",),
    *,
    n_sessions: int = 60,
    start: date = date(2026, 1, 5),
    seed: int = 7,
    base_price: float = 100.0,
    daily_vol: float = 0.01,
    drift: float = 0.0003,
    volume: float = 1_000_000.0,
    available_offset: timedelta = timedelta(hours=21),
) -> MarketData:
    """Deterministic synthetic daily bars on a weekday calendar (``synthetic=True``).

    Prices follow a seeded geometric random walk per instrument; ``available_at`` is the session
    date plus ``available_offset`` (UTC), before the default 22:00 UTC decision time.
    """
    rng = np.random.default_rng(seed)
    sessions = _weekdays(start, n_sessions)
    bars: list[Bar] = []
    for k, iid in enumerate(instruments):
        prev_close = base_price * (1.0 + 0.1 * k)
        for s in sessions:
            gap, ret = rng.normal(0.0, daily_vol / 4), rng.normal(drift, daily_vol)
            o = round(prev_close * (1.0 + gap), 4)
            c = round(o * (1.0 + ret), 4)
            hi, lo = round(max(o, c) * 1.002, 4), round(min(o, c) * 0.998, 4)
            vol = float(round(volume * (1.0 + 0.2 * rng.random())))
            bars.append(Bar(iid, s, c, datetime.combine(s, time(0), tzinfo=UTC) + available_offset, o, hi, lo, vol))
            prev_close = c
    return MarketData(sessions, bars, synthetic=True, dataset_id="synthetic/etf-daily/" + "-".join(instruments))
