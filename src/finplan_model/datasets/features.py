"""Point-in-time feature pipeline and look-ahead check (DS-03).

Every dataset observation carries ``available_at``. Features for a decision at time ``t`` (the
decision time of session ``s``) are computed from observations available at or before ``t``:

* ``alignment: asof`` (default) - an as-of join: only observations with ``session_date <= s`` and
  ``available_at <= t`` are visible, so an observation for day ``d`` retrieved after the decision
  time on ``d`` is excluded for that decision and used from the next decision onward;
* ``alignment: session`` with ``offset`` - the feature references the bar of a fixed session
  relative to ``s`` (``offset`` 0 = the decision session itself, 1 = the next session). Such a
  reference is only legitimate when the referenced observation is already available.

After computing, :func:`compute_features` verifies **every** observation each feature value
referenced: one whose ``available_at`` is after the decision time fails preparation with
``VALIDATION_FAILED`` naming the feature, the observation's session date and its availability
timestamp. With as-of features the check always passes; it catches buggy session-aligned features
(for example a next-session close, or today's close when today's bar arrived late).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from typing import Any

from finplan_model.core.clock import utc_iso
from finplan_model.core.errors import FinplanError

from .config import FeatureSpec

__all__ = ["DatasetBar", "compute_features", "decision_time"]


@dataclass(frozen=True)
class DatasetBar:
    """One completed daily observation in a prepared dataset, with its availability time."""

    instrument_id: str
    session_date: date
    close: float
    available_at: datetime
    open: float | None = None
    high: float | None = None
    low: float | None = None
    volume: float | None = None
    adj_close: float | None = None
    source_snapshot_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"instrument_id": self.instrument_id, "session_date": self.session_date.isoformat(), "kind": "completed_daily", "close": self.close, "available_at": utc_iso(self.available_at), "source_snapshot_id": self.source_snapshot_id}
        for k in ("open", "high", "low", "volume", "adj_close"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DatasetBar":
        from finplan_model.core.clock import parse_utc

        return cls(
            instrument_id=str(d["instrument_id"]),
            session_date=date.fromisoformat(str(d["session_date"])),
            close=float(d["close"]),
            available_at=parse_utc(str(d["available_at"])),
            open=None if d.get("open") is None else float(d["open"]),
            high=None if d.get("high") is None else float(d["high"]),
            low=None if d.get("low") is None else float(d["low"]),
            volume=None if d.get("volume") is None else float(d["volume"]),
            adj_close=None if d.get("adj_close") is None else float(d["adj_close"]),
            source_snapshot_id=str(d.get("source_snapshot_id", "")),
        )


def decision_time(session: date, at: time) -> datetime:
    return datetime.combine(session, at, tzinfo=UTC)


def _value(spec: FeatureSpec, bars: Sequence[DatasetBar]) -> float | None:
    if spec.kind == "close":
        return bars[-1].close if bars else None
    if len(bars) < spec.lookback + 1:
        return None
    closes = [b.close for b in bars[-(spec.lookback + 1) :]]
    if spec.kind == "trailing_return":
        return closes[-1] / closes[0] - 1.0
    rets = [closes[k] / closes[k - 1] - 1.0 for k in range(1, len(closes))]
    if len(rets) < 2:
        return None
    m = sum(rets) / len(rets)
    return math.sqrt(sum((r - m) ** 2 for r in rets) / (len(rets) - 1))


def _asof_refs(bars: Sequence[DatasetBar], upto: int, t: datetime, need: int) -> list[DatasetBar]:
    """The last ``need`` bars among ``bars[:upto]`` available at ``t`` (oldest first)."""
    out: list[DatasetBar] = []
    j = upto - 1
    while j >= 0 and len(out) < need:
        if bars[j].available_at <= t:
            out.append(bars[j])
        j -= 1
    out.reverse()
    return out


def compute_features(
    specs: Sequence[FeatureSpec],
    bars_by_instrument: Mapping[str, Sequence[DatasetBar]],
    sessions: Sequence[date],
    at: time,
) -> list[dict[str, Any]]:
    """Feature rows ``{session_date, instrument_id, values}`` for every session and instrument.

    Raises ``VALIDATION_FAILED`` (look-ahead) when a feature references an observation whose
    ``available_at`` is after the decision time.
    """
    rows: list[dict[str, Any]] = []
    session_index = {s: i for i, s in enumerate(sessions)}
    for iid in sorted(bars_by_instrument):
        bars = sorted(bars_by_instrument[iid], key=lambda b: b.session_date)
        by_session = {b.session_date: b for b in bars}
        p = 0
        for s in sessions:
            while p < len(bars) and bars[p].session_date <= s:
                p += 1
            t = decision_time(s, at)
            values: dict[str, float | None] = {}
            for spec in specs:
                need = 1 if spec.kind == "close" else spec.lookback + 1
                if spec.alignment == "asof":
                    refs = _asof_refs(bars, p, t, need)
                else:
                    i = session_index[s] + spec.offset
                    lo = i - (need - 1)
                    if i >= len(sessions) or lo < 0:
                        refs = []
                    else:
                        refs = [by_session[d] for d in sessions[lo : i + 1] if d in by_session]
                        if len(refs) != need:
                            refs = []
                for b in refs:
                    if b.available_at > t or b.session_date > s:
                        raise FinplanError.validation(
                            "look-ahead: a feature references an observation that is not available at the decision time",
                            pointer="/features/pipeline",
                            feature=spec.name,
                            instrument_id=iid,
                            observation_session_date=b.session_date.isoformat(),
                            observation_available_at=utc_iso(b.available_at),
                            decision_time=utc_iso(t),
                        )
                v = _value(spec, refs)
                values[spec.name] = v if v is None or math.isfinite(v) else None
            rows.append({"session_date": s.isoformat(), "instrument_id": iid, "values": values})
    rows.sort(key=lambda r: (r["session_date"], r["instrument_id"]))
    return rows
