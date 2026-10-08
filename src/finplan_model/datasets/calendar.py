"""Session lists for trading-day arithmetic (spec research-datasets; design D6).

FinanceModel has **no calendar library dependency**. Trading-day arithmetic (embargo lengths, fold
boundaries, missing-session detection) uses the session list carried with the snapshots:

* real ETF snapshots: the XNYS session list from the platform's pinned ``exchange_calendars``
  calendar, named by ``lineage.calendar_version`` (for example
  ``xnys-exchange_calendars-<library_version>-<YYYYMMDD>-<YYYYMMDD>``) and listed in the snapshot
  manifest under ``calendar.sessions`` (design FM-A5; the platform manifest of contracts 0.2.2 and 1.0.0 names
  the calendar but does not yet list its sessions - recorded as a contract gap, and a real snapshot
  without the list is refused rather than guessed);
* synthetic fixtures: :func:`fixture_session_list`, a rule-based synthetic calendar (weekdays minus
  rule-based holidays; never real exchange data), version ``fixture-synthetic-v1-<start>-<end>``
  like the platform's fixture calendar.
"""

from __future__ import annotations

import bisect
import calendar as _stdlib_calendar
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from finplan_model.core.errors import FinplanError

__all__ = [
    "CALENDAR_VERSION_RE",
    "SessionList",
    "add_months",
    "fixture_session_list",
    "calendar_coverage_gap",
    "fixture_calendar_version",
    "merge_session_lists",
    "parse_calendar_version",
]

#: ``<exchange>-<library>-<library_version>-<YYYYMMDD>-<YYYYMMDD>`` (platform calendar artifacts).
CALENDAR_VERSION_RE = re.compile(r"^(?P<exchange>[a-z0-9]+)-(?P<library>[a-z][a-z0-9_]*)-(?P<library_version>[0-9][A-Za-z0-9.+]*)-(?P<start>\d{8})-(?P<end>\d{8})\Z")


def parse_calendar_version(version: str) -> dict[str, str] | None:
    """Exchange, library and library version encoded in a platform calendar version string."""
    m = CALENDAR_VERSION_RE.match(version or "")
    if not m:
        return None
    return {"exchange": m["exchange"].upper(), "library": m["library"], "library_version": m["library_version"], "coverage_start": m["start"], "coverage_end": m["end"]}


def add_months(d: date, months: int) -> date:
    """``d`` plus ``months`` calendar months (day clamped to the target month's length)."""
    y, m = divmod(d.month - 1 + months, 12)
    year, month = d.year + y, m + 1
    return date(year, month, min(d.day, _stdlib_calendar.monthrange(year, month)[1]))


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _observed(d: date) -> date:
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _fixture_holidays(year: int) -> set[date]:
    """Synthetic rule-based holidays (not an exchange's real holiday list)."""
    return {
        _observed(date(year, 1, 1)),
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 11, 3, 4),  # fourth Thursday of November
        _observed(date(year, 12, 25)),
    }


def fixture_calendar_version(start: date, end: date) -> str:
    return f"fixture-synthetic-v1-{start:%Y%m%d}-{end:%Y%m%d}"


def fixture_session_list(start: date, end: date) -> list[date]:
    """Sessions of the synthetic fixture calendar in ``[start, end]`` (weekdays minus rule holidays)."""
    out: list[date] = []
    holidays: dict[int, set[date]] = {}
    d = start
    while d <= end:
        if d.weekday() < 5:
            hs = holidays.setdefault(d.year, _fixture_holidays(d.year))
            if d not in hs:
                out.append(d)
        d += timedelta(days=1)
    return out


@dataclass(frozen=True)
class SessionList:
    """An ordered, immutable list of exchange sessions with trading-day arithmetic."""

    exchange: str
    version: str
    sessions: tuple[date, ...]
    synthetic: bool = True
    #: Date range the session list is complete for (defaults to its first and last session).
    coverage: tuple[date, date] | None = None

    def __post_init__(self) -> None:
        if list(self.sessions) != sorted(set(self.sessions)):
            raise FinplanError.validation("calendar session list must be strictly increasing", pointer="/calendar/sessions")

    @classmethod
    def from_iso(cls, exchange: str, version: str, sessions: Iterable[str], *, synthetic: bool, coverage: Any = None) -> "SessionList":
        try:
            parsed = sorted({date.fromisoformat(str(s)) for s in sessions})
            cov = None if not coverage else (date.fromisoformat(str(coverage["start"])), date.fromisoformat(str(coverage["end"])))
        except (ValueError, KeyError, TypeError):
            raise FinplanError.validation("calendar session list or coverage contains a value that is not a date", pointer="/calendar/sessions") from None
        return cls(exchange, version, tuple(parsed), synthetic, cov)

    @property
    def covered(self) -> tuple[date, date] | None:
        if self.coverage is not None:
            return self.coverage
        return (self.sessions[0], self.sessions[-1]) if self.sessions else None

    def __len__(self) -> int:
        return len(self.sessions)

    def __contains__(self, d: object) -> bool:
        if not isinstance(d, date):
            return False
        i = bisect.bisect_left(self.sessions, d)
        return i < len(self.sessions) and self.sessions[i] == d

    def index(self, d: date) -> int:
        i = bisect.bisect_left(self.sessions, d)
        if i >= len(self.sessions) or self.sessions[i] != d:
            raise KeyError(d)
        return i

    def first_on_or_after(self, d: date) -> int | None:
        i = bisect.bisect_left(self.sessions, d)
        return i if i < len(self.sessions) else None

    def last_on_or_before(self, d: date) -> int | None:
        i = bisect.bisect_right(self.sessions, d) - 1
        return i if i >= 0 else None

    def between(self, start: date, end: date) -> list[date]:
        lo, hi = bisect.bisect_left(self.sessions, start), bisect.bisect_right(self.sessions, end)
        return list(self.sessions[lo:hi])

    def sessions_strictly_between(self, a: date, b: date) -> int:
        """Number of sessions ``s`` with ``a < s < b`` (the embargo gap between two ranges)."""
        return max(0, bisect.bisect_left(self.sessions, b) - bisect.bisect_right(self.sessions, a))

    def restricted(self, start: date, end: date) -> "SessionList":
        return SessionList(self.exchange, self.version, tuple(self.between(start, end)), self.synthetic, (start, end))

    def describe(self) -> dict[str, Any]:
        return {"exchange": self.exchange, "version": self.version, "synthetic": self.synthetic, "count": len(self.sessions), "start": self.sessions[0].isoformat() if self.sessions else None, "end": self.sessions[-1].isoformat() if self.sessions else None}

    def iso(self) -> list[str]:
        return [s.isoformat() for s in self.sessions]


def merge_session_lists(lists: Sequence[SessionList]) -> SessionList:
    """Union of the session lists of several snapshots (one calendar version required)."""
    if not lists:
        raise FinplanError.validation("no calendar session list", pointer="/calendar")
    versions = sorted({sl.version for sl in lists})
    exchanges = sorted({sl.exchange for sl in lists})
    if len(versions) > 1 or len(exchanges) > 1:
        raise FinplanError.validation("input snapshots use different session calendars", pointer="/input_snapshot_ids", calendar_versions=versions[:5], exchanges=exchanges)
    union = sorted({s for sl in lists for s in sl.sessions})
    return SessionList(exchanges[0], versions[0], tuple(union), all(sl.synthetic for sl in lists))


def calendar_coverage_gap(lists: Sequence[SessionList], start: date, end: date) -> date | None:
    """First day of ``[start, end]`` that no session list covers (None when fully covered)."""
    intervals = sorted(sl.covered for sl in lists if sl.covered is not None)
    cursor = start
    for a, b in intervals:
        if a > cursor:
            break
        if b >= cursor:
            cursor = b + timedelta(days=1)
        if cursor > end:
            return None
    return cursor if cursor <= end else None
