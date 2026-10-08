"""Injectable clocks. Code never calls ``datetime.now()`` directly; it receives a :class:`Clock`.

Tests use :class:`FrozenClock` so timestamps, ULIDs and approval expiries are deterministic.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Protocol, runtime_checkable

__all__ = ["Clock", "FrozenClock", "SystemClock", "parse_date", "parse_utc", "utc_iso"]


@runtime_checkable
class Clock(Protocol):
    def now(self) -> datetime:
        """Current time as a timezone-aware UTC datetime."""
        ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class FrozenClock:
    """A clock that only moves when told to (``advance``/``set``), or by ``tick`` on every read."""

    def __init__(self, start: datetime | str = "2026-01-05T00:00:00Z", *, tick: timedelta | None = None) -> None:
        self._now = parse_utc(start) if isinstance(start, str) else _aware(start)
        self._tick = tick

    def now(self) -> datetime:
        current = self._now
        if self._tick:
            self._now = self._now + self._tick
        return current

    def advance(self, delta: timedelta | None = None, **kwargs: float) -> datetime:
        self._now = self._now + (delta or timedelta(**kwargs))
        return self._now

    def set(self, when: datetime | str) -> None:
        self._now = parse_utc(when) if isinstance(when, str) else _aware(when)


def _aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise ValueError("naive datetimes are not allowed; use UTC-aware datetimes")
    return dt.astimezone(UTC)


def utc_iso(dt: datetime) -> str:
    """Contract timestamp: RFC 3339 UTC ending in ``Z`` (microseconds kept only when non-zero)."""
    dt = _aware(dt)
    if dt.microsecond:
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    return _aware(dt)


def parse_date(value: str | date) -> date:
    return value if isinstance(value, date) and not isinstance(value, datetime) else date.fromisoformat(str(value))
