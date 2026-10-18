"""Declared decision opportunities of forward experiments (slot validation, predecessors).

``hourly`` (the original forward experiments): hour-aligned UTC slots, one per hour.

``market_twice_daily_0945_1545_NY`` (``beta-competition-v1``): two decisions per weekday,
at 09:45 and 15:45 America/New_York wall-clock time (DST-correct; the UTC instant moves
with daylight saving). Weekends are rejected. Exchange holidays are not rejected here:
a holiday slot can still record decisions, but its quotes carry ``market_open=false``, so
the regular-session execution rule refuses every fill (explicit execution attempt).

Research collection for the language-model book runs 45 minutes earlier, at 09:00 and
15:00 America/New_York (``RESEARCH_TIMES``).
"""
from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
HOURLY = "hourly"
TWICE_DAILY = "market_twice_daily_0945_1545_NY"
DECISION_TIMES = {TWICE_DAILY: (time(9, 45), time(15, 45))}
RESEARCH_TIMES = (time(9, 0), time(15, 0))
OPPORTUNITIES = (HOURLY, TWICE_DAILY)


def parse_slot(value: str) -> datetime:
    """Timezone-aware instant in UTC; naive or malformed values are rejected."""
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("slot must be an ISO-8601 timestamp") from None
    if parsed.tzinfo is None:
        raise ValueError("aware slot timestamp required")
    return parsed.astimezone(UTC)


def format_slot(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _weekday_wall_clock(instant: datetime, allowed: tuple[time, ...], what: str) -> None:
    local = instant.astimezone(NY)
    if local.weekday() >= 5:
        raise ValueError(f"{what} must fall on a weekday in America/New_York")
    if local.second or local.microsecond or local.time().replace(second=0, microsecond=0) not in allowed:
        raise ValueError(f"{what} must be exactly " + " or ".join(t.strftime("%H:%M") for t in allowed) + " America/New_York")


def validate_slot(slot: str, opportunity: str = HOURLY) -> str:
    """Normalized ``...Z`` slot, or ValueError when it is not a declared opportunity."""
    if opportunity not in OPPORTUNITIES:
        raise ValueError(f"unsupported decision opportunity {opportunity!r}")
    instant = parse_slot(slot)
    if opportunity == HOURLY:
        if instant.minute or instant.second or instant.microsecond:
            raise ValueError("hourly slot must be hour aligned")
    else:
        _weekday_wall_clock(instant, DECISION_TIMES[opportunity], "decision slot")
    return format_slot(instant)


def validate_research_slot(slot: str) -> str:
    instant = parse_slot(slot)
    _weekday_wall_clock(instant, RESEARCH_TIMES, "research slot")
    return format_slot(instant)


def _previous_wall_clock(instant: datetime, allowed: tuple[time, ...]) -> datetime:
    local = instant.astimezone(NY)
    day = local.date()
    for _ in range(14):
        for wall in sorted(allowed, reverse=True):
            candidate = datetime.combine(day, wall, tzinfo=NY)
            if candidate < local and candidate.weekday() < 5:
                return candidate.astimezone(UTC)
        day -= timedelta(days=1)
    raise ValueError("no previous scheduled slot within two weeks")  # pragma: no cover - weekdays always exist


def previous_slot(slot: str, opportunity: str = HOURLY) -> str:
    """The declared opportunity strictly before ``slot``."""
    instant = parse_slot(validate_slot(slot, opportunity))
    if opportunity == HOURLY:
        return format_slot(instant - timedelta(hours=1))
    return format_slot(_previous_wall_clock(instant, DECISION_TIMES[opportunity]))


def research_slots_between(start: str, end: str) -> list[str]:
    """Research slots ``start <= slot <= end``, newest first (bounded to 14 days)."""
    lo, hi = parse_slot(start), parse_slot(end)
    if hi - lo > timedelta(days=14):
        raise ValueError("research slot window is bounded to 14 days")
    found, cursor = [], hi + timedelta(microseconds=1)
    while True:
        candidate = _previous_wall_clock(cursor, RESEARCH_TIMES)
        if candidate < lo:
            return found
        found.append(format_slot(candidate))
        cursor = candidate


def horizon_hour(slot: str) -> str:
    """Hour-aligned instant at or before ``slot`` (the platform ledger's registered horizon)."""
    instant = parse_slot(slot)
    return format_slot(instant.replace(minute=0, second=0, microsecond=0))
