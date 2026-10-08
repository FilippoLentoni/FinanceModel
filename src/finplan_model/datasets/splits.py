"""Chronological splits with embargo and walk-forward folds (DS-04, DS-05).

* :func:`resolve_splits` maps the configured ``train``, ``validation`` and ``holdout`` date ranges
  onto the dataset's sessions and checks that adjacent ranges are separated by at least the
  configured embargo, counted in **trading sessions** of the snapshot calendar (sessions strictly
  between the end of one range and the start of the next).
* :func:`walk_forward_folds` generates expanding or rolling folds over the pre-holdout span (from the
  start of training to the end of validation). Each fold's test window starts after its training
  window ends plus the embargo, so a fold trains or calibrates only on data before its test window.
  Consecutive test windows never overlap. A fold whose test window would run past the span is
  dropped (and counted).

Nothing here shuffles: evaluation splits are chronological only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from finplan_model.core.errors import FinplanError

from .calendar import SessionList, add_months
from .config import Length, PreparationConfig, SplitRange, WalkForwardConfig

__all__ = ["Fold", "ResolvedRange", "ResolvedSplits", "resolve_splits", "walk_forward_folds"]


@dataclass(frozen=True)
class ResolvedRange:
    start: date
    end: date
    sessions: int

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat(), "sessions": self.sessions}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ResolvedRange":
        return cls(date.fromisoformat(d["start"]), date.fromisoformat(d["end"]), int(d["sessions"]))


@dataclass(frozen=True)
class Fold:
    index: int
    train: ResolvedRange
    test: ResolvedRange
    embargo_sessions: int
    gap_sessions: int

    @property
    def fold_id(self) -> str:
        return f"fold-{self.index:03d}"

    def to_dict(self) -> dict[str, Any]:
        return {"fold_id": self.fold_id, "index": self.index, "train": self.train.to_dict(), "test": self.test.to_dict(), "embargo_sessions": self.embargo_sessions, "gap_sessions": self.gap_sessions}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Fold":
        return cls(int(d["index"]), ResolvedRange.from_dict(d["train"]), ResolvedRange.from_dict(d["test"]), int(d["embargo_sessions"]), int(d["gap_sessions"]))


@dataclass(frozen=True)
class ResolvedSplits:
    train: ResolvedRange
    validation: ResolvedRange
    holdout: ResolvedRange
    embargo_sessions: int
    gaps: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {"train": self.train.to_dict(), "validation": self.validation.to_dict(), "holdout": self.holdout.to_dict(), "embargo_sessions": self.embargo_sessions, "gap_sessions": dict(self.gaps)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ResolvedSplits":
        return cls(ResolvedRange.from_dict(d["train"]), ResolvedRange.from_dict(d["validation"]), ResolvedRange.from_dict(d["holdout"]), int(d["embargo_sessions"]), dict(d["gap_sessions"]))


def _resolve(cal: SessionList, name: str, r: SplitRange) -> ResolvedRange:
    sessions = cal.between(r.start, r.end)
    if not sessions:
        raise FinplanError.validation(f"splits.{name} contains no session of the dataset", pointer=f"/splits/{name}", start=r.start.isoformat(), end=r.end.isoformat())
    return ResolvedRange(sessions[0], sessions[-1], len(sessions))


def resolve_splits(cfg: PreparationConfig, cal: SessionList, full_calendar: SessionList | None = None) -> ResolvedSplits:
    """Resolve the configured ranges on ``cal`` (the dataset's sessions). With ``full_calendar``,
    a range reaching sessions the data does not cover (before its first or after its last
    session) is refused instead of silently truncated."""
    if cfg.splits is None:
        raise FinplanError.validation("historical datasets need train, validation and holdout splits", pointer="/splits")
    if full_calendar is not None:
        before = full_calendar.between(cfg.splits["train"].start, cal.sessions[0])[:-1] if cfg.splits["train"].start < cal.sessions[0] else []
        after = full_calendar.between(cal.sessions[-1], cfg.splits["holdout"].end)[1:] if cfg.splits["holdout"].end > cal.sessions[-1] else []
        if before or after:
            raise FinplanError.validation("split ranges include sessions outside the dataset's coverage", pointer="/splits", coverage_start=cal.sessions[0].isoformat(), coverage_end=cal.sessions[-1].isoformat())
    tr = _resolve(cal, "train", cfg.splits["train"])
    va = _resolve(cal, "validation", cfg.splits["validation"])
    ho = _resolve(cal, "holdout", cfg.splits["holdout"])
    emb = cfg.split_embargo_sessions
    gaps = {"train_validation": cal.sessions_strictly_between(tr.end, va.start), "validation_holdout": cal.sessions_strictly_between(va.end, ho.start)}
    for key, ptr in (("train_validation", "/splits/validation/start"), ("validation_holdout", "/splits/holdout/start")):
        if gaps[key] < emb:
            raise FinplanError.validation("adjacent split ranges are closer than the embargo", pointer=ptr, gap_sessions=gaps[key], embargo_sessions=emb, between=key)
    return ResolvedSplits(tr, va, ho, emb, gaps)


def _window_end(cal: SessionList, start_idx: int, length: Length) -> int | None:
    """Index of the last session of a window of ``length`` starting at ``start_idx`` (None if past the end)."""
    if length.unit == "sessions":
        end = start_idx + length.value - 1
        return end if end < len(cal) else None
    boundary = add_months(cal.sessions[start_idx], length.value)  # exclusive end date
    nominal_end = date.fromordinal(boundary.toordinal() - 1)
    if nominal_end > cal.sessions[-1]:
        return None  # the span ends before the window's nominal end: incomplete window
    idx = cal.last_on_or_before(nominal_end)
    return idx if idx is not None and idx >= start_idx else None


def walk_forward_folds(wf: WalkForwardConfig, span: SessionList) -> tuple[list[Fold], int]:
    """Folds over ``span`` (the pre-holdout sessions); returns ``(folds, dropped_incomplete)``."""
    folds: list[Fold] = []
    dropped = 0
    n = len(span)
    if n == 0:
        raise FinplanError.validation("walk-forward span has no sessions", pointer="/walk_forward")
    start_date = span.sessions[0]
    for k in range(100_000):
        if wf.train_length.unit == "sessions":
            train_end = wf.train_length.value + k * wf.step.value - 1
            train_start = 0 if wf.window == "expanding" else k * wf.step.value
            if train_end >= n:
                break
        else:
            boundary = add_months(start_date, wf.train_length.value + k * wf.step.value)
            if boundary > span.sessions[-1]:
                break
            te = span.last_on_or_before(date.fromordinal(boundary.toordinal() - 1))
            if te is None:
                break
            train_end = te
            if wf.window == "expanding":
                train_start = 0
            else:
                ts = span.first_on_or_after(add_months(start_date, k * wf.step.value))
                if ts is None or ts > train_end:
                    break
                train_start = ts
        test_start = train_end + wf.embargo_sessions + 1
        if folds:
            # consecutive test windows never overlap (calendar-unit rounding can make them touch)
            test_start = max(test_start, span.index(folds[-1].test.end) + 1)
        if test_start >= n:
            dropped += 1
            break
        test_end = _window_end(span, test_start, wf.test_length)
        if test_end is None:
            dropped += 1
            break
        gap = span.sessions_strictly_between(span.sessions[train_end], span.sessions[test_start])
        if gap < wf.embargo_sessions:  # pragma: no cover - guaranteed by construction
            raise FinplanError.internal("walk-forward fold violates its embargo", fold=k)
        folds.append(
            Fold(
                index=k,
                train=ResolvedRange(span.sessions[train_start], span.sessions[train_end], train_end - train_start + 1),
                test=ResolvedRange(span.sessions[test_start], span.sessions[test_end], test_end - test_start + 1),
                embargo_sessions=wf.embargo_sessions,
                gap_sessions=gap,
            )
        )
    if len(folds) < wf.min_folds:
        raise FinplanError.validation("walk-forward configuration yields fewer folds than min_folds over the pre-holdout span", pointer="/walk_forward", folds=len(folds), min_folds=wf.min_folds, span_start=span.sessions[0].isoformat(), span_end=span.sessions[-1].isoformat())
    return folds, dropped
