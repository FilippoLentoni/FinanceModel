"""Untouched holdout with logged access, and the prospective paper period (DS-06, DS-07).

Holdout
-------
The latest evaluation range of a historical dataset is stored in a separate file that only
:class:`HoldoutAccessor` reads. Every attempt is logged (granted or refused) with ``run_id``,
candidate ``model_version``, candidate family, purpose, time and ``correlation_id``:

* only runs with purpose ``holdout_evaluation`` may read it; ``research`` (model selection),
  ``tuning`` (and calibration) and ``production_candidate`` are refused with
  ``OPERATION_NOT_PERMITTED``;
* a holdout evaluation requires a **frozen** candidate (:class:`FrozenCandidate` with a
  ``model_version`` and a freeze time not after now); otherwise ``PRECONDITION_FAILED``
  (``candidate_not_frozen``).

:func:`holdout_access_report` lists every access of a dataset's holdout and flags it as **reused**
when the same candidate family has been evaluated on it in more than one run.

The log is a :class:`HoldoutAccessLog`; :class:`InMemoryHoldoutAccessLog` serves tests and
offline runs, and the control plane provides a run-table implementation (append-only).

Prospective paper period
------------------------
:func:`check_prospective_snapshots` accepts only snapshots ingested (retrieved **and** committed)
after the candidate freeze time; an older snapshot is rejected for the prospective period with
``VALIDATION_FAILED`` naming it. Prospective results are reported separately (period type
``prospective_paper``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from finplan_model.core.clock import parse_utc, utc_iso
from finplan_model.core.context import RunContext
from finplan_model.core.errors import FinplanError
from finplan_model.core.ids import require_id
from finplan_model.core.platform import SnapshotContent
from finplan_model.sim.market import MarketData

from .dataset import Dataset

__all__ = [
    "FrozenCandidate",
    "HoldoutAccessLog",
    "HoldoutAccessor",
    "HoldoutData",
    "InMemoryHoldoutAccessLog",
    "check_prospective_snapshots",
    "holdout_access_report",
]

HOLDOUT_PURPOSE = "holdout_evaluation"


@dataclass(frozen=True)
class FrozenCandidate:
    """A candidate whose code, parameters and artifacts were frozen before evaluation."""

    model_version: str
    family: str
    frozen_at: datetime
    frozen: bool = True

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FrozenCandidate":
        return cls(str(d["model_version"]), str(d["family"]), parse_utc(str(d["frozen_at"])), bool(d.get("frozen", True)))

    def to_dict(self) -> dict[str, Any]:
        return {"model_version": self.model_version, "family": self.family, "frozen_at": utc_iso(self.frozen_at), "frozen": self.frozen}


@runtime_checkable
class HoldoutAccessLog(Protocol):
    def append(self, entry: Mapping[str, Any]) -> None: ...

    def entries(self, dataset_id: str | None = None) -> list[dict[str, Any]]: ...


class InMemoryHoldoutAccessLog:
    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    def append(self, entry: Mapping[str, Any]) -> None:
        self._entries.append(dict(entry))

    def entries(self, dataset_id: str | None = None) -> list[dict[str, Any]]:
        return [dict(e) for e in self._entries if dataset_id is None or e.get("dataset_id") == dataset_id]


@dataclass(frozen=True)
class HoldoutData:
    market: MarketData
    start: str
    end: str
    access: dict[str, Any]


class HoldoutAccessor:
    """The single accessor of a dataset's holdout range."""

    def __init__(self, dataset: Dataset, log: HoldoutAccessLog) -> None:
        self.dataset = dataset
        self.log = log

    def _entry(self, ctx: RunContext, candidate: FrozenCandidate | None, outcome: str, reason: str | None) -> dict[str, Any]:
        e: dict[str, Any] = {
            "dataset_id": self.dataset.dataset_id,
            "run_id": ctx.run_id,
            "model_version": candidate.model_version if candidate else None,
            "family": candidate.family if candidate else None,
            "purpose": ctx.purpose,
            "outcome": outcome,
            "at": ctx.now_ts(),
            "correlation_id": ctx.correlation_id,
        }
        if reason:
            e["reason"] = reason
        return e

    def _refuse(self, ctx: RunContext, candidate: FrozenCandidate | None, err: FinplanError) -> FinplanError:
        entry = self._entry(ctx, candidate, "refused", str(err.details.get("reason") or err.code))
        self.log.append(entry)
        ctx.log("holdout_access_refused", dataset_id=self.dataset.dataset_id, reason=entry["reason"], purpose=ctx.purpose)
        return err

    def read(self, ctx: RunContext, candidate: FrozenCandidate | None) -> HoldoutData:
        bounds = self.dataset.holdout_bounds
        if bounds is None:
            raise FinplanError.precondition("this dataset has no holdout range", reason="dataset_has_no_holdout")
        if ctx.purpose != HOLDOUT_PURPOSE:
            raise self._refuse(ctx, candidate, FinplanError.not_permitted("the holdout is read only by holdout_evaluation runs; model selection, tuning and calibration never read it", reason="holdout_purpose_not_permitted", purpose=ctx.purpose))
        if ctx.run_id is None:
            raise self._refuse(ctx, candidate, FinplanError.precondition("a holdout read needs a run_id", reason="run_id_required"))
        if candidate is None or not candidate.frozen or candidate.frozen_at > ctx.clock.now():
            raise self._refuse(ctx, candidate, FinplanError.precondition("a holdout evaluation requires a frozen candidate", reason="candidate_not_frozen"))
        try:
            require_id("model_version", candidate.model_version)
        except FinplanError as exc:
            raise self._refuse(ctx, candidate, exc) from None
        entry = self._entry(ctx, candidate, "granted", None)
        entry["holdout"] = {"start": bounds.start.isoformat(), "end": bounds.end.isoformat()}
        self.log.append(entry)
        ctx.log("holdout_access_granted", dataset_id=self.dataset.dataset_id, model_version=candidate.model_version, family=candidate.family)
        return HoldoutData(self.dataset._market_with_holdout(), bounds.start.isoformat(), bounds.end.isoformat(), entry)


def holdout_access_report(log: HoldoutAccessLog, dataset_id: str) -> dict[str, Any]:
    """Every access of a dataset's holdout, and whether it was reused by a candidate family."""
    entries = sorted(log.entries(dataset_id), key=lambda e: (str(e.get("at")), str(e.get("run_id"))))
    granted = [e for e in entries if e.get("outcome") == "granted"]
    runs_by_family: dict[str, set[str]] = {}
    for e in granted:
        runs_by_family.setdefault(str(e.get("family")), set()).add(str(e.get("run_id")))
    reused = sorted(f for f, runs in runs_by_family.items() if len(runs) > 1)
    return {
        "dataset_id": dataset_id,
        "accesses": granted,
        "refused_attempts": [e for e in entries if e.get("outcome") == "refused"],
        "evaluations_by_family": {f: len(r) for f, r in sorted(runs_by_family.items())},
        "reused_families": reused,
        "reused": bool(reused),
    }


def check_prospective_snapshots(contents: Sequence[SnapshotContent], candidate: FrozenCandidate) -> None:
    """Refuse any snapshot ingested before the candidate freeze time (DS-07)."""
    for n, c in enumerate(contents):
        rec = c.snapshot.record
        stamps = [parse_utc(str(rec["created_at"]))]
        if c.snapshot.lineage.get("retrieved_at"):
            stamps.append(parse_utc(str(c.snapshot.lineage["retrieved_at"])))
        if min(stamps) <= candidate.frozen_at:
            raise FinplanError.validation(
                "snapshot was ingested before the candidate was frozen and cannot be used for the prospective paper period",
                pointer=f"/input_snapshot_ids/{n}",
                input_snapshot_id=c.snapshot.input_snapshot_id,
                ingested_at=utc_iso(min(stamps)),
                frozen_at=utc_iso(candidate.frozen_at),
                reason="snapshot_predates_candidate_freeze",
            )
