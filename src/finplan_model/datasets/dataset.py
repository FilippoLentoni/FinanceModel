"""A prepared research dataset, loaded from research storage by its lineage record.

:class:`Dataset` verifies the manifest checksum (the ``research_dataset`` trusted reference) and the
checksum of every file it reads, then exposes:

* :meth:`Dataset.market_data` - the **non-holdout** sessions as simulator
  :class:`~finplan_model.sim.market.MarketData` (bars with ``available_at``, one decision time per
  session, ``dataset_id`` and ``dataset_checksum`` = manifest checksum, ``synthetic`` flag);
* :meth:`Dataset.training_market` - only the sessions up to a fold's training end (a fold trains or
  calibrates only on data before its test window, DS-05);
* splits, walk-forward folds, holdout bounds, the prospective window and the feature rows.

Holdout bars live in a separate file that this class never loads on its own: the only way to read
them is :class:`~finplan_model.datasets.holdout.HoldoutAccessor`, which checks the run purpose and
the frozen candidate and logs every read (DS-06).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, time
from typing import Any

from finplan_model.core.artifacts import ArtifactRef, ArtifactStore, canonical_json_bytes, sha256_checksum
from finplan_model.core.errors import ErrorCode, FinplanError
from finplan_model.sim.market import Bar, MarketData

from .catalog import verify_record
from .features import DatasetBar
from .splits import Fold, ResolvedRange, ResolvedSplits

__all__ = ["Dataset"]

HOLDOUT_FILE = "holdout"


class Dataset:
    def __init__(self, record: Mapping[str, Any], manifest: Mapping[str, Any], files: Mapping[str, Any], store: ArtifactStore) -> None:
        self.record = dict(record)
        self.manifest = dict(manifest)
        self._files = dict(files)
        self._store = store

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, record: Mapping[str, Any], store: ArtifactStore) -> "Dataset":
        rec = verify_record(record)
        ref = ArtifactRef.from_dict(rec["dataset_ref"])
        manifest_bytes = store.get(ref)  # checksum verified against the research_dataset reference
        if sha256_checksum(manifest_bytes) != rec["manifest_checksum"]:
            raise FinplanError(ErrorCode.PRECONDITION_FAILED, "dataset manifest checksum does not match its record", details={"reason": "dataset_checksum_mismatch"})
        import json

        manifest = json.loads(manifest_bytes)
        files: dict[str, Any] = {}
        for f in manifest["files"]:
            if f["access"] == "holdout":
                continue
            files[f["name"]] = json.loads(store.get(cls._file_ref(f, rec)))
        return cls(rec, manifest, files, store)

    @staticmethod
    def _file_ref(f: Mapping[str, Any], rec: Mapping[str, Any]) -> ArtifactRef:
        return ArtifactRef(artifact_id=f["artifact_id"], kind="research_dataset_file", checksum=f["checksum"], content_type="application/json", size_bytes=f.get("size_bytes"), synthetic=True if rec.get("synthetic") else None, domain="finance")

    def _read_holdout_file(self) -> dict[str, Any]:
        """Holdout bars and features. Internal: use :class:`HoldoutAccessor`."""
        import json

        for f in self.manifest["files"]:
            if f["name"] == HOLDOUT_FILE:
                return json.loads(self._store.get(self._file_ref(f, self.record)))
        raise FinplanError.precondition("this dataset has no holdout range", reason="dataset_has_no_holdout")

    # ------------------------------------------------------------------ identity
    @property
    def dataset_id(self) -> str:
        return str(self.record["dataset_id"])

    @property
    def manifest_checksum(self) -> str:
        return str(self.record["manifest_checksum"])

    @property
    def ref(self) -> dict[str, Any]:
        return dict(self.record["dataset_ref"])

    @property
    def synthetic(self) -> bool:
        return bool(self.record["synthetic"])

    @property
    def period(self) -> str:
        return str(self.record["period"])

    @property
    def decision_time_utc(self) -> time:
        return time.fromisoformat(self.record["configuration"]["decision_time_utc"])

    # ------------------------------------------------------------------ structure
    @property
    def sessions(self) -> list[date]:
        """Non-holdout dataset sessions (excluded sessions removed)."""
        return [date.fromisoformat(s) for s in self._files["calendar"]["sessions"]]

    @property
    def splits(self) -> ResolvedSplits | None:
        sp = self.record.get("splits")
        return None if sp is None else ResolvedSplits.from_dict(sp)

    @property
    def folds(self) -> list[Fold]:
        return [Fold.from_dict(f) for f in self.record.get("folds") or []]

    @property
    def validation(self) -> ResolvedRange | None:
        sp = self.splits
        return None if sp is None else sp.validation

    @property
    def holdout_bounds(self) -> ResolvedRange | None:
        sp = self.splits
        return None if sp is None else sp.holdout

    @property
    def prospective_window(self) -> ResolvedRange | None:
        p = self.record.get("prospective")
        return None if not p else ResolvedRange.from_dict(p["window"])

    def bars(self) -> list[DatasetBar]:
        return [DatasetBar.from_dict(b) for b in self._files["observations"]["bars"]]

    def features(self) -> list[dict[str, Any]]:
        return list(self._files["features"]["rows"])

    # ------------------------------------------------------------------ market data
    def _market(self, bars: list[DatasetBar], sessions: list[date]) -> MarketData:
        return MarketData(
            sessions,
            [Bar(b.instrument_id, b.session_date, b.close, b.available_at, b.open, b.high, b.low, b.volume) for b in bars],
            decision_time_utc=self.decision_time_utc,
            synthetic=self.synthetic,
            dataset_id=self.dataset_id,
            dataset_checksum=self.manifest_checksum,
        )

    def market_data(self) -> MarketData:
        """Non-holdout market data (everything before the holdout range)."""
        return self._market(self.bars(), self.sessions)

    def training_market(self, fold: Fold) -> MarketData:
        """Only sessions up to the fold's training end: what a fold may train or calibrate on."""
        end = fold.train.end
        return self._market([b for b in self.bars() if b.session_date <= end], [s for s in self.sessions if s <= end])

    def _market_with_holdout(self) -> MarketData:
        h = self._read_holdout_file()
        bars = self.bars() + [DatasetBar.from_dict(b) for b in h["bars"]]
        sessions = self.sessions + [date.fromisoformat(s) for s in h["sessions"]]
        return self._market(bars, sessions)

    def describe(self) -> dict[str, Any]:
        """Identity block used by results and reports (no storage locations)."""
        return {
            "dataset_id": self.dataset_id,
            "dataset_ref": self.ref,
            "manifest_checksum": self.manifest_checksum,
            "configuration_id": self.record["configuration_id"],
            "synthetic": self.synthetic,
            "period": self.period,
        }


def canonical_checksum(doc: Any) -> str:
    return sha256_checksum(canonical_json_bytes(doc))
