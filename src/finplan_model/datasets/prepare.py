"""Deterministic, content-addressed dataset preparation (DS-01 to DS-13; tasks 3.1 to 3.8).

``prepare_dataset(input_snapshot_ids, config, reader=..., store=..., catalog=..., ctx=...)``:

1. computes the **dataset key** from the sorted ``input_snapshot_id`` list, the preparation
   ``configuration_id``, the container image digest and the preparation code version (and, for a
   prospective dataset, the candidate freeze); if the catalog already holds it, the existing
   dataset is **reused** and nothing is written (DS-01);
2. loads every snapshot through the approved-only, checksum-verified platform path and applies the
   data-source rules (fixture = synthetic only; real data needs approved real snapshots, else
   ``DEPENDENCY_UNAVAILABLE``) (DS-08, DS-11);
3. checks the instrument (configured ETF daily series only; index level and constituents refused,
   DS-09), copies provenance and quality flags into lineage and refuses real snapshots lacking any
   provenance field (DS-12, task 3.8);
4. merges observations: ``intraday_partial`` observations are never daily bars (a date with only an
   intraday observation fails), a completed observation on a non-session date fails, revisions
   follow the revision policy, and flagged or missing sessions are excluded or fail per the quality
   policy - never filled (DS-10, DS-12);
5. stamps each observation's ``available_at`` (``retrieved_at``: first retrieval, never before the
   session close; ``session_close``: the session close plus the publication lag, for historical
   backfills) and computes the point-in-time feature pipeline with the look-ahead check (DS-03);
6. resolves chronological splits with the embargo and generates walk-forward folds (DS-04, DS-05),
   or for a prospective dataset checks the candidate freeze (DS-07);
7. writes canonical-JSON files (holdout bars in their own file), a SHA-256 manifest of them (the
   ``research_dataset`` trusted reference: ``artifact_id`` = ``dataset_id``, ``checksum`` =
   manifest checksum) and the lineage record (DS-02).

Equal inputs, configuration and image digest give byte-identical files and the same manifest
checksum; only the lineage record's ``created_at`` differs, and the record is written once.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from finplan_model.core.artifacts import ArtifactRef, ArtifactStore, canonical_json_bytes, sha256_checksum
from finplan_model.core.clock import parse_utc, utc_iso
from finplan_model.core.context import RunContext
from finplan_model.core.errors import FinplanError
from finplan_model.core.platform import SnapshotContent, SnapshotReader

from .calendar import SessionList, calendar_coverage_gap, merge_session_lists
from .catalog import DatasetCatalog
from .config import PREPARATION_VERSION, PreparationConfig
from .dataset import HOLDOUT_FILE, Dataset
from .features import DatasetBar, compute_features
from .holdout import FrozenCandidate, check_prospective_snapshots
from .sources import REAL_DATA_SOURCE, check_instrument, load_input_snapshots, quality_affected_dates, snapshot_provenance
from .splits import ResolvedRange, resolve_splits, walk_forward_folds

__all__ = ["DATASET_FILE_KIND", "DATASET_KIND", "PreparedDataset", "dataset_key", "prepare_dataset"]

DATASET_KIND = "research_dataset"
DATASET_FILE_KIND = "research_dataset_file"
MANIFEST_VERSION = "research-dataset-manifest-v1"
RECORD_VERSION = "research-dataset-record-v1"
PERIODS = ("historical", "prospective_paper")
_PRICE_FIELDS = ("open", "high", "low", "close", "volume", "adj_close")


@dataclass(frozen=True)
class PreparedDataset:
    record: dict[str, Any]
    reused: bool

    @property
    def dataset_id(self) -> str:
        return str(self.record["dataset_id"])

    @property
    def manifest_checksum(self) -> str:
        return str(self.record["manifest_checksum"])

    @property
    def ref(self) -> dict[str, Any]:
        return dict(self.record["dataset_ref"])

    def load(self, store: ArtifactStore) -> Dataset:
        return Dataset.load(self.record, store)


def dataset_key(input_snapshot_ids: Sequence[str], configuration_id: str, image_digest: str | None, *, period: str = "historical", candidate: FrozenCandidate | None = None) -> str:
    doc: dict[str, Any] = {
        "preparation_version": PREPARATION_VERSION,
        "input_snapshot_ids": sorted(set(input_snapshot_ids)),
        "configuration_id": configuration_id,
        "image_digest": image_digest or "local",
        "period": period,
    }
    if candidate is not None:
        doc["candidate"] = candidate.to_dict()
    return hashlib.sha256(canonical_json_bytes(doc)).hexdigest()


def _session_close(cfg: PreparationConfig, d: date) -> datetime:
    return datetime.combine(d, cfg.assumed_close_utc, tzinfo=UTC)


def _available_at(cfg: PreparationConfig, d: date, retrieved_at: datetime) -> datetime:
    close = _session_close(cfg, d) + timedelta(minutes=cfg.publication_lag_minutes)
    if cfg.availability_mode == "session_close":
        return close
    return max(close, retrieved_at)


def _num(v: Any) -> float | None:
    return None if v is None else float(v)


def prepare_dataset(
    input_snapshot_ids: Sequence[str],
    config: PreparationConfig | Mapping[str, Any],
    *,
    reader: SnapshotReader,
    store: ArtifactStore,
    catalog: DatasetCatalog,
    ctx: RunContext,
    environment_data_source: str = "fixture",
    period: str = "historical",
    candidate: FrozenCandidate | None = None,
) -> PreparedDataset:
    cfg = config if isinstance(config, PreparationConfig) else PreparationConfig.from_dict(config)
    if period not in PERIODS:
        raise FinplanError.validation("period must be historical or prospective_paper", pointer="/period")
    if period == "prospective_paper":
        if candidate is None:
            raise FinplanError.precondition("a prospective paper dataset needs the frozen candidate", reason="candidate_not_frozen")
        if cfg.splits is not None:
            raise FinplanError.validation("a prospective paper dataset has no train/validation/holdout splits", pointer="/splits")
    elif cfg.splits is None:
        raise FinplanError.validation("historical datasets need train, validation and holdout splits", pointer="/splits")
    cfg_id = cfg.configuration_id
    key = dataset_key(input_snapshot_ids, cfg_id, ctx.image_digest, period=period, candidate=candidate)
    dataset_id = f"rds_{key[:40]}"

    existing = catalog.get(key)
    if existing is not None and store.exists(existing["dataset_ref"]):
        ctx.log("dataset_reused", dataset_id=dataset_id, manifest_checksum=existing["manifest_checksum"])
        return PreparedDataset(existing, reused=True)

    contents = load_input_snapshots(reader, input_snapshot_ids, cfg, environment_data_source=environment_data_source)
    if period == "prospective_paper":
        check_prospective_snapshots(contents, candidate)  # type: ignore[arg-type]
    real = cfg.data_source == REAL_DATA_SOURCE

    # ---- instrument, provenance, calendar
    domains = {(c.snapshot.record["domain"], c.snapshot.record["domain_schema_version"]) for c in contents}
    if len(domains) != 1:
        raise FinplanError.validation("input snapshots have different domains or domain schema versions", pointer="/input_snapshot_ids")
    domain, domain_schema_version = next(iter(domains))
    snap_entries: list[dict[str, Any]] = []
    calendars: list[SessionList] = []
    tickers: dict[str, str] = {}
    order = sorted(range(len(contents)), key=lambda i: (str(contents[i].snapshot.lineage.get("retrieved_at")), contents[i].snapshot.input_snapshot_id))
    for i, c in enumerate(contents):
        tickers[c.snapshot.input_snapshot_id] = check_instrument(c, cfg, i)
        prov, sessions = snapshot_provenance(c, cfg, i)
        calendars.append(sessions)
        rec = c.snapshot.record
        snap_entries.append(
            {
                "input_snapshot_id": c.snapshot.input_snapshot_id,
                "manifest_checksum": c.snapshot.manifest_checksum,
                "dataset_id": c.snapshot.dataset_id,
                "instrument_id": tickers[c.snapshot.input_snapshot_id],
                "status": rec["status"],
                "synthetic": bool(c.snapshot.synthetic),
                "created_at": rec["created_at"],
                "coverage": dict(rec["coverage"]),
                "quality_flags": sorted(rec.get("quality_flags") or []),
                "provenance": prov,
            }
        )
    cal_all = merge_session_lists(calendars)
    cov_start = min(date.fromisoformat(c.snapshot.record["coverage"]["start"]) for c in contents)
    cov_end = max(date.fromisoformat(c.snapshot.record["coverage"]["end"]) for c in contents)
    gap = calendar_coverage_gap(calendars, cov_start, cov_end)
    if gap is not None:
        # Without a session list for every day of the range, missing sessions could go unnoticed.
        raise FinplanError.validation("the snapshots' calendar session lists do not cover the dataset range", pointer="/input_snapshot_ids", uncovered_from=gap.isoformat(), reason="calendar_coverage_gap")
    cal = cal_all.restricted(cov_start, cov_end)
    if not len(cal):
        raise FinplanError.validation("the snapshots' calendar lists no session in their coverage", pointer="/input_snapshot_ids")

    # ---- observations (intraday never daily, sessions only, revisions)
    completed: dict[tuple[str, date], tuple[DatasetBar, dict[str, Any]]] = {}
    intraday_dates: dict[str, set[date]] = {}
    revised = 0
    intraday_dropped = 0
    for i in order:
        c = contents[i]
        sid = c.snapshot.input_snapshot_id
        retrieved = parse_utc(str(c.snapshot.lineage["retrieved_at"]))
        for obs in sorted(c.payload.get("observations") or [], key=lambda o: (o["session_date"], o["instrument_id"], o.get("kind", ""))):
            d = date.fromisoformat(str(obs["session_date"]))
            iid = str(obs["instrument_id"])
            if obs.get("kind") != "completed_daily":
                intraday_dropped += 1
                intraday_dates.setdefault(iid, set()).add(d)
                continue
            if d not in cal_all:
                raise FinplanError.validation("observation dated on a day that is not a session of the snapshot's exchange calendar", pointer=f"/input_snapshot_ids/{i}", input_snapshot_id=sid, instrument_id=iid, session_date=d.isoformat(), calendar_version=cal_all.version, reason="non_session_date")
            vals = {k: _num(obs.get(k)) for k in _PRICE_FIELDS}
            bar = DatasetBar(iid, d, float(obs["close"]), _available_at(cfg, d, retrieved), vals["open"], vals["high"], vals["low"], vals["volume"], vals["adj_close"], sid)
            prev = completed.get((iid, d))
            if prev is None:
                completed[(iid, d)] = (bar, vals)
            elif prev[1] != vals:
                if cfg.revisions == "fail":
                    raise FinplanError.validation("an observation was revised between snapshots and the revision policy is fail", pointer="/revisions", instrument_id=iid, session_date=d.isoformat())
                revised += 1  # first_seen: keep what was known first (point in time)
    for iid, dates in sorted(intraday_dates.items()):
        lacking = sorted(d for d in dates if (iid, d) not in completed)
        if lacking:
            raise FinplanError.validation("an intraday observation is not a completed daily bar, and no completed daily observation exists for that date", pointer="/input_snapshot_ids", instrument_id=iid, session_date=lacking[0].isoformat(), dates=[d.isoformat() for d in lacking[:20]], reason="intraday_only")

    # ---- quality flags and missing sessions (exclude or fail; never fill)
    affected: dict[str, set[str]] = {}
    for c in contents:
        for flag in sorted(set(c.snapshot.record.get("quality_flags") or []) & set(cfg.quality_flags)):
            for ds in quality_affected_dates(c, flag, cal_all):
                if cov_start.isoformat() <= ds <= cov_end.isoformat():
                    affected.setdefault(ds, set()).add(flag)
    missing: dict[str, list[str]] = {}
    for ticker in sorted(set(tickers.values())):
        for s in cal.sessions:
            if (ticker, s) not in completed:
                missing.setdefault(s.isoformat(), []).append(ticker)
                affected.setdefault(s.isoformat(), set()).add("missing_sessions")
    if affected and cfg.quality_policy == "fail":
        first = sorted(affected)
        raise FinplanError.validation("input snapshots carry quality flags or missing sessions and the quality policy is fail", pointer="/quality/policy", dates=first[:20], flags=sorted({f for fs in affected.values() for f in fs}), reason="quality_flagged_dates")
    excluded = sorted(affected)
    excluded_set = {date.fromisoformat(x) for x in excluded}
    sessions = [s for s in cal.sessions if s not in excluded_set]
    if len(sessions) < 2:
        raise FinplanError.validation("fewer than two usable sessions remain after quality exclusions", pointer="/quality")
    session_set = set(sessions)
    bars = sorted((b for (iid, d), (b, _v) in completed.items() if d in session_set), key=lambda b: (b.session_date, b.instrument_id))

    # ---- splits / folds / prospective window
    splits_doc: dict[str, Any] | None = None
    folds_doc: list[dict[str, Any]] = []
    dropped = 0
    prospective: dict[str, Any] | None = None
    holdout_start: date | None = None
    if period == "historical":
        rs = resolve_splits(cfg, cal, cal_all)
        splits_doc = rs.to_dict()
        holdout_start = rs.holdout.start
        if cfg.walk_forward is not None:
            span = cal.restricted(rs.train.start, rs.validation.end)
            folds, dropped = walk_forward_folds(cfg.walk_forward, span)
            folds_doc = [f.to_dict() for f in folds]
    else:
        assert candidate is not None
        window = [s for s in sessions if s > candidate.frozen_at.date()]
        if len(window) < 2:
            raise FinplanError.validation("the prospective period has fewer than two sessions after the candidate freeze", pointer="/input_snapshot_ids")
        prospective = {"window": ResolvedRange(window[0], window[-1], len(window)).to_dict(), "candidate": candidate.to_dict()}

    # ---- features (point in time, look-ahead check)
    by_inst: dict[str, list[DatasetBar]] = {}
    for b in bars:
        by_inst.setdefault(b.instrument_id, []).append(b)
    feature_rows = compute_features(cfg.features, by_inst, sessions, cfg.decision_time_utc)

    # ---- files
    synthetic = all(e["synthetic"] for e in snap_entries)
    general_sessions = [s for s in sessions if holdout_start is None or s < holdout_start]
    holdout_sessions = [s for s in sessions if holdout_start is not None and s >= holdout_start]
    gen_set = set(general_sessions)
    files: list[tuple[str, str, dict[str, Any]]] = [
        ("calendar", "general", {"file": "calendar", "exchange": cal.exchange, "version": cal.version, "synthetic": cal.synthetic, "sessions": [s.isoformat() for s in general_sessions], "excluded_sessions": excluded, "synthetic_data": synthetic}),
        ("observations", "general", {"file": "observations", "bars": [b.to_dict() for b in bars if b.session_date in gen_set], "synthetic": synthetic}),
        ("features", "general", {"file": "features", "names": [f.name for f in cfg.features], "rows": [r for r in feature_rows if date.fromisoformat(r["session_date"]) in gen_set], "synthetic": synthetic}),
        ("splits", "general", {"file": "splits", "period": period, "splits": splits_doc, "folds": folds_doc, "dropped_incomplete_folds": dropped, "prospective": prospective, "synthetic": synthetic}),
    ]
    if holdout_sessions:
        hset = set(holdout_sessions)
        files.append((HOLDOUT_FILE, "holdout", {"file": HOLDOUT_FILE, "sessions": [s.isoformat() for s in holdout_sessions], "bars": [b.to_dict() for b in bars if b.session_date in hset], "features": [r for r in feature_rows if date.fromisoformat(r["session_date"]) in hset], "synthetic": synthetic}))
    file_entries = []
    for name, access, doc in files:
        ref = store.put_json(doc, kind=DATASET_FILE_KIND, synthetic=True if synthetic else None, domain=domain)
        file_entries.append({"name": name, "access": access, "artifact_id": ref.artifact_id, "checksum": ref.checksum, "size_bytes": ref.size_bytes})
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "dataset_key": key,
        "preparation_version": PREPARATION_VERSION,
        "configuration_id": cfg_id,
        "image_digest": ctx.image_digest,
        "contract_version": ctx.contract_version,
        "period": period,
        "input_snapshots": [{"input_snapshot_id": e["input_snapshot_id"], "manifest_checksum": e["manifest_checksum"]} for e in sorted(snap_entries, key=lambda e: e["input_snapshot_id"])],
        "files": file_entries,
        "synthetic": synthetic,
    }
    manifest_ref: ArtifactRef = store.put(canonical_json_bytes(manifest), kind=DATASET_KIND, artifact_id=dataset_id, synthetic=True if synthetic else None, domain=domain)

    record: dict[str, Any] = {
        "record_version": RECORD_VERSION,
        "dataset_id": dataset_id,
        "dataset_key": key,
        "dataset_ref": manifest_ref.to_dict(),
        "manifest_checksum": manifest_ref.checksum,
        "input_snapshots": sorted(snap_entries, key=lambda e: e["input_snapshot_id"]),
        "configuration_id": cfg_id,
        "configuration": cfg.to_dict(),
        "preparation_version": PREPARATION_VERSION,
        "image_digest": ctx.image_digest,
        "contract_version": ctx.contract_version,
        "domain": domain,
        "domain_schema_version": domain_schema_version,
        "data_source": cfg.data_source,
        "instrument": {"tickers": list(cfg.tickers), "dataset_ids": sorted(cfg.expected_dataset_ids()), "dataset_kind": cfg.dataset_kind, "asset_class": cfg.asset_class, "calendar": cfg.calendar, "currency": cfg.currency, "granularity": cfg.granularity},
        "calendar": {**cal.describe(), "library": snap_entries[0]["provenance"]["calendar"]["library"], "library_version": snap_entries[0]["provenance"]["calendar"]["library_version"]},
        "coverage": {"start": sessions[0].isoformat(), "end": sessions[-1].isoformat(), "sessions": len(sessions)},
        "availability": {"mode": cfg.availability_mode, "assumed_close_utc": cfg.assumed_close_utc.strftime("%H:%M"), "publication_lag_minutes": cfg.publication_lag_minutes, "decision_time_utc": cfg.decision_time_utc.strftime("%H:%M")},
        "quality": {
            "policy": cfg.quality_policy,
            "considered_flags": list(cfg.quality_flags),
            "flags_by_snapshot": {e["input_snapshot_id"]: e["quality_flags"] for e in snap_entries if e["quality_flags"]},
            "excluded_sessions": excluded,
            "excluded_reasons": {d: sorted(fs) for d, fs in sorted(affected.items())},
            "missing_sessions": {d: t for d, t in sorted(missing.items())},
        },
        "revisions": {"policy": cfg.revisions, "revised_observations": revised},
        "intraday_dropped": intraday_dropped,
        "period": period,
        "splits": splits_doc,
        "folds": folds_doc,
        "dropped_incomplete_folds": dropped,
        "prospective": prospective,
        "synthetic": synthetic,
        "real_data": real,
        "created_at": ctx.now_ts(),
        "created_by_run_id": ctx.run_id,
        "correlation_id": ctx.correlation_id,
    }
    stored, created = catalog.create(key, record)
    ctx.log("dataset_prepared" if created else "dataset_reused", dataset_id=dataset_id, manifest_checksum=manifest_ref.checksum, synthetic=synthetic, sessions=len(sessions), excluded_sessions=len(excluded))
    return PreparedDataset(stored, reused=not created)


def manifest_checksum_of(manifest: Mapping[str, Any]) -> str:
    return sha256_checksum(canonical_json_bytes(manifest))
