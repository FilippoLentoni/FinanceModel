"""Input snapshot resolution and checks for dataset preparation (DS-09, DS-11, DS-12 lineage).

* :func:`load_input_snapshots` reads snapshots **only** through the platform snapshot path
  (:class:`~finplan_model.core.platform.SnapshotReader`: approved-only, every checksum verified)
  and enforces the data-source rules:

  - ``data_source: fixture`` (phase 1): every snapshot must be synthetic; a real snapshot is
    ``VALIDATION_FAILED``;
  - ``data_source: platform_snapshots`` (real ETF data): the environment must have approved real
    ETF snapshots (``config/<env>.json`` ``instrument.data_source``); otherwise, and whenever the
    requested snapshot is missing or not approved, the request fails with
    ``DEPENDENCY_UNAVAILABLE`` (``no_approved_real_snapshot``) while fixture-backed preparation
    stays available (DS-11). A synthetic snapshot offered for a real-data request is
    ``VALIDATION_FAILED``.

* :func:`check_instrument` refuses a snapshot whose dataset or instrument is not the configured ETF
  daily series - for example the S&P 500 index level or the constituent universe - with
  ``VALIDATION_FAILED`` naming the mismatch (DS-09).
* :func:`snapshot_provenance` copies the provider lineage (adapter, pinned library and version,
  retrieval timestamp) and the exchange calendar (exchange, calendar version, calendar library and
  version, session list) from a snapshot. For a **real** snapshot every one of them is required;
  a missing field is ``VALIDATION_FAILED`` naming it (contract gap FM-A5 if the platform does not
  provide it), never filled by calling a provider.

FinanceModel never calls a market-data provider: there is no provider client anywhere in this
package (the build-stage provider guard enforces it, DS-12).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from finplan_model.core.errors import ErrorCode, FinplanError
from finplan_model.core.platform import SnapshotContent, SnapshotReader

from .calendar import SessionList, parse_calendar_version
from .config import PreparationConfig

__all__ = ["REAL_DATA_SOURCE", "check_instrument", "load_input_snapshots", "snapshot_provenance"]

REAL_DATA_SOURCE = "platform_snapshots"
#: Providers that only ever serve synthetic data (tests, CI, the mock provider).
SYNTHETIC_PROVIDERS = ("fixture", "mock")
#: Provenance a real snapshot must carry (copied into the dataset lineage record).
REQUIRED_REAL_LINEAGE = ("provider", "provider_library", "library_version", "retrieved_at", "calendar_version")


def _no_real(reason_detail: str, **details: Any) -> FinplanError:
    return FinplanError.dependency_unavailable(
        "no approved real ETF snapshot is available in this environment; fixture-backed preparation remains available",
        reason="no_approved_real_snapshot",
        detail=reason_detail,
        **details,
    )


def load_input_snapshots(reader: SnapshotReader, input_snapshot_ids: Sequence[str], cfg: PreparationConfig, *, environment_data_source: str = "fixture") -> list[SnapshotContent]:
    ids = sorted(set(input_snapshot_ids))
    real = cfg.data_source == REAL_DATA_SOURCE
    if real and environment_data_source != REAL_DATA_SOURCE:
        raise _no_real("environment_serves_fixtures_only")
    if not ids:
        if real:
            raise _no_real("no_snapshot_requested")
        raise FinplanError.validation("dataset preparation needs at least one input_snapshot_id", pointer="/input_snapshot_ids")
    out: list[SnapshotContent] = []
    for n, sid in enumerate(ids):
        try:
            content = reader.load(sid)
        except FinplanError as exc:
            if real and (exc.code == ErrorCode.NOT_FOUND or (exc.code == ErrorCode.PRECONDITION_FAILED and exc.details.get("reason") == "snapshot_not_approved")):
                raise _no_real("snapshot_missing_or_not_approved", input_snapshot_id=sid) from None
            raise
        synthetic = bool(content.snapshot.synthetic)
        if real and synthetic:
            raise FinplanError.validation("a real-data preparation was given a synthetic snapshot", pointer=f"/input_snapshot_ids/{n}", input_snapshot_id=sid)
        if not real and not synthetic:
            raise FinplanError.validation("phase 1 datasets are prepared only from synthetic fixture snapshots (data_source fixture)", pointer=f"/input_snapshot_ids/{n}", input_snapshot_id=sid)
        if not real and str(content.snapshot.lineage.get("provider")) not in SYNTHETIC_PROVIDERS:
            raise FinplanError.validation("fixture preparation accepts only the fixture or mock provider", pointer=f"/input_snapshot_ids/{n}", input_snapshot_id=sid)
        out.append(content)
    return out


def check_instrument(content: SnapshotContent, cfg: PreparationConfig, index: int = 0) -> str:
    """The configured ticker this snapshot carries; ``VALIDATION_FAILED`` on any instrument mismatch."""
    expected = cfg.expected_dataset_ids()
    sid = content.snapshot.input_snapshot_id
    rec_ds = content.snapshot.dataset_id
    pay_ds = str(content.payload.get("dataset_id", ""))
    pointer = f"/input_snapshot_ids/{index}"
    if rec_ds not in expected or pay_ds != rec_ds:
        raise FinplanError.validation(
            "instrument mismatch: the snapshot is not the configured ETF daily series (the index level and the constituent universe are distinct datasets)",
            pointer=pointer,
            input_snapshot_id=sid,
            expected_dataset_ids=sorted(expected),
            snapshot_dataset_id=rec_ds or pay_ds,
        )
    ticker = expected[rec_ds]
    for inst in content.payload.get("instruments") or []:
        if inst.get("instrument_id") != ticker or inst.get("asset_class") != cfg.asset_class:
            raise FinplanError.validation(
                "instrument mismatch: the snapshot's instrument is not the configured ETF",
                pointer=pointer,
                input_snapshot_id=sid,
                expected_instrument_id=ticker,
                expected_asset_class=cfg.asset_class,
                snapshot_instrument_id=str(inst.get("instrument_id")),
                snapshot_asset_class=str(inst.get("asset_class")),
            )
    for obs in content.payload.get("observations") or []:
        if obs.get("instrument_id") != ticker:
            raise FinplanError.validation("instrument mismatch: an observation belongs to another instrument", pointer=pointer, input_snapshot_id=sid, expected_instrument_id=ticker, observation_instrument_id=str(obs.get("instrument_id")))
    return ticker


def snapshot_provenance(content: SnapshotContent, cfg: PreparationConfig, index: int = 0) -> tuple[dict[str, Any], SessionList]:
    """Provenance copied into the lineage record, plus the snapshot's calendar session list."""
    sid = content.snapshot.input_snapshot_id
    synthetic = bool(content.snapshot.synthetic)
    lineage = dict(content.snapshot.lineage)
    cal = dict(content.manifest.get("calendar") or {})
    pointer = f"/input_snapshot_ids/{index}"
    missing: list[str] = []
    if not synthetic:
        missing += [f"lineage.{k}" for k in REQUIRED_REAL_LINEAGE if not lineage.get(k)]
        if lineage.get("provider") in SYNTHETIC_PROVIDERS:
            missing.append("lineage.provider (real provider adapter)")
    if not cal.get("exchange"):
        missing.append("calendar.exchange")
    if not cal.get("version"):
        missing.append("calendar.version")
    if not isinstance(cal.get("sessions"), list):
        missing.append("calendar.sessions")
    parsed = parse_calendar_version(str(lineage.get("calendar_version") or cal.get("version") or ""))
    if not synthetic and parsed is None:
        missing.append("calendar library and version (lineage.calendar_version '<exchange>-<library>-<version>-<start>-<end>')")
    if missing:
        raise FinplanError.validation("snapshot provenance is incomplete; FinanceModel never fills it by calling a provider", pointer=pointer, input_snapshot_id=sid, missing_fields=missing, reason="snapshot_provenance_incomplete")
    if str(cal["exchange"]) != cfg.calendar:
        raise FinplanError.validation("snapshot calendar exchange differs from the configured calendar", pointer=pointer, input_snapshot_id=sid, expected=cfg.calendar, snapshot_calendar=str(cal["exchange"]))
    if not synthetic and lineage.get("calendar_version") != cal.get("version"):
        raise FinplanError.validation("snapshot lineage calendar_version differs from the manifest calendar version", pointer=pointer, input_snapshot_id=sid)
    sessions = SessionList.from_iso(str(cal["exchange"]), str(cal["version"]), cal["sessions"], synthetic=bool(cal.get("synthetic", synthetic)), coverage=cal.get("coverage"))
    if not synthetic and sessions.synthetic:
        raise FinplanError.validation("a real snapshot uses the synthetic fixture calendar", pointer=pointer, input_snapshot_id=sid)
    prov: dict[str, Any] = {
        "provider": lineage.get("provider"),
        "provider_library": lineage.get("provider_library"),
        "library_version": lineage.get("library_version"),
        "retrieved_at": lineage.get("retrieved_at"),
        "calendar": {
            "exchange": str(cal["exchange"]),
            "version": str(cal["version"]),
            "library": parsed["library"] if parsed else None,
            "library_version": parsed["library_version"] if parsed else None,
            "synthetic": sessions.synthetic,
            "sessions": len(sessions),
        },
    }
    return prov, sessions


def quality_affected_dates(content: SnapshotContent, flag: str, sessions: SessionList) -> list[str]:
    """Dates a platform quality flag applies to: from ``quality_details[flag]`` when it lists dates,
    otherwise every session the snapshot covers (the platform did not say which dates)."""
    details: Mapping[str, Any] = content.snapshot.record.get("quality_details") or content.manifest.get("quality_details") or {}
    d = details.get(flag)
    if isinstance(d, Mapping):
        d = d.get("dates") or d.get("session_dates")
    if isinstance(d, list) and d and all(isinstance(x, str) for x in d):
        return sorted(set(d))
    if isinstance(d, list) and d and all(isinstance(x, Mapping) and "session_date" in x for x in d):
        return sorted({str(x["session_date"]) for x in d})
    cov = content.snapshot.record.get("coverage") or {}
    try:
        from datetime import date

        return [s.isoformat() for s in sessions.between(date.fromisoformat(cov["start"]), date.fromisoformat(cov["end"]))]
    except (KeyError, ValueError):
        return []
