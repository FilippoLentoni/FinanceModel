"""Synthetic ETF-shaped snapshot fixtures and the mock provider (DS-08, DS-11; design D6).

Phase 1 datasets are prepared only from **synthetic** snapshots built from the shared contract
fixtures (``finplan-contracts`` ``observation/valid/completed-daily.json`` and
``snapshot-payload/valid/etf-daily.json``, read from the pinned package at run time, never copied).
Every generated observation, payload, manifest and record carries ``synthetic: true``. Values are
a seeded random walk; nothing here is retrieved market data (the repository is public).

:class:`MockSnapshotProvider` is the mock provider: it publishes those synthetic snapshots through
the in-process :class:`~finplan_model.core.platform.FixturePlatformClient`, so dataset preparation
reads them exactly like approved platform snapshots (``SnapshotReader``: approved-only, checksums
verified). CI and build-stage tests use only this provider; no network request is ever made.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta
from functools import lru_cache
from typing import Any

import numpy as np

from finplan_model.core.clock import Clock, FrozenClock, utc_iso
from finplan_model.core.ids import IdMinter
from finplan_model.core.platform import FixturePlatformClient, SnapshotReader, build_synthetic_snapshot

from .calendar import SessionList, fixture_calendar_version, fixture_session_list

__all__ = ["MockSnapshotProvider", "contract_fixture", "fixture_calendar", "synthetic_etf_observations", "synthetic_etf_snapshot"]

FIXTURE_PROVIDER = "fixture"
FIXTURE_LIBRARY = "finplan-fixture-provider"
FIXTURE_LIBRARY_VERSION = "0.0.0"


@lru_cache(maxsize=None)
def _fixture_text(schema: str, name: str) -> str:
    from finplan_contracts.schemas import load_store

    return (load_store().fixtures_dir(schema) / "valid" / name).read_text(encoding="utf-8")


def contract_fixture(schema: str, name: str) -> dict[str, Any]:
    """A *valid* shared contract fixture, read from the pinned package (never copied)."""
    return json.loads(_fixture_text(schema, name))


def fixture_calendar(start: date = date(2018, 1, 1), end: date = date(2027, 12, 31)) -> SessionList:
    """The synthetic fixture calendar (exchange XNYS shape, rule-based holidays, synthetic)."""
    return SessionList("XNYS", fixture_calendar_version(start, end), tuple(fixture_session_list(start, end)), synthetic=True, coverage=(start, end))


def synthetic_etf_observations(
    ticker: str,
    sessions: Sequence[date],
    *,
    seed: int = 7,
    base_price: float = 100.0,
    daily_vol: float = 0.01,
    drift: float = 0.0003,
    volume: int = 5_000_000,
    shocks: Mapping[date, float] | None = None,
) -> list[dict[str, Any]]:
    """Completed daily observations shaped like the contract ``observation`` fixture (synthetic).

    ``shocks`` adds a one-day return on given sessions (for stress fixtures).
    """
    template = contract_fixture("observation", "completed-daily.json")
    rng = np.random.default_rng(seed)
    out: list[dict[str, Any]] = []
    prev = float(base_price)
    for s in sessions:
        gap, ret = rng.normal(0.0, daily_vol / 4), rng.normal(drift, daily_vol)
        if shocks and s in shocks:
            ret += shocks[s]
        o = round(prev * (1.0 + gap), 4)
        c = round(max(o * (1.0 + ret), 0.01), 4)
        obs = {k: template[k] for k in template if k not in ("instrument_id", "session_date", "open", "high", "low", "close", "volume", "adj_close")}
        obs.update(
            instrument_id=ticker,
            session_date=s.isoformat(),
            kind="completed_daily",
            open=o,
            high=round(max(o, c) * 1.002, 4),
            low=round(min(o, c) * 0.998, 4),
            close=c,
            volume=int(volume * (1.0 + 0.2 * float(rng.random()))),
            adj_close=c,
            synthetic=True,
        )
        if "session_status" in template:
            obs["session_status"] = "regular"
        out.append(obs)
        prev = c
    return out


def synthetic_etf_snapshot(
    observations: Sequence[Mapping[str, Any]],
    *,
    input_snapshot_id: str,
    calendar: SessionList,
    retrieved_at: datetime | str,
    created_at: datetime | str | None = None,
    status: str = "approved",
    ticker: str | None = None,
    dataset_kind: str = "etf-daily",
    asset_class: str = "etf",
    provider: str = FIXTURE_PROVIDER,
    lineage_extra: Mapping[str, Any] | None = None,
    quality_flags: Iterable[str] = (),
    quality_details: Mapping[str, Any] | None = None,
    calendar_block: Mapping[str, Any] | None = None,
    synthetic: bool = True,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """A platform-shaped snapshot (record + artifact bytes) for one instrument's observations.

    The manifest's ``calendar`` block carries the exchange, the calendar version and the session
    list covering the snapshot (FM-A5). ``synthetic`` stays true for anything committed here.
    """
    if ticker is None:
        ticker = str(observations[0]["instrument_id"]) if observations else "SPY"
    inst = dict(contract_fixture("snapshot-payload", "etf-daily.json")["instruments"][0])
    inst.update(instrument_id=ticker, asset_class=asset_class, name=f"Synthetic {ticker} series ({dataset_kind})")
    if synthetic:
        inst["synthetic"] = True
    else:
        inst.pop("synthetic", None)
    payload = {"dataset_id": f"finance/{dataset_kind}/{ticker}", "calendar": calendar.exchange, "instruments": [inst], "observations": [dict(o) for o in observations]}
    if not synthetic:
        for o in payload["observations"]:
            o.pop("synthetic", None)
    cov = calendar.covered
    cal_block = (
        dict(calendar_block)
        if calendar_block is not None
        else {"exchange": calendar.exchange, "version": calendar.version, "synthetic": calendar.synthetic, "coverage": {"start": cov[0].isoformat(), "end": cov[1].isoformat()} if cov else None, "sessions": calendar.iso()}
    )
    lineage: dict[str, Any] = {"provider_library": FIXTURE_LIBRARY, "library_version": FIXTURE_LIBRARY_VERSION, "calendar_version": calendar.version}
    lineage.update(dict(lineage_extra or {}))
    lineage = {k: v for k, v in lineage.items() if v is not None}  # None removes a field (tests of incomplete provenance)
    ra = retrieved_at if isinstance(retrieved_at, str) else utc_iso(retrieved_at)
    ca = None if created_at is None else (created_at if isinstance(created_at, str) else utc_iso(created_at))
    return build_synthetic_snapshot(
        payload,
        input_snapshot_id=input_snapshot_id,
        status=status,
        provider=provider,
        retrieved_at=ra,
        created_at=ca,
        lineage_extra=lineage,
        quality_flags=quality_flags,
        quality_details=quality_details,
        calendar=cal_block,
        synthetic=synthetic,
    )


class MockSnapshotProvider:
    """The mock market-data provider of phase 1: publishes synthetic ETF snapshots to the fixture
    platform client and hands out a :class:`SnapshotReader` over it (no network)."""

    def __init__(self, client: FixturePlatformClient | None = None, *, clock: Clock | None = None, ids: IdMinter | None = None, calendar: SessionList | None = None) -> None:
        self.clock = clock or FrozenClock("2026-01-05T00:00:00Z")
        self.client = client or FixturePlatformClient(clock=self.clock)
        self.ids = ids or IdMinter.seeded(self.clock, 4242)
        self.calendar = calendar or fixture_calendar()

    @property
    def reader(self) -> SnapshotReader:
        return SnapshotReader(self.client)

    def new_snapshot_id(self) -> str:
        return f"snap_{self.ids.ulid()}"

    def publish(self, observations: Sequence[Mapping[str, Any]], **kwargs: Any) -> str:
        sid = kwargs.pop("input_snapshot_id", None) or self.new_snapshot_id()
        rec, blobs = synthetic_etf_snapshot(observations, input_snapshot_id=sid, calendar=kwargs.pop("calendar", self.calendar), **kwargs)
        return self.client.add_snapshot(rec, blobs)

    def sessions(self, start: date, end: date) -> list[date]:
        return self.calendar.between(start, end)

    def publish_backfill(self, ticker: str, start: date, end: date, *, seed: int = 7, retrieved_at: datetime | None = None, **series_kwargs: Any) -> str:
        """One snapshot covering ``[start, end]``, retrieved after the last session (a backfill)."""
        sessions = self.sessions(start, end)
        obs = synthetic_etf_observations(ticker, sessions, seed=seed, **series_kwargs)
        ra = retrieved_at or datetime.combine(sessions[-1] + timedelta(days=1), time(13, 0), tzinfo=UTC)
        return self.publish(obs, retrieved_at=ra)

    def publish_daily(
        self,
        ticker: str,
        start: date,
        end: date,
        *,
        seed: int = 7,
        retrieval_time: time = time(21, 15),
        late: Mapping[date, datetime] | None = None,
        skip: Iterable[date] = (),
        **series_kwargs: Any,
    ) -> list[str]:
        """One snapshot per session (daily ingestion), retrieved at ``retrieval_time`` UTC on the
        session date unless listed in ``late``; sessions in ``skip`` get no snapshot."""
        sessions = self.sessions(start, end)
        obs = synthetic_etf_observations(ticker, sessions, seed=seed, **series_kwargs)
        skipped = set(skip)
        ids = []
        for s, o in zip(sessions, obs):
            if s in skipped:
                continue
            ra = (late or {}).get(s) or datetime.combine(s, retrieval_time, tzinfo=UTC)
            ids.append(self.publish([o], retrieved_at=ra))
        return ids
