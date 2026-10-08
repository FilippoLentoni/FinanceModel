"""Approved snapshot -> :class:`~finplan_model.sim.market.MarketData` inside a job (task 6.2).

The snapshot is resolved through the platform API by ``input_snapshot_id`` only (never a
caller-supplied location), refused unless ``approved``, and every artifact's SHA-256 is recomputed
and compared with the snapshot manifest **before any strategy code runs** (WS-03, WS-04): a mismatch
raises ``PRECONDITION_FAILED`` (``snapshot_checksum_mismatch``) and the job writes a failed result.

Conversion to market data prefers the research-datasets hook when task group 3 provides one
(``finplan_model.datasets.market_from_snapshot(content, ...)``). Otherwise observations become
completed daily bars available at the session date plus :data:`DEFAULT_AVAILABILITY_OFFSET`
(the synthetic fixtures' convention), ``intraday_partial`` observations are dropped, and the
session calendar is the snapshot's own session list.
"""

from __future__ import annotations

import importlib
from datetime import UTC, date, datetime, time, timedelta

from finplan_model.core.platform import PlatformClient, SnapshotContent, SnapshotReader
from finplan_model.sim.market import Bar, MarketData

__all__ = ["DEFAULT_AVAILABILITY_OFFSET", "load_market", "market_from_content"]

DEFAULT_AVAILABILITY_OFFSET = timedelta(hours=21)


def market_from_content(content: SnapshotContent) -> MarketData:
    try:
        ds = importlib.import_module("finplan_model.datasets")
        hook = getattr(ds, "market_from_snapshot", None)
    except ImportError:  # pragma: no cover
        hook = None
    if callable(hook):
        return hook(content)
    obs = list(content.payload.get("observations", []))
    bars = []
    sessions: set[date] = set()
    for o in obs:
        d = date.fromisoformat(str(o["session_date"]))
        sessions.add(d)
        available = datetime.combine(d, time(0), tzinfo=UTC) + DEFAULT_AVAILABILITY_OFFSET
        bars.append(Bar.from_observation(o, available))
    calendar = content.manifest.get("calendar") or {}
    for s in calendar.get("sessions", []) if isinstance(calendar, dict) else []:
        sessions.add(date.fromisoformat(str(s)))
    return MarketData(
        sorted(sessions),
        bars,
        synthetic=bool(content.snapshot.synthetic or content.payload.get("synthetic")),
        dataset_id=content.snapshot.dataset_id or str(content.payload.get("dataset_id") or "") or None,
        dataset_checksum=content.snapshot.manifest_checksum,
    )


def load_market(platform: PlatformClient, input_snapshot_id: str) -> tuple[MarketData, SnapshotContent]:
    """Resolve (approved only), download, verify every checksum, then build market data."""
    content = SnapshotReader(platform).load(input_snapshot_id)
    return market_from_content(content), content
