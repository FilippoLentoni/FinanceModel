"""Synthetic snapshot fixtures and a trivial fixture strategy for job-container tests.

All data here is synthetic (``synthetic: true``); nothing is retrieved market data (DS-13)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from finplan_model.core.platform import FixturePlatformClient, build_synthetic_snapshot
from finplan_model.sim.market import synthetic_market

SID = "snap_01KDVDNAZ83BAMMYCEGWF33DPM"


def synthetic_payload(instruments: tuple[str, ...] = ("AGG", "SPY"), n_sessions: int = 60) -> dict[str, Any]:
    market = synthetic_market(instruments, n_sessions=n_sessions, seed=11)
    obs = []
    for iid in market.instruments:
        for b in market.bars_of(iid):
            obs.append({"instrument_id": iid, "session_date": b.session_date.isoformat(), "kind": "completed_daily", "open": b.open, "high": b.high, "low": b.low, "close": b.close, "volume": int(b.volume or 0), "synthetic": True})
    return {
        "dataset_id": "finance/etf-daily/" + "-".join(instruments),
        "calendar": "XNYS",
        "instruments": [{"instrument_id": i, "asset_class": "etf", "currency": "USD", "synthetic": True} for i in instruments],
        "observations": obs,
        "synthetic": True,
    }


def platform_with_snapshot(status: str = "approved", sid: str = SID) -> FixturePlatformClient:
    p = FixturePlatformClient()
    rec, blobs = build_synthetic_snapshot(synthetic_payload(), input_snapshot_id=sid, status=status)
    p.add_snapshot(rec, blobs)
    return p


class FixtureStatic:
    """Equal weights over the visible universe at every rebalance; counts its decisions."""

    calls = 0

    def __init__(self, name: str = "fixture_static") -> None:
        self.name = name

    def decide(self, view: Any, holdings: Any) -> Mapping[str, Any]:
        type(self).calls += 1
        ids = list(view.instruments)
        return {"weights": {i: 1.0 / len(ids) for i in ids}, "cash": 0.0}


class FixtureCash:
    def __init__(self, name: str) -> None:
        self.name = name

    def decide(self, view: Any, holdings: Any) -> Mapping[str, Any]:
        return {"weights": {i: 0.0 for i in view.instruments}, "cash": 1.0, "solution_status": "not_applicable"}
