"""Helpers for dataset tests: synthetic snapshots through the mock provider (values are invented)."""

from __future__ import annotations

import copy
from datetime import UTC, date, datetime
from typing import Any

from finplan_model.core.artifacts import InMemoryArtifactStore
from finplan_model.core.clock import FrozenClock
from finplan_model.core.context import RunContext
from finplan_model.datasets import InMemoryDatasetCatalog, MockSnapshotProvider, prepare_dataset

#: A short historical configuration on 2020-2022 synthetic sessions (session_close availability:
#: one backfill snapshot per instrument).
BASE_CONFIG: dict[str, Any] = {
    "instrument": {"tickers": ["SPY"]},
    "availability": {"mode": "session_close"},
    "features": {"lookback_sessions": 20, "label_horizon_sessions": 20},
    "splits": {
        "train": {"start": "2020-01-01", "end": "2021-06-30"},
        "validation": {"start": "2021-08-02", "end": "2022-03-31"},
        "holdout": {"start": "2022-05-02", "end": "2022-12-31"},
        "embargo_sessions": 20,
    },
    "walk_forward": {"window": "expanding", "train_length": {"months": 12}, "test_length": {"months": 3}, "step": {"months": 3}, "embargo_sessions": 20},
}


def config(**overrides: Any) -> dict[str, Any]:
    cfg = copy.deepcopy(BASE_CONFIG)
    for k, v in overrides.items():
        if v is None:
            cfg.pop(k, None)
        elif isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    return cfg


def context(seed: int = 1, start: str = "2026-03-01T00:00:00Z", **kw: Any) -> RunContext:
    return RunContext.for_tests(seed=seed, start=start, **kw)


def provider(ctx: RunContext | None = None) -> MockSnapshotProvider:
    clock = ctx.clock if ctx is not None else FrozenClock("2026-03-01T00:00:00Z")
    return MockSnapshotProvider(clock=clock)


class Env:
    """A mock provider, an artifact store and a dataset catalog sharing one run context."""

    def __init__(self, seed: int = 1) -> None:
        self.ctx = context(seed)
        self.provider = provider(self.ctx)
        self.store = InMemoryArtifactStore()
        self.catalog = InMemoryDatasetCatalog()

    def backfill(self, ticker: str = "SPY", start: date = date(2020, 1, 1), end: date = date(2022, 12, 31), **kw: Any) -> str:
        return self.provider.publish_backfill(ticker, start, end, **kw)

    def prepare(self, ids: list[str], cfg: dict[str, Any] | None = None, **kw: Any):  # noqa: ANN201
        return prepare_dataset(ids, cfg or config(), reader=self.provider.reader, store=self.store, catalog=self.catalog, ctx=kw.pop("ctx", self.ctx), **kw)


def utc(y: int, m: int, d: int, hh: int = 0, mm: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=UTC)
