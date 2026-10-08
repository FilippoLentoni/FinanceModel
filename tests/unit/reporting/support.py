"""Shared benchmark fixtures for reporting tests (synthetic data only)."""

from __future__ import annotations

import math
from datetime import date
from typing import Any

from finplan_model.datasets import FrozenCandidate, HoldoutAccessor, InMemoryHoldoutAccessLog, holdout_access_report
from finplan_model.reporting import run_benchmark

from ..datasets.support import Env, config, utc

SIM = {"rebalance_frequency": "monthly"}


def benchmark(strategies: list[dict[str, Any]], periods: list[str] | None = None, *, seed: int = 1, store: bool = True) -> dict[str, Any]:
    """Prepare a two-instrument synthetic dataset and run a benchmark (holdout included by default)."""
    env = Env(seed=seed)
    ids = [env.backfill("SPY", seed=1), env.backfill("AGG", seed=2, daily_vol=0.003)]
    ds = env.prepare(ids, config(instrument={"tickers": ["SPY", "AGG"]})).load(env.store)
    ctx = env.ctx.with_run(env.ctx.ids.run_id(), purpose="holdout_evaluation")
    log = InMemoryHoldoutAccessLog()
    cand = FrozenCandidate(ctx.ids.model_version(), "classical_optimizer", utc(2026, 2, 1))
    run = run_benchmark({"strategies": strategies, "periods": periods or ["walk_forward", "validation", "holdout"]}, ds, SIM, ctx=ctx, holdout_accessor=HoldoutAccessor(ds, log), candidate=cand, store=env.store if store else None)
    cost = {"run_id": ctx.run_id, "instance_type": "ml.m5.xlarge", "instance_count": 1, "billed_runtime_seconds": None, "estimated_usd": 0.12, "actual_usd": None}
    return {"env": env, "dataset": ds, "ctx": ctx, "run": run, "access": holdout_access_report(log, ds.dataset_id), "cost": cost, "log": log}


# ----------------------------------------------------------------- hand-computable fixture
def hand_prices(n: int) -> list[float]:
    """Invented prices with ups and downs (open = high = low = close)."""
    return [round(100.0 + 5.0 * math.sin(k / 3.0) + 0.2 * k, 2) for k in range(n)]


def hand_observations(sessions: list[date]) -> list[dict[str, Any]]:
    out = []
    for s, p in zip(sessions, hand_prices(len(sessions))):
        out.append({"instrument_id": "SPY", "session_date": s.isoformat(), "kind": "completed_daily", "session_status": "regular", "open": p, "high": p, "low": p, "close": p, "volume": 10_000_000, "adj_close": p, "synthetic": True})
    return out
