"""DS-06 (untouched holdout with logged access) and DS-07 (prospective paper period); task 3.5."""

from __future__ import annotations

from datetime import date

import pytest

from finplan_model.core.errors import FinplanError
from finplan_model.datasets import FrozenCandidate, HoldoutAccessor, InMemoryHoldoutAccessLog, holdout_access_report

from .support import Env, config, utc


@pytest.fixture(scope="module")
def prepared():
    env = Env()
    prep = env.prepare([env.backfill()])
    return env, prep.load(env.store)


def _candidate(env, family="classical_optimizer", frozen_at=None):
    return FrozenCandidate(env.ctx.ids.model_version(), family, frozen_at or utc(2026, 2, 1))


def _ctx(env, purpose):
    return env.ctx.with_run(env.ctx.ids.run_id(), purpose=purpose)


@pytest.mark.parametrize("purpose", ["tuning", "research", "production_candidate"])
def test_non_holdout_purposes_are_refused_and_logged(prepared, purpose):
    env, ds = prepared
    log = InMemoryHoldoutAccessLog()
    with pytest.raises(FinplanError) as ei:
        HoldoutAccessor(ds, log).read(_ctx(env, purpose), _candidate(env))
    assert ei.value.code == "OPERATION_NOT_PERMITTED"
    (entry,) = log.entries()
    assert entry["outcome"] == "refused" and entry["purpose"] == purpose and entry["run_id"].startswith("run_")


def test_holdout_evaluation_needs_a_frozen_candidate(prepared):
    env, ds = prepared
    log = InMemoryHoldoutAccessLog()
    acc = HoldoutAccessor(ds, log)
    for cand in (None, FrozenCandidate(env.ctx.ids.model_version(), "x", utc(2026, 2, 1), frozen=False), _candidate(env, frozen_at=utc(2027, 1, 1))):
        with pytest.raises(FinplanError) as ei:
            acc.read(_ctx(env, "holdout_evaluation"), cand)
        assert ei.value.code == "PRECONDITION_FAILED" and ei.value.details["reason"] == "candidate_not_frozen"
    assert [e["outcome"] for e in log.entries()] == ["refused"] * 3


def test_granted_read_is_logged_and_returns_only_holdout_window(prepared):
    env, ds = prepared
    log = InMemoryHoldoutAccessLog()
    ctx = _ctx(env, "holdout_evaluation")
    cand = _candidate(env)
    data = HoldoutAccessor(ds, log).read(ctx, cand)
    assert data.start == ds.holdout_bounds.start.isoformat() and data.end == ds.holdout_bounds.end.isoformat()
    assert data.market.sessions[-1] == ds.holdout_bounds.end
    (entry,) = log.entries(ds.dataset_id)
    assert entry["outcome"] == "granted" and entry["run_id"] == ctx.run_id and entry["model_version"] == cand.model_version and entry["purpose"] == "holdout_evaluation"


def test_repeated_holdout_use_by_a_family_is_flagged(prepared):
    env, ds = prepared
    log = InMemoryHoldoutAccessLog()
    acc = HoldoutAccessor(ds, log)
    acc.read(_ctx(env, "holdout_evaluation"), _candidate(env, "classical_optimizer"))
    report = holdout_access_report(log, ds.dataset_id)
    assert report["reused"] is False and len(report["accesses"]) == 1
    acc.read(_ctx(env, "holdout_evaluation"), _candidate(env, "classical_optimizer"))
    acc.read(_ctx(env, "holdout_evaluation"), _candidate(env, "rl"))
    report = holdout_access_report(log, ds.dataset_id)
    assert report["reused"] is True and report["reused_families"] == ["classical_optimizer"]
    assert len(report["accesses"]) == 3 and report["evaluations_by_family"] == {"classical_optimizer": 2, "rl": 1}


def test_prospective_period_rejects_snapshot_ingested_before_freeze():
    env = Env()
    old = env.provider.publish_backfill("SPY", date(2026, 1, 5), date(2026, 1, 30))  # retrieved 2026-01-31
    new = env.provider.publish_daily("SPY", date(2026, 2, 9), date(2026, 2, 27))
    cand = FrozenCandidate(env.ctx.ids.model_version(), "classical_optimizer", utc(2026, 2, 6))
    cfg = config(splits=None, walk_forward=None, availability={"mode": "retrieved_at"}, features={"lookback_sessions": 3, "label_horizon_sessions": 1})
    with pytest.raises(FinplanError) as ei:
        env.prepare([old, *new], cfg, period="prospective_paper", candidate=cand)
    assert ei.value.code == "VALIDATION_FAILED" and ei.value.details["input_snapshot_id"] == old
    prep = env.prepare(new, cfg, period="prospective_paper", candidate=cand)
    assert prep.record["period"] == "prospective_paper"
    assert prep.record["prospective"]["window"]["start"] > "2026-02-06"
    assert prep.record["splits"] is None and prep.record["folds"] == []


def test_prospective_dataset_requires_a_candidate():
    env = Env()
    ids = env.provider.publish_daily("SPY", date(2026, 2, 9), date(2026, 2, 13))
    with pytest.raises(FinplanError):
        env.prepare(ids, config(splits=None, walk_forward=None), period="prospective_paper")
