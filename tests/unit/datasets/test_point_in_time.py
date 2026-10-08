"""DS-03: point-in-time availability, the as-of feature join and the look-ahead check; task 3.3."""

from __future__ import annotations

from datetime import date, time

import pytest

from finplan_model.core.errors import FinplanError
from finplan_model.datasets import Dataset
from finplan_model.datasets.features import decision_time

from .support import Env, config, utc

D0, D1 = date(2026, 1, 5), date(2026, 1, 30)
LATE_DAY = date(2026, 1, 14)


def _daily_config(**overrides):
    base = config(
        availability={"mode": "retrieved_at"},
        features={"lookback_sessions": 3, "label_horizon_sessions": 1, "pipeline": [{"name": "last_close", "kind": "close"}, {"name": "ret_3", "kind": "trailing_return", "lookback": 3}]},
        splits={"train": {"start": "2026-01-05", "end": "2026-01-16"}, "validation": {"start": "2026-01-21", "end": "2026-01-23"}, "holdout": {"start": "2026-01-28", "end": "2026-01-30"}, "embargo_sessions": 1},
        walk_forward=None,
    )
    base.update(overrides)
    return base


def _late_env():
    env = Env()
    # Daily ingestion retrieved at 21:15 UTC; the bar of LATE_DAY was retrieved after the 22:00 decision.
    ids = env.provider.publish_daily("SPY", D0, D1, late={LATE_DAY: utc(2026, 1, 14, 23, 30)})
    return env, ids


def test_every_observation_carries_its_availability_time():
    env, ids = _late_env()
    ds = env.prepare(ids, _daily_config()).load(env.store)
    bars = {b.session_date: b for b in ds.bars()}
    assert bars[date(2026, 1, 13)].available_at == utc(2026, 1, 13, 21, 30)  # close + lag (retrieved 21:15)
    assert bars[LATE_DAY].available_at == utc(2026, 1, 14, 23, 30)  # retrieved after the decision time


def test_late_observation_excluded_for_that_decision_and_used_from_the_next():
    env, ids = _late_env()
    ds = env.prepare(ids, _daily_config()).load(env.store)
    bars = {b.session_date: b for b in ds.bars()}
    rows = {(r["session_date"]): r["values"] for r in ds.features()}
    assert rows["2026-01-14"]["last_close"] == bars[date(2026, 1, 13)].close  # late bar invisible on d
    assert rows["2026-01-15"]["last_close"] == bars[date(2026, 1, 15)].close
    # the trailing return on d+1 includes the late bar (it is available by then)
    closes = [bars[d].close for d in (date(2026, 1, 12), date(2026, 1, 13), LATE_DAY, date(2026, 1, 15))]
    assert rows["2026-01-15"]["ret_3"] == pytest.approx(closes[-1] / closes[0] - 1.0)
    # the simulator's point-in-time view agrees
    market = ds.market_data()
    assert market.view(LATE_DAY).latest("SPY") == bars[date(2026, 1, 13)].close
    assert market.view(date(2026, 1, 15)).history("SPY")[-2] == (LATE_DAY, bars[LATE_DAY].close)


def test_look_ahead_feature_fails_naming_feature_and_timestamp():
    env, ids = _late_env()
    cfg = _daily_config(features={"lookback_sessions": 3, "label_horizon_sessions": 1, "pipeline": [{"name": "close_today", "kind": "close", "alignment": "session", "offset": 0}]})
    with pytest.raises(FinplanError) as ei:
        env.prepare(ids, cfg)
    e = ei.value
    assert e.code == "VALIDATION_FAILED"
    assert e.details["feature"] == "close_today"
    assert e.details["observation_session_date"] == LATE_DAY.isoformat()
    assert e.details["observation_available_at"] == "2026-01-14T23:30:00Z"
    assert e.details["decision_time"] == "2026-01-14T22:00:00Z"


def test_session_aligned_feature_on_timely_data_passes_and_next_session_fails():
    env = Env()
    ids = env.provider.publish_daily("SPY", D0, D1)
    ok = _daily_config(features={"lookback_sessions": 3, "label_horizon_sessions": 1, "pipeline": [{"name": "close_today", "kind": "close", "alignment": "session", "offset": 0}]})
    env.prepare(ids, ok)
    bad = _daily_config(features={"lookback_sessions": 3, "label_horizon_sessions": 1, "pipeline": [{"name": "close_tomorrow", "kind": "close", "alignment": "session", "offset": 1}]})
    with pytest.raises(FinplanError) as ei:
        env.prepare(ids, bad)
    assert ei.value.details["feature"] == "close_tomorrow"


def test_intraday_partial_never_used_as_a_completed_daily_bar():
    env = Env()
    sessions = env.provider.sessions(D0, date(2026, 1, 16))
    from finplan_model.datasets import synthetic_etf_observations

    obs = synthetic_etf_observations("SPY", sessions, seed=3)
    intraday = dict(obs[3], kind="intraday_partial", close=obs[3]["close"] * 1.5)
    sid = env.provider.publish(obs + [intraday], retrieved_at=utc(2026, 1, 17, 13))
    cfg = _daily_config(availability={"mode": "session_close"}, splits={"train": {"start": "2026-01-05", "end": "2026-01-08"}, "validation": {"start": "2026-01-12", "end": "2026-01-12"}, "holdout": {"start": "2026-01-14", "end": "2026-01-16"}, "embargo_sessions": 1})
    prep = env.prepare([sid], cfg)
    ds = Dataset.load(prep.record, env.store)
    assert prep.record["intraday_dropped"] == 1
    bar = next(b for b in ds.bars() if b.session_date == sessions[3])
    assert bar.close == obs[3]["close"]


def test_decision_time_default_is_22_utc():
    assert decision_time(date(2026, 1, 5), time(22, 0)) == utc(2026, 1, 5, 22)
