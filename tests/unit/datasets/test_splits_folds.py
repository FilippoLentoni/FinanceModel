"""DS-04 (chronological splits with embargo) and DS-05 (walk-forward folds); task 3.4."""

from __future__ import annotations

from datetime import date

import pytest

from finplan_model.core.errors import FinplanError
from finplan_model.datasets import PreparationConfig, add_months, fixture_session_list
from finplan_model.datasets.calendar import SessionList

from .support import Env, config


def test_overlapping_split_is_rejected():
    cfg = config(splits={"validation": {"start": "2021-06-01", "end": "2022-03-31"}})
    with pytest.raises(FinplanError) as ei:
        PreparationConfig.from_dict(cfg)
    assert ei.value.code == "VALIDATION_FAILED" and ei.value.details["pointer"] == "/splits/validation/start"


def test_holdout_must_be_the_latest_range():
    with pytest.raises(FinplanError) as ei:
        PreparationConfig.from_dict(config(splits={"holdout": {"start": "2021-01-04", "end": "2021-03-31"}}))
    assert ei.value.details["pointer"] == "/splits/holdout/start"


def test_shuffling_is_not_an_option():
    with pytest.raises(FinplanError) as ei:
        PreparationConfig.from_dict({**config(), "shuffle": True})
    assert ei.value.details["pointer"] == "/shuffle"
    with pytest.raises(FinplanError):
        PreparationConfig.from_dict(config(splits={"shuffle": True}))


def test_embargo_shorter_than_label_horizon_is_rejected():
    with pytest.raises(FinplanError) as ei:
        PreparationConfig.from_dict(config(splits={"embargo_sessions": 19}))
    assert ei.value.details["required_sessions"] == 20
    with pytest.raises(FinplanError):
        PreparationConfig.from_dict(config(walk_forward={"embargo_sessions": 5}))


def test_embargo_of_20_trading_days_separates_train_and_validation():
    env = Env()
    prep = env.prepare([env.backfill()])
    sp = prep.record["splits"]
    cal = SessionList("XNYS", "v", tuple(fixture_session_list(date(2020, 1, 1), date(2022, 12, 31))))
    gap = cal.sessions_strictly_between(date.fromisoformat(sp["train"]["end"]), date.fromisoformat(sp["validation"]["start"]))
    assert gap >= 20 and sp["gap_sessions"]["train_validation"] == gap
    assert sp["gap_sessions"]["validation_holdout"] >= 20


def test_ranges_closer_than_the_embargo_in_trading_days_are_rejected():
    env = Env()
    sid = env.backfill()
    # validation starts 10 trading sessions after the end of training (calendar order is fine)
    cfg = config(splits={"train": {"start": "2020-01-01", "end": "2021-06-30"}, "validation": {"start": "2021-07-16", "end": "2022-03-31"}})
    with pytest.raises(FinplanError) as ei:
        env.prepare([sid], cfg)
    assert ei.value.details["between"] == "train_validation" and ei.value.details["gap_sessions"] < 20


def test_three_year_expanding_window_with_six_month_steps():
    env = Env()
    sid = env.backfill(start=date(2018, 1, 1), end=date(2023, 12, 31))
    cfg = config(
        splits={"train": {"start": "2018-01-01", "end": "2022-06-30"}, "validation": {"start": "2022-08-01", "end": "2023-03-31"}, "holdout": {"start": "2023-05-01", "end": "2023-12-31"}, "embargo_sessions": 20},
        walk_forward={"window": "expanding", "train_length": {"years": 3}, "test_length": {"months": 6}, "step": {"months": 6}, "embargo_sessions": 20},
    )
    prep = env.prepare([sid], cfg)
    folds = prep.record["folds"]
    assert len(folds) >= 3
    cal = SessionList("XNYS", "v", tuple(fixture_session_list(date(2018, 1, 1), date(2023, 12, 31))))
    first = date(2018, 1, 2)
    for k, f in enumerate(folds):
        tr_start, tr_end = date.fromisoformat(f["train"]["start"]), date.fromisoformat(f["train"]["end"])
        te_start, te_end = date.fromisoformat(f["test"]["start"]), date.fromisoformat(f["test"]["end"])
        assert tr_start == first  # expanding
        assert tr_end < add_months(first, 36 + 6 * k)  # train window grows by six months per fold
        assert cal.sessions_strictly_between(tr_end, te_start) >= 20  # test starts after train end + embargo
        assert te_start > tr_end and te_end >= te_start
        assert te_end <= date(2023, 3, 31)  # never reaches the holdout
        if k:
            assert te_start > date.fromisoformat(folds[k - 1]["test"]["end"])  # disjoint test windows
    ds = prep.load(env.store)
    assert [f.fold_id for f in ds.folds] == [f["fold_id"] for f in folds]  # recorded in the dataset record


def test_rolling_window_in_sessions_and_training_market():
    env = Env()
    sid = env.backfill()
    prep = env.prepare([sid], config(walk_forward={"window": "rolling", "train_length": {"sessions": 120}, "test_length": {"sessions": 40}, "step": {"sessions": 40}, "embargo_sessions": 20}))
    ds = prep.load(env.store)
    folds = ds.folds
    assert len(folds) >= 3 and all(f.train.sessions == 120 for f in folds)
    assert all(f.gap_sessions == 20 for f in folds)
    tm = ds.training_market(folds[1])
    assert tm.sessions[-1] == folds[1].train.end  # a fold trains only on data before its test window
    assert all(b.session_date <= folds[1].train.end for i in tm.instruments for b in tm.bars_of(i))


def test_too_few_folds_is_refused():
    env = Env()
    with pytest.raises(FinplanError) as ei:
        env.prepare([env.backfill()], config(walk_forward={"min_folds": 50}))
    assert ei.value.details["pointer"] == "/walk_forward"
