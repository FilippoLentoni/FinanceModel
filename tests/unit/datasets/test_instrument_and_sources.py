"""DS-09 (initial instrument), DS-10 (completed daily only), DS-11 (mock provider until approved real
snapshots exist), DS-12 lineage and quality flags; tasks 3.6 and 3.8.

"Real" snapshots here are built in memory by the test (invented values, ``synthetic`` false only to
model the platform's real-data record shape); nothing retrieved is ever committed."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from finplan_model.core.errors import FinplanError
from finplan_model.datasets import PreparationConfig, synthetic_etf_observations

from .support import Env, config, utc

SHORT = {"train": {"start": "2026-01-05", "end": "2026-01-16"}, "validation": {"start": "2026-01-21", "end": "2026-01-23"}, "holdout": {"start": "2026-01-28", "end": "2026-01-30"}, "embargo_sessions": 1}


def _cfg(**kw: Any) -> dict[str, Any]:
    base = config(splits=SHORT, walk_forward=None, features={"lookback_sessions": 3, "label_horizon_sessions": 1})
    base.update(kw)
    return base


def _obs(env, start=date(2026, 1, 5), end=date(2026, 1, 30), ticker="SPY"):
    return synthetic_etf_observations(ticker, env.provider.sessions(start, end), seed=5)


REAL_LINEAGE = {"provider_library": "yfinance", "library_version": "0.2.66", "calendar_version": "xnys-exchange_calendars-4.13.2-20180101-20271231"}


def _real(env, obs, *, lineage=None, calendar_block=None, provider="yfinance", **kw):
    cal = env.provider.calendar
    block = calendar_block if calendar_block is not None else {"exchange": "XNYS", "version": REAL_LINEAGE["calendar_version"], "synthetic": False, "coverage": {"start": "2018-01-01", "end": "2027-12-31"}, "sessions": cal.iso()}
    return env.provider.publish([{k: v for k, v in o.items() if k != "synthetic"} for o in obs], retrieved_at=utc(2026, 1, 31, 13), provider=provider, lineage_extra=REAL_LINEAGE if lineage is None else lineage, calendar_block=block, synthetic=False, **kw)


# --------------------------------------------------------------------- DS-09
@pytest.mark.parametrize("dataset_kind,asset_class,ticker", [("index-level", "index", "SPX"), ("index-constituents", "equity", "SP500")])
def test_index_level_or_constituents_offered_for_the_etf_series_fails(dataset_kind, asset_class, ticker):
    env = Env()
    sid = env.provider.publish(_obs(env, ticker=ticker), retrieved_at=utc(2026, 1, 31, 13), ticker=ticker, dataset_kind=dataset_kind, asset_class=asset_class)
    with pytest.raises(FinplanError) as ei:
        env.prepare([sid], _cfg())
    e = ei.value
    assert e.code == "VALIDATION_FAILED" and "instrument mismatch" in e.message
    assert e.details["snapshot_dataset_id"] == f"finance/{dataset_kind}/{ticker}"
    assert e.details["expected_dataset_ids"] == ["finance/etf-daily/SPY"]


def test_etf_asset_class_mismatch_fails():
    env = Env()
    sid = env.provider.publish(_obs(env), retrieved_at=utc(2026, 1, 31, 13), asset_class="index")
    with pytest.raises(FinplanError) as ei:
        env.prepare([sid], _cfg())
    assert ei.value.details["snapshot_asset_class"] == "index"


def test_preparation_config_is_the_etf_daily_series_only():
    for inst in ({"dataset_kind": "index-level"}, {"asset_class": "index"}, {"granularity": "intraday"}):
        with pytest.raises(FinplanError):
            PreparationConfig.from_dict(_cfg(instrument={"tickers": ["SPY"], **inst}))


# --------------------------------------------------------------------- DS-10
def test_intraday_only_date_fails_naming_the_date():
    env = Env()
    obs = _obs(env)
    obs[4] = dict(obs[4], kind="intraday_partial")
    sid = env.provider.publish(obs, retrieved_at=utc(2026, 1, 31, 13))
    with pytest.raises(FinplanError) as ei:
        env.prepare([sid], _cfg())
    assert ei.value.code == "VALIDATION_FAILED" and ei.value.details["session_date"] == obs[4]["session_date"]
    assert ei.value.details["reason"] == "intraday_only"


def test_observation_on_a_non_session_date_fails_naming_the_date():
    env = Env()
    obs = _obs(env)
    obs.append(dict(obs[-1], session_date="2026-01-31"))  # a Saturday
    sid = env.provider.publish(obs, retrieved_at=utc(2026, 2, 1, 13))
    with pytest.raises(FinplanError) as ei:
        env.prepare([sid], _cfg())
    assert ei.value.details["session_date"] == "2026-01-31" and ei.value.details["reason"] == "non_session_date"


# --------------------------------------------------------------------- DS-11
def test_real_data_request_without_approved_real_snapshot_is_dependency_unavailable():
    env = Env()
    real_cfg = _cfg(data_source="platform_snapshots")
    # environment still serves fixtures only
    with pytest.raises(FinplanError) as ei:
        env.prepare([env.provider.new_snapshot_id()], real_cfg)
    assert ei.value.code == "DEPENDENCY_UNAVAILABLE" and ei.value.details["reason"] == "no_approved_real_snapshot"
    # environment configured for real data, but the snapshot is missing or not approved
    with pytest.raises(FinplanError) as ei:
        env.prepare([env.provider.new_snapshot_id()], real_cfg, environment_data_source="platform_snapshots")
    assert ei.value.code == "DEPENDENCY_UNAVAILABLE"
    committed = _real(env, _obs(env), status="committed")
    with pytest.raises(FinplanError) as ei:
        env.prepare([committed], real_cfg, environment_data_source="platform_snapshots")
    assert ei.value.code == "DEPENDENCY_UNAVAILABLE"
    # fixture-backed preparation remains available
    assert env.prepare([env.provider.publish(_obs(env), retrieved_at=utc(2026, 1, 31, 13))], _cfg()).record["synthetic"] is True


def test_ci_uses_only_the_mock_provider_and_no_network():
    env = Env()
    sid = env.provider.publish(_obs(env), retrieved_at=utc(2026, 1, 31, 13))
    env.prepare([sid], _cfg())
    assert {c[0] for c in env.provider.client.calls} <= {"get_snapshot", "download"}  # in-process mock only
    import socket

    with pytest.raises(Exception):  # the offline harness blocks any network connection
        socket.create_connection(("query1.example.invalid", 443), timeout=1)


def test_fixture_preparation_refuses_real_snapshots():
    env = Env()
    sid = _real(env, _obs(env))
    with pytest.raises(FinplanError) as ei:
        env.prepare([sid], _cfg())
    assert ei.value.code == "VALIDATION_FAILED"


# --------------------------------------------------------------------- DS-12 lineage (task 3.8)
def test_real_snapshot_provenance_is_copied_into_lineage():
    env = Env()
    sid = _real(env, _obs(env))
    prep = env.prepare([sid], _cfg(data_source="platform_snapshots"), environment_data_source="platform_snapshots")
    rec = prep.record
    (snap,) = rec["input_snapshots"]
    prov = snap["provenance"]
    assert prov["provider"] == "yfinance" and prov["provider_library"] == "yfinance" and prov["library_version"] == "0.2.66"
    assert prov["retrieved_at"] == "2026-01-31T13:00:00Z"
    assert prov["calendar"] == {"exchange": "XNYS", "version": REAL_LINEAGE["calendar_version"], "library": "exchange_calendars", "library_version": "4.13.2", "synthetic": False, "sessions": len(env.provider.calendar)}
    assert rec["synthetic"] is False and rec["real_data"] is True
    assert rec["calendar"]["library"] == "exchange_calendars"


@pytest.mark.parametrize("drop,missing", [("library_version", "lineage.library_version"), ("provider_library", "lineage.provider_library"), ("calendar_version", "lineage.calendar_version")])
def test_real_snapshot_missing_provenance_fails(drop, missing):
    env = Env()
    lineage = {**REAL_LINEAGE, drop: None}
    sid = _real(env, _obs(env), lineage=lineage)
    with pytest.raises(FinplanError) as ei:
        env.prepare([sid], _cfg(data_source="platform_snapshots"), environment_data_source="platform_snapshots")
    assert ei.value.code == "VALIDATION_FAILED" and missing in ei.value.details["missing_fields"]


def test_real_snapshot_without_session_list_fails_as_contract_gap():
    env = Env()
    sid = _real(env, _obs(env), calendar_block={"exchange": "XNYS", "version": REAL_LINEAGE["calendar_version"], "synthetic": False})
    with pytest.raises(FinplanError) as ei:
        env.prepare([sid], _cfg(data_source="platform_snapshots"), environment_data_source="platform_snapshots")
    assert "calendar.sessions" in ei.value.details["missing_fields"]


def test_quality_flagged_dates_are_excluded_or_fail_never_filled():
    env = Env()
    obs = _obs(env)
    flagged = obs[6]["session_date"]
    gap = obs[8]["session_date"]
    kept = [o for o in obs if o["session_date"] != gap]
    sid = env.provider.publish(kept, retrieved_at=utc(2026, 1, 31, 13), quality_flags=["partial_response", "missing_sessions"], quality_details={"partial_response": [flagged], "missing_sessions": [gap]})
    prep = env.prepare([sid], _cfg())
    q = prep.record["quality"]
    assert q["excluded_sessions"] == sorted([flagged, gap])
    assert q["excluded_reasons"][flagged] == ["partial_response"] and q["excluded_reasons"][gap] == ["missing_sessions"]
    assert q["flags_by_snapshot"][sid] == ["missing_sessions", "partial_response"]
    ds = prep.load(env.store)
    assert date.fromisoformat(flagged) not in ds.sessions and date.fromisoformat(gap) not in ds.sessions
    assert all(b.session_date.isoformat() not in (flagged, gap) for b in ds.bars())  # nothing filled
    with pytest.raises(FinplanError) as ei:
        env.prepare([sid], _cfg(quality={"policy": "fail"}))
    assert ei.value.details["reason"] == "quality_flagged_dates" and flagged in ei.value.details["dates"]


def test_unflagged_missing_session_is_detected_from_the_calendar():
    env = Env()
    skip = date(2026, 1, 13)
    ids = env.provider.publish_daily("SPY", date(2026, 1, 5), date(2026, 1, 30), skip=[skip])
    prep = env.prepare(ids, _cfg(availability={"mode": "retrieved_at"}))
    assert prep.record["quality"]["missing_sessions"] == {"2026-01-13": ["SPY"]}
    assert "2026-01-13" in prep.record["quality"]["excluded_sessions"]


def test_revisions_keep_the_first_seen_values_or_fail():
    env = Env()
    obs = _obs(env)
    first = env.provider.publish(obs, retrieved_at=utc(2026, 1, 31, 13))
    revised = [dict(o, close=o["close"] + 1.0) if o["session_date"] == obs[2]["session_date"] else o for o in obs]
    second = env.provider.publish(revised, retrieved_at=utc(2026, 2, 2, 13))
    prep = env.prepare([second, first], _cfg())
    assert prep.record["revisions"]["revised_observations"] == 1
    bar = next(b for b in prep.load(env.store).bars() if b.session_date.isoformat() == obs[2]["session_date"])
    assert bar.close == obs[2]["close"] and bar.source_snapshot_id == first
    with pytest.raises(FinplanError):
        env.prepare([first, second], _cfg(revisions="fail"))
