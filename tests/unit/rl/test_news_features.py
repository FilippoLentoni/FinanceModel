import copy

import pytest

from finplan_model.core.errors import FinplanError
from finplan_model.rl.news import features_at, freeze_sources, require_coverage


def records():
    return freeze_sources([{"url": "https://example.org/earnings", "title": "Company beats earnings with record profit",
                            "instrument_ids": ["AAPL"], "first_seen_at": "2026-10-12T12:00:00Z"}],
                          retrieved_at="2026-10-13T12:00:00Z")


def test_old_headline_cannot_be_backfilled_into_historical_state():
    r = records()
    assert features_at(r, "AAPL", decision_time="2026-10-12T22:00:00Z")["missing"] == 1
    f = features_at(r, "AAPL", decision_time="2026-10-13T22:00:00Z")
    assert f["missing"] == 0 and f["sentiment"] == 1 and f["earnings_event"] == 1
    assert f["age_hours"] == 34  # Source age, not time since this machine fetched it.
    assert features_at(r, "NVDA", decision_time="2026-10-13T22:00:00Z")["missing"] == 1
    with pytest.raises(FinplanError):
        require_coverage(r, ["AAPL"], ["2026-10-12T22:00:00Z"])


def test_revised_content_and_unsupported_sources_fail_closed():
    r = records()
    changed = copy.deepcopy(r)
    changed[0]["title"] = "Revised after the event"
    with pytest.raises(FinplanError):
        features_at(changed, "AAPL", decision_time="2026-10-13T22:00:00Z")
    assert freeze_sources([{"url": "https://127.0.0.1/private", "title": "private"}], retrieved_at="2026-10-13T12:00:00Z") == []
    source = {"url": "https://example.org/future", "title": "Future news", "first_seen_at": "2026-10-14T00:00:00Z"}
    assert freeze_sources([source], retrieved_at="2026-10-13T12:00:00Z") == []


def test_repeated_collection_and_honest_revisions_do_not_double_count_or_backfill():
    original = records()
    revised = freeze_sources([{"url": "https://example.org/earnings", "title": "Company misses earnings with loss",
                               "instrument_ids": ["AAPL"], "first_seen_at": "2026-10-12T12:00:00Z"}],
                             retrieved_at="2026-10-14T12:00:00Z")
    combined = original + original + revised
    before = features_at(combined, "AAPL", decision_time="2026-10-13T22:00:00Z")
    after = features_at(combined, "AAPL", decision_time="2026-10-14T22:00:00Z")
    assert before["count"] == after["count"] == 1
    assert before["sentiment"] == 1 and after["sentiment"] == -1
    assert features_at(combined, "AAPL", decision_time="2026-10-16T22:00:00Z")["missing"] == 1
