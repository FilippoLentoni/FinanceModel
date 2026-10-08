"""DS-14: Yahoo-derived data leaves FinanceModel only from research runs and only as bucketed text
descriptors (task 3.9a; user decision 15d)."""

from __future__ import annotations

import logging
from datetime import date

import pytest

from finplan_model.core.errors import FinplanError
from finplan_model.datasets import fixture_calendar, synthetic_etf_observations
from finplan_model.datasets.outbound import check_outbound_payload, guarded_send, raw_values_of

from .support import context


@pytest.fixture(scope="module")
def observations():
    sessions = fixture_calendar().between(date(2025, 11, 3), date(2026, 1, 30))
    return synthetic_etf_observations("SPY", sessions, seed=21, base_price=470.0)


class Sender:
    def __init__(self) -> None:
        self.sent: list[object] = []

    def __call__(self, payload: object) -> str:
        self.sent.append(payload)
        return "ok"


DESCRIPTOR = {
    "instrument": "SPY",
    "descriptors": ["trend: moderately up over the last month", "volatility regime: low (bucket 2 of 5)", "drawdown from peak: small", "momentum: positive, weakening"],
}


def test_raw_series_payload_rejected_without_network_and_logged_without_values(observations, caplog):
    ctx = context(purpose="research").with_run(context().ids.run_id())
    closes = [o["adj_close"] for o in observations[-30:]]
    payload = {"instrument": "SPY", "prompt": "Last 30 adjusted closes: " + ", ".join(f"{c:.2f}" for c in closes)}
    send = Sender()
    with caplog.at_level(logging.INFO, logger="finplan_model"):
        with pytest.raises(FinplanError) as ei:
            guarded_send(payload, send, ctx=ctx, raw_values=raw_values_of(observations))
    assert ei.value.code == "VALIDATION_FAILED" and ei.value.details["reason"] == "outbound_payload_rejected"
    assert set(ei.value.details["violations"]) >= {"numeric_run", "raw_value"}
    assert send.sent == []  # no request was made
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "outbound_payload_rejected" in logged and ctx.run_id in logged
    assert f"{closes[0]:.2f}" not in logged and "adjusted closes" not in logged


def test_raw_numeric_array_rejected(observations):
    ctx = context(purpose="tuning")
    with pytest.raises(FinplanError):
        check_outbound_payload({"closes": [o["close"] for o in observations[-30:]]}, ctx=ctx, raw_values=raw_values_of(observations))


def test_single_raw_price_value_rejected_even_rounded(observations):
    ctx = context(purpose="research")
    price = observations[-1]["close"]
    with pytest.raises(FinplanError) as ei:
        check_outbound_payload({"text": f"SPY closed near {round(price, 1)} yesterday"}, ctx=ctx, raw_values=raw_values_of(observations))
    assert "raw_value" in ei.value.details["violations"]


@pytest.mark.parametrize(
    "payload,violation",
    [
        ({"text": "| day | close |\n|---|---|\n| 1 | up |\n| 2 | down |"}, "table"),
        ({"text": "<table><tr><td>x</td></tr></table>"}, "table"),
        ({"attachments": [{"name": "spy.csv"}]}, "attachment"),
        ({"blob": b"\x00\x01"}, "attachment"),
        ({"text": "A" * 400}, "attachment"),
        ({"text": "trend up " * 3000}, "bulk"),
    ],
)
def test_tables_attachments_and_bulk_rejected(payload, violation):
    with pytest.raises(FinplanError) as ei:
        check_outbound_payload(payload, ctx=context(purpose="research"))
    assert violation in ei.value.details["violations"]


@pytest.mark.parametrize("purpose", ["research", "tuning", "holdout_evaluation"])
def test_descriptor_payload_passes_for_research_purposes(observations, purpose):
    send = Sender()
    assert guarded_send(DESCRIPTOR, send, ctx=context(purpose=purpose), raw_values=raw_values_of(observations)) == "ok"
    assert send.sent == [DESCRIPTOR]


def test_production_candidate_runs_never_send_yahoo_derived_data(observations):
    send = Sender()
    with pytest.raises(FinplanError) as ei:
        guarded_send(DESCRIPTOR, send, ctx=context(purpose="production_candidate"), raw_values=raw_values_of(observations))
    assert ei.value.code == "OPERATION_NOT_PERMITTED" and send.sent == []
