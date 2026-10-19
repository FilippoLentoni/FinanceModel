"""Agent-tool books must equal the direct classical books for the same slot (non-zero holdings)."""
import copy
from datetime import timedelta

import pytest

from finplan_model.classical_api import handle
from finplan_model.forward.hourly import _agent_tool_adapter, _portfolio_holdings
from finplan_model.forward.runtime import evaluate_approach

UNIVERSE = ["AAPL", "NVDA", "VOO"]


def _source(context, index):
    days = context.dates[:index + 1]
    rows = []
    for i in UNIVERSE:
        for b in context.market.bars_of(i):
            if b.session_date in days:
                rows.append({"instrument_id": i, "session_date": b.session_date.isoformat(), "close": b.close, "open": b.open,
                             "high": b.high, "low": b.low, "volume": b.volume, "kind": "completed_daily",
                             "available_at": context.market.decision_time(b.session_date).isoformat()})
    # Executable raw closes differ from the model-native feature basis (dividends/splits),
    # so marking at the wrong basis would change the solver's current weights.
    raw = [{**r, "close": r["close"] * 1.03} for r in rows]
    return {"investable_universe": UNIVERSE, "input_snapshot_id": context.sid, "observations": rows, "execution_observations": raw,
            "dataset_id": "finance/equity-etf-daily/research-universe", "checksum": "sha256:" + "0" * 64}


def _book(source):
    last = {r["instrument_id"]: r["close"] for r in source["observations"]}
    positions = [{"instrument_id": "AAPL", "quantity": 3000 / last["AAPL"]}, {"instrument_id": "NVDA", "quantity": 5000 / last["NVDA"]},
                 {"instrument_id": "VOO", "quantity": 0.}]
    return {"portfolio_id": "pf_parity", "revision": 3,
            "paper_state": {"positions": positions, "cash_balance": 2000., "high_watermark": 12000.}}


@pytest.mark.parametrize("model", ["min_variance", "cvar"])
def test_agent_tool_receives_the_direct_books_inputs_and_returns_its_weights(context, model):
    index = 150
    source = _source(context, index)
    holdings = _portfolio_holdings(_book(source), source)
    cutoff = (context.market.decision_time(context.dates[index]) + timedelta(hours=1)).isoformat()
    direct_capture, agent_capture, sent = [], [], []

    direct = evaluate_approach("classical_" + model, source, copy.deepcopy(holdings), cutoff=cutoff, run_id="mr_d", input_id="mi_d",
                               capture=direct_capture.append)

    def invoke(tool, arguments, prompt, before):
        before({"payload": {"tool_request": {"name": tool, "arguments": arguments}}})
        sent.append(copy.deepcopy(arguments))
        doc = handle({"environment": "beta", "operation": tool, "request": arguments, "instruments": arguments["instruments"]}, context.service)
        rec = doc["recommendation"]
        return {"status": "completed", "tools_called": [tool], "target_weights": rec["target_weights"],
                "cash_weight": rec["cash_weight"], "served_strategy": rec["strategy"]}

    agent = evaluate_approach("agent_" + model, source, copy.deepcopy(holdings), cutoff=cutoff, run_id="mr_a", input_id="mi_a",
                              capture=agent_capture.append,
                              external_adapters={"agent_tool": lambda b, s, h, before: _agent_tool_adapter(b, s, h, before, invoke=invoke)})
    assert direct["status"] == agent["status"] == "recommended"

    # The request's holdings are exactly the direct solver's current weights / cash.
    prepared = direct_capture[0]["numeric"]["prepared"]
    request = sent[0]["holdings"]
    assert [w["instrument_id"] for w in request["weights"]] == prepared["instruments"] == UNIVERSE
    assert [w["weight"] for w in request["weights"]] == prepared["current_weights"]
    assert request["cash_weight"] + sum(prepared["current_weights"]) == pytest.approx(1., abs=1e-12)
    assert prepared["current_weights"][0] > 0 and prepared["current_weights"][1] > 0
    # The runner's raw-close holdings differ: sending them would not be the direct book's input.
    assert [w["weight"] for w in holdings["weights"]] != prepared["current_weights"]
    assert request["high_watermark"] >= request["portfolio_value"]
    stored = context.service.store.list(kind="recommendation", limit=1)[0]["solve_inputs"]
    assert stored["current_weights"] == prepared["current_weights"]

    by_id = lambda out: [w["weight"] for w in out["target_weights"]]
    assert by_id(agent) == pytest.approx(by_id(direct), abs=1e-12)
    assert agent["cash_weight"] == pytest.approx(direct["cash_weight"], abs=1e-12)
