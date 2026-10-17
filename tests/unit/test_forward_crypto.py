"""Controlled fixtures for native crypto adapter, never forward performance."""
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from finplan_model import crypto_research as native
from finplan_model.forward import crypto
from finplan_model.forward.runtime import evaluate_approach


END = datetime(2026, 1, 3, tzinfo=UTC)
CUTOFF = native.iso(END + timedelta(minutes=10))


def dataset():
    rows = {p: [{"start": native.iso(END-timedelta(hours=48-i)),
                 "open": 100+j*50+i, "close": 100.5+j*50+i,
                 "low": 99+j*50+i, "high": 102+j*50+i, "volume": 10}
                for i in range(48)] for j, p in enumerate(native.PRODUCTS)}
    return {"dataset_id": "coinbase-crypto-hourly-v1", "as_of": native.iso(END),
            "available_at": native.iso(END+timedelta(minutes=5)), "granularity_seconds": 3600,
            "market_data_synthetic": True, "products": native.PRODUCTS.copy(), "bars": rows,
            "sources": [], "dataset_checksum": "sha256:"+"a"*64, "contract_version": "1.7.0"}


def holdings():
    return {"positions": [], "cash_balance": 100000., "revision": 1}


@pytest.mark.parametrize("name", native.BOOKS)
def test_every_native_approach_runs_its_own_boundary(name):
    seen = []
    result = crypto.evaluate_crypto("crypto_"+name, dataset(), holdings(), cutoff=CUTOFF,
                                    capture=seen.append, allow_synthetic=True)
    assert len(seen) == 1
    assert seen[0]["approach_id"] == "crypto_"+name
    assert sum(result["weights"].values())+result["cash"] == pytest.approx(1)
    assert not result["forecast"]["available"]
    assert not seen[0]["news"]["consumed"]
    if name in native.ALGORITHMS:
        assert max(result["weights"].values()) <= .600001
        assert result["cash"] == pytest.approx(.1)


@pytest.mark.parametrize("name", native.ALGORITHMS)
def test_exact_native_math_matches_existing_implementation(name):
    data = dataset()
    h = {"positions": [{"instrument_id": "BTC-USD", "quantity": 150.}], "cash_balance": 60000., "revision": 3}
    books = {a: {"quantities": {"BTC-USD": 150., "ETH-USD": 0.}, "cash": 60000.} for a in native.ALGORITHMS}
    expected = native.recommendations(data, books)[name]
    seen = []
    actual = crypto.evaluate_crypto("crypto_"+name, data, h, cutoff=CUTOFF, capture=seen.append, allow_synthetic=True)
    assert actual["weights"] == pytest.approx(expected["weights"], abs=1e-10)
    assert actual["cash"] == pytest.approx(expected["cash_weight"])
    prepared = seen[0]
    close = np.array([[r["close"] for r in data["bars"][p]] for p in native.PRODUCTS]).T
    returns = close[1:]/close[:-1]-1
    assert prepared["scenarios"] == returns.tolist()
    assert prepared["covariance"] == (np.cov(returns, rowvar=False)+np.eye(2)*1e-10).tolist()
    assert prepared["expected_returns"] == returns.mean(axis=0).tolist()
    assert prepared["holdings"]["revision"] == 3
    data["bars"]["BTC-USD"][0]["close"] = 999
    assert prepared["price_history"]["BTC-USD"][0]["close"] == 100.5


def test_capture_completes_before_solver_and_failure_prevents_compute(monkeypatch):
    called = []
    original = crypto.solve
    captured = []
    def solver(inputs):
        assert captured
        assert inputs["scenarios"] == captured[0]["scenarios"]
        called.append(inputs)
        return original(inputs)
    monkeypatch.setattr(crypto, "solve", solver)
    crypto.evaluate_crypto("crypto_min_variance", dataset(), holdings(), cutoff=CUTOFF,
                           capture=captured.append, allow_synthetic=True)
    assert len(called) == 1
    def failed_capture(payload):
        raise RuntimeError("durable save rejected")
    with pytest.raises(RuntimeError, match="durable save"):
        crypto.evaluate_crypto("crypto_min_variance", dataset(), holdings(), cutoff=CUTOFF,
                               capture=failed_capture, allow_synthetic=True)
    assert len(called) == 1


@pytest.mark.parametrize("failure", ["missing", "future", "stale", "synthetic", "gap", "holdings"])
def test_unavailable_inputs_capture_honest_precondition_before_any_solver(failure, monkeypatch):
    data, h = dataset(), holdings()
    cutoff = CUTOFF
    if failure == "missing": data = None
    if failure == "future": cutoff = native.iso(END)
    if failure == "stale": cutoff = native.iso(END+timedelta(hours=3))
    if failure == "gap": data["bars"]["ETH-USD"].pop(2)
    if failure == "holdings": h["positions"] = [{"instrument_id": "SPY", "quantity": 1.}]
    monkeypatch.setattr(crypto, "solve", lambda inputs: pytest.fail("must not invoke solver"))
    captures = []
    with pytest.raises(Exception):
        crypto.evaluate_crypto("crypto_cvar", data, h, cutoff=cutoff, capture=captures.append,
                               allow_synthetic=failure != "synthetic")
    assert len(captures) == 1
    assert captures[0]["invocation_state"] == "not_invoked"
    assert "reason" in captures[0]


def test_cash_needs_no_prices_and_buyhold_does_not_rebalance_drift():
    cash = crypto.evaluate_crypto("crypto_cash", None, {}, cutoff=CUTOFF, capture=lambda p: None)
    assert cash["cash"] == 1
    h = {"positions": [{"instrument_id": "BTC-USD", "quantity": 100.}], "cash_balance": 300., "revision": 4}
    result = crypto.evaluate_crypto("crypto_btc_buy_hold", dataset(), h, cutoff=CUTOFF,
                                    capture=lambda p: None, allow_synthetic=True)
    assert result["status"] == "hold" and result["solution_status"] == "no_effect"
    assert result["cash"] > 0


def test_native_fallback_is_captured_and_uses_same_objective(monkeypatch):
    monkeypatch.setattr(crypto, "solve", lambda i: {"status": "infeasible"})
    captured = []
    result = crypto.evaluate_crypto("crypto_mean_variance", dataset(), holdings(), cutoff=CUTOFF,
                                    capture=captured.append, allow_synthetic=True)
    assert captured[0]["fallback"]["bounds"] == [.3, .6]
    assert result["solution_status"] == "feasible"
    assert result["diagnostics"]["fallback"] == "bounded_scalar_same_objective"


def test_runtime_external_adapter_captures_exact_native_values_and_intents():
    source = {"investable_universe": native.PRODUCTS, "native_crypto_dataset": dataset(), "trigger": "controlled_replay"}
    seen = []
    result = evaluate_approach("crypto_min_variance", source, holdings(), cutoff=CUTOFF,
                              run_id="test_run", input_id="test_input", capture=seen.append,
                              external_adapters={"crypto_native": crypto.crypto_adapter})
    assert result["status"] == "recommended", result
    assert len(seen) == 1 and seen[0]["numeric"]["prepared"]["boundary"] == "native_crypto_classical_solve"
    assert seen[0]["numeric"]["prepared"]["native_source"]["as_of"] == dataset()["as_of"]
    assert all(i["action"] == "buy" for i in result["intents"])
    assert not seen[0]["news"]["consumed"]
    source["investable_universe"] = ["SPY"]
    bad = evaluate_approach("crypto_min_variance", source, holdings(), cutoff=CUTOFF,
                           run_id="other_run", input_id="other_input", capture=lambda p: None,
                           external_adapters={"crypto_native": crypto.crypto_adapter})
    assert bad["status"] == "unavailable" and "cohort" in bad["reason"]
