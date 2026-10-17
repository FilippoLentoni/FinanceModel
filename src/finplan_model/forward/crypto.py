"""Native BTC/ETH decision adapters for a separately labelled 24/7 cohort.

This module consumes only the platform's persisted hourly dataset. It does not
collect data, train, execute historical candles, or own a portfolio ledger. The
common point evaluator executes these weights at subsequently captured quotes
under the cohort's explicitly declared fee policy.
"""
from __future__ import annotations

import copy
import math
from datetime import UTC, datetime
from typing import Callable

import numpy as np
from scipy.optimize import minimize_scalar

from finplan_model.classical.math import objective, settings, solve
from finplan_model.core.errors import FinplanError
from finplan_model.crypto_research import BOOKS, IMPLEMENTATION, PRODUCTS, validate_dataset

ADAPTER_VERSION = "native-crypto-boundary/1"


def _unavailable(reason):
    raise FinplanError.precondition(reason, reason=reason)


def _book(holdings, prices):
    rows = holdings.get("positions", [])
    ids = [r["instrument_id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) - set(PRODUCTS):
        _unavailable("crypto_holdings_universe_or_duplicate")
    quantities = {p: 0.0 for p in PRODUCTS}
    for row in rows:
        quantities[row["instrument_id"]] = float(row["quantity"])
    cash = holdings.get("cash_balance", holdings.get("cash"))
    if cash is None:
        _unavailable("crypto_cash_unavailable")
    cash = float(cash)
    if any(not math.isfinite(v) or v < 0 for v in [cash, *quantities.values()]):
        _unavailable("crypto_invalid_long_only_book")
    value = cash + sum(quantities[p] * prices[p] for p in PRODUCTS)
    if value <= 0:
        _unavailable("crypto_nonpositive_book_value")
    return quantities, cash, value, [quantities[p] * prices[p] / value for p in PRODUCTS]


def evaluate_crypto(approach_id: str, dataset: dict | None, holdings: dict, *, cutoff: str,
                    capture: Callable[[dict], None], allow_synthetic: bool = False):
    """Capture the exact prepared representation *before* a rule/solver executes.

    ``capture`` must durably save before returning; its exception prevents the
    solve and propagates. Inputs are copied, so later source mutations cannot
    rewrite a captured decision. A failed precondition is captured as not_invoked
    before its error propagates to the normal hourly failure/status envelope.
    """
    name = approach_id.removeprefix("crypto_")
    if approach_id != "crypto_" + name or name not in BOOKS:
        raise ValueError("unknown native crypto approach")
    captured = False

    def before(payload):
        nonlocal captured
        # Set first so capture failures are not followed by an alternate capture.
        captured = True
        capture(copy.deepcopy(payload))

    context = {"adapter_version": ADAPTER_VERSION, "policy_version": IMPLEMENTATION,
               "approach_id": approach_id, "instruments": list(PRODUCTS),
               "cadence": "hourly_crypto_24x7", "availability_cutoff": cutoff,
               "news": {"consumed": False, "reason": "News not consumed"},
               "forecast": {"available": False, "reason": "Allocation is not a value forecast"}}
    try:
        now = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
        if now.tzinfo is None:
            _unavailable("crypto_cutoff_timezone_required")
        if name == "cash":
            before({**context, "boundary": "native_crypto_constant_cash", "parameters": {},
                    "holdings_consumed": False, "price_history_consumed": False,
                    "invocation_state": "invoking"})
            return {"weights": {p: 0.0 for p in PRODUCTS}, "cash": 1.0,
                    "solution_status": "not_applicable", "policy_id": IMPLEMENTATION + "/cash",
                    "forecast": {"available": False, "status": "not_available"}}
        if dataset is None:
            _unavailable("native_crypto_dataset_missing")
        validate_dataset(dataset, now.astimezone(UTC))
        if dataset["market_data_synthetic"] and not allow_synthetic:
            _unavailable("synthetic_crypto_dataset_not_forward_eligible")
        prices = {p: float(dataset["bars"][p][-1]["close"]) for p in PRODUCTS}
        quantities, cash, value, current = _book(holdings, prices)
        native = {"dataset_id": dataset["dataset_id"], "dataset_checksum": dataset["dataset_checksum"],
                  "as_of": dataset["as_of"], "available_at": dataset["available_at"],
                  "market_data_synthetic": dataset["market_data_synthetic"],
                  "provider_sources": copy.deepcopy(dataset.get("sources", [])),
                  "granularity_seconds": 3600, "symbols_order": list(PRODUCTS)}
        book = {"positions": copy.deepcopy(holdings.get("positions", [])), "cash_balance": cash,
                "revision": holdings.get("revision"), "mark_prices": prices,
                "mark_as_of": dataset["as_of"], "portfolio_value": value,
                "current_weights": current}
        if name in ("btc_buy_hold", "equal_buy_hold"):
            invested = any(q > 0 for q in quantities.values())
            rule = {"rule": "retain_actual_positions_after_initial_allocation",
                    "initial_weights": [1.0, 0.0] if name == "btc_buy_hold" else [0.5, 0.5],
                    "initialized_from_actual_positions": invested}
            before({**context, "boundary": "native_crypto_buy_hold_control", "parameters": rule,
                    "native_source": native, "holdings": book, "eligible_prices": prices,
                    "price_history_consumed": False, "invocation_state": "invoking"})
            weights = current if invested else rule["initial_weights"]
            return {"weights": dict(zip(PRODUCTS, weights)), "cash": cash / value if invested else 0.0,
                    "status": "hold" if invested else "recommended",
                    "solution_status": "no_effect" if invested else "not_applicable",
                    "policy_id": IMPLEMENTATION + "/" + name,
                    "forecast": {"available": False, "status": "not_available"}}
        px = np.array([[r["close"] for r in dataset["bars"][p]] for p in PRODUCTS], dtype=float).T
        returns = px[1:] / px[:-1] - 1
        cfg = settings({"lookback_days": min(167, len(returns)), "horizon_sessions": 1,
                        "cash_weight": .1, "max_weight": .6, "turnover_penalty": .001})
        inputs = {"algorithm": name, "instruments": list(PRODUCTS), "settings": cfg,
                  "current_weights": current, "covariance": (np.cov(returns, rowvar=False) + np.eye(2) * 1e-10).tolist(),
                  "expected_returns": returns.mean(axis=0).tolist(), "scenarios": returns.tolist()}
        before({**context, **inputs, "boundary": "native_crypto_classical_solve",
                "native_source": native, "holdings": book, "invocation_state": "invoking",
                "price_history_consumed": True,
                "price_history": {p: [{"start": r["start"], "close": r["close"]} for r in dataset["bars"][p]] for p in PRODUCTS},
                "transformation": {"version": ADAPTER_VERSION, "dtype": "float64", "ordering": "ascending UTC hourly start; columns BTC-USD,ETH-USD",
                                   "returns": "close[t]/close[t-1]-1; unitless per hour",
                                   "expected_returns": "arithmetic mean of hourly returns",
                                   "covariance": "sample covariance ddof=1 + identity*1e-10; hourly return squared",
                                   "settings_time_unit": "native hours; legacy setting names days/sessions retained without daily resampling",
                                   "normalization": "none"},
                "fallback": {"method": "bounded_scalar_same_objective", "bounds": [.3, .6],
                             "xatol": 1e-12, "candidates": [.3, .45, .6], "second_weight": ".9-first_weight"}})
        result = solve(copy.deepcopy(inputs))
        if result["status"] not in ("optimal", "feasible"):
            fit = minimize_scalar(lambda x: -objective(inputs, [float(x), .9-float(x)]),
                                  bounds=(.3, .6), method="bounded", options={"xatol": 1e-12})
            candidates = [.3, .45, .6] + ([float(fit.x)] if fit.success else [])
            x = max(candidates, key=lambda item: objective(inputs, [item, .9-item]))
            if np.isfinite(objective(inputs, [x, .9-x])):
                result = {"status": "feasible", "weights": [x, .9-x], "fallback": "bounded_scalar_same_objective"}
        if result["status"] not in ("optimal", "feasible"):
            _unavailable("native_crypto_solver_" + result["status"])
        weights = result["weights"]
        if isinstance(weights, dict):
            weights = [weights[p] for p in PRODUCTS]
        if len(weights) != 2 or not np.isfinite(weights).all() or min(weights) < 0 or max(weights) > .600001 or abs(sum(weights)-.9) > 1e-6:
            raise ValueError("native crypto solver returned invalid allocation")
        return {"weights": dict(zip(PRODUCTS, map(float, weights))), "cash": 1-float(sum(weights)),
                "solution_status": result["status"], "policy_id": IMPLEMENTATION + "/" + name,
                "decision_as_of": dataset["as_of"], "diagnostics": result,
                "forecast": {"available": False, "status": "not_available",
                             "reason": "Native historical hourly allocation objective is not a calibrated portfolio forecast"}}
    except Exception as exc:
        if not captured:
            before({**context, "boundary": "not_invoked_native_crypto_precondition", "invocation_state": "not_invoked",
                    "error_type": type(exc).__name__, "reason": str(exc),
                    "native_source_supplied": copy.deepcopy(dataset), "holdings": copy.deepcopy(holdings)})
        raise


def crypto_adapter(base, source, holdings, before):
    """External adapter signature consumed by ``evaluate_approach``.

    Historical hourly bars are inference inputs only, never execution quotes.
    A compatible common cohort must be exactly the native two-product universe.
    """
    if list(source.get("investable_universe", [])) != PRODUCTS:
        before({"boundary": "not_invoked_crypto_cohort_mismatch", "invocation_state": "not_invoked",
                "supplied_universe": source.get("investable_universe"), "required_universe": list(PRODUCTS)})
        _unavailable("native_crypto_incompatible_comparison_cohort")
    return evaluate_crypto(base["approach_id"], source.get("native_crypto_dataset"), holdings,
                           cutoff=base["availability_cutoff"], capture=before,
                           allow_synthetic=source.get("trigger") == "controlled_replay")
