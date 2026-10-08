"""Compact strategy comparison carried in the job result (``payload.benchmark``).

The research-workspace run artifact holds the full evidence (NAV, fills, decisions) but callers of
``get_experiment_result`` cannot read it (by design). This module derives, from the same
:class:`~finplan_model.evaluate.EvaluationResult` objects, a small summary that the job result
document carries next to the contract sections:

* one row per evaluated strategy (the configured strategy first, then each control) with
  ``total_return``, ``cagr``, ``ann_volatility``, ``sharpe``, ``max_drawdown``, ``turnover``,
  ``transaction_cost`` (base currency, USD by default) and ``transaction_cost_fraction``;
* the strategy's ``final_weights`` (end of the last session, after price drift) and
  ``average_weights`` (mean of the end-of-session weights over every session of the window), per
  instrument, plus the matching cash weights;
* the evaluation window, the risk-free assumption and a ``units`` legend.

Contract placement: ``finance/v1/run-result-payload.json`` (contract 1.1.0) is an *open* object
(no ``additionalProperties: false``) and its ``performance`` section only admits numbers, so the
comparison goes under the payload key ``benchmark``; no contract change is needed.

Units (no math changed, labels only): ``turnover`` is the simulator's cumulative traded notional
(buys plus sells, absolute) divided by the initial capital, so ``1.0`` means one full portfolio's
worth traded over the window; for a NAV near its starting value it equals the cumulative sum of
absolute weight changes. ``transaction_cost`` is the cumulative fees + spread + slippage in the
simulation's ``base_currency`` (USD by default) on ``initial_capital``; ``transaction_cost_fraction``
is the same amount divided by the initial capital. Example: turnover 19.6 on USD 100,000 is USD 1.96M
traded; at 1 bp fee + 1 bp half spread that is about USD 392 (0.39 % of the starting capital).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from finplan_model.sim.market import MarketData

from .strategy_resolver import CONTROLS

__all__ = ["MAX_WEIGHT_ENTRIES", "UNITS", "benchmark_section", "strategy_weights"]

SCHEMA = "finplan.benchmark_comparison/1"
#: Per-map cap on instrument entries (keeps the result well inside the run item size limit).
MAX_WEIGHT_ENTRIES = 40
_DIGITS = 6

UNITS: dict[str, str] = {
    "total_return": "fraction; net of costs, final NAV / initial capital - 1",
    "cagr": "fraction per year; (1 + total_return) ** (periods_per_year / session_returns) - 1",
    "ann_volatility": "fraction per year; sample std of session returns x sqrt(periods_per_year)",
    "sharpe": "annualized; (mean session return - per-session risk-free) / std x sqrt(periods_per_year); null when volatility is 0",
    "max_drawdown": "positive fraction; largest peak-to-trough NAV decline (0.12 = -12 %)",
    "turnover": "multiple of initial capital; cumulative traded notional (buys + sells) / initial capital (1.0 = one full portfolio traded)",
    "transaction_cost": "base_currency amount; cumulative fees + spread + slippage, relative to initial_capital",
    "transaction_cost_fraction": "fraction of initial capital; transaction_cost / initial_capital",
    "weights": "fraction of NAV at the session close; final = last session, average = mean over every session of the window; cash reported separately",
}


def _num(x: Any) -> float | None:
    if x is None or isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        return None
    return round(float(x), _DIGITS) + 0.0


def strategy_weights(sim: Any, market: MarketData) -> tuple[dict[str, float], float, dict[str, float], float]:
    """(final weights, final cash weight, average weights, average cash weight) from the ledger.

    Holdings are rebuilt from the fills; each session is marked at its close (the last known close
    when a bar is missing), exactly as the simulator marks its NAV.
    """
    fills_by_session: dict[str, list[Any]] = {}
    for f in sim.fills:
        fills_by_session.setdefault(f.session_date, []).append(f)
    shares: dict[str, float] = {}
    marks: dict[str, float] = {}
    sums: dict[str, float] = {}
    cash_sum = 0.0
    last: dict[str, float] = {}
    last_cash = 1.0
    n = 0
    for row in sim.nav:
        s_iso = row["session_date"]
        for f in fills_by_session.get(s_iso, ()):
            shares[f.instrument_id] = shares.get(f.instrument_id, 0.0) + f.quantity
        s = date.fromisoformat(s_iso)
        for i in shares:
            bar = market.bar(i, s)
            if bar is not None and bar.close is not None:
                marks[i] = float(bar.close)
        value = float(row["value"])
        if value <= 0:
            continue
        w = {i: q * marks.get(i, 0.0) / value for i, q in shares.items() if q}
        cw = float(row["cash"]) / value
        for i, x in w.items():
            sums[i] = sums.get(i, 0.0) + x
        cash_sum += cw
        last, last_cash, n = w, cw, n + 1
    avg = {i: x / n for i, x in sums.items()} if n else {}
    return last, last_cash, avg, (cash_sum / n if n else 1.0)


def _compact(weights: Mapping[str, float]) -> tuple[dict[str, float], bool]:
    items = [(i, r) for i, x in weights.items() if (r := _num(x)) is not None and r != 0.0]
    items.sort(key=lambda kv: (-abs(kv[1]), kv[0]))
    truncated = len(items) > MAX_WEIGHT_ENTRIES
    return dict(sorted(items[:MAX_WEIGHT_ENTRIES])), truncated


def _row(res: Any, market: MarketData, *, primary: str) -> dict[str, Any]:
    sim, met = res.simulation, res.metrics
    cfg = sim.config
    init = float(cfg.initial_cash)
    total_return = float(met.get("net_cumulative_return") or 0.0)
    periods = max(len(sim.nav) - 1, 1)
    cagr = -1.0 if total_return <= -1.0 else (1.0 + total_return) ** (cfg.periods_per_year / periods) - 1.0
    costs = float(sim.summary.get("total_costs") or 0.0)
    final_w, final_cash, avg_w, avg_cash = strategy_weights(sim, market)
    fw, f_trunc = _compact(final_w)
    aw, a_trunc = _compact(avg_w)
    row: dict[str, Any] = {
        "strategy": res.strategy,
        "role": "control" if res.strategy in CONTROLS else "optimizer",
        "primary": res.strategy == primary,
        "solution_status": res.solution_status,
        "metrics": {
            "total_return": _num(total_return),
            "cagr": _num(cagr),
            "ann_volatility": _num(met.get("annualized_volatility")),
            "sharpe": _num(met.get("sharpe_ratio")),
            "max_drawdown": _num(met.get("max_drawdown")),
            "turnover": _num(met.get("turnover")),
            "transaction_cost": _num(costs),
            "transaction_cost_fraction": _num(costs / init if init else 0.0),
        },
        "final_weights": fw,
        "final_cash_weight": _num(final_cash),
        "average_weights": aw,
        "average_cash_weight": _num(avg_cash),
    }
    if f_trunc or a_trunc:
        row["weights_truncated_to"] = MAX_WEIGHT_ENTRIES
    return row


def benchmark_section(results: Sequence[Any], market: MarketData, *, primary: str) -> dict[str, Any]:
    """The ``payload.benchmark`` block for one or more comparable evaluation results."""
    first = results[0]
    cfg = first.simulation.config
    win = first.evaluation_window
    return {
        "schema": SCHEMA,
        "primary_strategy": primary,
        "evaluation_window": {"start": win["start"], "end": win["end"], "sessions": len(first.simulation.nav)},
        "periods_per_year": int(cfg.periods_per_year),
        "base_currency": str(cfg.base_currency),
        "initial_capital": _num(cfg.initial_cash),
        "risk_free": {"annual_rate": _num(cfg.cash_rate_annual), "source": str(first.metrics.get("sharpe_risk_free_source") or "configured_cash_rate")},
        "units": dict(UNITS),
        "strategies": [_row(r, market, primary=primary) for r in results],
    }
