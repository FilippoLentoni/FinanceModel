"""Portfolio metrics computed by the common evaluator from a NAV series (deterministic, pure Python).

Conventions (documented in ``docs/simulator.md``):

* returns are simple session-to-session returns of the NAV;
* annualized volatility = sample standard deviation of session returns x sqrt(periods_per_year);
* Sharpe = (mean session return - per-session risk-free) / sample std x sqrt(periods_per_year),
  with the risk-free source stated by the caller (default: the configured cash rate);
* maximum drawdown = largest peak-to-trough decline of the NAV, as a positive fraction.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = ["annualized_volatility", "cumulative_return", "max_drawdown", "period_returns", "sharpe_ratio"]


def period_returns(values: Sequence[float]) -> list[float]:
    return [values[k] / values[k - 1] - 1.0 for k in range(1, len(values)) if values[k - 1] != 0]


def cumulative_return(values: Sequence[float]) -> float:
    if not values or values[0] == 0:
        return 0.0
    return values[-1] / values[0] - 1.0


def max_drawdown(values: Sequence[float]) -> float:
    peak, mdd = -math.inf, 0.0
    for v in values:
        peak = max(peak, v)
        if peak > 0:
            mdd = max(mdd, (peak - v) / peak)
    return mdd


def _std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def annualized_volatility(values: Sequence[float], periods_per_year: int = 252) -> float:
    return _std(period_returns(values)) * math.sqrt(periods_per_year)


def sharpe_ratio(values: Sequence[float], periods_per_year: int = 252, risk_free_annual: float = 0.0) -> float | None:
    """None when volatility is zero (undefined), never infinity."""
    r = period_returns(values)
    sd = _std(r)
    if not r or sd == 0:
        return None
    rf = (1.0 + risk_free_annual) ** (1.0 / periods_per_year) - 1.0
    return (sum(r) / len(r) - rf) / sd * math.sqrt(periods_per_year)
