"""Point-in-time return, mean and covariance estimators (BASE-02, BASE-03).

Every estimator reads returns **only** through the decision's
:class:`~finplan_model.sim.market.PointInTimeView` (``view.returns(lookback)``), so an estimate at
decision time ``t`` uses only observations available at or before ``t``. Estimators are named in
the strategy configuration:

* covariance: ``sample`` (unbiased sample covariance), ``ledoit_wolf`` (shrinkage toward a scaled
  identity, Ledoit and Wolf 2004), ``diagonal`` (sample variances only), ``ewma`` (exponentially
  weighted, ``ewma_halflife`` sessions);
* expected return: ``historical_mean``, ``ewma_mean`` (``ewma_halflife``), ``zero``.

All estimates are per session (not annualized); optimizers compare quantities on the same scale.
"""

from __future__ import annotations

import numpy as np

from finplan_model.core.errors import FinplanError
from finplan_model.sim.market import PointInTimeView

__all__ = ["COVARIANCE_ESTIMATORS", "MEAN_ESTIMATORS", "estimate_covariance", "estimate_mean", "pit_returns"]

COVARIANCE_ESTIMATORS = ("sample", "ledoit_wolf", "diagonal", "ewma")
MEAN_ESTIMATORS = ("historical_mean", "ewma_mean", "zero")


def pit_returns(view: PointInTimeView, lookback: int) -> np.ndarray:
    """``T x N`` simple returns over the last ``lookback`` periods visible at the decision time."""
    _dates, r = view.returns(lookback)
    return np.asarray(r, dtype=float)


def _ewma_weights(t: int, halflife: float) -> np.ndarray:
    lam = 0.5 ** (1.0 / halflife)
    w = lam ** np.arange(t - 1, -1, -1, dtype=float)
    return w / w.sum()


def estimate_mean(r: np.ndarray, method: str, *, halflife: float = 20.0) -> np.ndarray:
    if method == "zero":
        return np.zeros(r.shape[1])
    if method == "historical_mean":
        return r.mean(axis=0)
    if method == "ewma_mean":
        return _ewma_weights(r.shape[0], halflife) @ r
    raise FinplanError.validation("unknown expected-return estimator", pointer="/params/return_estimator")


def estimate_covariance(r: np.ndarray, method: str, *, halflife: float = 20.0) -> np.ndarray:
    t, n = r.shape
    if t < 2:
        raise FinplanError.validation("covariance needs at least two return observations", pointer="/params/lookback")
    x = r - r.mean(axis=0)
    sample = (x.T @ x) / (t - 1)
    if method == "sample":
        cov = sample
    elif method == "diagonal":
        cov = np.diag(np.diag(sample))
    elif method == "ewma":
        w = _ewma_weights(t, halflife)
        mu = w @ r
        xc = r - mu
        cov = (xc * w[:, None]).T @ xc / max(1e-300, 1.0 - float((w**2).sum()))
    elif method == "ledoit_wolf":
        s = (x.T @ x) / t  # MLE covariance used by the Ledoit-Wolf formula
        mu = float(np.trace(s)) / n
        target = mu * np.eye(n)
        d2 = float(((s - target) ** 2).sum())
        b2 = sum(float(((np.outer(row, row) - s) ** 2).sum()) for row in x) / (t * t)
        b2 = min(b2, d2)
        shrink = 0.0 if d2 == 0 else b2 / d2
        cov = shrink * target + (1.0 - shrink) * s
    else:
        raise FinplanError.validation("unknown covariance estimator", pointer="/params/covariance_estimator")
    cov = (cov + cov.T) / 2.0
    return cov
