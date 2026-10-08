"""Pre-flight cost estimate and category budget check (spec job-execution-controls, "Pre-flight cost
estimate and budget check", "Category budget allocation"; design D4; CTL-07, CTL-10).

* :func:`estimate` - the upper bound ``price_per_hour x max_runtime_hours x instance_count +
  storage_estimate``. Prices come only from ``/finplan/<env>/financemodel/config/instance-prices``;
  a missing price, a missing retrieval date or a price older than ``cost.price_max_age_days`` blocks
  the job with ``PRECONDITION_FAILED``.
* :func:`check_budget` - the job type's declared ``budget_category`` (missing -> ``PRECONDITION_FAILED``
  ``budget_category_missing``) checked by the contract package's own pre-flight
  (:func:`finplan_contracts.budget.preflight`) against the shared allocation
  (``/finplan/shared/financialplanning/config/budget-allocation``) minus this environment's spend in
  that category, and refused while ``budget-state`` reports the 100% deny action active.
  The refusal is ``BUDGET_EXCEEDED`` (never retryable) naming the category, the estimate and the
  remaining allocation. Funds of another category (``gpu``) are never used.
* :func:`run_spend_usd` - what a run counts against its category: the actual billed cost when known,
  otherwise the estimate (an upper bound), and nothing for a run that ended before its SageMaker job
  started (design D4: "non-cancelled-before-start runs").
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from finplan_contracts import budget as contract_budget

from finplan_model.core.clock import parse_utc, utc_iso
from finplan_model.core.config import EnvConfig, JobTypeConfig
from finplan_model.core.errors import ErrorCode, FinplanError

__all__ = ["CostEstimate", "check_budget", "estimate", "run_spend_usd", "spend_records"]


@dataclass(frozen=True)
class CostEstimate:
    estimated_usd_upper_bound: float
    price_retrieved_at: str
    budget_category: str
    usd_per_hour: float
    max_runtime_seconds: int
    instance_count: int
    instance_type: str
    storage_usd: float

    def block(self, remaining_allocation_usd: float, *, synthetic: bool | None = None) -> dict[str, Any]:
        """The contract ``cost-estimate`` block."""
        d: dict[str, Any] = {
            "estimated_usd_upper_bound": self.estimated_usd_upper_bound,
            "price_retrieved_at": self.price_retrieved_at,
            "remaining_allocation_usd": round(remaining_allocation_usd, 6),
            "budget_category": self.budget_category,
        }
        if synthetic:
            d["synthetic"] = True
        return d


def _ceil6(x: float) -> float:
    return math.ceil(x * 1_000_000 - 1e-9) / 1_000_000


def estimate(cfg: EnvConfig, job_type: JobTypeConfig, *, instance_type: str, instance_count: int, max_runtime_seconds: int, prices: Mapping[str, Any] | None, now: datetime) -> CostEstimate:
    if not job_type.budget_category:
        raise FinplanError.precondition("the job type declares no budget category; paid jobs need one", reason="budget_category_missing", job_type=job_type.name)
    table = (prices or {}).get("usd_per_hour") if isinstance(prices, Mapping) else None
    price = table.get(instance_type) if isinstance(table, Mapping) else None
    if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(float(price)) or float(price) < 0:
        raise FinplanError.precondition("no price is configured for the requested instance type", reason="instance_price_missing", instance_type=instance_type)
    retrieved_raw = (prices or {}).get("retrieved_at")
    try:
        retrieved = parse_utc(str(retrieved_raw))
    except (TypeError, ValueError):
        raise FinplanError.precondition("the instance price has no valid retrieval date", reason="instance_price_undated", instance_type=instance_type) from None
    max_age = timedelta(days=float(cfg.cost.get("price_max_age_days", 30)))
    if now - retrieved > max_age:
        raise FinplanError.precondition("the configured instance price is older than the allowed age", reason="instance_price_stale", instance_type=instance_type, max_age_days=float(cfg.cost.get("price_max_age_days", 30)))
    storage = float(cfg.cost.get("storage_estimate_usd", 0.0))
    total = _ceil6(float(price) * (max_runtime_seconds / 3600.0) * instance_count + storage)
    return CostEstimate(total, utc_iso(retrieved), job_type.budget_category, float(price), int(max_runtime_seconds), int(instance_count), instance_type, storage)


def run_spend_usd(run: Mapping[str, Any]) -> float:
    """USD a run counts against its category (actual when billed, else the upper-bound estimate)."""
    if run.get("actual_cost_usd") is not None:
        return float(run["actual_cost_usd"])
    if run.get("dry_run"):
        return 0.0
    started = bool(run.get("job_started"))
    if run.get("state") in ("cancelled", "failed") and not started:
        return 0.0
    est = (run.get("cost_estimate") or {}).get("estimated_usd_upper_bound")
    return float(est or 0.0)


def spend_records(runs: Iterable[Mapping[str, Any]], *, exclude_run_id: str | None = None) -> list[dict[str, Any]]:
    out = []
    for r in runs:
        if r.get("run_id") == exclude_run_id or not r.get("budget_category"):
            continue
        usd = run_spend_usd(r)
        if usd > 0:
            out.append({"usd": usd, "category": r["budget_category"], "billed_by": "aws"})
    return out


def check_budget(est: CostEstimate, *, allocation: Mapping[str, float], budget_state: Any, runs: Iterable[Mapping[str, Any]], correlation_id: str, exclude_run_id: str | None = None) -> float:
    """Contract pre-flight; returns the remaining allocation of the category or raises."""
    res = contract_budget.preflight(
        est.budget_category,
        est.estimated_usd_upper_bound,
        allocation,
        spend_records(runs, exclude_run_id=exclude_run_id),
        budget_state=budget_state,
        correlation_id=correlation_id,
    )
    if res.allowed:
        return float(res.remaining_allocation_usd or 0.0)
    err = res.error or {}
    details = {k: v for k, v in (err.get("details") or {}).items() if k in ("budget_category", "estimated_usd_upper_bound", "remaining_allocation_usd", "budget_state", "pointer", "field", "problems")}
    code = str(err.get("code") or ErrorCode.BUDGET_EXCEEDED)
    if code == ErrorCode.VALIDATION_FAILED:
        # An invalid shared allocation refuses all paid work; it is a precondition of this call.
        raise FinplanError.precondition("the shared budget allocation is invalid; no paid work starts", reason="budget_allocation_invalid")
    raise FinplanError(code, str(err.get("message") or "budget exceeded"), details=details)
