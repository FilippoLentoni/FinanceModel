"""Runtime settings the control plane reads at request time (never committed values).

Everything that differs per deployment or that an operator changes lives in SSM, never in this
repository (prices, the auto-approve threshold, role references, the research storage reference):

==============================================  ============================================================
setting                                         parameter
==============================================  ============================================================
instance prices (JSON)                          ``/finplan/<env>/financemodel/config/instance-prices``
auto-approve threshold (USD, number)            ``/finplan/<env>/financemodel/config/auto-approve-usd``
lease limits (JSON ``{"cpu": 1, "gpu": 0}``)    ``/finplan/<env>/financemodel/config/lease-limits``
production-candidate principals (JSON list)     ``/finplan/<env>/financemodel/config/production-candidate-principals``
approver role reference (role name or ARN)      ``/finplan/<env>/financemodel/config/approver-role-ref``
research storage reference (bucket)             ``/finplan/<env>/financemodel/config/research-storage-ref``
production strategy (JSON, read uncached)       ``/finplan/<env>/financemodel/config/production-strategy``
job role reference (role ARN)                   ``/finplan/<env>/financemodel/job/job-role-ref``
job definition per job type (JSON)              ``/finplan/<env>/financemodel/job/<job-type-kebab>``
budget allocation (JSON, FinancialPlanning)     ``/finplan/shared/financialplanning/config/budget-allocation``
budget state (JSON, FinancialPlanning)          ``/finplan/shared/financialplanning/config/budget-state``
==============================================  ============================================================

Instance prices document (written by an operator from current AWS pricing, design D4)::

    {"retrieved_at": "2026-10-07T00:00:00Z", "currency": "USD", "usd_per_hour": {"ml.m5.xlarge": <price>}}

Job definition document (published by the deploy for every deployed job type, DEP-03)::

    {"job_type": "run_backtest", "image_uri": "<repository-uri>@sha256:<digest>", "deployed": true}

Missing optional values fall back to configuration defaults (``config/<env>.json``). A value that
cannot be read (throttling, access denied) fails closed with ``DEPENDENCY_UNAVAILABLE``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from finplan_contracts import budget as contract_budget
from finplan_contracts import ssm as contract_ssm

from finplan_model.core.config import EnvConfig
from finplan_model.core.errors import FinplanError

__all__ = ["RuntimeSettings", "SettingsProvider", "SsmSettings", "StaticSettings", "job_definition_parameter"]


def job_definition_parameter(env: str, job_type: str) -> str:
    """``/finplan/<env>/financemodel/job/<job-type-kebab>`` (DEP-04 ``job/*`` references)."""
    return contract_ssm.build(env, "financemodel", "job", job_type.replace("_", "-"))


@runtime_checkable
class SettingsProvider(Protocol):
    def instance_prices(self) -> Mapping[str, Any] | None: ...

    def auto_approve_usd(self) -> float: ...

    def lease_limits(self) -> dict[str, int]: ...

    def budget_allocation(self) -> Mapping[str, float]: ...

    def budget_state(self) -> Any: ...

    def production_candidate_principals(self) -> list[str]: ...

    def approver_role_name(self) -> str | None: ...

    def job_definition(self, job_type: str) -> Mapping[str, Any] | None: ...

    def job_role_arn(self) -> str | None: ...

    def research_storage(self) -> str | None: ...

    def production_strategy(self) -> str | None: ...


RuntimeSettings = SettingsProvider


def _role_name(value: str | None) -> str | None:
    if not value:
        return None
    v = value.strip()
    if v.startswith("arn:"):
        return v.rsplit("/", 1)[-1]
    return v


@dataclass
class StaticSettings:
    """Settings held in memory (unit tests, local runs). Defaults mirror configuration defaults."""

    cfg: EnvConfig
    prices: Mapping[str, Any] | None = None
    auto_approve: float | None = None
    leases: Mapping[str, int] | None = None
    allocation: Mapping[str, float] | None = None
    state: Any = None
    production_principals: list[str] = field(default_factory=list)
    approver_role: str | None = None
    job_definitions: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    job_role: str | None = None
    storage: str | None = None
    strategy_raw: str | None = None

    def instance_prices(self) -> Mapping[str, Any] | None:
        return self.prices

    def auto_approve_usd(self) -> float:
        return float(self.cfg.approval["auto_approve_usd_default"] if self.auto_approve is None else self.auto_approve)

    def lease_limits(self) -> dict[str, int]:
        base = {k: int(v) for k, v in self.cfg.lease["max_holders"].items()}
        base.update({k: int(v) for k, v in (self.leases or {}).items()})
        return base

    def budget_allocation(self) -> Mapping[str, float]:
        return dict(contract_budget.DEFAULT_ALLOCATION if self.allocation is None else self.allocation)

    def budget_state(self) -> Any:
        return self.state

    def production_candidate_principals(self) -> list[str]:
        return list(self.production_principals)

    def approver_role_name(self) -> str | None:
        return _role_name(self.approver_role)

    def job_definition(self, job_type: str) -> Mapping[str, Any] | None:
        return self.job_definitions.get(job_type)

    def job_role_arn(self) -> str | None:
        return self.job_role

    def research_storage(self) -> str | None:
        return self.storage

    def production_strategy(self) -> str | None:
        return self.strategy_raw


class SsmSettings:
    """Reads the settings from SSM (``ssm:GetParameter`` only) with a short in-process cache."""

    def __init__(self, ssm_client: Any, cfg: EnvConfig, *, ttl_seconds: float = 60.0, monotonic: Callable[[], float] = time.monotonic) -> None:
        self.ssm = ssm_client
        self.cfg = cfg
        self.ttl = ttl_seconds
        self._now = monotonic
        self._cache: dict[str, tuple[float, str | None]] = {}

    def _get(self, name: str) -> str | None:
        hit = self._cache.get(name)
        if hit is not None and self._now() - hit[0] < self.ttl:
            return hit[1]
        try:
            value: str | None = self.ssm.get_parameter(Name=name)["Parameter"]["Value"]
        except Exception as exc:  # noqa: BLE001 - classify below; fail closed
            code = getattr(exc, "response", {}).get("Error", {}).get("Code") if hasattr(exc, "response") else None
            if code != "ParameterNotFound":
                raise FinplanError.dependency_unavailable("a runtime setting could not be read; retry", retryable=True, setting=name.rsplit("/", 1)[-1]) from None
            value = None
        self._cache[name] = (self._now(), value)
        return value

    def _json(self, name: str) -> Any:
        raw = self._get(name)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            raise FinplanError.precondition("a runtime setting is not valid JSON", reason="setting_invalid", setting=name.rsplit("/", 1)[-1]) from None

    def instance_prices(self) -> Mapping[str, Any] | None:
        return self._json(self.cfg.ssm["instance_prices"])

    def auto_approve_usd(self) -> float:
        raw = self._get(self.cfg.ssm["auto_approve_usd"])
        if raw is None:
            return float(self.cfg.approval["auto_approve_usd_default"])
        try:
            v = float(raw)
        except ValueError:
            raise FinplanError.precondition("auto-approve threshold is not a number", reason="setting_invalid", setting="auto-approve-usd") from None
        if v < 0:
            raise FinplanError.precondition("auto-approve threshold must be >= 0", reason="setting_invalid", setting="auto-approve-usd")
        return v

    def lease_limits(self) -> dict[str, int]:
        base = {k: int(v) for k, v in self.cfg.lease["max_holders"].items()}
        doc = self._json(self.cfg.ssm["lease_limits"]) or {}
        if not isinstance(doc, Mapping) or not all(isinstance(v, int) and v >= 0 for v in doc.values()):
            raise FinplanError.precondition("lease limits must map instance classes to non-negative integers", reason="setting_invalid", setting="lease-limits")
        base.update({str(k): int(v) for k, v in doc.items()})
        return base

    def budget_allocation(self) -> Mapping[str, float]:
        doc = self._json(self.cfg.ssm["budget_allocation"])
        return dict(contract_budget.DEFAULT_ALLOCATION) if doc is None else dict(doc)

    def budget_state(self) -> Any:
        return self._get(self.cfg.ssm["budget_state"])

    def production_candidate_principals(self) -> list[str]:
        doc = self._json(self.cfg.ssm["production_candidate_principals"]) or []
        if not isinstance(doc, list) or not all(isinstance(x, str) for x in doc):
            raise FinplanError.precondition("production-candidate principals must be a JSON list of role names", reason="setting_invalid", setting="production-candidate-principals")
        return [str(_role_name(x)) for x in doc]

    def approver_role_name(self) -> str | None:
        return _role_name(self._get(self.cfg.ssm["approver_role_ref"]))

    def job_definition(self, job_type: str) -> Mapping[str, Any] | None:
        doc = self._json(job_definition_parameter(self.cfg.env, job_type))
        if isinstance(doc, Mapping) and job_type == "swarm_mode_a":
            image = self._get(self.cfg.ssm_name("config", "vllm-image"))
            if not image:
                return None
            doc = {**doc, "image_uri": image, "image_digest": image.rsplit("@", 1)[-1]}
        return doc if isinstance(doc, Mapping) else None

    def job_role_arn(self) -> str | None:
        return self._get(self.cfg.ssm["job_role_ref"])

    def research_storage(self) -> str | None:
        raw = self._get(self.cfg.ssm["research_storage_ref"])
        if raw is None:
            return None
        if raw.lstrip().startswith("{"):
            try:
                return str(json.loads(raw).get("bucket") or "") or None
            except json.JSONDecodeError:
                return None
        return raw.strip()

    def production_strategy(self) -> str | None:
        """Raw production-strategy value, read without the cache (a cleared key stops the next job)."""
        from .production_strategy import SsmStrategyParameter

        return SsmStrategyParameter(self.ssm, self.cfg.ssm["production_strategy"]).read()
