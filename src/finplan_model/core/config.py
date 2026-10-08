"""FinanceModel configuration: ``config/<env>.json`` (beta, gamma, prod) and ``config/shared.json``.

Configuration holds settings, defaults and SSM parameter *names* only: never account IDs, ARNs,
bucket, role or endpoint values, prices or secrets (the leak scan enforces it). Live values such as
instance prices, the auto-approve threshold or the research-storage reference are read at run time
from ``/finplan/<env>/financemodel/config/*``; the account-level budget allocation and state are
read from ``/finplan/shared/financialplanning/config/{budget-allocation,budget-state}``.

Every value labeled a default here is configuration pending user review (design FM-OQ-3, FM-OQ-5),
not a market fact.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from finplan_contracts import ssm as contract_ssm

__all__ = [
    "BUDGET_CATEGORIES",
    "CONFIG_DIR",
    "ConfigError",
    "DEPLOYED_ENVIRONMENTS",
    "EnvConfig",
    "JobTypeConfig",
    "is_gpu_instance_type",
    "load_all",
    "load_config",
    "load_shared_config",
    "validate_config",
]

REPO = "financemodel"
CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"
DEPLOYED_ENVIRONMENTS = ("beta", "gamma", "prod")
#: Budget categories FinanceModel jobs may draw on (contracts D11): cpu_research (USD 7 default),
#: gpu (USD 25 default, used only by the next change).
BUDGET_CATEGORIES = ("cpu_research", "gpu")
_INSTANCE_RE = re.compile(r"^ml\.[a-z][a-z0-9]*\.[a-z0-9]+\Z")
_GPU_FAMILIES = re.compile(r"^ml\.(p[0-9]|g[0-9]|inf[0-9]|trn[0-9]|dl[0-9])")
_REQUIRED_TOP = ("environment", "region", "phase", "served_contract_majors", "instrument", "job_types", "lease", "queue", "approval", "cost", "idempotency", "retention", "simulation_defaults")


class ConfigError(ValueError):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


def is_gpu_instance_type(instance_type: str) -> bool:
    """True for SageMaker GPU/accelerator families (p*, g*, inf*, trn*, dl*)."""
    return bool(_GPU_FAMILIES.match(instance_type))


@dataclass(frozen=True)
class JobTypeConfig:
    name: str
    default_runtime_seconds: int
    max_runtime_seconds: int
    instance_types: tuple[str, ...]
    default_instance_type: str
    max_instance_count: int
    budget_category: str | None
    compute_class: str
    deployed: bool
    entry_point: str
    #: Per-kind approval ceiling (USD): runs estimated at or below it need no human approval.
    auto_approve_usd: float | None = None
    #: Build cost check (DRJ-05): the upper-bound estimate at ``cost_check_usd_per_hour`` must not exceed it.
    cost_cap_usd: float | None = None
    cost_check_usd_per_hour: float | None = None

    @classmethod
    def from_dict(cls, name: str, d: Mapping[str, Any]) -> "JobTypeConfig":
        return cls(
            name=name,
            default_runtime_seconds=int(d["default_runtime_seconds"]),
            max_runtime_seconds=int(d["max_runtime_seconds"]),
            instance_types=tuple(d["instance_types"]),
            default_instance_type=str(d["default_instance_type"]),
            max_instance_count=int(d.get("max_instance_count", 1)),
            budget_category=d.get("budget_category"),
            compute_class=str(d.get("compute_class", "cpu")),
            deployed=bool(d.get("deployed", True)),
            entry_point=str(d.get("entry_point", name)),
            auto_approve_usd=None if d.get("auto_approve_usd") is None else float(d["auto_approve_usd"]),
            cost_cap_usd=None if d.get("cost_cap_usd") is None else float(d["cost_cap_usd"]),
            cost_check_usd_per_hour=None if d.get("cost_check_usd_per_hour") is None else float(d["cost_check_usd_per_hour"]),
        )


@dataclass(frozen=True)
class EnvConfig:
    env: str
    raw: Mapping[str, Any] = field(repr=False)

    @property
    def region(self) -> str:
        return str(self.raw["region"])

    @property
    def phase(self) -> int:
        return int(self.raw["phase"])

    @property
    def served_contract_majors(self) -> tuple[int, ...]:
        return tuple(int(m) for m in self.raw["served_contract_majors"])

    @property
    def instrument(self) -> Mapping[str, Any]:
        return self.raw["instrument"]

    @property
    def job_types(self) -> dict[str, JobTypeConfig]:
        return {k: JobTypeConfig.from_dict(k, v) for k, v in self.raw["job_types"].items()}

    def job_type(self, name: str) -> JobTypeConfig | None:
        d = self.raw["job_types"].get(name)
        return JobTypeConfig.from_dict(name, d) if d is not None else None

    @property
    def lease(self) -> Mapping[str, Any]:
        return self.raw["lease"]

    @property
    def queue(self) -> Mapping[str, Any]:
        return self.raw["queue"]

    @property
    def approval(self) -> Mapping[str, Any]:
        return self.raw["approval"]

    @property
    def cost(self) -> Mapping[str, Any]:
        return self.raw["cost"]

    @property
    def idempotency(self) -> Mapping[str, Any]:
        return self.raw["idempotency"]

    @property
    def retention(self) -> Mapping[str, Any]:
        return self.raw["retention"]

    @property
    def simulation_defaults(self) -> Mapping[str, Any]:
        return self.raw["simulation_defaults"]

    # ------------------------------------------------------------------ SSM names
    def ssm_name(self, category: str, name: str) -> str:
        """``/finplan/<env>/financemodel/<category>/<name>`` (contract naming)."""
        return contract_ssm.build(self.env, REPO, category, name)

    def platform_ssm_name(self, category: str, name: str) -> str:
        """A FinancialPlanning-owned reference in this environment (read-only for FinanceModel)."""
        return contract_ssm.build(self.env, "financialplanning", category, name)

    @property
    def ssm(self) -> dict[str, str]:
        """Every parameter name FinanceModel reads or writes in this environment."""
        own = {k: self.ssm_name(*v) for k, v in _OWN_PARAMS.items()}
        platform = {k: self.platform_ssm_name(*v) for k, v in _PLATFORM_PARAMS.items()}
        return {**own, **platform, "budget_allocation": contract_ssm.build("shared", "financialplanning", "config", "budget-allocation"), "budget_state": contract_ssm.BUDGET_STATE_PARAMETER}


_OWN_PARAMS = {
    "job_endpoint": ("api", "job-endpoint"),
    "registry_ref": ("model", "registry-ref"),
    "job_role_ref": ("job", "job-role-ref"),
    "job_api_role_ref": ("job", "job-api-role-ref"),
    "release_manifest": ("release", "manifest"),
    "current_release_id": ("release", "current-release-id"),
    "budget_enforced_role_names": ("config", "budget-enforced-role-names"),
    "instance_prices": ("config", "instance-prices"),
    "auto_approve_usd": ("config", "auto-approve-usd"),
    "lease_limits": ("config", "lease-limits"),
    "research_storage_ref": ("config", "research-storage-ref"),
    "approver_role_ref": ("config", "approver-role-ref"),
    "production_candidate_principals": ("config", "production-candidate-principals"),
    "production_strategy": ("config", "production-strategy"),
}
_PLATFORM_PARAMS = {
    "plan_endpoint": ("api", "plan-endpoint"),
    "run_staging_ref": ("config", "run-staging-ref"),
}


def validate_config(env: str, raw: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    for key in _REQUIRED_TOP:
        if key not in raw:
            problems.append(f"{env}: missing key {key!r}")
    if problems:
        return problems
    if raw["environment"] != env:
        problems.append(f"{env}: environment field is {raw['environment']!r}")
    if not re.fullmatch(r"[a-z]{2}-[a-z]+-[0-9]", str(raw["region"])):
        problems.append(f"{env}: region {raw['region']!r} is not a region name")
    if raw["phase"] != 1:
        problems.append(f"{env}: this change implements phase 1 only")
    if not raw["served_contract_majors"] or not all(isinstance(m, int) and m >= 0 for m in raw["served_contract_majors"]):
        problems.append(f"{env}: served_contract_majors must be non-negative integers")
    inst = raw["instrument"]
    if inst.get("data_source") not in ("fixture", "platform_snapshots"):
        problems.append(f"{env}: instrument.data_source must be 'fixture' or 'platform_snapshots'")
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9._-]{0,31}", str(inst.get("ticker", ""))):
        problems.append(f"{env}: instrument.ticker is not an instrument id")
    if inst.get("granularity") != "daily":
        problems.append(f"{env}: instrument.granularity must be 'daily' (no intraday in phases 1 and 2)")
    for name, jt in raw["job_types"].items():
        p = f"{env}: job_types.{name}"
        try:
            j = JobTypeConfig.from_dict(name, jt)
        except (KeyError, TypeError, ValueError) as exc:
            problems.append(f"{p}: {exc}")
            continue
        if not 0 < j.default_runtime_seconds <= j.max_runtime_seconds:
            problems.append(f"{p}: need 0 < default_runtime_seconds <= max_runtime_seconds")
        if j.default_instance_type not in j.instance_types:
            problems.append(f"{p}: default_instance_type not in instance_types")
        for it in j.instance_types:
            if not _INSTANCE_RE.match(it):
                problems.append(f"{p}: {it!r} is not a SageMaker instance type")
            if j.compute_class == "cpu" and is_gpu_instance_type(it):
                problems.append(f"{p}: CPU job type lists GPU instance type {it}")
        if j.budget_category is not None and j.budget_category not in BUDGET_CATEGORIES:
            problems.append(f"{p}: unknown budget category {j.budget_category!r}")
        if j.compute_class not in ("cpu", "gpu"):
            problems.append(f"{p}: compute_class must be cpu or gpu")
        if j.max_instance_count < 1:
            problems.append(f"{p}: max_instance_count must be >= 1")
        problems += daily_cost_problems(j, prefix=p)
    lease = raw["lease"]
    if not all(isinstance(v, int) and v >= 0 for v in lease.get("max_holders", {}).values()):
        problems.append(f"{env}: lease.max_holders values must be non-negative integers")
    if int(raw["queue"].get("max_depth", 0)) < 1:
        problems.append(f"{env}: queue.max_depth must be >= 1")
    if float(raw["approval"].get("auto_approve_usd_default", -1)) < 0:
        problems.append(f"{env}: approval.auto_approve_usd_default must be >= 0")
    if int(raw["idempotency"].get("retention_days", 0)) < 7:
        problems.append(f"{env}: idempotency.retention_days must be at least 7 (contracts)")
    try:
        from finplan_model.sim.config import SimulationConfig

        SimulationConfig.from_dict(raw["simulation_defaults"])
    except Exception as exc:  # noqa: BLE001 - report, do not crash
        problems.append(f"{env}: simulation_defaults invalid: {exc}")
    return problems


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(env: str, config_dir: Path | None = None) -> EnvConfig:
    if env not in DEPLOYED_ENVIRONMENTS:
        raise ConfigError([f"unknown environment {env!r}"])
    raw = _read((config_dir or CONFIG_DIR) / f"{env}.json")
    problems = validate_config(env, raw)
    if problems:
        raise ConfigError(problems)
    return EnvConfig(env=env, raw=raw)


def load_all(config_dir: Path | None = None) -> dict[str, EnvConfig]:
    out: dict[str, EnvConfig] = {}
    problems: list[str] = []
    for env in DEPLOYED_ENVIRONMENTS:
        try:
            out[env] = load_config(env, config_dir)
        except ConfigError as exc:
            problems += exc.problems
    if problems:
        raise ConfigError(problems)
    return out


def load_shared_config(config_dir: Path | None = None) -> dict[str, Any]:
    raw = _read((config_dir or CONFIG_DIR) / "shared.json")
    problems = []
    for key in ("region", "repo", "source"):
        if key not in raw:
            problems.append(f"shared: missing key {key!r}")
    if raw.get("repo") != REPO:
        problems.append("shared: repo must be 'financemodel'")
    if problems:
        raise ConfigError(problems)
    return raw


#: Hard ceiling of the daily recommendation job's upper-bound estimate (design M4).
DAILY_COST_CEILING_USD = 0.15


def daily_cost_problems(j: JobTypeConfig, *, prefix: str = "") -> list[str]:
    """DRJ-05 build check: the ``daily_recommendation`` estimate (one instance, max runtime, the
    configured planning price) must stay within USD 0.15 and within its auto-approve ceiling."""
    if j.name != "daily_recommendation":
        return []
    out: list[str] = []
    if j.max_runtime_seconds > 1800 or j.default_instance_type != "ml.m5.xlarge" or j.max_instance_count != 1 or j.budget_category != "cpu_research":
        out.append(f"{prefix}: daily_recommendation must be one ml.m5.xlarge, cpu_research, at most 1800 s")
    if j.cost_check_usd_per_hour is None or j.cost_cap_usd is None or j.auto_approve_usd is None:
        return [*out, f"{prefix}: daily_recommendation needs cost_check_usd_per_hour, cost_cap_usd and auto_approve_usd"]
    est = j.cost_check_usd_per_hour * j.max_runtime_seconds / 3600.0 * j.max_instance_count
    if j.cost_cap_usd > DAILY_COST_CEILING_USD + 1e-12:
        out.append(f"{prefix}: cost_cap_usd {j.cost_cap_usd} exceeds the USD {DAILY_COST_CEILING_USD} ceiling")
    if est > min(j.cost_cap_usd, DAILY_COST_CEILING_USD) + 1e-12:
        out.append(f"{prefix}: daily_recommendation estimate USD {est:.4f} exceeds the USD {min(j.cost_cap_usd, DAILY_COST_CEILING_USD)} cap")
    if est > j.auto_approve_usd + 1e-12:
        out.append(f"{prefix}: daily_recommendation estimate USD {est:.4f} exceeds its auto-approve threshold USD {j.auto_approve_usd}")
    return out
