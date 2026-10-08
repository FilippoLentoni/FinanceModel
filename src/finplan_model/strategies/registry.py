"""Baseline strategy registry, control auto-add and the CPU-only rule (BASE-01, BASE-06).

* :data:`BASELINES` maps each strategy name to its class; :func:`build_strategy` validates the
  parameters (``VALIDATION_FAILED`` with a pointer into the request) and binds the simulation's
  constraint set.
* :data:`CONTROLS` are ``cash``, ``buy_and_hold`` and ``equal_weight``; :func:`with_controls` adds
  any control a benchmark request omits and reports which were added (they are listed in the
  report).
* :func:`descriptor` gives the registry identity (name, family, parameter schema version, whether
  the strategy predicts or uses randomness, compute class) the model registry mints
  ``model_version`` from (code + image digest + parameter schema).
* :func:`require_cpu_instance` and :func:`validate_baseline_submission`: controls and classical
  optimizers run as CPU SageMaker Processing Jobs only; a GPU (or other accelerator) instance type
  is ``VALIDATION_FAILED`` (BASE-06).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from finplan_model.core.config import EnvConfig, is_gpu_instance_type
from finplan_model.core.errors import FinplanError
from finplan_model.sim.config import ConstraintSet

from .base import BaselineStrategy
from .controls import BuyAndHold, CashControl, EqualWeight
from .optimizers import MeanVariance, MinVariance, ScenarioCVaR

__all__ = ["BASELINES", "BASELINE_JOB_TYPES", "CONTROLS", "build_strategy", "descriptor", "require_cpu_instance", "validate_baseline_submission", "with_controls"]

BASELINES: dict[str, type[BaselineStrategy]] = {c.name: c for c in (CashControl, BuyAndHold, EqualWeight, MinVariance, MeanVariance, ScenarioCVaR)}
CONTROLS = ("cash", "buy_and_hold", "equal_weight")
#: Job types that run controls and classical optimizers (CPU image ``financemodel-cpu``).
BASELINE_JOB_TYPES = ("prepare_dataset", "run_backtest", "run_benchmark", "report")


def build_strategy(name: str, params: Mapping[str, Any] | None = None, *, constraints: ConstraintSet | None = None, pointer: str = "") -> BaselineStrategy:
    cls = BASELINES.get(name)
    if cls is None:
        raise FinplanError.validation("unknown strategy", pointer=f"{pointer}/name", strategy=str(name)[:64], known=sorted(BASELINES))
    try:
        return cls(params, constraints=constraints)
    except FinplanError as exc:
        if exc.code == "VALIDATION_FAILED" and pointer:
            raise FinplanError.validation(exc.message, pointer=pointer + str(exc.details.get("pointer", "")), **{k: v for k, v in exc.details.items() if k != "pointer"}) from None
        raise


def descriptor(name: str) -> dict[str, Any]:
    cls = BASELINES[name]
    return {"strategy": name, "family": cls.family, "param_schema_version": cls.param_schema_version, "makes_predictions": cls.makes_predictions, "has_randomness": cls.has_randomness, "compute_class": "cpu", "control": name in CONTROLS}


def with_controls(specs: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """The requested strategy specs plus every missing control (appended, in control order)."""
    out = [dict(s) for s in specs]
    present = {s.get("name") for s in out}
    added = [c for c in CONTROLS if c not in present]
    out += [{"name": c, "params": {}, "auto_added": True} for c in added]
    return out, added


def require_cpu_instance(instance_type: str, *, strategy: str | None = None, pointer: str = "/instance_type") -> str:
    if is_gpu_instance_type(instance_type):
        raise FinplanError.validation("controls and classical optimizers run on CPU instance types only; GPU instances are not allowed", pointer=pointer, instance_type=instance_type, strategy=strategy)
    return instance_type


def validate_baseline_submission(submission: Mapping[str, Any], env: EnvConfig | None = None) -> None:
    """Check the compute request of a baseline job (``job_type``, ``instance_type``, ``strategies``)."""
    job_type = str(submission.get("job_type", ""))
    if job_type not in BASELINE_JOB_TYPES:
        raise FinplanError.validation("not a baseline job type", pointer="/job_type", job_type=job_type[:64])
    for n, s in enumerate(submission.get("strategies") or []):
        name = s.get("name") if isinstance(s, Mapping) else s
        if name not in BASELINES:
            raise FinplanError.validation("unknown strategy", pointer=f"/strategies/{n}/name")
    it = submission.get("instance_type")
    if it is None:
        return
    require_cpu_instance(str(it))
    if env is not None:
        jt = env.job_type(job_type)
        if jt is not None and (jt.compute_class != "cpu" or str(it) not in jt.instance_types):
            raise FinplanError.validation("instance type is not allowed for this job type", pointer="/instance_type", instance_type=str(it), allowed=list(jt.instance_types))
