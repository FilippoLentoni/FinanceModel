"""Baseline strategies: controls and classical optimizers (spec baseline-strategies; task group 4).
See ``docs/strategies.md``."""

from .base import BaselineStrategy
from .controls import BuyAndHold, CashControl, EqualWeight
from .estimators import COVARIANCE_ESTIMATORS, MEAN_ESTIMATORS, estimate_covariance, estimate_mean
from .optimizers import SOLVER_STATUS_MAP, MeanVariance, MinVariance, ScenarioCVaR, generate_scenarios
from .outcome import job_outcome
from .registry import BASELINE_JOB_TYPES, BASELINES, CONTROLS, build_strategy, descriptor, require_cpu_instance, validate_baseline_submission, with_controls

__all__ = [
    "BASELINES",
    "BASELINE_JOB_TYPES",
    "BaselineStrategy",
    "BuyAndHold",
    "CONTROLS",
    "COVARIANCE_ESTIMATORS",
    "CashControl",
    "EqualWeight",
    "MEAN_ESTIMATORS",
    "MeanVariance",
    "MinVariance",
    "SOLVER_STATUS_MAP",
    "ScenarioCVaR",
    "build_strategy",
    "descriptor",
    "estimate_covariance",
    "estimate_mean",
    "generate_scenarios",
    "job_outcome",
    "require_cpu_instance",
    "validate_baseline_submission",
    "with_controls",
]
