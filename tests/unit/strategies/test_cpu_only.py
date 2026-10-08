"""BASE-06: controls and classical optimizers run on CPU instance types only; task 4.6."""

from __future__ import annotations

import pytest

from finplan_model.core.config import load_config
from finplan_model.core.errors import FinplanError
from finplan_model.strategies import BASELINES, descriptor, require_cpu_instance, validate_baseline_submission


@pytest.mark.parametrize("instance_type", ["ml.g5.xlarge", "ml.p4d.24xlarge", "ml.g6.2xlarge", "ml.inf2.xlarge", "ml.trn1.2xlarge"])
def test_gpu_instance_for_a_baseline_is_rejected(instance_type):
    with pytest.raises(FinplanError) as ei:
        validate_baseline_submission({"job_type": "run_benchmark", "instance_type": instance_type, "strategies": ["min_variance"]}, load_config("beta"))
    assert ei.value.code == "VALIDATION_FAILED" and ei.value.details["pointer"] == "/instance_type"
    with pytest.raises(FinplanError):
        require_cpu_instance(instance_type, strategy="cash")


def test_cpu_instance_for_a_baseline_is_accepted():
    validate_baseline_submission({"job_type": "run_backtest", "instance_type": "ml.m5.xlarge", "strategies": ["scenario_cvar"]}, load_config("beta"))
    with pytest.raises(FinplanError):  # a CPU type that is not configured for the job type
        validate_baseline_submission({"job_type": "run_backtest", "instance_type": "ml.c5.large"}, load_config("beta"))


def test_every_baseline_declares_cpu_compute():
    for name in BASELINES:
        assert descriptor(name)["compute_class"] == "cpu"
    for env in ("beta", "gamma", "prod"):
        cfg = load_config(env)
        for jt in ("prepare_dataset", "run_backtest", "run_benchmark", "report"):
            assert cfg.job_type(jt).compute_class == "cpu"
