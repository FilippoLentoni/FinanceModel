"""``submit_job`` request validation (spec experiment-job-interface, "Submission validation",
"Asynchronous submission mints run_id", "Fixture-backed availability"; JOB-02, JOB-03, JOB-09).

Order of checks (every rejection creates no run and starts no job):

1. caller-supplied ``run_id`` -> ``VALIDATION_FAILED`` (``/run_id``): FinanceModel mints it;
2. any storage location (``s3://``, S3 ARNs, bucket hosts, ``file://``) anywhere in the body ->
   ``VALIDATION_FAILED`` pointing at the field, without echoing the value;
3. ``contract_version`` whose major is not served -> ``UNSUPPORTED_CONTRACT_VERSION`` listing the
   served majors;
4. the contract schema ``core/v1/job-submission.json`` (domain and ``domain_schema_version``
   registered, finance configuration payload, purpose, idempotency key, ...);
5. a caller ``configuration_id`` must equal the computed one;
6. job type: unknown -> ``VALIDATION_FAILED``; announced but not deployed here (or no published job
   definition) -> ``DEPENDENCY_UNAVAILABLE``, ``retryable`` false;
7. strategy unknown -> ``VALIDATION_FAILED``; ``compute_class`` must match the job type;
8. runtime above the job type's ceiling -> ``VALIDATION_FAILED`` (CTL-01).

Purpose authorization (``FORBIDDEN``) and cost/budget checks follow in the service.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from finplan_model.core.config import EnvConfig, JobTypeConfig
from finplan_model.core.errors import ErrorCode, FinplanError
from finplan_model.core.ids import configuration_id
from finplan_model.core.outcome import require_valid
from finplan_model.jobs.strategy_resolver import known_strategies

__all__ = ["PLANNED_JOB_TYPES", "SELECTION_JOB_TYPE", "STRATEGY_JOB_TYPES", "Submission", "find_storage_location", "validate_submission"]

#: Job types announced by later changes (add-learning-and-llm-strategies). Submitting one before it
#: is deployed in an environment gives DEPENDENCY_UNAVAILABLE (JOB-09), not VALIDATION_FAILED.
PLANNED_JOB_TYPES = ("rl_train", "rl_evaluate", "rl_weight_staging", "swarm_mode_a", "swarm_mode_b", "jev_backtest")
#: Offline model selection (decision 27): every family on one snapshot, one SageMaker Training job.
SELECTION_JOB_TYPE = "model_selection"
#: Job types whose configuration names a strategy that must exist.
STRATEGY_JOB_TYPES = ("run_backtest", "run_benchmark", "daily_recommendation")

_STORAGE_RE = re.compile(r"(?i)^\s*(s3|s3a|s3n|gs|file|hdfs|ftp)://|arn:aws[a-z-]*:s3:|\.s3[.-][a-z0-9-]*\.amazonaws\.com|^\s*/(tmp|opt|home|var|mnt)/")
_SEMVER_MAJOR = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")


def _esc(token: Any) -> str:
    return str(token).replace("~", "~0").replace("/", "~1")


def find_storage_location(node: Any, ptr: str = "") -> str | None:
    """JSON pointer of the first value that looks like a storage location, else ``None``."""
    if isinstance(node, str):
        return ptr if _STORAGE_RE.search(node) else None
    if isinstance(node, Mapping):
        for k, v in node.items():
            hit = find_storage_location(v, f"{ptr}/{_esc(k)}")
            if hit is not None:
                return hit
    elif isinstance(node, list):
        for i, v in enumerate(node):
            hit = find_storage_location(v, f"{ptr}/{i}")
            if hit is not None:
                return hit
    return None


@dataclass(frozen=True)
class Submission:
    body: dict[str, Any]
    job_type: JobTypeConfig
    configuration_id: str
    max_runtime_seconds: int
    instance_type: str
    instance_count: int
    dry_run: bool
    synthetic: bool

    @property
    def purpose(self) -> str:
        return str(self.body["purpose"])


def validate_submission(body: Any, cfg: EnvConfig, *, job_definition_published: Any = None) -> Submission:
    """Validate a ``submit_job`` body; ``job_definition_published(job_type) -> bool`` checks deployment."""
    if not isinstance(body, Mapping):
        raise FinplanError.validation("the request body must be a JSON object", pointer="")
    if "run_id" in body:
        raise FinplanError.validation("run_id is minted by FinanceModel and must not be supplied", pointer="/run_id")
    hit = find_storage_location(body)
    if hit is not None:
        raise FinplanError.validation("storage locations are not accepted; inputs are referenced by identifier", pointer=hit)
    served = list(cfg.served_contract_majors)
    cv = body.get("contract_version")
    if isinstance(cv, str):
        m = _SEMVER_MAJOR.match(cv)
        if m and int(m.group(1)) not in served:
            raise FinplanError(ErrorCode.UNSUPPORTED_CONTRACT_VERSION, "the declared contract major is not served", details={"served_contract_majors": served})
    require_valid(dict(body), "job-submission")
    computed = configuration_id(body["configuration"])
    if body.get("configuration_id") not in (None, computed):
        raise FinplanError.validation("configuration_id does not match the canonical configuration", pointer="/configuration_id")
    name = str(body["job_type"])
    jt = cfg.job_type(name)
    if jt is None:
        if name in PLANNED_JOB_TYPES:
            raise FinplanError.dependency_unavailable("this job type is not deployed in this environment", retryable=False, job_type=name)
        raise FinplanError.validation("unknown job type", pointer="/job_type")
    if not jt.deployed or (job_definition_published is not None and not job_definition_published(name)):
        raise FinplanError.dependency_unavailable("this job type is not deployed in this environment", retryable=False, job_type=name)
    payload = body["configuration"]["payload"]
    if name in STRATEGY_JOB_TYPES and str(payload.get("strategy")) not in known_strategies():
        raise FinplanError.validation("unknown strategy", pointer="/configuration/payload/strategy")
    if name in (SELECTION_JOB_TYPE, "recursive_evaluate"):
        # The protocol (splits, grids, seeds) is configuration frozen at submission; the request only
        # names the experiment, the universe and the shared simulation settings.
        if payload.get("strategy") != SELECTION_JOB_TYPE:
            raise FinplanError.validation("model_selection configurations name strategy model_selection", pointer="/configuration/payload/strategy")
        if body["purpose"] not in ("research", "holdout_evaluation"):
            raise FinplanError.validation("model_selection runs with purpose research or holdout_evaluation", pointer="/purpose")
    if name in ("swarm_mode_a", "jev_backtest", "rl_weight_staging"):
        expected = "qwen_swarm" if name == "swarm_mode_a" else "jev" if name == "jev_backtest" else "qwen_weights"
        if payload.get("strategy") != expected:
            raise FinplanError.validation("benchmark configuration does not name its identified strategy", pointer="/configuration/payload/strategy")
        if body["purpose"] not in ("research", "holdout_evaluation"):
            raise FinplanError.validation("identified benchmarks accept research purposes only", pointer="/purpose")
        if name in ("swarm_mode_a", "jev_backtest") and payload.get("rebalance_frequency") != "monthly":
            raise FinplanError.validation("the initial bounded LLM benchmark decides monthly", pointer="/configuration/payload/rebalance_frequency")
    if body.get("compute_class") not in (None, jt.compute_class):
        raise FinplanError.validation("compute_class does not match the job type", pointer="/compute_class")
    runtime = int(body.get("max_runtime_seconds") or jt.default_runtime_seconds)
    if runtime > jt.max_runtime_seconds:
        raise FinplanError.validation("requested runtime is above the job type's maximum", pointer="/max_runtime_seconds", max_runtime_seconds=jt.max_runtime_seconds)
    synthetic = bool(body.get("synthetic") or body["configuration"].get("synthetic"))
    return Submission(dict(body), jt, computed, runtime, jt.default_instance_type, 1, bool(body["dry_run"]), synthetic)
