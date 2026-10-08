"""SageMaker Processing and Training Job requests and status mapping (design D1, D2, D5).

Job types run as SageMaker **Processing** jobs unless their configuration says ``sagemaker_job:
training`` (``model_selection``: RL learners run only in SageMaker Training jobs, spec rl-strategies
"Training as SageMaker Training Jobs"). :data:`JOB_KINDS` names the API operations and fields of each
kind; :func:`build_training_request` is the ``CreateTrainingJob`` twin of
:func:`build_processing_request` (same image, entry point, environment, tags and stopping rule; the
mandatory output path is the research workspace's ``scratch/training-output/`` prefix (expires with
the scratch lifecycle; the job role already writes ``scratch/*``), where SageMaker writes only its own
empty model archive: the job writes its results through the run hand-off like every other job).

* :func:`build_processing_request` - the ``CreateProcessingJob`` request for one run attempt:
  digest-pinned image (DEP-03), a stopping condition equal to the run's approved runtime and never
  above the job type's ceiling (CTL-01), CPU instance types only (BASE-06), the contract
  cost-allocation tags ``project``, ``owner-repo``, ``environment``, ``logical-role`` and ``run-id``
  (CTL-09), and only identifiers in the container environment (no storage paths, no secrets).
* :func:`job_name` / :func:`run_id_from_job_name` - deterministic names per attempt, so a retried
  start of the same attempt is idempotent (``ResourceInUse`` means "already started").
* :func:`classify_start_error` - ``quota`` (``ResourceLimitExceeded``: re-queue with backoff, CTL-04),
  ``throttled`` (throttling or 5xx: bounded start retries, CTL-06), ``exists`` or ``fatal``.
* :func:`terminal_outcome` - SageMaker status -> run completion: ``Completed`` -> ``succeeded``
  (``solution_status`` comes from the job's own result file, never from SageMaker), ``Failed`` ->
  ``failed``, ``Stopped`` -> ``cancelled`` when the run asked for it, ``timed_out`` when the
  stopping condition (``MaxRuntimeInSeconds``) fired.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from finplan_contracts import ssm as contract_ssm

from finplan_model.core.config import is_gpu_instance_type
from finplan_model.core.errors import FinplanError

__all__ = [
    "CONTAINER_ENTRYPOINT",
    "JOB_KINDS",
    "TRAINING_OUTPUT_PREFIX",
    "build_job_request",
    "build_training_request",
    "job_kind",
    "job_status",
    "JOB_LOGICAL_ROLE",
    "SAGEMAKER_TERMINAL",
    "build_processing_request",
    "classify_start_error",
    "job_name",
    "run_id_from_job_name",
    "terminal_outcome",
]

#: ``python -m finplan_model.jobs <job_type>`` inside the ``financemodel-cpu`` image.
CONTAINER_ENTRYPOINT = ["python", "-m", "finplan_model.jobs"]
JOB_LOGICAL_ROLE = "research-job"
SAGEMAKER_TERMINAL = ("Completed", "Failed", "Stopped")
_DIGEST_URI = re.compile(r"^[a-z0-9][a-z0-9.\-/:]*@sha256:[0-9a-f]{64}\Z")
_NAME_RE = re.compile(r"^fm-(beta|gamma|prod)-run-([0-9a-hjkmnp-tv-z]{26})-a([0-9]{1,3})\Z")
_MAX_RUNTIME_MARKERS = ("maxruntime", "max runtime", "maxruntimeinseconds", "stopping condition")


TRAINING_OUTPUT_PREFIX = "scratch/training-output"
#: Per SageMaker job kind: client method names and description/event field names.
JOB_KINDS: dict[str, dict[str, str]] = {
    "processing": {"create": "create_processing_job", "describe": "describe_processing_job", "stop": "stop_processing_job", "name": "ProcessingJobName", "status": "ProcessingJobStatus", "start": "ProcessingStartTime", "end": "ProcessingEndTime", "detail_type": "SageMaker Processing Job State Change"},
    "training": {"create": "create_training_job", "describe": "describe_training_job", "stop": "stop_training_job", "name": "TrainingJobName", "status": "TrainingJobStatus", "start": "TrainingStartTime", "end": "TrainingEndTime", "detail_type": "SageMaker Training Job State Change"},
}


def job_kind(run: Mapping[str, Any]) -> str:
    """The SageMaker job kind of a run (runs recorded before the field existed are processing jobs)."""
    kind = str(run.get("sagemaker_job") or "processing")
    return kind if kind in JOB_KINDS else "processing"


def job_status(desc: Mapping[str, Any] | None, kind: str) -> str | None:
    if not desc:
        return None
    status = desc.get(JOB_KINDS[kind]["status"])
    return None if status is None else str(status)


def job_name(environment: str, run_id: str, attempt: int) -> str:
    name = f"fm-{environment}-{run_id.replace('_', '-').lower()}-a{int(attempt)}"
    if len(name) > 63:  # pragma: no cover - run ids are fixed length
        raise ValueError("processing job name too long")
    return name


def run_id_from_job_name(name: str) -> tuple[str, str, int] | None:
    """``(environment, run_id, attempt)`` of a FinanceModel job name, else ``None``."""
    m = _NAME_RE.match(name or "")
    if not m:
        return None
    return m.group(1), "run_" + m.group(2).upper(), int(m.group(3))


def build_processing_request(
    run: Mapping[str, Any],
    *,
    attempt: int,
    image_uri: str,
    role_arn: str,
    run_spec_checksum: str,
    volume_size_gb: int = 10,
) -> dict[str, Any]:
    env = str(run["environment"])
    if not _DIGEST_URI.match(image_uri or ""):
        raise FinplanError.precondition("job definitions must reference images by digest", reason="image_not_digest_pinned", job_type=run["job_type"])
    if not role_arn:
        raise FinplanError.precondition("the job role reference is not published", reason="job_role_missing")
    instance_type = str(run["instance_type"])
    if run.get("compute_class", "cpu") == "cpu" and is_gpu_instance_type(instance_type):
        raise FinplanError.validation("GPU instance types are not allowed for CPU job types", pointer="/instance_type")
    runtime = int(run["max_runtime_seconds"])
    ceiling = int(run["runtime_ceiling_seconds"])
    if not 0 < runtime <= ceiling:
        raise FinplanError.validation("requested runtime is above the job type's maximum", pointer="/max_runtime_seconds", max_runtime_seconds=ceiling)
    tags = contract_ssm.cost_allocation_tags("financemodel", env, JOB_LOGICAL_ROLE, run_id=str(run["run_id"]))
    digest = image_uri.rsplit("@", 1)[1]
    environment = {
        "FINPLAN_ENVIRONMENT": env,
        "FINPLAN_RUN_ID": str(run["run_id"]),
        "FINPLAN_JOB_TYPE": str(run["job_type"]),
        "FINPLAN_CORRELATION_ID": str(run["correlation_id"]),
        "FINPLAN_RUN_SPEC_SHA256": run_spec_checksum,
        "FINPLAN_IMAGE_DIGEST": digest,
    }
    return {
        "ProcessingJobName": job_name(env, str(run["run_id"]), attempt),
        "ProcessingResources": {"ClusterConfig": {"InstanceCount": int(run["instance_count"]), "InstanceType": instance_type, "VolumeSizeInGB": int(volume_size_gb)}},
        "StoppingCondition": {"MaxRuntimeInSeconds": runtime},
        "AppSpecification": {"ImageUri": image_uri, "ContainerEntrypoint": list(CONTAINER_ENTRYPOINT), "ContainerArguments": [str(run["job_type"])]},
        "Environment": environment,
        "NetworkConfig": {"EnableInterContainerTrafficEncryption": True, "EnableNetworkIsolation": False},
        "RoleArn": role_arn,
        "Tags": [{"Key": k, "Value": v} for k, v in tags.items()],
    }


def build_training_request(
    run: Mapping[str, Any],
    *,
    attempt: int,
    image_uri: str,
    role_arn: str,
    run_spec_checksum: str,
    output_bucket: str | None,
    volume_size_gb: int = 10,
) -> dict[str, Any]:
    """``CreateTrainingJob`` for one run attempt: the processing request's image, entry point,
    environment, stopping condition, CPU check and cost tags, as a SageMaker Training job."""
    if not output_bucket:
        raise FinplanError.precondition("the research storage reference is not published (training jobs need an output path)", reason="research_storage_missing")
    proc = build_processing_request(run, attempt=attempt, image_uri=image_uri, role_arn=role_arn, run_spec_checksum=run_spec_checksum, volume_size_gb=volume_size_gb)
    cluster = proc["ProcessingResources"]["ClusterConfig"]
    return {
        "TrainingJobName": proc["ProcessingJobName"],
        "AlgorithmSpecification": {"TrainingImage": image_uri, "TrainingInputMode": "File", "ContainerEntrypoint": list(CONTAINER_ENTRYPOINT), "ContainerArguments": [str(run["job_type"])]},
        "RoleArn": role_arn,
        "OutputDataConfig": {"S3OutputPath": f"s3://{output_bucket}/{TRAINING_OUTPUT_PREFIX}/"},
        "ResourceConfig": {"InstanceType": cluster["InstanceType"], "InstanceCount": cluster["InstanceCount"], "VolumeSizeInGB": cluster["VolumeSizeInGB"]},
        "StoppingCondition": {"MaxRuntimeInSeconds": proc["StoppingCondition"]["MaxRuntimeInSeconds"]},
        "Environment": dict(proc["Environment"]),
        "EnableNetworkIsolation": False,
        "EnableInterContainerTrafficEncryption": False,
        # no Debugger profiling (it would write profiler output and could start rule jobs)
        "ProfilerConfig": {"DisableProfiler": True},
        "Tags": list(proc["Tags"]),
    }


def build_job_request(run: Mapping[str, Any], *, attempt: int, image_uri: str, role_arn: str, run_spec_checksum: str, output_bucket: str | None = None) -> tuple[str, dict[str, Any]]:
    """``(kind, request)`` for the run's SageMaker job kind."""
    kind = job_kind(run)
    if kind == "training":
        return kind, build_training_request(run, attempt=attempt, image_uri=image_uri, role_arn=role_arn, run_spec_checksum=run_spec_checksum, output_bucket=output_bucket)
    return kind, build_processing_request(run, attempt=attempt, image_uri=image_uri, role_arn=role_arn, run_spec_checksum=run_spec_checksum)


def _error_code(exc: BaseException) -> str:
    resp = getattr(exc, "response", None)
    if isinstance(resp, Mapping):
        return str((resp.get("Error") or {}).get("Code") or "")
    return ""


def _http_status(exc: BaseException) -> int:
    resp = getattr(exc, "response", None)
    if isinstance(resp, Mapping):
        return int((resp.get("ResponseMetadata") or {}).get("HTTPStatusCode") or 0)
    return 0


def classify_start_error(exc: BaseException) -> str:
    code = _error_code(exc)
    if code == "ResourceLimitExceeded":
        return "quota"
    if code in ("ThrottlingException", "Throttling", "TooManyRequestsException", "ServiceUnavailable", "InternalFailure", "InternalServerError") or _http_status(exc) >= 500:
        return "throttled"
    if code == "ResourceInUse":
        return "exists"
    if code in ("AccessDeniedException", "AccessDenied"):
        return "denied"
    return "fatal"


def _mentions_max_runtime(desc: Mapping[str, Any]) -> bool:
    text = " ".join(str(desc.get(k) or "") for k in ("FailureReason", "ExitMessage", "SecondaryStatus")).lower()
    return any(m in text for m in _MAX_RUNTIME_MARKERS)


def terminal_outcome(status: str, desc: Mapping[str, Any], *, cancel_requested: bool) -> str | None:
    """Run completion for a SageMaker status, or ``None`` while the job is not terminal."""
    if status == "Completed":
        return "succeeded"
    if status == "Failed":
        return "timed_out" if _mentions_max_runtime(desc) else "failed"
    if status == "Stopped":
        if cancel_requested:
            return "cancelled"
        return "timed_out" if _mentions_max_runtime(desc) or desc.get("_elapsed_reached_limit") else "cancelled"
    return None
