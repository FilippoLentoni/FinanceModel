"""In-memory fake of the SageMaker Processing and Training API subset the control plane uses.

The offline harness blocks every real (and moto) SageMaker call, so control-plane tests inject this
fake instead. Failures are scripted with :meth:`FakeSageMaker.fail_next` using botocore
``ClientError`` codes (``ResourceLimitExceeded``, ``ThrottlingException`` ...).
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Any

from botocore.exceptions import ClientError

__all__ = ["FakeSageMaker"]


class FakeSageMaker:
    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._failures: deque[tuple[str, str]] = deque()

    def fail_next(self, operation: str, code: str) -> None:
        self._failures.append((operation, code))

    def _maybe_fail(self, operation: str) -> None:
        if self._failures and self._failures[0][0] == operation:
            _, code = self._failures.popleft()
            raise ClientError({"Error": {"Code": code, "Message": f"scripted {code}"}, "ResponseMetadata": {"HTTPStatusCode": 400}}, operation)

    def create_processing_job(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("CreateProcessingJob", copy.deepcopy(kwargs)))
        self._maybe_fail("CreateProcessingJob")
        name = kwargs["ProcessingJobName"]
        if name in self.jobs:
            raise ClientError({"Error": {"Code": "ResourceInUse", "Message": "exists"}}, "CreateProcessingJob")
        self.jobs[name] = {**kwargs, "ProcessingJobStatus": "InProgress"}
        return {"ProcessingJobArn": f"arn:aws:sagemaker:us-east-2:<account-id>:processing-job/{name}"}

    def stop_processing_job(self, ProcessingJobName: str) -> dict[str, Any]:  # noqa: N803 - boto3 casing
        self.calls.append(("StopProcessingJob", {"ProcessingJobName": ProcessingJobName}))
        self._maybe_fail("StopProcessingJob")
        self.jobs[ProcessingJobName]["ProcessingJobStatus"] = "Stopping"
        return {}

    def describe_processing_job(self, ProcessingJobName: str) -> dict[str, Any]:  # noqa: N803
        self.calls.append(("DescribeProcessingJob", {"ProcessingJobName": ProcessingJobName}))
        self._maybe_fail("DescribeProcessingJob")
        job = self.jobs.get(ProcessingJobName)
        if job is None:
            raise ClientError({"Error": {"Code": "ResourceNotFound", "Message": "missing"}}, "DescribeProcessingJob")
        return copy.deepcopy(job)

    def create_training_job(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("CreateTrainingJob", copy.deepcopy(kwargs)))
        self._maybe_fail("CreateTrainingJob")
        name = kwargs["TrainingJobName"]
        if name in self.jobs:
            raise ClientError({"Error": {"Code": "ResourceInUse", "Message": "exists"}}, "CreateTrainingJob")
        self.jobs[name] = {**kwargs, "TrainingJobStatus": "InProgress", "SecondaryStatus": "Starting"}
        return {"TrainingJobArn": f"arn:aws:sagemaker:us-east-2:<account-id>:training-job/{name}"}

    def stop_training_job(self, TrainingJobName: str) -> dict[str, Any]:  # noqa: N803
        self.calls.append(("StopTrainingJob", {"TrainingJobName": TrainingJobName}))
        self._maybe_fail("StopTrainingJob")
        self.jobs[TrainingJobName]["TrainingJobStatus"] = "Stopping"
        return {}

    def describe_training_job(self, TrainingJobName: str) -> dict[str, Any]:  # noqa: N803
        self.calls.append(("DescribeTrainingJob", {"TrainingJobName": TrainingJobName}))
        self._maybe_fail("DescribeTrainingJob")
        job = self.jobs.get(TrainingJobName)
        if job is None or "TrainingJobName" not in job:
            raise ClientError({"Error": {"Code": "ValidationException", "Message": "Requested resource not found."}}, "DescribeTrainingJob")
        return copy.deepcopy(job)

    def set_status(self, name: str, status: str, **extra: Any) -> None:
        """Move a job to ``Completed``, ``Failed``, ``Stopped`` ... (``ExitMessage``, ``FailureReason``)."""
        key = "TrainingJobStatus" if "TrainingJobName" in self.jobs[name] else "ProcessingJobStatus"
        self.jobs[name].update({key: status}, **extra)
